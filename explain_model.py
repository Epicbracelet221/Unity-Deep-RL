import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# STEP A — Load the checkpoint and inspect its shape
# ----------------------------------------------------------------------
def flatten_tensors(obj, prefix=""):
    """
    ML-Agents checkpoints often nest the actual weight tensors a few
    dict-levels deep (e.g. under 'policy', 'optimizer', etc). This walks
    the WHOLE loaded object, however deep, and collects every tensor it
    finds, with its full dotted path as the key. Non-tensor values
    (ints, floats, None, strings) are skipped automatically.
    """
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else str(k)
            flat.update(flatten_tensors(v, full_key))
    elif hasattr(obj, "shape"):  # this is a tensor
        flat[prefix] = obj
    # anything else (int, float, str, None, list of non-tensors) is skipped
    return flat


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = flatten_tensors(ckpt)

    print(f"\n--- Found {len(state_dict)} tensors in checkpoint ---")
    for k, v in state_dict.items():
        print(f"  {k:70s} {tuple(v.shape)}")
    print("---------------------------------\n")

    if not state_dict:
        raise RuntimeError(
            "No tensors found at all — the .pt file may be empty or in an "
            "unexpected format. Paste me this script's full output and I'll adjust."
        )
    return state_dict


# ----------------------------------------------------------------------
# STEP B — Rebuild the network architecture to match the checkpoint
# ----------------------------------------------------------------------
class PolicyNet(nn.Module):
    """
    Mirrors ML-Agents' default PPO network:
    obs -> Linear -> activation -> Linear -> activation -> mu head (action mean)
    Layer sizes are read directly from the checkpoint, so this works for
    the 1D, 2D, or Ray Perception models without editing anything.
    """
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super().__init__()
        self.layer0 = nn.Linear(obs_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.act = nn.SiLU()  # ML-Agents' default "swish" activation

    def forward(self, x):
        x = self.act(self.layer0(x))
        x = self.act(self.layer2(x))
        return self.mu_head(x)  # predicted action mean (X, Z movement)


def build_model_from_checkpoint(state_dict):
    # Find the encoder's first layer weight to learn obs_dim & hidden_dim
    body_keys = [k for k in state_dict if "_body_endoder.seq_layers.0.weight" in k]
    mu_keys = [k for k in state_dict if "_continuous_distribution.mu.weight" in k]
    if not body_keys or not mu_keys:
        raise RuntimeError(
            "Couldn't find expected keys. Run this script once just to print "
            "the checkpoint keys, then tell me the exact names and I'll adjust."
        )

    body0_key = body_keys[0]
    hidden_key = body0_key.replace(".0.weight", ".2.weight")
    mu_key = mu_keys[0]

    obs_dim = state_dict[body0_key].shape[1]
    hidden_dim = state_dict[body0_key].shape[0]
    action_dim = state_dict[mu_key].shape[0]

    print(f"Detected: obs_dim={obs_dim}, hidden_dim={hidden_dim}, action_dim={action_dim}\n")

    model = PolicyNet(obs_dim, hidden_dim, action_dim)
    model.layer0.weight.data = state_dict[body0_key].clone()
    model.layer0.bias.data = state_dict[body0_key.replace("weight", "bias")].clone()
    model.layer2.weight.data = state_dict[hidden_key].clone()
    model.layer2.bias.data = state_dict[hidden_key.replace("weight", "bias")].clone()
    model.mu_head.weight.data = state_dict[mu_key].clone()
    model.mu_head.bias.data = state_dict[mu_key.replace("weight", "bias")].clone()
    model.eval()
    return model, obs_dim, action_dim


# ----------------------------------------------------------------------
# STEP C — Give the model observations to react to (real where possible)
# ----------------------------------------------------------------------
def infer_obs_structure(obs_dim):
    """
    Figures out how many 'manual' observations (rel_x, rel_y, rel_z) vs.
    ray-sensor blocks (4 floats each: hitTarget, hitWall, hasHit, distance)
    make up the total obs_dim. Tries 3 manual obs first (confirmed from
    AgentController.cs: sensor.AddObservation(Vector3) adds x,y,z), then
    2, then 1, then 0 — whichever cleanly divides the rest into ray
    blocks of 4. Falls back to "unknown" if nothing divides evenly.
    """
    for n_manual in [3, 2, 1, 0]:
        remainder = obs_dim - n_manual
        if remainder >= 0 and remainder % 4 == 0:
            return n_manual, remainder // 4
    return None, None  # couldn't infer a clean structure


def make_feature_labels(obs_dim):
    n_manual, n_rays = infer_obs_structure(obs_dim)
    if n_manual is None:
        return [f"obs_{i}" for i in range(obs_dim)]

    # Confirmed order from CollectObservations(): sensor.AddObservation(relativePosition)
    # adds (x, y, z) in that exact order.
    manual_names = ["rel_x_to_target", "rel_y_to_target", "rel_z_to_target"][:n_manual]
    labels = list(manual_names)
    for r in range(n_rays):
        name = f"Ray{r + 1}"
        labels += [f"{name}_hitTarget", f"{name}_hitWall", f"{name}_hasHit", f"{name}_distance"]

    assert len(labels) == obs_dim, "Label count mismatch — this should never happen now."
    return labels


def load_real_observations(csv_path, feature_labels):
    """
    Reads obs_log.csv. Two possible formats, auto-detected by column names:

    FULL (preferred — from the updated AgentController.cs that casts real
    rays): every name in feature_labels (rel_x_to_target -> 'rel_x',
    Ray6_hitWall -> 'Ray6_hitWall', etc.) is present as a CSV column.
    Returns a complete (N, obs_dim) array of 100% real observations.

    PARTIAL (from the earlier logging version): only rel_x/rel_y/rel_z were
    logged. Returns an (N, 3) array; the caller fills in the remaining ray
    dimensions synthetically.

    Note the CSV uses short names (rel_x, rel_y, rel_z) while our internal
    labels use long names (rel_x_to_target, ...) for the first 3 columns —
    this function maps between them.
    """
    import csv as csv_module
    with open(csv_path, newline="") as f:
        reader = csv_module.DictReader(f)
        available_cols = set(reader.fieldnames)
        rows = list(reader)

    short_to_long = {"rel_x": "rel_x_to_target", "rel_y": "rel_y_to_target", "rel_z": "rel_z_to_target"}
    csv_column_for_label = {}
    for label in feature_labels:
        if label in short_to_long.values():
            short = [k for k, v in short_to_long.items() if v == label][0]
            csv_column_for_label[label] = short
        else:
            csv_column_for_label[label] = label

    full_available = all(csv_column_for_label[label] in available_cols for label in feature_labels)

    if full_available:
        data = np.array(
            [[float(row[csv_column_for_label[label]]) for label in feature_labels] for row in rows],
            dtype=np.float32,
        )
        print(f"Loaded FULL real observations ({data.shape[1]} dims, 100% real — including real ray-sensor "
              f"readings) for {len(data)} steps from {csv_path}\n")
        return data, True
    else:
        manual_cols = [c for c in ["rel_x", "rel_y", "rel_z"] if c in available_cols]
        data = np.array([[float(row[c]) for c in manual_cols] for row in rows], dtype=np.float32)
        print(f"Loaded PARTIAL real observations (only {manual_cols} found — this CSV predates real "
              f"ray-sensor logging) for {len(data)} steps from {csv_path}")
        print(f"  rel_x range: [{data[:,0].min():.2f}, {data[:,0].max():.2f}]")
        if data.shape[1] > 1:
            print(f"  rel_z range: [{data[:,-1].min():.2f}, {data[:,-1].max():.2f}]")
        print()
        return data, False


def make_observation_batch(obs_dim, n_samples=500, seed=0, real_full_data=None, real_manual_data=None):
    """
    Builds the observation batch fed into the network for saliency.
    Priority order:
      1. real_full_data provided -> sample complete real rows directly (best).
      2. real_manual_data provided -> real rel_x/y/z + synthetic ray blocks.
      3. neither -> fully synthetic (least faithful, but still runs).
    """
    rng = np.random.default_rng(seed)

    if real_full_data is not None:
        idx = rng.integers(0, len(real_full_data), n_samples)
        obs = real_full_data[idx].astype(np.float32)
        return torch.tensor(obs, requires_grad=True)

    obs = np.zeros((n_samples, obs_dim), dtype=np.float32)
    n_manual, n_rays = infer_obs_structure(obs_dim)
    if n_manual is None:
        obs[:, :] = rng.uniform(-1, 1, (n_samples, obs_dim))
        return torch.tensor(obs, requires_grad=True)

    if real_manual_data is not None and real_manual_data.shape[1] >= n_manual:
        idx = rng.integers(0, len(real_manual_data), n_samples)
        obs[:, :n_manual] = real_manual_data[idx, :n_manual]
    else:
        for m in range(n_manual):
            obs[:, m] = rng.uniform(-10, 10, n_samples)

    ray_start = n_manual
    for r in range(n_rays):
        base = ray_start + r * 4
        hit_target = rng.integers(0, 2, n_samples)
        hit_wall = np.where(hit_target == 1, 0, rng.integers(0, 2, n_samples))
        has_hit = np.maximum(hit_target, hit_wall)
        dist = np.where(has_hit == 1, rng.uniform(0.1, 1.0, n_samples), 1.0)
        obs[:, base + 0] = hit_target
        obs[:, base + 1] = hit_wall
        obs[:, base + 2] = has_hit
        obs[:, base + 3] = dist
    return torch.tensor(obs, requires_grad=True)


# ----------------------------------------------------------------------
# STEP D — Compute saliency: gradient of the action w.r.t. each input
# ----------------------------------------------------------------------
def compute_saliency(model, obs_batch):
    output = model(obs_batch)             # shape: [batch, action_dim]
    # Summing the whole output tensor and calling backward() once is safe here:
    # each row of obs_batch only affects its own row of output (no cross-batch
    # mixing happens inside a standard feed-forward net), so the resulting
    # .grad still gives correct PER-SAMPLE gradients, just computed in one pass.
    output.sum().backward()
    grads = obs_batch.grad.detach().numpy()
    importance = np.mean(np.abs(grads), axis=0)   # average absolute influence per input
    return importance


# ----------------------------------------------------------------------
# STEP E — Plot the results
# ----------------------------------------------------------------------
def plot_importance(labels, importance, out_path="feature_importance.png"):
    order = np.argsort(importance)[::-1]
    labels_sorted = [labels[i] for i in order]
    values_sorted = importance[order]

    plt.figure(figsize=(9, max(4, len(labels) * 0.28)))
    colors = plt.cm.viridis(values_sorted / values_sorted.max())
    plt.barh(labels_sorted[::-1], values_sorted[::-1], color=colors[::-1])
    plt.xlabel("Average |gradient| — influence on movement decision")
    plt.title("Which observations drive the agent's decisions?\n(Saliency over the trained policy)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved chart to: {out_path}")


# ----------------------------------------------------------------------
# STEP F — Aggregate importance per ray (sum its 4 sub-features)
# ----------------------------------------------------------------------
def aggregate_by_ray(labels, importance):
    """
    Groups every 'RayN_xxx' label together and sums their importance,
    giving one number per physical ray direction — much easier to reason
    about than 36 individual sub-feature scores. Non-ray labels
    (rel_x_to_target, extra_manual_obs, etc.) are kept as their own entries.
    """
    groups = {}
    for label, val in zip(labels, importance):
        if label.startswith("Ray") and "_" in label:
            ray_name = label.split("_")[0]   # e.g. "Ray6_hitWall" -> "Ray6"
        else:
            ray_name = label                  # manual obs stay ungrouped
        groups[ray_name] = groups.get(ray_name, 0.0) + val
    # Sort by importance, descending
    return dict(sorted(groups.items(), key=lambda kv: kv[1], reverse=True))


def plot_ray_summary(ray_importance, out_path="feature_importance_by_ray.png"):
    names = list(ray_importance.keys())
    values = list(ray_importance.values())

    plt.figure(figsize=(8, max(4, len(names) * 0.35)))
    colors = plt.cm.plasma(np.array(values) / max(values))
    plt.barh(names[::-1], values[::-1], color=colors[::-1])
    plt.xlabel("Summed importance across all sub-features")
    plt.title("Which ray (or manual obs) matters most overall?\n(Sub-features grouped per ray)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    print(f"Saved chart to: {out_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python explain_model.py <path_to_checkpoint.pt> [optional: path_to_obs_log.csv]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None

    state_dict = load_checkpoint(ckpt_path)
    model, obs_dim, action_dim = build_model_from_checkpoint(state_dict)
    labels = make_feature_labels(obs_dim)

    real_full_data, real_manual_data = None, None
    if csv_path:
        data, is_full = load_real_observations(csv_path, labels)
        if is_full:
            real_full_data = data
        else:
            real_manual_data = data
    else:
        print("No CSV provided — using synthetic values for everything.\n"
              "(Pass obs_log.csv as a 2nd argument to use real logged data instead.)\n")

    obs_batch = make_observation_batch(
        obs_dim, n_samples=500, real_full_data=real_full_data, real_manual_data=real_manual_data
    )

    importance = compute_saliency(model, obs_batch)

    print("Feature importance ranking:")
    for i in np.argsort(importance)[::-1]:
        print(f"  {labels[i]:20s} {importance[i]:.4f}")

    plot_importance(labels, importance)

    print("\nPer-ray summary (sub-features grouped):")
    ray_importance = aggregate_by_ray(labels, importance)
    for name, val in ray_importance.items():
        print(f"  {name:20s} {val:.4f}")

    plot_ray_summary(ray_importance)