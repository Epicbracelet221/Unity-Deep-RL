# 🚀 Autonomous Navigation with Deep Reinforcement Learning
### A Unity ML-Agents Sandbox (Mars Rover Prototype)

![Unity](https://img.shields.io/badge/Engine-Unity_2022.3-black?logo=unity&style=for-the-badge)
![ML-Agents](https://img.shields.io/badge/ML--Agents-Toolkit-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-yellow?logo=python&style=for-the-badge)
![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C?logo=pytorch&style=for-the-badge)

---

## 🧠 Overview

This repository features a **Deep Reinforcement Learning (DRL)** agent trained to perform autonomous navigation and target acquisition in a bounded 3D environment. Built using the **Unity ML-Agents Toolkit**, this project serves as a foundational prototype for autonomous rover exploration (akin to Mars Rover navigation). 

The agent learns purely through trial and error, utilizing **Proximal Policy Optimization (PPO)** to map hybrid sensory inputs (simulated LiDAR and GPS) to continuous physical control outputs (tank-style locomotion).

---

## 🔬 Technical Architecture

### 1. Hybrid Perception System (Observation Space)
The agent utilizes a multimodal observation space to understand its environment:
* **Ray Perception Sensors (Simulated LiDAR):** 
  * 7 raycasts spread across a 220° horizontal field of view.
  * Dynamically detects bounding walls and science targets.
  * Injects 28 continuous values into the neural network (4 floats per ray: target hit, wall hit, hit boolean, normalized distance).
* **Local Vector Observations (Simulated GPS/Compass):** 
  * Transforms the global target coordinate into the agent's **local coordinate space** using `InverseTransformPoint`.
  * Provides 2 continuous values (normalized local X and Z) to ensure the agent always knows the target's relative bearing, even when occluded from the raycasts.

### 2. Continuous Control (Action Space)
The agent operates via a continuous action space tailored for **kinematic tank controls**:
* **Action [0] (Rotation):** Controls angular velocity around the Y-axis.
* **Action [1] (Thrust):** Controls forward translation along the local Z-axis via Rigidbody physics (`rb.MovePosition`).

### 3. Reward Function Engineering
The reward topology is designed to prevent local optima and encourage time-efficient routing:
* **Sparse Terminal Rewards:** 
  * `+5.0` for successfully intersecting the target.
  * `-2.0` penalty for boundary wall collisions.
* **Dense Shaping Rewards:** 
  * `-0.001` existential time penalty per step (encourages trajectory optimization).
  * `-0.001 * distance` proximity penalty (provides a dense gradient towards the target in early training phases).

### 4. Parallelized Training Regimen
To maximize sample efficiency and decorrelate batches, the training scene utilizes **21 parallel environments**. This architecture dramatically increases experience collection throughput, allowing the PPO algorithm to converge significantly faster on a stable policy.

---

## ⚙️ Hyperparameter Configuration (PPO)

The agent is trained using a highly tuned Proximal Policy Optimization setup:

* **Batch Size:** 1024
* **Buffer Size:** 10,240
* **Learning Rate:** 3e-4 (Linear Decay)
* **Hidden Units:** 128 (2 Layers)
* **Max Steps:** 500,000
* **Time Scale:** 20x (Accelerated Simulation)

---

## 📦 Setup & Reproduction

### Prerequisites
* Unity 2022.3.x LTS
* Python 3.10+
* ML-Agents Toolkit (Release 20+)

### Running Inference (Pre-trained Model)
1. Open `SampleScene` in Unity.
2. Select the `Agent` prefab.
3. Ensure the **Behavior Type** in `Behavior Parameters` is set to `Default`.
4. Press **Play** in the Unity Editor to watch the ONNX model control the agent in real-time.

### Initiating a Training Run
1. Activate the Python virtual environment:
   ```bash
   MLvenv\Scripts\activate
   ```
2. Launch the ML-Agents training process:
   ```bash
   mlagents-learn results/MoreMovements1/configuration.yaml --run-id=NewTrainingRun
   ```
3. Press **Play** in the Unity Editor to commence simulation.
4. Monitor metrics via TensorBoard:
   ```bash
   tensorboard --logdir results
   ```

---

## 🔮 Future Roadmap

<<<<<<< HEAD
* **Multi-Target Acquisition:** Transitioning from single-pellet retrieval to dynamic multi-waypoint navigation.
* **Intrinsic Curiosity Module (ICM):** Implementing curiosity-driven exploration to handle extremely sparse reward topologies.
* **Dynamic Obstacle Avoidance:** Introducing procedural static and moving obstacles to stress-test the Ray Perception module.
* **Curriculum Learning:** Incrementally scaling arena size and target distance based on the agent's ELO/success rate.
=======
```bash
# Activate virtual environment
MLvenv\Scripts\activate

# Start training
mlagents-learn --run-id=DeepRlRun1
```

Then press **Play ▶️ in Unity** to begin training.

---

## 📈 Project Status

> ⚠️ This is currently a **base project setup**

The repository contains the foundational implementation.
It will be continuously updated with improvements and new features.

---

## 🔮 Future Plans

* Improved reward engineering
* Faster and more stable training
* Advanced agent behaviors
* Complex environments
* TensorBoard integration
* Model optimization for inference

---

## 🤝 Updates

This repository will evolve as the project develops.
New features, improvements, and experiments will be pushed regularly.

---

## 📄 License

This project is intended for learning and experimentation.
>>>>>>> f4ed831d7cf404bc294ee1a19b84353213931449
