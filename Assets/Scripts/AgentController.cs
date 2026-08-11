using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.IO;

public class AgentController : Agent
{
    [SerializeField] private Transform target;
    [SerializeField] private float moveSpeed = 5f;
    [SerializeField] private bool logObservations = false;   // only tick this ON for ONE agent
    [Header("Real Ray-Sensor Logging (for explainability)")]    
    [SerializeField] private Transform raySensorOrigin;       // drag the "Ray" child object here
    [SerializeField] private string[] detectableTags = { "Target", "Wall" };  // CONFIRM order matches Inspector
    [SerializeField] private int raysPerDirection = 4;
    [SerializeField] private float maxRayDegrees = 51.6f;
    [SerializeField] private float sphereCastRadius = 0.34f;
    [SerializeField] private float rayLength = 20f;
    [SerializeField] private LayerMask rayLayerMask;           // set in Inspector to match your sensor's mask

    private Rigidbody rb;
    private StreamWriter logWriter;
    private int stepCounter = 0;

    public override void Initialize()
        {
            rb = GetComponent<Rigidbody>();

            if (logObservations)
            {
                string path = Path.Combine(Application.dataPath, "..", "obs_log.csv");
                logWriter = new StreamWriter(path, false); // false = overwrite each run
                var header = new System.Text.StringBuilder("step,rel_x,rel_y,rel_z");
                for (int r = 1; r <= 2 * raysPerDirection + 1; r++)
                    header.Append($",Ray{r}_hitTarget,Ray{r}_hitWall,Ray{r}_hasHit,Ray{r}_distance");
                header.Append(",action_rotate,action_forward");
                logWriter.WriteLine(header.ToString());
                Debug.Log("Logging observations to: " + path);
            }
        }

    public override void OnEpisodeBegin()
    {
        //agent
        transform.localPosition = new Vector3(Random.Range(6f,-6f),1.1f ,Random.Range(6f,-6f));

        //pellet
        target.localPosition = new Vector3(Random.Range(6f,-6f),1.1f ,Random.Range(6f,-6f));



    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 relativePosition = target.localPosition - transform.localPosition;
        sensor.AddObservation(relativePosition);
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        float moveRotate = actions.ContinuousActions[0];
        float moveForward = actions.ContinuousActions[1];

        rb.MovePosition(transform.position + transform.forward * moveForward * moveSpeed * Time.deltaTime);
        transform.Rotate(0f, moveRotate * moveSpeed, 0f , Space.Self);

        if (logObservations && logWriter != null)
        {
            Vector3 rel = target.localPosition - transform.localPosition;
            var row = new System.Text.StringBuilder();
            row.Append(stepCounter).Append(',').Append(rel.x).Append(',').Append(rel.y).Append(',').Append(rel.z);

            float[] angles = GetRayAngles(raysPerDirection, maxRayDegrees);
            foreach (float angle in angles)
            {
                var (ht, hw, hh, dist) = CastOneRay(angle);
                row.Append(',').Append(ht).Append(',').Append(hw).Append(',').Append(hh).Append(',').Append(dist);
            }

            row.Append(',').Append(moveRotate).Append(',').Append(moveForward);
            logWriter.WriteLine(row.ToString());
            stepCounter++;
        }
        /*
        Vector3 velocity = new Vector3(moveX, 0f, moveZ);
        velocity = velocity.normalized * Time.deltaTime * moveSpeed;

        transform.localPosition += velocity;
        */

        AddReward(-0.001f); //penalty per step to encourage faster solutions

        float distance = Vector3.Distance(transform.localPosition, target.localPosition);
        AddReward(-distance * 0.001f); //small penalty based on distance to encourage getting closer to the target
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> continuousActions = actionsOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Horizontal");
        continuousActions[1] = Input.GetAxisRaw("Vertical");
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.tag == "Target")
        {
            AddReward(5f);
            EndEpisode();
        }
        else if (other.CompareTag("Wall"))
        {
            SetReward(-2f);
            EndEpisode();
        }
    }

    // Reproduces ML-Agents' own ray-angle formula, so our logged rays match
// exactly what the trained sensor actually saw.
private float[] GetRayAngles(int raysPerDir, float maxDegrees)
{
    var angles = new float[2 * raysPerDir + 1];
    float delta = maxDegrees / raysPerDir;
    angles[0] = 90f; // center ray
    for (int i = 0; i < raysPerDir; i++)
    {
        angles[2 * i + 1] = 90f - (i + 1) * delta; // right side
        angles[2 * i + 2] = 90f + (i + 1) * delta; // left side
    }
    return angles;
}

// Casts ONE ray at the given angle and returns the same 4 values
// ML-Agents' Ray Perception Sensor would report: hitTarget, hitWall, hasHit, distance
private (float hitTarget, float hitWall, float hasHit, float distance) CastOneRay(float angleDegrees)
{
    float rad = angleDegrees * Mathf.Deg2Rad;
    Vector3 localDir = new Vector3(Mathf.Cos(rad), 0f, Mathf.Sin(rad));
    Vector3 worldDir = raySensorOrigin.TransformDirection(localDir);
    Vector3 origin = raySensorOrigin.position;

    float hitTarget = 0f, hitWall = 0f, hasHit = 0f, distance = 1f; // default = nothing hit

    RaycastHit hit;
    bool didHit = sphereCastRadius > 0f
        ? Physics.SphereCast(origin, sphereCastRadius, worldDir, out hit, rayLength, rayLayerMask)
        : Physics.Raycast(origin, worldDir, out hit, rayLength, rayLayerMask);

    if (didHit)
    {
        hasHit = 1f;
        distance = hit.distance / rayLength;
        if (hit.collider.CompareTag(detectableTags[0])) hitTarget = 1f;
        else if (detectableTags.Length > 1 && hit.collider.CompareTag(detectableTags[1])) hitWall = 1f;
    }
    return (hitTarget, hitWall, hasHit, distance);
}

    private void OnApplicationQuit()
    {
        if (logWriter != null)
        {
            logWriter.Close();
        }
    }
}
