using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{

    [Header("ACCELERATION (0: min, 1: max)")]
    [Range(0f, 1f)]
    public float acc;
    [Header("TURN (-1: left, 1: right)")]
    [Range(-1f, 1f)]
    public float turn;
    [Header("FITNESS")]
    public float fitness = 0f;
    [Header("DISTANCE")]
    public float distance = 0f;

    private bool started = false;
    private bool logged = false;
    private float time = 0f;
    private float avgSpeed = 0f;
    private float distanceMultiplier = 1.5f;
    private float avgSpeedMultiplier = 0.2f;
    private float sensorMultiplier = 0.1f;

    private Vector3 initPosition, initRotation, lastPosition;
    private float flSensor, fSensor, frSensor;
    private NeuralNetworkController network;


    void Awake()
    {
        initPosition = transform.position;
        initRotation = transform.eulerAngles;
    }

    void FixedUpdate()
    {
        if (started)
        {
            CalculateSensors();
            // Debug.Log("flSensor: " + flSensor + " / fSensor: " + fSensor + " / frSensor: " + frSensor);
            lastPosition = transform.position;

            (acc, turn) = network.Run(flSensor, fSensor, frSensor);

            Move(acc, turn);
            time += Time.deltaTime;
            CalculateFitness();

            if (!logged && distance > 300f)
            {
                logged = true;
                network.LogValues();
            }
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == "wall")
        {
            if (fitness == 0f)
            {
                // Debug.LogError("Collision happened with no fitness, ignoring");
            }
            else
            {
                started = false;
                logged = false;
                FindObjectOfType<GeneticAlgorithmController>().CarCrashed(fitness);
            }
        }
    }

    public void StartWithNetwork(NeuralNetworkController currentNetwork)
    {
        network = currentNetwork;

        transform.position = initPosition;
        transform.eulerAngles = initRotation;

        fitness = 0f;
        time = 0f;
        distance = 0f;
        avgSpeed = 0f;
        lastPosition = initPosition;
        started = true;
    }

    private void Move(float v, float h)
    {
        transform.position += transform.TransformDirection(Vector3.Lerp(Vector3.zero, new Vector3(0, 0, v * 10f), 0.02f));
        transform.eulerAngles += new Vector3(0, (h * 45) * 0.02f, 0);
    }

    private void CalculateSensors()
    {
        Vector3 fl = transform.forward - transform.right;
        Vector3 f = transform.forward;
        Vector3 fr = transform.forward + transform.right;

        Ray ray = new Ray(transform.position, fl);
        RaycastHit hit;

        if (Physics.Raycast(ray, out hit)) {
            flSensor = hit.distance / 20;
            Debug.DrawLine(ray.origin, hit.point, Color.red);
        }

        ray.direction = f;
        if (Physics.Raycast(ray, out hit))
        {
            fSensor = hit.distance / 20;
            Debug.DrawLine(ray.origin, hit.point, Color.red);
        }

        ray.direction = fr;
        if (Physics.Raycast(ray, out hit))
        {
            frSensor = hit.distance / 20;
            Debug.DrawLine(ray.origin, hit.point, Color.red);
        }
    }

    private void CalculateFitness()
    {
        distance += Vector3.Distance(transform.position, lastPosition);
        avgSpeed = distance / time;

        float distanceFitness = distance * distanceMultiplier;
        float avgSpeedFitness = avgSpeed * avgSpeedMultiplier;
        float sensorsFitness = (flSensor + fSensor + frSensor) * sensorMultiplier / 3;
        fitness = distanceFitness + avgSpeedFitness + sensorsFitness;
    }

}
