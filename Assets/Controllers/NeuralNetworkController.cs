using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;

using Random = UnityEngine.Random;

public class NeuralNetworkController : MonoBehaviour
{

    public float fitness;

    private Matrix<float> inputLayer = Matrix<float>.Build.Dense(1, 3);
    private Matrix<float> hiddenLayer = Matrix<float>.Build.Dense(1, 2);
    private Matrix<float> outputLayer = Matrix<float>.Build.Dense(1, 2);
    private Matrix<float> inputWeights = Matrix<float>.Build.Dense(3, 2);
    private Matrix<float> hiddenWeights = Matrix<float>.Build.Dense(2, 2);
    private float inputBias;
    private float hiddentBias;

    public void Initialize()
    {
        InitializeFitness();
        InitializeLayers();

        inputWeights.Clear();
        hiddenWeights.Clear();

        RandomizeWeights(inputWeights);
        //inputWeights[0, 0] = 0.8201492f;
        //inputWeights[0, 1] = -1f;
        //inputWeights[1, 0] = -0.8790882f;
        //inputWeights[1, 1] = -0.1724356f;
        //inputWeights[2, 0] = 1f;
        //inputWeights[2, 1] = 0.8973196f;
        RandomizeWeights(hiddenWeights);
        //hiddenWeights[0, 0] = 0.271293f;
        //hiddenWeights[0, 1] = -1f;
        //hiddenWeights[1, 0] = 0.7970545f;
        //hiddenWeights[1, 1] = 0.07970011f;

        inputBias = Random.Range(-1f, 1f);
        //inputBias = -0.6048596f;
        hiddentBias = Random.Range(-1f, 1f);
        //hiddentBias = -0.6229837f;
    }

    public void InitializeFitness()
    {
        fitness = 0f;
    }

    public void InitializeLayers()
    {
        inputLayer.Clear();
        hiddenLayer.Clear();
        outputLayer.Clear();
    }

    private void RandomizeWeights(Matrix<float> weights)
    {
        for (int r = 0; r < weights.RowCount; r++)
        {
            for (int c = 0; c < weights.ColumnCount; c++)
            {
                weights[r, c] = Random.Range(-1f, 1f);
            }
        }
    }

    public void CrossoverWeights(NeuralNetworkController nncA, NeuralNetworkController nncB)
    {
        for (int r = 0; r < inputWeights.RowCount; r++)
        {
            for (int c = 0; c < inputWeights.ColumnCount; c++)
            {
                if (Random.Range(0, 2) == 1)
                {
                    inputWeights[r, c] = nncA.inputWeights[r, c];
                }
                else
                {
                    inputWeights[r, c] = nncB.inputWeights[r, c];
                }
            }
        }

        for (int r = 0; r < hiddenWeights.RowCount; r++)
        {
            for (int c = 0; c < hiddenWeights.ColumnCount; c++)
            {
                if (Random.Range(0, 2) == 1)
                {
                    hiddenWeights[r, c] = nncA.hiddenWeights[r, c];
                }
                else
                {
                    hiddenWeights[r, c] = nncB.hiddenWeights[r, c];
                }
            }
        }
    }

    public void CrossoverBiases(NeuralNetworkController nncA, NeuralNetworkController nncB)
    {
        if (Random.Range(0, 2) == 1)
        {
            inputBias = nncA.inputBias;
            hiddentBias = nncB.hiddentBias;
        }
        else
        {
            inputBias = nncB.inputBias;
            hiddentBias = nncA.hiddentBias;
        }
    }

    public void MutateRandomWeights()
    {
        int r = Random.Range(0, inputWeights.RowCount);
        int c = Random.Range(0, inputWeights.ColumnCount);
        inputWeights[r, c] = Mathf.Clamp(inputWeights[r, c] + Random.Range(-1f, 1f), -1f, 1f);

        r = Random.Range(0, hiddenWeights.RowCount);
        c = Random.Range(0, hiddenWeights.ColumnCount);
        hiddenWeights[r, c] = Mathf.Clamp(hiddenWeights[r, c] + Random.Range(-1f, 1f), -1f, 1f);
    }

    public (float, float) Run(float input1, float input2, float input3)
    {
        inputLayer[0, 0] = input1;
        inputLayer[0, 1] = input2;
        inputLayer[0, 2] = input3;

        inputLayer = inputLayer.PointwiseTanh();
        hiddenLayer = (inputLayer * inputWeights + inputBias).PointwiseTanh();
        outputLayer = (hiddenLayer * hiddenWeights + hiddentBias).PointwiseTanh();

        return (Sigmoid(outputLayer[0, 0]), (float)Math.Tanh(outputLayer[0, 1]));
    }

    public void LogValues()
    {
        string log = "inputWeights: ";
        for (int r = 0; r < inputWeights.RowCount; r++)
        {
            for (int c = 0; c < inputWeights.ColumnCount; c++)
            {
                log += "[" + r + "," + c + "]" + inputWeights[r, c];
                log += ", ";
            }
        }
        Debug.Log(log);

        log = "hiddenWeights: ";
        for (int r = 0; r < hiddenWeights.RowCount; r++)
        {
            for (int c = 0; c < hiddenWeights.ColumnCount; c++)
            {
                log += "[" + r + "," + c + "]" + hiddenWeights[r, c];
                log += ", ";
            }
        }

        Debug.Log(log);
        Debug.Log("inputBias: " + inputBias);
        Debug.Log("hiddentBias: " + hiddentBias);
    }

    private float Sigmoid(float x)
    {
        return 1 / (1 + Mathf.Exp(-x));
    }

}
