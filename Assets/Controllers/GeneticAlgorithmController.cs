using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Random = UnityEngine.Random;

public class GeneticAlgorithmController : MonoBehaviour
{

    [Header("POPULATION QUANTITY")]
    public int populationQuantity = 20;
    [Header("CURRENT GENERATION")]
    public int generationNo = 1;
    [Header("CURRENT GENOME")]
    public int genomeNo = 1;

    private int topGenomes = 2;
    private float mutationRate = 0.1f;
    private NeuralNetworkController[] population;
    private NeuralNetworkComparer neuralNetworkComparer = new NeuralNetworkComparer();

    private void Start()
    {
        InitializePopulation();
        StartCarWithCurrentGenome();
    }

    private void InitializePopulation()
    {
        population = new NeuralNetworkController[populationQuantity];

        for (int i = 0; i < population.Length; i++)
        {
            population[i] = gameObject.AddComponent<NeuralNetworkController>();
            population[i].Initialize();
        }
    }

    private void InitializeNextPopulation(NeuralNetworkController[] selectedGenomes)
    {
        generationNo++;
        genomeNo = 1;

        for (int i = 0; i < population.Length; i++)
        {
            population[i].InitializeFitness();
            population[i].InitializeLayers();
        }

        Crossover(selectedGenomes);
        Mutation();
    }

    private void StartCarWithCurrentGenome()
    {
        FindObjectOfType<CarController>().StartWithNetwork(population[genomeNo - 1]);
    }

    public void CarCrashed(float fitness)
    {
        population[genomeNo - 1].fitness = fitness;

        if (genomeNo < populationQuantity)
        {
            genomeNo++;
            StartCarWithCurrentGenome();
        }
        else
        {
            Array.Sort(population, neuralNetworkComparer);

            Debug.Log(
                "-- GENERATION #" + generationNo + " --\n" +
                "MAX AND MIN FITNESS: " + population[0].fitness + " >>> " + population[populationQuantity - 1].fitness
            );

            NeuralNetworkController[] selectedGenomes = RouletteWheelSelection();
            InitializeNextPopulation(selectedGenomes);
            StartCarWithCurrentGenome();
        }
    }

    private NeuralNetworkController[] RouletteWheelSelection()
    {
        float allFitness = 0f;
        for (int i = 0; i < population.Length; i++)
        {
            allFitness += population[i].fitness;
        }

        float[] normalizedFitness = new float[populationQuantity];
        float[] accumulatedFitness = new float[populationQuantity];
        float allPreviousNormalizedFitness = 0f;
        for (int i = population.Length - 1; i >= 0; i--)
        {
            normalizedFitness[i] = population[i].fitness / allFitness;
            accumulatedFitness[i] = normalizedFitness[i] + allPreviousNormalizedFitness;
            allPreviousNormalizedFitness += normalizedFitness[i];
        }

        NeuralNetworkController[] selectedGenomes = new NeuralNetworkController[populationQuantity];
        for (int i = 0; i < topGenomes; i++)
        {
            selectedGenomes[i] = population[i];
        }

        int selectedIndex = topGenomes;
        float selectionTreshold;
        while (selectedIndex < populationQuantity) {
            selectionTreshold = Random.Range(0f, 1f);
            for (int i = accumulatedFitness.Length - 1; i >= 0; i--)
            {
                if (accumulatedFitness[i] >= selectionTreshold) {
                    selectedGenomes[selectedIndex] = population[i];
                    selectedIndex++;
                    break;
                }
                else if (i == 0)
                {
                    Debug.LogError("Genome not selected with threshold: " + selectionTreshold + ", rounding problem?");
                }
            }
        }

        Array.Sort(selectedGenomes, neuralNetworkComparer);

        return selectedGenomes;
    }

    private void Crossover(NeuralNetworkController[] selectedGenomes)
    {
        for (int i = 0; i < selectedGenomes.Length - 1; i++)
        {
            NeuralNetworkController parentA = selectedGenomes[i];
            NeuralNetworkController parentB = selectedGenomes[Random.Range(0, populationQuantity)];
            population[i].CrossoverWeights(parentA, parentB);
            population[i].CrossoverBiases(parentA, parentB);
        }
    }

    private void Mutation()
    {
        for (int i = 0; i < population.Length - 1; i++)
        {
            if (mutationRate > Random.Range(0f, 1f))
            {
                population[i].MutateRandomWeights();
            }
        }
    }

}

public class NeuralNetworkComparer : IComparer<NeuralNetworkController>
{
    public int Compare(NeuralNetworkController nncA, NeuralNetworkController nncB)
    {
        return nncB.fitness.CompareTo(nncA.fitness);
    }
}
