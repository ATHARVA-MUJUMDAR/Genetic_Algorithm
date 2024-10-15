#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

const int POP_SIZE = 100;          // Population size
const int N = 50;                  // Length of vector x (genes per chromosome)
const int M = 20;                  // Length of vector y
const int MAX_GEN = 1000;          // Maximum number of generations
const double CROSSOVER_RATE = 0.7; // Crossover probability
const double MUTATION_RATE = 0.01; // Mutation probability
const int K = 5;                   // Number of non-zero entries in x
const double lambda1 = 0.1;        // Regularization parameter for L1 norm
const double lambda2 = 0.01;       // Regularization parameter for L2 norm

// Random number generator for Gaussian and uniform distributions
random_device rd;
mt19937 gen(rd());
normal_distribution<> H_dist(0.0, 1.0);    // Mean 0, Variance 1 for H
normal_distribution<> x_dist(0.05, 1.0);   // Mean 0.05, Variance 1 for non-zero x
uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution for random numbers between 0 and 1

// Function to generate a random sparse vector x with Gaussian non-zero values
vector<double> generateSparseVector(int N, int K)
{
    vector<double> x(N, 0.0);
    for (int i = 0; i < K; i++)
    {
        int index = rand() % N;
        x[index] = x_dist(gen); // Random non-zero value from N(0.05, 1)
    }
    return x;
}

// Original cost function: J(x) = ||y - Hx||_2^2
double costFunction(const vector<double> &y, const vector<vector<double>> &H, const vector<double> &x)
{
    double error = 0.0;

    // Compute the residual y - Hx
    vector<double> residual(y.size(), 0.0);
    for (int i = 0; i < y.size(); ++i)
    {
        residual[i] = y[i];
        for (int j = 0; j < x.size(); ++j)
        {
            residual[i] -= H[i][j] * x[j];
        }
        error += residual[i] * residual[i]; // ||y - Hx||_2^2
    }

    // Return the cost (reconstruction error)
    return error;
}

// Elastic Net cost function: J(x) = ||y - Hx||_2^2 + λ1 ||x||_1 + λ2 ||x||_2^2
double elasticNetCostFunction(const vector<double> &y, const vector<vector<double>> &H, const vector<double> &x)
{
    double error = costFunction(y, H, x); // ||y - Hx||_2^2

    // Add L1 regularization term (sparsity-promoting)
    double l1_norm = 0.0;
    for (double xi : x)
    {
        l1_norm += abs(xi); // ||x||_1
    }

    // Add L2 regularization term (stability-promoting)
    double l2_norm = 0.0;
    for (double xi : x)
    {
        l2_norm += xi * xi; // ||x||_2^2
    }

    // Return total Elastic Net cost
    return error + lambda1 * l1_norm + lambda2 * l2_norm;
}

// Selection operator (tournament selection)
vector<double> tournamentSelection(const vector<vector<double>> &population, const vector<double> &fitness, const vector<double> &y, const vector<vector<double>> &H, bool useElasticNet)
{
    int tournamentSize = 3;
    vector<double> best = population[rand() % POP_SIZE];

    for (int i = 1; i < tournamentSize; i++)
    {
        int randomIndex = rand() % POP_SIZE;
        double cost = useElasticNet ? elasticNetCostFunction(y, H, population[randomIndex]) : costFunction(y, H, population[randomIndex]);
        if (fitness[randomIndex] < cost)
        {
            best = population[randomIndex];
        }
    }

    return best;
}

// Crossover operator (single-point crossover)
void crossover(vector<double> &parent1, vector<double> &parent2)
{
    if (dis(gen) < CROSSOVER_RATE)
    {
        int point = rand() % N;
        for (int i = point; i < N; i++)
        {
            swap(parent1[i], parent2[i]);
        }
    }
}

// Mutation operator
void mutate(vector<double> &chromosome)
{
    for (int i = 0; i < N; i++)
    {
        if (dis(gen) < MUTATION_RATE)
        {
            chromosome[i] += dis(gen) - 0.5; // Small random mutation
        }
    }
}

// Genetic Algorithm
vector<double> geneticAlgorithm(const vector<double> &y, const vector<vector<double>> &H, bool useElasticNet)
{
    // Step 1: Initialize population
    vector<vector<double>> population(POP_SIZE, vector<double>(N));
    vector<double> fitness(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++)
    {
        population[i] = generateSparseVector(N, K);
        fitness[i] = useElasticNet ? elasticNetCostFunction(y, H, population[i]) : costFunction(y, H, population[i]);
    }

    // Step 2: Evolution loop
    for (int gen = 0; gen < MAX_GEN; gen++)
    {
        vector<vector<double>> newPopulation(POP_SIZE);

        // Step 3: Selection, Crossover, and Mutation
        for (int i = 0; i < POP_SIZE; i += 2)
        {
            vector<double> parent1 = tournamentSelection(population, fitness, y, H, useElasticNet);
            vector<double> parent2 = tournamentSelection(population, fitness, y, H, useElasticNet);
            crossover(parent1, parent2);
            mutate(parent1);
            mutate(parent2);
            newPopulation[i] = parent1;
            newPopulation[i + 1] = parent2;
        }

        // Step 4: Evaluate new population
        for (int i = 0; i < POP_SIZE; i++)
        {
            fitness[i] = useElasticNet ? elasticNetCostFunction(y, H, newPopulation[i]) : costFunction(y, H, newPopulation[i]);
        }

        population = newPopulation;

        // Step 5: Check for convergence (can implement stopping condition here)
        if (gen % 100 == 0)
        {
            cout << "Generation " << gen << ", Best Cost: " << *min_element(fitness.begin(), fitness.end()) << endl;
        }
    }

    // Step 6: Return the best solution found
    int bestIndex = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    return population[bestIndex];
}

int main()
{
    freopen("output.txt", "w", stdout);
    // Randomly initialize dictionary matrix H and measurement vector y
    vector<vector<double>> H(M, vector<double>(N));
    vector<double> y(M, 0.0);

    // Generate random Gaussian dictionary matrix H
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            H[i][j] = H_dist(gen); // H follows N(0, 1)
        }
    }

    // Generate random measurement vector y (noisy observation)
    for (int i = 0; i < M; i++)
    {
        y[i] = dis(gen); // Example random measurement, can replace with real data
    }

    // Apply Genetic Algorithm with original cost function
    cout << "Running GA with original cost function..." << endl;
    vector<double> bestSolutionOriginal = geneticAlgorithm(y, H, false);

    // Output the best solution for the original cost function
    cout << "Best Solution (Original Cost Function):" << endl;
    for (double xi : bestSolutionOriginal)
    {
        cout << xi << " ";
    }
    cout << endl;

    // Apply Genetic Algorithm with Elastic Net cost function
    cout << "Running GA with Elastic Net cost function..." << endl;
    vector<double> bestSolutionElasticNet = geneticAlgorithm(y, H, true);

    // Output the best solution for the Elastic Net cost function
    cout << "Best Solution (Elastic Net Cost Function):" << endl;
    for (double xi : bestSolutionElasticNet)
    {
        cout << xi << " ";
    }
    cout << endl;

    return 0;
}
