#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

const int POP_SIZE = 100;  // Population size
const int N = 50;          // Length of vector x (genes per chromosome)
const int M = 20;          // Length of vector y
const int MAX_GEN = 1000;  // Maximum number of generations
const double CROSSOVER_RATE = 0.7;  // Crossover probability
const double MUTATION_RATE = 0.01;  // Mutation probability
const int K = 5;  // Number of non-zero entries in x

// Random number generator for Gaussian and uniform distributions
random_device rd;
mt19937 gen(rd());
normal_distribution<> H_dist(0.0, 1.0); // Mean 0, Variance 1 for H
normal_distribution<> x_dist(0.05, 1.0); // Mean 0.05, Variance 1 for non-zero x
uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution for random numbers between 0 and 1

// Function to generate a random sparse vector x with Gaussian non-zero values
vector<double> generateSparseVector(int N, int K) {
    vector<double> x(N, 0.0);
    for (int i = 0; i < K; i++) {
        int index = rand() % N;
        x[index] = x_dist(gen);  // Random non-zero value from N(0.05, 1)
    }
    return x;
}

// Function to calculate the cost (fitness) J(x) = ||y - Hx||_2^2
double costFunction(const vector<double>& y, const vector<vector<double>>& H, const vector<double>& x) {
    double error = 0.0;

    // Compute the residual y - Hx
    vector<double> residual(y.size(), 0.0);
    for (int i = 0; i < y.size(); ++i) {
        residual[i] = y[i];
        for (int j = 0; j < x.size(); ++j) {
            residual[i] -= H[i][j] * x[j];
        }
        error += residual[i] * residual[i];  // ||y - Hx||_2^2
    }

    // Return the cost (reconstruction error)
    return error;
}

// Selection operator (tournament selection)
vector<double> tournamentSelection(const vector<vector<double>>& population, const vector<double>& fitness, const vector<double>& y, const vector<vector<double>>& H) {
    int tournamentSize = 3;
    vector<double> best = population[rand() % POP_SIZE];

    for (int i = 1; i < tournamentSize; i++) {
        int randomIndex = rand() % POP_SIZE;
        if (fitness[randomIndex] < costFunction(y, H, population[randomIndex])) {
            best = population[randomIndex];
        }
    }

    return best;
}

// Crossover operator (single-point crossover)
void crossover(vector<double>& parent1, vector<double>& parent2) {
    if (dis(gen) < CROSSOVER_RATE) {
        int point = rand() % N;
        for (int i = point; i < N; i++) {
            swap(parent1[i], parent2[i]);
        }
    }
}

// Mutation operator
void mutate(vector<double>& chromosome) {
    for (int i = 0; i < N; i++) {
        if (dis(gen) < MUTATION_RATE) {
            chromosome[i] += dis(gen) - 0.5;  // Small random mutation
        }
    }
}

// Genetic Algorithm
vector<double> geneticAlgorithm(const vector<double>& y, const vector<vector<double>>& H) {
    // Step 1: Initialize population
    vector<vector<double>> population(POP_SIZE, vector<double>(N));
    vector<double> fitness(POP_SIZE);
    for (int i = 0; i < POP_SIZE; i++) {
        population[i] = generateSparseVector(N, K);
        fitness[i] = costFunction(y, H, population[i]);
    }

    // Step 2: Evolution loop
    for (int gen = 0; gen < MAX_GEN; gen++) {
        vector<vector<double>> newPopulation(POP_SIZE);

        // Step 3: Selection, Crossover, and Mutation
        for (int i = 0; i < POP_SIZE; i += 2) {
            vector<double> parent1 = tournamentSelection(population, fitness, y, H);
            vector<double> parent2 = tournamentSelection(population, fitness, y, H);
            crossover(parent1, parent2);
            mutate(parent1);
            mutate(parent2);
            newPopulation[i] = parent1;
            newPopulation[i + 1] = parent2;
        }

        // Step 4: Evaluate new population
        for (int i = 0; i < POP_SIZE; i++) {
            fitness[i] = costFunction(y, H, newPopulation[i]);
        }

        population = newPopulation;

        // Step 5: Check for convergence (can implement stopping condition here)
        if (gen % 100 == 0) {
            cout << "Generation " << gen << ", Best Cost: " << *min_element(fitness.begin(), fitness.end()) << endl;
        }
    }

    // Step 6: Return the best solution found
    int bestIndex = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    return population[bestIndex];
}

int main() {
    // Randomly initialize dictionary matrix H and measurement vector y
    vector<vector<double>> H(M, vector<double>(N));
    vector<double> y(M, 0.0);

    // Generate random Gaussian dictionary matrix H
    for (int i = 0; i < M; i++) {
        
            for (int j = 0; j < N; j++) {
            H[i][j] = H_dist(gen);  // H follows N(0, 1)
        }
    }

    // Generate random measurement vector y (noisy observation)
    for (int i = 0; i < M; i++) {
        y[i] = dis(gen);  // Example random measurement, can replace with real data
    }

    // Apply Genetic Algorithm to find sparse x
    vector<double> bestSolution = geneticAlgorithm(y, H);

    // Output the best solution
    cout << "Best Solution (Sparse x):" << endl;
    for (double val : bestSolution) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}

