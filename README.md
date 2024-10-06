# Report on Solving Sparse Representation Using Genetic Algorithm (GA)
## Problem Statement
Consider a system y=Hx+n, where y is the M x 1 measurement vector, H is M x N dictionary matrix, M << N.  x is N  x 1 desired vector, which is sparse with K non-zero entries. N is additive noise modelled as additive white Gaussian. Each element of H is independent and identically distributed Gaussian. Given the measurement y and dictionary H, find the weights x using the Genetic Algorithm (GA). Choose suitable cost function and genetic operators.

## Introduction
In numerous real-world applications, we encounter the problem of solving under-determined systems of equations where the number of measurements is much smaller than the number of unknowns. One approach to solve such systems is to leverage the fact that the unknown solution is often sparse, meaning that only a few components are non-zero. This property is commonly exploited in fields like signal processing, image compression, and compressed sensing.

In this report, we aim to solve the sparse signal recovery problem using a Genetic Algorithm (GA). The system in question is represented by the equation y = Hx + n, where:

•	y is an (M * 1) measurement vector,
•	H is an (M * N) dictionary matrix (with M << N),
•	x is an (N * 1) sparse desired vector with only K non-zero entries,
•	n represents additive white Gaussian noise.

Our task is to recover the sparse vector x from the measurement vector y and the dictionary matrix H using a Genetic Algorithm. The Genetic Algorithm (GA) is an evolutionary optimization technique inspired by the principles of natural selection and genetics. We will explore how GA can be employed to minimize the error between the observed measurement y and the modelled signal Hx, thus leading to an accurate estimation of the sparse vector x.

## Problem Formulation
The key problem in this context is sparse signal recovery. Given the system y = Hx + n, where y is the measurement vector, H is a known matrix, and n is additive white Gaussian noise, we aim to recover the unknown vector x, which has the property of sparsity. This means that most elements in x are zero, with only a few non-zero elements.

The system is under-determined, meaning that the number of measurements M is smaller than the number of unknowns N (i.e., M << N). This makes the system challenging to solve because multiple solutions exist. However, the sparsity assumption significantly reduces the number of plausible solutions, and by leveraging optimization techniques, such as GA, we can efficiently find a solution that matches the data.

### Sparse Vector Generation
The sparse vector x contains mostly zero elements, with only K non-zero entries. To generate x, we randomly select K positions within the vector and assign them non-zero values drawn from a Gaussian distribution N(0.05, 1), while the remaining entries are set to zero. This is done using the function generateSparseVector(), which ensures that x maintains its sparsity throughout the GA execution.

### Cost Function
The cost function is designed to minimize the error between the observed measurement vector y and the modeled signal Hx. The error is measured using the squared Euclidean norm (or l2 - norm), i.e.,
J(x) = || y – Hx || 2-2

This cost function represents the sum of squared differences between the actual measurement vector and the reconstructed signal. Our goal is to find the sparse vector x that minimizes this cost function.

## Genetic Algorithm (GA) Approach
A Genetic Algorithm (GA) is an iterative search heuristic that mimics the process of natural evolution. It operates by evolving a population of candidate solutions (called chromosomes) through selection, crossover, and mutation operations. In our case, each chromosome represents a candidate sparse vector x, and the objective is to evolve the population over several generations to minimize the cost function J(x).

### GA Components

#### Population Initialization
We initialize the GA with a population of 100 individuals, where everyone represents a sparse vector x with N genes (corresponding to the N-dimensional sparse vector). Each individual is initialized using the function generateSparseVector() to ensure sparsity (i.e., only K non-zero entries).

#### Fitness Function
The fitness of everyone in the population is evaluated using the cost function J(x) = || y – Hx || 2-2. Individuals with lower cost values are considered fitter, as they provide a better approximation to the measured vector y.

#### Selection
We use tournament selection to select individuals for reproduction. In tournament selection, a subset of individuals is chosen randomly from the population, and the fittest individual in this subset is selected as a parent. This process is repeated to select two parents for crossover. The selected parents are more likely to produce fitter offspring, leading to the propagation of desirable traits in the population.

#### Crossover
Crossover is the process by which two parents combine their genetic information to produce offspring. In this implementation, we use single-point crossover, where a random crossover point is selected, and the genes after the crossover point are swapped between the two parents. The crossover rate is set to 0.7, meaning that 70% of the population will undergo crossover in each generation.

#### Mutation
Mutation introduces random changes to the genes of an individual to maintain diversity in the population. Each gene has a small probability of mutating (in this case, 0.01). If a gene mutates, it is perturbed by a small random value drawn from a uniform distribution. This helps prevent premature convergence to suboptimal solutions.

## Execution of GA
The Genetic Algorithm is executed for a maximum of 1000 generations. In each generation, the following steps are performed:

1. Selection: Two parents are selected from the population using tournament selection.
2. Crossover: The parents undergo crossover with a probability of 0.7 to produce two offspring.
3. Mutation: Each offspring has a small probability (0.01) of mutation, which introduces random perturbations to its genes.
4. Evaluation: The new population is evaluated using the cost function.
5. Replacement: The old population is replaced by the new population.
At regular intervals (every 100 generations), the algorithm prints the best cost value observed so far, allowing us to monitor the progress of the GA.


## Experimentation
### Experimental Setup
The following parameters were used for the Genetic Algorithm:

- Population Size: 100
- Number of Generations: 1000
- Crossover Rate: 0.7
- Mutation Rate: 0.01
- Number of Non-zero Entries (K): 5
- Measurement Vector (y) Length (M): 20
- Sparse Vector (x) Length (N): 50

The dictionary matrix H was generated randomly, with each element drawn from a standard Gaussian distribution N(0, 1). Similarly, the measurement vector y was generated as a noisy observation by adding a small amount of random noise to the true signal Hx.

## Results and Visualizations
The Genetic Algorithm was able to recover a sparse approximation of x that closely matched the original signal after several generations. Over the course of 1000 generations, the cost function consistently decreased, indicating that the GA was effectively minimizing the reconstruction error.

## Visualization of Cost Function:
The plot of the cost function over generations shows a rapid decline in the early stages of the algorithm, followed by a slower convergence towards the end. This behaviour is typical of Genetic Algorithms, where initial improvements are made quickly, but the algorithm takes longer to refine the solution as it approaches the global minimum.

## Conclusions
The experiment demonstrated that the Genetic Algorithm is a viable approach for solving the sparse signal recovery problem in under-determined systems. The algorithm successfully minimized the reconstruction error by evolving a population of candidate sparse vectors over multiple generations. 
The main advantages of using GA for this problem are its ability to handle non-convex optimization landscapes and its robustness in exploring a large search space. However, GAs tends to converge slowly as the population approaches the optimal solution, which is evident from the gradual reduction in the cost function in the later stages of the experiment.
In future work, it would be interesting to explore other optimization techniques, such as Simulated Annealing or Particle Swarm Optimization, to compare their performance with GA in this context. Additionally, the GA could be further optimized by fine-tuning parameters such as the mutation rate and crossover probability.
