# ternary-fitness-c

C implementation of ternary fitness evaluation.

## Components

- **FitnessFunction** — evaluate fitness of a ternary genome against a target (match-based, hamming-based, custom)
- **PopulationFitness** — evaluate fitness for an entire population with summary statistics
- **FitnessLandscape** — configurable landscape with Gaussian peaks and valleys
- **SelectionPressure** — scale raw fitness values (rank, exponential, sigma-scaling, power)
- **Elitism** — select top N agents by fitness

## Build & Test

```bash
gcc -o test_fitness tests/test_fitness.c src/ternary_fitness.c -lm -Wall -O2
./test_fitness
```

## API

```c
#include "src/ternary_fitness.h"

// Fitness function — match fraction against target
double tf_fitness_match(const tf_genome_t *genome, const tf_genome_t *target, const double *weights);
double tf_fitness_hamming(const tf_genome_t *genome, const tf_genome_t *target, const double *weights);

// Population evaluation
void tf_population_fitness(const tf_genome_t *pop, size_t pop_size,
                           const tf_genome_t *target, const double *weights,
                           tf_fitness_fn fn, tf_pop_fitness_t *out);

// Landscape with peaks and valleys
void tf_landscape_init(tf_landscape_t *land);
void tf_landscape_add_peak(tf_landscape_t *land, const tf_genome_t *genome, double bonus, double sigma);
void tf_landscape_add_valley(tf_landscape_t *land, const tf_genome_t *genome, double penalty, double sigma);
double tf_landscape_fitness(const tf_landscape_t *land, const tf_genome_t *genome,
                            const tf_genome_t *target, const double *weights);

// Selection pressure
double tf_apply_pressure(const tf_pop_fitness_t *raw, tf_selection_pressure_t sp, double *scaled);

// Elitism
size_t tf_elitism(const tf_pop_fitness_t *pop_fitness, const tf_genome_t *pop,
                  size_t pop_size, size_t n_elite, tf_genome_t *elite_out, double *elite_fitness_out);
```
