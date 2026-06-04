/* ternary-fitness-c: Ternary fitness evaluation library */
#ifndef TERNARY_FITNESS_H
#define TERNARY_FITNESS_H

#include <stddef.h>

/* Ternary digit: -1 (avoid), 0 (neutral), +1 (choose) */
typedef enum {
    TF_AVOID   = -1,
    TF_NEUTRAL =  0,
    TF_CHOOSE  =  1
} tf_ternary_t;

/* A ternary genome (array of ternary digits) */
#define TF_MAX_GENOME_LEN 256
#define TF_MAX_POP_SIZE   10000

typedef struct {
    tf_ternary_t genes[TF_MAX_GENOME_LEN];
    size_t       len;
} tf_genome_t;

/* ---- FitnessFunction ---- */
/* Evaluate the fitness of a single ternary genome against a target genome.
 * Fitness = (number of matching positions) / genome_length.
 * Optionally apply per-position weights (NULL = uniform). */
typedef double (*tf_fitness_fn)(const tf_genome_t *genome,
                                const tf_genome_t *target,
                                const double *weights);

double tf_fitness_match(const tf_genome_t *genome,
                        const tf_genome_t *target,
                        const double *weights);

/* Hamming-distance fitness: 1 - (hamming_distance / len) */
double tf_fitness_hamming(const tf_genome_t *genome,
                          const tf_genome_t *target,
                          const double *weights);

/* ---- PopulationFitness ---- */
typedef struct {
    double   fitnesses[TF_MAX_POP_SIZE];
    size_t   count;
    double   total;
    double   mean;
    double   min;
    double   max;
    size_t   best_idx;
    size_t   worst_idx;
} tf_pop_fitness_t;

/* Evaluate fitness for an entire population. Returns summary stats. */
void tf_population_fitness(const tf_genome_t *pop, size_t pop_size,
                           const tf_genome_t *target,
                           const double *weights,
                           tf_fitness_fn fn,
                           tf_pop_fitness_t *out);

/* ---- FitnessLandscape ---- */
/* A configurable landscape with peaks and valleys.
 * Peaks are specific genomes with associated fitness bonuses.
 * Valleys are genomes with fitness penalties.
 * Base fitness comes from matching against a target. */

#define TF_MAX_PEAKS    64

typedef struct {
    tf_genome_t genome;
    double      bonus;   /* fitness bonus when close to this peak */
    double      sigma;   /* influence radius (hamming distance) */
} tf_peak_t;

typedef struct {
    tf_genome_t genome;
    double      penalty; /* fitness penalty when close to this valley */
    double      sigma;   /* influence radius */
} tf_valley_t;

#define TF_MAX_VALLEYS 64

typedef struct {
    tf_peak_t   peaks[TF_MAX_PEAKS];
    size_t      n_peaks;
    tf_valley_t valleys[TF_MAX_VALLEYS];
    size_t      n_valleys;
    double      base_fitness_scale; /* multiplier on base fitness (default 1.0) */
} tf_landscape_t;

/* Initialize a landscape with default settings */
void tf_landscape_init(tf_landscape_t *land);

/* Add a peak to the landscape */
void tf_landscape_add_peak(tf_landscape_t *land,
                            const tf_genome_t *genome,
                            double bonus, double sigma);

/* Add a valley to the landscape */
void tf_landscape_add_valley(tf_landscape_t *land,
                              const tf_genome_t *genome,
                              double penalty, double sigma);

/* Evaluate fitness on the landscape for a single genome */
double tf_landscape_fitness(const tf_landscape_t *land,
                            const tf_genome_t *genome,
                            const tf_genome_t *target,
                            const double *weights);

/* ---- SelectionPressure ---- */
/* Scale raw fitness values to create selection pressure.
 * Methods: rank, exponential, sigma-scaling, tournament-style. */

typedef enum {
    TF_PRESSURE_RANK,       /* Rank-based: 1 + (rank-1)/(n-1) * (k-1) */
    TF_PRESSURE_EXPONENTIAL, /* exp(s * raw_fitness) */
    TF_PRESSURE_SIGMA,      /* 1 + (raw - mean) / (2 * stddev) */
    TF_PRESSURE_POWER       /* raw^k */
} tf_pressure_method_t;

typedef struct {
    tf_pressure_method_t method;
    double               param;  /* method-specific parameter (k, s, etc.) */
} tf_selection_pressure_t;

/* Apply selection pressure to a population's raw fitness values.
 * scaled[] must have room for pop_size elements.
 * Returns the sum of scaled fitnesses. */
double tf_apply_pressure(const tf_pop_fitness_t *raw,
                         tf_selection_pressure_t sp,
                         double *scaled);

/* ---- Elitism ---- */
/* Select top N agents by fitness from a population.
 * indices[] must have room for n_elite elements.
 * Returns actual number selected (min(n_elite, pop_size)). */
size_t tf_elitism(const tf_pop_fitness_t *pop_fitness,
                  const tf_genome_t *pop, size_t pop_size,
                  size_t n_elite,
                  tf_genome_t *elite_out, double *elite_fitness_out);

/* Sort population indices by fitness (descending). indices[] = pop_size elements. */
void tf_sort_by_fitness(const tf_pop_fitness_t *pf, size_t *indices, size_t n);

#endif /* TERNARY_FITNESS_H */
