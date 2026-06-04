/* ternary-fitness-c: Implementation */
#include "ternary_fitness.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

/* ---- Internal helpers ---- */

/* Hamming distance between two genomes (matching length only) */
static size_t hamming_dist(const tf_genome_t *a, const tf_genome_t *b) {
    size_t min_len = a->len < b->len ? a->len : b->len;
    size_t d = 0;
    for (size_t i = 0; i < min_len; i++) {
        if (a->genes[i] != b->genes[i]) d++;
    }
    d += (a->len > b->len ? a->len : b->len) - min_len;
    return d;
}

/* ---- FitnessFunction ---- */

double tf_fitness_match(const tf_genome_t *genome,
                        const tf_genome_t *target,
                        const double *weights) {
    if (genome->len == 0) return 0.0;
    size_t min_len = genome->len < target->len ? genome->len : target->len;
    double score = 0.0;
    double w_total = 0.0;
    for (size_t i = 0; i < min_len; i++) {
        double w = weights ? weights[i] : 1.0;
        if (genome->genes[i] == target->genes[i]) {
            score += w;
        }
        w_total += w;
    }
    return w_total > 0.0 ? score / w_total : 0.0;
}

double tf_fitness_hamming(const tf_genome_t *genome,
                          const tf_genome_t *target,
                          const double *weights) {
    if (genome->len == 0) return 0.0;
    size_t d = hamming_dist(genome, target);
    return 1.0 - (double)d / (double)(genome->len > target->len ? genome->len : target->len);
}

/* ---- PopulationFitness ---- */

void tf_population_fitness(const tf_genome_t *pop, size_t pop_size,
                           const tf_genome_t *target,
                           const double *weights,
                           tf_fitness_fn fn,
                           tf_pop_fitness_t *out) {
    memset(out, 0, sizeof(*out));
    if (pop_size == 0) return;

    out->count = pop_size;
    out->total = 0.0;
    out->min = DBL_MAX;
    out->max = -DBL_MAX;
    out->best_idx = 0;
    out->worst_idx = 0;

    for (size_t i = 0; i < pop_size; i++) {
        double f = fn(&pop[i], target, weights);
        out->fitnesses[i] = f;
        out->total += f;
        if (f > out->max) { out->max = f; out->best_idx = i; }
        if (f < out->min) { out->min = f; out->worst_idx = i; }
    }
    out->mean = out->total / (double)pop_size;
}

/* ---- FitnessLandscape ---- */

void tf_landscape_init(tf_landscape_t *land) {
    memset(land, 0, sizeof(*land));
    land->base_fitness_scale = 1.0;
}

void tf_landscape_add_peak(tf_landscape_t *land,
                           const tf_genome_t *genome,
                           double bonus, double sigma) {
    if (land->n_peaks >= TF_MAX_PEAKS) return;
    land->peaks[land->n_peaks].genome = *genome;
    land->peaks[land->n_peaks].bonus = bonus;
    land->peaks[land->n_peaks].sigma = sigma > 0.0 ? sigma : 1.0;
    land->n_peaks++;
}

void tf_landscape_add_valley(tf_landscape_t *land,
                             const tf_genome_t *genome,
                             double penalty, double sigma) {
    if (land->n_valleys >= TF_MAX_VALLEYS) return;
    land->valleys[land->n_valleys].genome = *genome;
    land->valleys[land->n_valleys].penalty = penalty;
    land->valleys[land->n_valleys].sigma = sigma > 0.0 ? sigma : 1.0;
    land->n_valleys++;
}

double tf_landscape_fitness(const tf_landscape_t *land,
                            const tf_genome_t *genome,
                            const tf_genome_t *target,
                            const double *weights) {
    /* Start with base matching fitness */
    double base = tf_fitness_match(genome, target, weights);
    double fitness = base * land->base_fitness_scale;

    /* Add peak bonuses (Gaussian decay) */
    for (size_t i = 0; i < land->n_peaks; i++) {
        double d = (double)hamming_dist(genome, &land->peaks[i].genome);
        double sigma = land->peaks[i].sigma;
        fitness += land->peaks[i].bonus * exp(-(d * d) / (2.0 * sigma * sigma));
    }

    /* Subtract valley penalties (Gaussian decay) */
    for (size_t i = 0; i < land->n_valleys; i++) {
        double d = (double)hamming_dist(genome, &land->valleys[i].genome);
        double sigma = land->valleys[i].sigma;
        fitness -= land->valleys[i].penalty * exp(-(d * d) / (2.0 * sigma * sigma));
    }

    return fitness;
}

/* ---- SelectionPressure ---- */

double tf_apply_pressure(const tf_pop_fitness_t *raw,
                         tf_selection_pressure_t sp,
                         double *scaled) {
    if (raw->count == 0) return 0.0;

    double sum = 0.0;

    switch (sp.method) {
    case TF_PRESSURE_RANK: {
        /* Sort indices by fitness descending to assign ranks */
        size_t n = raw->count;
        /* Build index array */
        size_t *idx = (size_t *)malloc(n * sizeof(size_t));
        for (size_t i = 0; i < n; i++) idx[i] = i;

        /* Simple insertion sort */
        for (size_t i = 1; i < n; i++) {
            size_t key = idx[i];
            size_t j = i;
            while (j > 0 && raw->fitnesses[idx[j-1]] < raw->fitnesses[key]) {
                idx[j] = idx[j-1];
                j--;
            }
            idx[j] = key;
        }

        /* k = param (selection intensity), default 2.0 */
        double k = sp.param > 0.0 ? sp.param : 2.0;
        for (size_t rank = 0; rank < n; rank++) {
            double sv = k - ((double)rank / (n > 1 ? (double)(n - 1) : 1.0)) * (k - 1.0);
            scaled[idx[rank]] = sv;
            sum += sv;
        }
        free(idx);
        break;
    }
    case TF_PRESSURE_EXPONENTIAL: {
        double s = sp.param != 0.0 ? sp.param : 1.0;
        for (size_t i = 0; i < raw->count; i++) {
            scaled[i] = exp(s * raw->fitnesses[i]);
            sum += scaled[i];
        }
        break;
    }
    case TF_PRESSURE_SIGMA: {
        /* Compute stddev */
        double var = 0.0;
        for (size_t i = 0; i < raw->count; i++) {
            double diff = raw->fitnesses[i] - raw->mean;
            var += diff * diff;
        }
        double stddev = sqrt(var / (double)raw->count);
        for (size_t i = 0; i < raw->count; i++) {
            if (stddev > 1e-12) {
                scaled[i] = 1.0 + (raw->fitnesses[i] - raw->mean) / (2.0 * stddev);
            } else {
                scaled[i] = 1.0;
            }
            if (scaled[i] < 0.0) scaled[i] = 0.0; /* floor at 0 */
            sum += scaled[i];
        }
        break;
    }
    case TF_PRESSURE_POWER: {
        double pw = sp.param > 0.0 ? sp.param : 2.0;
        for (size_t i = 0; i < raw->count; i++) {
            double base = raw->fitnesses[i] > 0.0 ? raw->fitnesses[i] : 0.0;
            scaled[i] = pow(base, pw);
            sum += scaled[i];
        }
        break;
    }
    }

    return sum;
}

/* ---- Elitism ---- */

/* Comparison for qsort */
typedef struct {
    double fitness;
    size_t idx;
} _fit_idx_t;

static int _fit_cmp(const void *a, const void *b) {
    double fa = ((const _fit_idx_t *)a)->fitness;
    double fb = ((const _fit_idx_t *)b)->fitness;
    if (fb > fa) return 1;
    if (fb < fa) return -1;
    return 0;
}

size_t tf_elitism(const tf_pop_fitness_t *pop_fitness,
                  const tf_genome_t *pop, size_t pop_size,
                  size_t n_elite,
                  tf_genome_t *elite_out, double *elite_fitness_out) {
    size_t n = n_elite < pop_size ? n_elite : pop_size;
    if (n == 0) return 0;

    _fit_idx_t *fi = (_fit_idx_t *)malloc(pop_size * sizeof(_fit_idx_t));
    for (size_t i = 0; i < pop_size; i++) {
        fi[i].fitness = pop_fitness->fitnesses[i];
        fi[i].idx = i;
    }
    qsort(fi, pop_size, sizeof(_fit_idx_t), _fit_cmp);

    for (size_t i = 0; i < n; i++) {
        elite_out[i] = pop[fi[i].idx];
        if (elite_fitness_out) elite_fitness_out[i] = fi[i].fitness;
    }
    free(fi);
    return n;
}

void tf_sort_by_fitness(const tf_pop_fitness_t *pf, size_t *indices, size_t n) {
    _fit_idx_t *fi = (_fit_idx_t *)malloc(n * sizeof(_fit_idx_t));
    for (size_t i = 0; i < n; i++) {
        fi[i].fitness = pf->fitnesses[i];
        fi[i].idx = i;
    }
    qsort(fi, n, sizeof(_fit_idx_t), _fit_cmp);
    for (size_t i = 0; i < n; i++) {
        indices[i] = fi[i].idx;
    }
    free(fi);
}
