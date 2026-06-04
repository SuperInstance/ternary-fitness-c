/* ternary-fitness-c: Fitness landscapes for ternary agent strategies */
#ifndef TERNARY_FITNESS_H
#define TERNARY_FITNESS_H

#include <stddef.h>

#define TF_MAX_STRATEGY 64
#define TF_MAX_STRATEGIES 10000

typedef enum {
    TF_AVOID  = -1,
    TF_UNKNOWN = 0,
    TF_CHOOSE = 1
} tf_action_t;

/* Ternary strategy: sequence of {-1, 0, +1} */
typedef struct {
    tf_action_t actions[TF_MAX_STRATEGY];
    size_t len;
} tf_strategy_t;

/* Environment: reward per (state, action) */
typedef struct {
    double rewards[TF_MAX_STRATEGY][3]; /* [state][action+1] */
    size_t n_states;
} tf_environment_t;

/* Strategy evaluation */
double tf_strategy_fitness(const tf_strategy_t *s, const tf_environment_t *env);
int tf_strategy_active_count(const tf_strategy_t *s);

/* Entropy */
double tf_shannon_entropy(const tf_strategy_t *s);
double tf_normalized_entropy(const tf_strategy_t *s);
double tf_hamming_distance(const tf_strategy_t *a, const tf_strategy_t *b);

/* Exhaustive search */
typedef struct {
    tf_strategy_t strategies[TF_MAX_STRATEGIES];
    double fitnesses[TF_MAX_STRATEGIES];
    size_t count;
} tf_search_result_t;

size_t tf_exhaustive_search(size_t n, const tf_environment_t *env, tf_search_result_t *result);
int tf_find_best(const tf_search_result_t *result, tf_strategy_t *best, double *best_fitness);

/* Fitness landscape */
typedef struct {
    size_t n_peaks;
    size_t n_basins;
    double max_fitness;
    double min_fitness;
    double mean_fitness;
} tf_landscape_info_t;

tf_landscape_info_t tf_analyze_landscape(const tf_search_result_t *result);

/* Population diversity */
double tf_population_diversity(const tf_strategy_t *pop, size_t n);

#endif /* TERNARY_FITNESS_H */
