/* ternary-fitness-c: Implementation */
#include "ternary_fitness.h"
#include <math.h>
#include <string.h>
#include <float.h>

double tf_strategy_fitness(const tf_strategy_t *s, const tf_environment_t *env) {
    double total = 0.0;
    size_t limit = s->len < env->n_states ? s->len : env->n_states;
    for (size_t i = 0; i < limit; i++) {
        int idx = (int)s->actions[i] + 1; /* -1→0, 0→1, 1→2 */
        if (idx >= 0 && idx < 3) {
            total += env->rewards[i][idx];
        }
    }
    return total;
}

int tf_strategy_active_count(const tf_strategy_t *s) {
    int count = 0;
    for (size_t i = 0; i < s->len; i++) {
        if (s->actions[i] == TF_CHOOSE) count++;
    }
    return count;
}

double tf_shannon_entropy(const tf_strategy_t *s) {
    int counts[3] = {0, 0, 0};
    for (size_t i = 0; i < s->len; i++) {
        int idx = (int)s->actions[i] + 1;
        if (idx >= 0 && idx < 3) counts[idx]++;
    }
    double h = 0.0;
    double total = (double)s->len;
    for (int i = 0; i < 3; i++) {
        if (counts[i] > 0) {
            double p = counts[i] / total;
            h -= p * log2(p);
        }
    }
    return h;
}

double tf_normalized_entropy(const tf_strategy_t *s) {
    if (s->len == 0) return 0.0;
    double h = tf_shannon_entropy(s);
    double max_h = log2(3.0); /* 3 outcomes */
    return h / max_h;
}

double tf_hamming_distance(const tf_strategy_t *a, const tf_strategy_t *b) {
    size_t min_len = a->len < b->len ? a->len : b->len;
    double dist = 0.0;
    for (size_t i = 0; i < min_len; i++) {
        if (a->actions[i] != b->actions[i]) dist += 1.0;
    }
    /* Extra length counts as differences */
    dist += (double)((a->len > b->len ? a->len : b->len) - min_len);
    return dist;
}

static void int_to_ternary(int val, size_t n, tf_strategy_t *s) {
    s->len = n;
    for (size_t i = 0; i < n; i++) {
        int digit = val % 3;
        val /= 3;
        s->actions[i] = (tf_action_t)(digit - 1); /* 0→-1, 1→0, 2→1 */
    }
}

size_t tf_exhaustive_search(size_t n, const tf_environment_t *env, tf_search_result_t *result) {
    /* Total strategies = 3^n */
    size_t total = 1;
    for (size_t i = 0; i < n; i++) {
        total *= 3;
        if (total > TF_MAX_STRATEGIES) {
            total = TF_MAX_STRATEGIES;
            break;
        }
    }
    
    result->count = 0;
    for (size_t i = 0; i < total; i++) {
        int_to_ternary((int)i, n, &result->strategies[i]);
        result->fitnesses[i] = tf_strategy_fitness(&result->strategies[i], env);
        result->count++;
    }
    return result->count;
}

int tf_find_best(const tf_search_result_t *result, tf_strategy_t *best, double *best_fitness) {
    if (result->count == 0) return 0;
    
    *best_fitness = -DBL_MAX;
    size_t best_idx = 0;
    for (size_t i = 0; i < result->count; i++) {
        if (result->fitnesses[i] > *best_fitness) {
            *best_fitness = result->fitnesses[i];
            best_idx = i;
        }
    }
    *best = result->strategies[best_idx];
    return 1;
}

tf_landscape_info_t tf_analyze_landscape(const tf_search_result_t *result) {
    tf_landscape_info_t info = {0, 0, -DBL_MAX, DBL_MAX, 0.0};
    if (result->count == 0) return info;
    
    double sum = 0.0;
    for (size_t i = 0; i < result->count; i++) {
        double f = result->fitnesses[i];
        sum += f;
        if (f > info.max_fitness) info.max_fitness = f;
        if (f < info.min_fitness) info.min_fitness = f;
    }
    info.mean_fitness = sum / (double)result->count;
    
    /* Count peaks: strategies with fitness above mean */
    for (size_t i = 0; i < result->count; i++) {
        double f = result->fitnesses[i];
        if (f > info.mean_fitness) info.n_peaks++;
    }
    
    info.n_basins = result->count - info.n_peaks;
    return info;
}

double tf_population_diversity(const tf_strategy_t *pop, size_t n) {
    if (n < 2) return 0.0;
    double total_dist = 0.0;
    size_t pairs = 0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            total_dist += tf_hamming_distance(&pop[i], &pop[j]);
            pairs++;
        }
    }
    return pairs > 0 ? total_dist / (double)pairs : 0.0;
}
