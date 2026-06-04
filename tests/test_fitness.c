/* Tests for ternary-fitness-c */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/ternary_fitness.h"

#define ASSERT_FEQ(a, b, tol) do { \
    if (fabs((a) - (b)) > (tol)) { \
        fprintf(stderr, "FAIL %s:%d: %f != %f\n", __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 1; \
    } \
} while(0)

int test_strategy_fitness() {
    tf_strategy_t s = {.len = 4, .actions = {TF_CHOOSE, TF_CHOOSE, TF_CHOOSE, TF_CHOOSE}};
    tf_environment_t env = {.n_states = 4};
    for (int i = 0; i < 4; i++) {
        env.rewards[i][0] = -1.0; /* avoid */
        env.rewards[i][1] =  0.0; /* unknown */
        env.rewards[i][2] =  1.0; /* choose */
    }
    double f = tf_strategy_fitness(&s, &env);
    ASSERT_FEQ(f, 4.0, 0.001);
    return 0;
}

int test_strategy_fitness_mixed() {
    tf_strategy_t s = {.len = 3, .actions = {TF_AVOID, TF_UNKNOWN, TF_CHOOSE}};
    tf_environment_t env = {.n_states = 3};
    for (int i = 0; i < 3; i++) {
        env.rewards[i][0] = -1.0;
        env.rewards[i][1] = 0.5;
        env.rewards[i][2] = 1.0;
    }
    double f = tf_strategy_fitness(&s, &env);
    ASSERT_FEQ(f, 0.5, 0.001); /* -1 + 0.5 + 1 = 0.5 */
    return 0;
}

int test_active_count() {
    tf_strategy_t s = {.len = 5, .actions = {TF_CHOOSE, TF_AVOID, TF_CHOOSE, TF_UNKNOWN, TF_CHOOSE}};
    assert(tf_strategy_active_count(&s) == 3);
    return 0;
}

int test_active_count_zero() {
    tf_strategy_t s = {.len = 3, .actions = {TF_AVOID, TF_UNKNOWN, TF_AVOID}};
    assert(tf_strategy_active_count(&s) == 0);
    return 0;
}

int test_shannon_entropy_uniform() {
    tf_strategy_t s = {.len = 9};
    for (int i = 0; i < 9; i++) {
        s.actions[i] = (tf_action_t)((i % 3) - 1);
    }
    double h = tf_shannon_entropy(&s);
    ASSERT_FEQ(h, log2(3.0), 0.01);
    return 0;
}

int test_shannon_entropy_pure() {
    tf_strategy_t s = {.len = 5};
    for (int i = 0; i < 5; i++) s.actions[i] = TF_CHOOSE;
    double h = tf_shannon_entropy(&s);
    ASSERT_FEQ(h, 0.0, 0.001);
    return 0;
}

int test_normalized_entropy() {
    tf_strategy_t s = {.len = 9};
    for (int i = 0; i < 9; i++) s.actions[i] = (tf_action_t)((i % 3) - 1);
    double h = tf_normalized_entropy(&s);
    ASSERT_FEQ(h, 1.0, 0.01);
    return 0;
}

int test_hamming_distance_same() {
    tf_strategy_t a = {.len = 4, .actions = {TF_AVOID, TF_UNKNOWN, TF_CHOOSE, TF_AVOID}};
    tf_strategy_t b = {.len = 4, .actions = {TF_AVOID, TF_UNKNOWN, TF_CHOOSE, TF_AVOID}};
    ASSERT_FEQ(tf_hamming_distance(&a, &b), 0.0, 0.001);
    return 0;
}

int test_hamming_distance_different() {
    tf_strategy_t a = {.len = 4, .actions = {TF_AVOID, TF_UNKNOWN, TF_CHOOSE, TF_AVOID}};
    tf_strategy_t b = {.len = 4, .actions = {TF_CHOOSE, TF_CHOOSE, TF_AVOID, TF_AVOID}};
    ASSERT_FEQ(tf_hamming_distance(&a, &b), 3.0, 0.001);
    return 0;
}

int test_hamming_distance_different_length() {
    tf_strategy_t a = {.len = 3, .actions = {TF_AVOID, TF_UNKNOWN, TF_CHOOSE}};
    tf_strategy_t b = {.len = 5, .actions = {TF_AVOID, TF_UNKNOWN, TF_CHOOSE, TF_AVOID, TF_CHOOSE}};
    ASSERT_FEQ(tf_hamming_distance(&a, &b), 2.0, 0.001);
    return 0;
}

int test_exhaustive_81() {
    /* 3^4 = 81 strategies */
    tf_environment_t env = {.n_states = 4};
    for (int i = 0; i < 4; i++) {
        env.rewards[i][0] = -1.0;
        env.rewards[i][1] = 0.0;
        env.rewards[i][2] = 1.0;
    }
    tf_search_result_t result;
    size_t count = tf_exhaustive_search(4, &env, &result);
    assert(count == 81);
    
    tf_strategy_t best;
    double best_fitness;
    assert(tf_find_best(&result, &best, &best_fitness));
    ASSERT_FEQ(best_fitness, 4.0, 0.001); /* all CHOOSE */
    return 0;
}

int test_exhaustive_best() {
    tf_environment_t env = {.n_states = 3};
    /* Non-uniform rewards: make AVOID best for state 0 */
    env.rewards[0][0] = 2.0;  /* avoid */
    env.rewards[0][1] = 0.0;
    env.rewards[0][2] = -1.0;
    env.rewards[1][0] = -1.0;
    env.rewards[1][1] = 0.0;
    env.rewards[1][2] = 1.0;
    env.rewards[2][0] = -1.0;
    env.rewards[2][1] = 3.0;  /* unknown is best */
    env.rewards[2][2] = 1.0;
    
    tf_search_result_t result;
    tf_exhaustive_search(3, &env, &result);
    assert(result.count == 27);
    
    tf_strategy_t best;
    double best_fitness;
    assert(tf_find_best(&result, &best, &best_fitness));
    ASSERT_FEQ(best_fitness, 6.0, 0.001); /* 2 + 1 + 3 */
    assert(best.actions[0] == TF_AVOID);
    assert(best.actions[1] == TF_CHOOSE);
    assert(best.actions[2] == TF_UNKNOWN);
    return 0;
}

int test_landscape_analysis() {
    tf_environment_t env = {.n_states = 3};
    for (int i = 0; i < 3; i++) {
        env.rewards[i][0] = -1.0;
        env.rewards[i][1] = 0.0;
        env.rewards[i][2] = 1.0;
    }
    tf_search_result_t result;
    tf_exhaustive_search(3, &env, &result);
    
    tf_landscape_info_t info = tf_analyze_landscape(&result);
    ASSERT_FEQ(info.max_fitness, 3.0, 0.001);
    ASSERT_FEQ(info.min_fitness, -3.0, 0.001);
    ASSERT_FEQ(info.mean_fitness, 0.0, 0.001);
    assert(info.n_peaks > 0);
    return 0;
}

int test_population_diversity_single() {
    tf_strategy_t pop[1];
    pop[0].len = 3;
    pop[0].actions[0] = TF_CHOOSE;
    pop[0].actions[1] = TF_AVOID;
    pop[0].actions[2] = TF_UNKNOWN;
    ASSERT_FEQ(tf_population_diversity(pop, 1), 0.0, 0.001);
    return 0;
}

int test_population_diversity_identical() {
    tf_strategy_t pop[3];
    for (int i = 0; i < 3; i++) {
        pop[i].len = 3;
        pop[i].actions[0] = TF_CHOOSE;
        pop[i].actions[1] = TF_AVOID;
        pop[i].actions[2] = TF_UNKNOWN;
    }
    ASSERT_FEQ(tf_population_diversity(pop, 3), 0.0, 0.001);
    return 0;
}

int test_population_diversity_diverse() {
    tf_strategy_t pop[3];
    pop[0] = (tf_strategy_t){.len = 3, .actions = {TF_AVOID, TF_AVOID, TF_AVOID}};
    pop[1] = (tf_strategy_t){.len = 3, .actions = {TF_CHOOSE, TF_CHOOSE, TF_CHOOSE}};
    pop[2] = (tf_strategy_t){.len = 3, .actions = {TF_UNKNOWN, TF_UNKNOWN, TF_UNKNOWN}};
    double div = tf_population_diversity(pop, 3);
    assert(div > 0.0);
    /* 3 pairs, each distance = 3 */
    ASSERT_FEQ(div, 3.0, 0.001);
    return 0;
}

int test_entropy_two_values() {
    tf_strategy_t s = {.len = 4, .actions = {TF_AVOID, TF_AVOID, TF_CHOOSE, TF_CHOOSE}};
    double h = tf_shannon_entropy(&s);
    ASSERT_FEQ(h, 1.0, 0.01); /* log2(2) = 1 */
    return 0;
}

int main(void) {
    int failures = 0;
    #define RUN(name) do { \
        int r = test_##name(); \
        if (r == 0) printf("  PASS: %s\n", #name); \
        else { printf("  FAIL: %s\n", #name); failures++; } \
    } while(0)
    
    printf("Running tests...\n");
    RUN(strategy_fitness);
    RUN(strategy_fitness_mixed);
    RUN(active_count);
    RUN(active_count_zero);
    RUN(shannon_entropy_uniform);
    RUN(shannon_entropy_pure);
    RUN(normalized_entropy);
    RUN(hamming_distance_same);
    RUN(hamming_distance_different);
    RUN(hamming_distance_different_length);
    RUN(exhaustive_81);
    RUN(exhaustive_best);
    RUN(landscape_analysis);
    RUN(population_diversity_single);
    RUN(population_diversity_identical);
    RUN(population_diversity_diverse);
    RUN(entropy_two_values);
    
    int n = 17;
    printf("\n%d/%d passed\n", n - failures, n);
    return failures;
}
