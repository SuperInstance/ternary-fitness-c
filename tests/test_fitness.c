/* Tests for ternary-fitness-c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../src/ternary_fitness.h"

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while(0)

#define ASSERT_FEQ(a, b, tol) do { \
    if (fabs((a) - (b)) > (tol)) { \
        fprintf(stderr, "FAIL %s:%d: %f != %f (tol %g)\n", \
                __FILE__, __LINE__, (double)(a), (double)(b), (double)(tol)); \
        return 1; \
    } \
} while(0)

/* Helper: build a genome from an array of ints */
static tf_genome_t make_genome(const int *vals, size_t n) {
    tf_genome_t g;
    g.len = n;
    for (size_t i = 0; i < n; i++) g.genes[i] = (tf_ternary_t)vals[i];
    return g;
}

/* ---- Test 1: FitnessFunction — perfect match ---- */
int test_fitness_perfect_match(void) {
    int v[] = {1, -1, 0, 1};
    tf_genome_t genome = make_genome(v, 4);
    tf_genome_t target = make_genome(v, 4);
    double f = tf_fitness_match(&genome, &target, NULL);
    ASSERT_FEQ(f, 1.0, 0.001);
    return 0;
}

/* ---- Test 2: FitnessFunction — no match ---- */
int test_fitness_no_match(void) {
    int gv[] = {1, 1, 1, 1};
    int tv[] = {-1, -1, -1, -1};
    tf_genome_t genome = make_genome(gv, 4);
    tf_genome_t target = make_genome(tv, 4);
    double f = tf_fitness_match(&genome, &target, NULL);
    ASSERT_FEQ(f, 0.0, 0.001);
    return 0;
}

/* ---- Test 3: FitnessFunction — weighted ---- */
int test_fitness_weighted(void) {
    int gv[] = {1, -1, 0};
    int tv[] = {1, -1, 0};
    double w[] = {2.0, 3.0, 5.0};
    tf_genome_t genome = make_genome(gv, 3);
    tf_genome_t target = make_genome(tv, 3);
    double f = tf_fitness_match(&genome, &target, w);
    ASSERT_FEQ(f, 1.0, 0.001);

    /* Partial match: first two match, third doesn't */
    int tv2[] = {1, -1, 1};
    target = make_genome(tv2, 3);
    f = tf_fitness_match(&genome, &target, w);
    ASSERT_FEQ(f, 5.0 / 10.0, 0.001); /* (2+3)/10 */
    return 0;
}

/* ---- Test 4: FitnessFunction — hamming fitness ---- */
int test_fitness_hamming(void) {
    int gv[] = {1, -1, 0, 1};
    int tv[] = {1, 1, 0, -1};
    tf_genome_t genome = make_genome(gv, 4);
    tf_genome_t target = make_genome(tv, 4);
    double f = tf_fitness_hamming(&genome, &target, NULL);
    /* 2 matches out of 4 → hamming dist = 2 → 1 - 2/4 = 0.5 */
    ASSERT_FEQ(f, 0.5, 0.001);
    return 0;
}

/* ---- Test 5: PopulationFitness — basic stats ---- */
int test_population_fitness_stats(void) {
    int tv[] = {1, 1, 1};
    tf_genome_t target = make_genome(tv, 3);

    int g0[] = {1, 1, 1};  /* perfect: 1.0 */
    int g1[] = {-1, -1, -1}; /* none: 0.0 */
    int g2[] = {1, -1, 1};  /* 2/3 ≈ 0.667 */

    tf_genome_t pop[3];
    pop[0] = make_genome(g0, 3);
    pop[1] = make_genome(g1, 3);
    pop[2] = make_genome(g2, 3);

    tf_pop_fitness_t pf;
    tf_population_fitness(pop, 3, &target, NULL, tf_fitness_match, &pf);

    ASSERT(pf.count == 3);
    ASSERT_FEQ(pf.max, 1.0, 0.001);
    ASSERT_FEQ(pf.min, 0.0, 0.001);
    ASSERT_FEQ(pf.mean, (1.0 + 0.0 + 2.0/3.0) / 3.0, 0.01);
    ASSERT(pf.best_idx == 0);
    ASSERT(pf.worst_idx == 1);
    return 0;
}

/* ---- Test 6: PopulationFitness — single individual ---- */
int test_population_single(void) {
    int tv[] = {1, 0, -1};
    tf_genome_t target = make_genome(tv, 3);
    tf_genome_t pop[1];
    pop[0] = make_genome(tv, 3);

    tf_pop_fitness_t pf;
    tf_population_fitness(pop, 1, &target, NULL, tf_fitness_match, &pf);

    ASSERT(pf.count == 1);
    ASSERT_FEQ(pf.max, 1.0, 0.001);
    ASSERT_FEQ(pf.min, 1.0, 0.001);
    ASSERT_FEQ(pf.mean, 1.0, 0.001);
    return 0;
}

/* ---- Test 7: FitnessLandscape — peak boosts nearby genome ---- */
int test_landscape_peak(void) {
    int tv[] = {1, 1, 1, 1};
    tf_genome_t target = make_genome(tv, 4);

    int pv[] = {1, 1, 1, 1};
    tf_genome_t peak_genome = make_genome(pv, 4);

    tf_landscape_t land;
    tf_landscape_init(&land);
    tf_landscape_add_peak(&land, &peak_genome, 2.0, 2.0);

    /* Genome exactly on peak */
    tf_genome_t g = make_genome(pv, 4);
    double f = tf_landscape_fitness(&land, &g, &target, NULL);
    ASSERT(f > 1.0); /* base(1.0) + peak bonus → > 1 */
    ASSERT_FEQ(f, 3.0, 0.01); /* 1.0 + 2.0 * exp(0) = 3.0 */
    return 0;
}

/* ---- Test 8: FitnessLandscape — valley penalizes nearby genome ---- */
int test_landscape_valley(void) {
    int tv[] = {1, 1, 1, 1};
    tf_genome_t target = make_genome(tv, 4);

    int vv[] = {1, 1, 1, 1};
    tf_genome_t valley_genome = make_genome(vv, 4);

    tf_landscape_t land;
    tf_landscape_init(&land);
    tf_landscape_add_valley(&land, &valley_genome, 1.5, 2.0);

    tf_genome_t g = make_genome(vv, 4);
    double f = tf_landscape_fitness(&land, &g, &target, NULL);
    /* base(1.0) - valley penalty → 1.0 - 1.5 = -0.5 */
    ASSERT_FEQ(f, -0.5, 0.01);
    return 0;
}

/* ---- Test 9: FitnessLandscape — peak + valley combined ---- */
int test_landscape_combined(void) {
    int tv[] = {1, 1, 1, 1};
    tf_genome_t target = make_genome(tv, 4);

    int pv[] = {-1, -1, -1, -1};
    tf_genome_t peak_g = make_genome(pv, 4);

    int vv[] = {1, 0, 1, 0};
    tf_genome_t valley_g = make_genome(vv, 4);

    tf_landscape_t land;
    tf_landscape_init(&land);
    tf_landscape_add_peak(&land, &peak_g, 3.0, 3.0);
    tf_landscape_add_valley(&land, &valley_g, 2.0, 1.0);

    /* Genome at peak: match=0, peak bonus = 3.0, valley: far enough for small penalty */
    tf_genome_t g = make_genome(pv, 4);
    double f = tf_landscape_fitness(&land, &g, &target, NULL);
    /* base=0, peak=3.0*exp(0)=3.0, valley: hamming=3 → penalty=2*exp(-9/2)≈0.022 */
    ASSERT(f > 2.9);
    return 0;
}

/* ---- Test 10: SelectionPressure — rank-based ---- */
int test_pressure_rank(void) {
    tf_pop_fitness_t pf;
    pf.count = 4;
    pf.fitnesses[0] = 0.9;
    pf.fitnesses[1] = 0.3;
    pf.fitnesses[2] = 0.7;
    pf.fitnesses[3] = 0.5;
    pf.mean = 0.6;

    tf_selection_pressure_t sp = {TF_PRESSURE_RANK, 2.0};
    double scaled[4];
    double sum = tf_apply_pressure(&pf, sp, scaled);

    /* Rank 0 (best=idx0) gets 2.0, rank 3 (worst=idx1) gets 1.0 */
    ASSERT_FEQ(scaled[0], 2.0, 0.001);     /* fitness 0.9 → rank 0 → 2.0 */
    ASSERT_FEQ(scaled[1], 1.0, 0.001);     /* fitness 0.3 → rank 3 → 1.0 */
    ASSERT(sum > 0.0);
    return 0;
}

/* ---- Test 11: SelectionPressure — exponential ---- */
int test_pressure_exponential(void) {
    tf_pop_fitness_t pf;
    pf.count = 3;
    pf.fitnesses[0] = 1.0;
    pf.fitnesses[1] = 0.5;
    pf.fitnesses[2] = 0.0;
    pf.mean = 0.5;

    tf_selection_pressure_t sp = {TF_PRESSURE_EXPONENTIAL, 2.0};
    double scaled[3];
    tf_apply_pressure(&pf, sp, scaled);

    ASSERT(scaled[0] > scaled[1]);
    ASSERT(scaled[1] > scaled[2]);
    ASSERT_FEQ(scaled[0], exp(2.0), 0.01);
    ASSERT_FEQ(scaled[2], exp(0.0), 0.01);
    return 0;
}

/* ---- Test 12: SelectionPressure — sigma scaling ---- */
int test_pressure_sigma(void) {
    tf_pop_fitness_t pf;
    pf.count = 3;
    pf.fitnesses[0] = 1.0;
    pf.fitnesses[1] = 0.5;
    pf.fitnesses[2] = 0.0;
    pf.mean = 0.5;

    tf_selection_pressure_t sp = {TF_PRESSURE_SIGMA, 0.0};
    double scaled[3];
    tf_apply_pressure(&pf, sp, scaled);

    /* mean=0.5, var=((0.5^2)+(0^2)+(0.5^2))/3=0.1667, stddev≈0.408 */
    /* Above mean gets >1, below gets <1 */
    ASSERT(scaled[0] > 1.0);
    ASSERT(scaled[2] < 1.0);
    return 0;
}

/* ---- Test 13: SelectionPressure — power ---- */
int test_pressure_power(void) {
    tf_pop_fitness_t pf;
    pf.count = 2;
    pf.fitnesses[0] = 0.5;
    pf.fitnesses[1] = 0.9;
    pf.mean = 0.7;

    tf_selection_pressure_t sp = {TF_PRESSURE_POWER, 2.0};
    double scaled[2];
    tf_apply_pressure(&pf, sp, scaled);

    ASSERT_FEQ(scaled[0], 0.25, 0.001);   /* 0.5^2 */
    ASSERT_FEQ(scaled[1], 0.81, 0.001);   /* 0.9^2 */
    return 0;
}

/* ---- Test 14: Elitism — select top 3 from 5 ---- */
int test_elitism_basic(void) {
    int tv[] = {1, 1, 1};
    tf_genome_t target = make_genome(tv, 3);

    tf_genome_t pop[5];
    int g0[] = {1, 1, 1};   /* match 1.0 */
    int g1[] = {-1, -1, -1}; /* 0.0 */
    int g2[] = {1, -1, 1};   /* 2/3 */
    int g3[] = {0, 1, 1};    /* 2/3 */
    int g4[] = {1, 0, 0};    /* 1/3 */

    pop[0] = make_genome(g0, 3);
    pop[1] = make_genome(g1, 3);
    pop[2] = make_genome(g2, 3);
    pop[3] = make_genome(g3, 3);
    pop[4] = make_genome(g4, 3);

    tf_pop_fitness_t pf;
    tf_population_fitness(pop, 5, &target, NULL, tf_fitness_match, &pf);

    tf_genome_t elite[3];
    double elite_f[3];
    size_t n = tf_elitism(&pf, pop, 5, 3, elite, elite_f);

    ASSERT(n == 3);
    ASSERT_FEQ(elite_f[0], 1.0, 0.001);
    /* The top 3 by fitness: 1.0, 2/3, 2/3 */
    ASSERT(elite_f[1] >= elite_f[2]);
    return 0;
}

/* ---- Test 15: Elitism — n_elite > pop_size ---- */
int test_elitism_too_many(void) {
    int tv[] = {1};
    tf_genome_t target = make_genome(tv, 1);

    tf_genome_t pop[2];
    pop[0] = make_genome((int[]){1}, 1);
    pop[1] = make_genome((int[]){-1}, 1);

    tf_pop_fitness_t pf;
    tf_population_fitness(pop, 2, &target, NULL, tf_fitness_match, &pf);

    tf_genome_t elite[5];
    size_t n = tf_elitism(&pf, pop, 2, 5, elite, NULL);
    ASSERT(n == 2); /* capped at pop_size */
    return 0;
}

/* ---- Test 16: sort_by_fitness ---- */
int test_sort_by_fitness(void) {
    tf_pop_fitness_t pf;
    pf.count = 4;
    pf.fitnesses[0] = 0.2;
    pf.fitnesses[1] = 0.8;
    pf.fitnesses[2] = 0.5;
    pf.fitnesses[3] = 0.1;

    size_t indices[4];
    tf_sort_by_fitness(&pf, indices, 4);

    ASSERT(indices[0] == 1); /* 0.8 */
    ASSERT(indices[1] == 2); /* 0.5 */
    ASSERT(indices[2] == 0); /* 0.2 */
    ASSERT(indices[3] == 3); /* 0.1 */
    return 0;
}

/* ---- Test 17: Landscape — no peaks/valleys = base fitness ---- */
int test_landscape_plain(void) {
    int tv[] = {1, -1, 0};
    tf_genome_t target = make_genome(tv, 3);

    tf_landscape_t land;
    tf_landscape_init(&land);

    int gv[] = {1, -1, 0};
    tf_genome_t g = make_genome(gv, 3);
    double f = tf_landscape_fitness(&land, &g, &target, NULL);
    ASSERT_FEQ(f, 1.0, 0.001); /* just base fitness */
    return 0;
}

/* ---- Test 18: Empty genome fitness ---- */
int test_empty_genome(void) {
    tf_genome_t g = {.len = 0};
    tf_genome_t t = {.len = 0};
    double f = tf_fitness_match(&g, &t, NULL);
    ASSERT_FEQ(f, 0.0, 0.001);
    return 0;
}

int main(void) {
    int failures = 0;
    #define RUN(name) do { \
        int r = test_##name(); \
        if (r == 0) printf("  PASS: %s\n", #name); \
        else { printf("  FAIL: %s\n", #name); failures++; } \
    } while(0)

    printf("Running ternary-fitness-c tests...\n\n");

    RUN(fitness_perfect_match);
    RUN(fitness_no_match);
    RUN(fitness_weighted);
    RUN(fitness_hamming);
    RUN(population_fitness_stats);
    RUN(population_single);
    RUN(landscape_peak);
    RUN(landscape_valley);
    RUN(landscape_combined);
    RUN(pressure_rank);
    RUN(pressure_exponential);
    RUN(pressure_sigma);
    RUN(pressure_power);
    RUN(elitism_basic);
    RUN(elitism_too_many);
    RUN(sort_by_fitness);
    RUN(landscape_plain);
    RUN(empty_genome);

    int total = 18;
    printf("\n%d/%d passed\n", total - failures, total);
    return failures;
}
