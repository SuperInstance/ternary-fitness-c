# Ternary Fitness (C)

**Ternary Fitness** is a C library for evaluating the fitness of ternary genomes — arrays of {-1, 0, +1} values — against target patterns and configurable fitness landscapes with peaks, valleys, and selection pressure methods. It provides the evaluation engine for ternary genetic algorithms and evolutionary strategy optimization.

## Why It Matters

Genetic algorithms need a fitness function to guide selection — without one, evolution is random search. For ternary genomes, fitness functions have unique properties: the search space is 3ⁿ (vs. 2ⁿ for binary), the neutral state (0) creates a "drift" dimension absent in binary GA, and fitness landscapes can have 3ⁿ local optima. This library provides four fitness models — exact match, Hamming distance, configurable landscapes with peaks/valleys, and selection pressure scaling — covering the range of evaluation strategies needed for ternary evolution experiments. The C implementation runs at native speed, critical when evaluating millions of genomes per generation.

## How It Works

### Fitness Functions

**Exact Match**: fraction of positions matching the target:
```
f_match = Σ 1[gene_i = target_i] / L
```

**Hamming Distance**: normalized distance:
```
f_hamming = 1 - (Σ 1[gene_i ≠ target_i] / L)
```

With optional per-position weights `wᵢ`:
```
f_weighted = Σ wᵢ · 1[gene_i = target_i] / Σ wᵢ
```

### Fitness Landscape

A `tf_landscape_t` defines a multi-modal surface with peaks and valleys:

```c
typedef struct {
    tf_genome_t genome;   // peak center
    double bonus;         // fitness bonus
    double sigma;         // influence radius (Hamming distance)
} tf_peak_t;
```

A genome's landscape fitness combines base fitness (match to target) with Gaussian peaks and valleys:

```
f_landscape = f_base · scale + Σ peak_i · exp(-d²/(2σ_i²)) - Σ valley_j · exp(-d²/(2σ_j²))
```

where `d` is the Hamming distance to each peak/valley center. This creates rugged landscapes with multiple optima — ideal for testing GA escape capabilities.

### Selection Pressure

Raw fitness values are transformed to control selection intensity:

| Method | Formula | Effect |
|--------|---------|--------|
| Rank | `1 + (rank-1)/(N-1) · (k-1)` | Linear rank pressure |
| Exponential | `exp(s · f_raw)` | Strong pressure on best |
| Sigma | `1 + (f_raw - μ) / (2σ)` | Adaptive to population variance |
| Power | `f_raw^k` | Amplifies differences |

High pressure converges fast but risks premature optimization; low pressure maintains diversity.

### Elitism

The `tf_elitism` function preserves the top N agents across generations:

```
elite = top_N(population, by=fitness_desc)
```

This prevents the GA from losing the best solution found due to selection noise.

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Match fitness | O(L) | O(1) |
| Landscape fitness | O(L·P) | O(P) |
| Population fitness | O(N·L·P) | O(N) |
| Selection pressure | O(N log N) | O(N) |
| Elitism | O(N log N) | O(N) |

N = population size, L = genome length, P = number of peaks.

## Quick Start

```c
#include "ternary_fitness.h"
#include <stdio.h>

int main(void) {
    tf_genome_t target = {.genes = {1, 1, -1, 0, 1, -1}, .len = 6};
    tf_genome_t candidate = {.genes = {1, 0, -1, 0, 1, 1}, .len = 6};

    double fit = tf_fitness_match(&candidate, &target, NULL);
    printf("Match fitness: %.3f\n", fit);  // 0.667 (4 of 6 match)

    // Landscape with a peak
    tf_landscape_t land;
    tf_landscape_init(&land);
    tf_landscape_add_peak(&land, &target, 5.0, 3.0);
    double land_fit = tf_landscape_fitness(&land, &candidate, &target, NULL);
    printf("Landscape fitness: %.3f\n", land_fit);

    return 0;
}
```

Compile: `gcc -lm -o demo src/ternary_fitness.c && ./demo`

## API

| Function | Description |
|----------|-------------|
| `tf_fitness_match` | Position-match fitness |
| `tf_fitness_hamming` | Hamming-distance fitness |
| `tf_population_fitness` | Evaluate entire population |
| `tf_landscape_init/add_peak/add_valley` | Configure landscape |
| `tf_landscape_fitness` | Evaluate on landscape |
| `tf_apply_pressure` | Scale raw fitness by pressure method |
| `tf_elitism` | Select top-N agents |
| `tf_sort_by_fitness` | Sort population indices descending |

## Architecture Notes

Ternary Fitness is the selection engine for γ in the γ + η = C framework — it measures how well a ternary genome constructs solutions (γ). The landscape's peaks represent target competence C configurations; the valleys represent η (avoidance zones). Selection pressure controls the γ/η balance — high pressure exploits known peaks (γ-dominated), low pressure explores around valleys (η-balanced). Elitism preserves the maximum C discovered so far. See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md).

## References

1. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. — Fitness landscapes and selection operators.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
3. Mitchell, M. (1996). *An Introduction to Genetic Algorithms*. MIT Press. — On fitness landscape design.

## License

MIT
