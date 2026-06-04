# ternary-fitness-c

C implementation of fitness landscapes for ternary agent strategies {-1, 0, +1}.

Cross-language companion to [ternary-fitness](https://github.com/SuperInstance/ternary-fitness) (Rust).

## API

- `tf_strategy_t` — ternary strategy vector
- `tf_environment_t` — reward matrix per (state, action)
- `tf_strategy_fitness()` — cumulative reward computation
- `tf_shannon_entropy()` / `tf_normalized_entropy()` — strategy entropy
- `tf_hamming_distance()` — strategy distance
- `tf_exhaustive_search()` — enumerate all 3^n strategies
- `tf_analyze_landscape()` — peaks, fitness range, mean
- `tf_population_diversity()` — mean pairwise Hamming distance

## Build & Test

```bash
gcc -o test_fitness tests/test_fitness.c src/ternary_fitness.c -lm -Wall -O2
./test_fitness
```

17 tests passing. No external dependencies.

## License

MIT
