# Future Integration: ternary-fitness-c

## Current State
C implementation of ternary fitness evaluation for evolutionary systems. Evaluates agent fitness on ternary genomes with configurable fitness landscapes and selection pressure.

## Integration Opportunities

### With evolution-ternary-c
`evolution-ternary-c` provides the evolutionary operators (selection, crossover, mutation); `ternary-fitness-c` provides the evaluation. Together on ESP32: a complete evolutionary optimization loop running on bare metal. Agents evolve their ternary strategies in-situ on the microcontroller.

### With ternary-cell (Cell Fitness Evaluation)
Cells need fitness evaluation for the `gc` phase — which cells to keep, which to prune. The C port evaluates cell fitness on-device using the same metrics as the Rust `ternary-fitness` crate. Low-fitness cells are candidates for apoptosis.

### With ternary-evolution-advanced
The Rust crate provides advanced algorithms (DE, NSGA-II); the C port provides basic evaluation for embedded deployment. Cross-language pipeline: use advanced Rust optimization to find good fitness functions, compile them to C for edge deployment.

## Potential in Mature Systems
In room-as-codespace, edge devices evaluate their own agent fitness locally. No need to send genomes to the cloud for evaluation. The C port runs the fitness landscape directly on ESP32. Results are reported back to PLATO for fleet-wide optimization, but the evaluation is local and fast.

## Cross-Pollination Ideas
- Fitness evaluation as a room health metric — average fitness of room agents indicates room quality
- Cross-language fitness consistency: ensure C and Rust produce identical rankings
- Configurable fitness landscapes per room type via compiled policy

## Dependencies for Next Steps
- Integration with evolution-ternary-c for complete on-device evolution
- Fitness landscape configuration format (JSON → C struct)
- Benchmarking: C fitness evaluation throughput on ESP32
