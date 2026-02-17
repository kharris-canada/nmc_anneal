# Algorithms

## Simulated Annealing

The annealing algorithm models disorder formation through stochastic local swaps.

### Steps

1. Select two lattice sites
2. Attempt species swap
3. Compute local energy change
4. Accept or reject using Metropolis criterion
5. Gradually change temperature according to user-selected path

### Properties

- Conserves global composition
- Local-energy-based → efficient scaling
- Produces realistic short-range ordering

---

## Local Energy Calculation

Energy is computed from deviations in local oxygen charge balance.

For each oxygen coordination:

- Sum neighboring cation charges
- Compare to ideal neutrality
- Add quadratic penalty

Only local updates are required after each swap, enabling efficient computation.

---

## Delithiation Algorithm

Lithium removal proceeds deterministically:

1. Compute local lithium site energies
2. Rank sites by instability
3. Remove lithium from highest-energy sites
4. Update local charge environment
5. Repeat

This produces physically meaningful lithium ordering evolution.

---

## NMR Spectrum Generation

1. Identify lithium sites
2. Determine local environment
3. Assign chemical shift
4. Add Gaussian/Lorentzian broadening
5. Sum contributions

Produces synthetic MAS NMR spectrum for comparison with experiment.
