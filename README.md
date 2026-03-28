# Controlled Benchmark of Three Boundary-Enforcement Schemes for the Immersed-Boundary Lattice Boltzmann Method

This repository contains the source code, processed benchmark data, and figures supporting the paper:

> **H. Jo**, "Controlled benchmark of three boundary-enforcement schemes for the immersed-boundary lattice Boltzmann method," *Physics of Fluids* (submitted, 2026).

## Repository contents

| Directory | Description |
|-----------|-------------|
| `iblbm/` | Core IB-LBM solver (LBM + IBM modules) |
| `scenarios/` | Benchmark scenario definitions (steady, oscillating, sedimentation, Taylor-Green) |
| `scripts/` | Analysis script for sedimentation canonical cases |
| `data/` | Processed benchmark outputs (`status.json`, `sedimentation_history.json`, reference CSVs) |
| `figures/` | Final paper figures (Fig. 1-11) |

## Three boundary-enforcement schemes

- **DF** (Direct Forcing) — Peskin-type explicit forcing
- **MDF** (Multi-Direct Forcing) — Iterative correction (N = 5, 10)
- **DFC** (Distribution Function Correction) — Post-collision distribution correction

Each scheme is tested with two delta-function kernels: 3-point hat and 4-point Peskin (P4).

## Requirements

```bash
pip install -r requirements.txt
```

## Data structure

```
data/
├── df_benchmarks/       # DF results (steady, oscillating, Taylor-Green)
├── mdf_benchmarks/      # MDF results (iter5, iter10)
├── dfc_benchmarks/      # DFC results
├── grid_sensitivity/    # Grid convergence study (Re=40, N=801)
└── sedimentation_canonical/   # Particle sedimentation (hat/P4 × 3 density ratios)
```

## What is excluded

Large raw field outputs (`velocity_field.npz`, videos) are excluded due to file size. These are available from the corresponding author upon reasonable request.

## Author

- **Hongju Jo** — Independent Researcher, Seoul, South Korea
- Correspondence: jhjoo3217@yonsei.ac.kr

## License

MIT License — see [LICENSE](./LICENSE).
