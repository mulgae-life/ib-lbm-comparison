"""정상류 시나리오 (Re=40, Cd ≈ 1.5-1.6).

MATLAB 참조: IBLBM_steady_DF_CPU.m
"""

from iblbm.config import SimConfig
from iblbm.solver import run

cfg = SimConfig(
    Re=40,
    NN=1601,
    xmax=1.0,
    ymax=1.0,
    cylinder_center=(0.4, 0.5),
    cylinder_D_ratio=1 / 40,
    lattice_u=0.1,
    inflow_u=0.1,
    bc_type="velocity_inlet",
    ibm_method="DF",
    max_steps=800_000,
    convergence_threshold=1e-5,
    convergence_start=100_000,
    check_interval=100,
    use_convergence=True,
    marker_spacing_factor=2/3,
    nx_formula="standard",
)

if __name__ == "__main__":
    result = run(cfg)
    print(f"\nFinal Cd = {result['Cd_history'][-1]:.6f}")
    print(f"Final Cl = {result['Cl_history'][-1]:.6f}")
