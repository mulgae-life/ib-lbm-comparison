"""이동 시나리오 (Re=20, 등속 이동).

MATLAB 참조: IBLBM_moving_DF_GPU.m
"""

from iblbm.config import SimConfig
from iblbm.solver import run

cfg = SimConfig(
    Re=20,
    NN=601,
    xmax=2.0,
    ymax=1.0,
    cylinder_center=(0.45, 0.5),
    cylinder_D_ratio=1 / 20,
    lattice_u=0.1,
    inflow_u=0.1,
    bc_type="velocity_inlet",
    ibm_method="DF",
    max_steps=800_000,
    use_convergence=False,
    check_interval=100,
    motion_type="translating",
    marker_spacing_factor=0.5,
    nx_formula="standard",
)

if __name__ == "__main__":
    result = run(cfg)
    print(f"\nFinal Cd = {result['Cd_history'][-1]:.6f}")
    print(f"Final Cl = {result['Cl_history'][-1]:.6f}")
