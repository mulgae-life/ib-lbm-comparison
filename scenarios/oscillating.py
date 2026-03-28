"""진동 시나리오 (Re=100, KC=5).

MATLAB 참조: IBLBM_oscillating_MDF_GPU.m
"""

from iblbm.config import SimConfig
from iblbm.solver import run

cfg = SimConfig(
    Re=100,
    NN=400,
    xmax=1.5,
    ymax=1.0,
    cylinder_center=(0.75, 0.5),
    cylinder_D_ratio=1 / 20,
    lattice_u=0.1,
    inflow_u=0.0,
    bc_type="open_boundary",
    ibm_method="MDF",
    max_steps=15_500,
    use_convergence=False,
    check_interval=100,
    motion_type="oscillating",
    KC=5.0,
    marker_spacing_factor=2.0 / 3.0,  # T5: 진동은 Δs ≈ 0.667 dx
    nx_formula="simple",  # T4: nx = NN*xmax
)

if __name__ == "__main__":
    result = run(cfg)
