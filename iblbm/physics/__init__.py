"""실험 상황별 물리 모듈.

각 시나리오(운동, 침강 등)의 물리 로직을 모듈별로 분리.
solver.py는 이 패키지의 함수를 호출만 수행한다.

모듈:
  motion         기존 운동 (oscillating/translating/rotating)
  sedimentation  단일 입자 침강 (Velocity Verlet, 유체력 추출, 중력)
"""

from .motion import update_oscillating, update_rotating, update_translating
from .sedimentation import (
    check_domain_bounds,
    compute_desired_velocity,
    compute_net_gravity,
    extract_hydro_force,
    init_sedimentation_state,
    update_markers,
    verlet_full_step,
    verlet_half_step,
)

__all__ = [
    # motion
    "update_oscillating",
    "update_rotating",
    "update_translating",
    # sedimentation
    "check_domain_bounds",
    "compute_desired_velocity",
    "compute_net_gravity",
    "extract_hydro_force",
    "init_sedimentation_state",
    "update_markers",
    "verlet_full_step",
    "verlet_half_step",
]
