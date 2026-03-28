"""침강 물리 모듈.

단일 입자 침강의 핵심 로직:
  - 중력/부력 계산
  - IBM 유체력 추출 (DF/MDF: fib 합산, DFC: dfc_force_lagr 합산)
  - Velocity Verlet 반스텝/전스텝
  - 마커 갱신 + desired_velocity 변환
  - SimState 초기화 헬퍼
  - 도메인 경계 검사
"""

from __future__ import annotations

import numpy as np

from ..config import SimConfig


# =====================================================================
# 물리 상수 계산
# =====================================================================

def compute_net_gravity(
    rho_ratio: float, r_lattice: float, g_lattice: float,
) -> np.ndarray:
    """순 중력 (중력 - 부력) 계산 — Reduced Pressure 방식.

    Reduced Pressure: 정수압을 NS 방정식에서 분리.
    유체 방정식에 중력 없음 (∵ g·H/cs² >> 1로 Full Pressure 부적합).
    대신 입자에 순 중력(중력-부력)을 직접 적용:

      F_net = -(rho_s - rho_f) * V * g * ey
            = -(rho_ratio - 1) * pi * r^2 * g_lattice * ey

    IBM이 전달하는 유체력(drag)이 이 순 중력과 평형을 이루면 종단 속도.

    부호 규약: y 아래 방향이 음수.

    Returns:
        (2,) 순 중력 벡터 [Fx=0, Fy<0]
    """
    return np.array([
        0.0,
        -(rho_ratio - 1.0) * np.pi * r_lattice**2 * g_lattice,
    ])


# =====================================================================
# 유체력 추출
# =====================================================================

def extract_hydro_force(
    fib: np.ndarray | None,
    dfc_force_lagr: np.ndarray | None,
    ibm_method: str,
    use_gpu: bool,
) -> np.ndarray:
    """IBM 유체력 추출 -- 입자에 작용하는 힘 반환.

    DF/MDF: fib = 유체에 가한 체적력 → Newton 3법칙으로 부호 반전
            F_on_particle = -sum(fib)

    DFC: dfc_force_lagr = -dS * λ * (deviation @ e)
         이미 '-' 부호가 포함되어 "입자에 작용하는 힘"으로 정의됨
         F_on_particle = +sum(dfc_force_lagr) (부호 반전 불필요)

    주의: 기존 Cd/Cl 진단(compute_cd_cl_dfc)에서는 abs()를 사용하여
    부호와 무관했으나, 입자 역학에서는 부호가 critical.

    Returns:
        (2,) 입자에 작용하는 유체력 (CPU numpy 배열)
    """
    if ibm_method == "DFC":
        if dfc_force_lagr is None:
            raise ValueError("DFC인데 dfc_force_lagr가 None")
        # DFC: dfc_force_lagr는 이미 입자에 작용하는 힘 (부호 반전 불필요)
        F_on_particle = np.array([
            float(dfc_force_lagr[:, 0].sum()),
            float(dfc_force_lagr[:, 1].sum()),
        ])
    else:
        if fib is None:
            raise ValueError(f"{ibm_method}인데 fib가 None")
        # DF/MDF: fib는 유체에 가한 힘 → Newton 3법칙으로 부호 반전
        F_on_fluid = np.array([
            float(fib[:, 0].sum()),
            float(fib[:, 1].sum()),
        ])
        F_on_particle = -F_on_fluid

    return F_on_particle


# =====================================================================
# Velocity Verlet 적분
# =====================================================================

def verlet_half_step(
    pos: np.ndarray,          # (2,) 현재 위치 (도메인 좌표)
    vel: np.ndarray,          # (2,) 현재 속도 (격자 단위)
    force_hydro: np.ndarray,  # (2,) 현재 유체력 (격자 단위)
    mass: float,
    rho_ratio: float,
    r_lattice: float,
    g_lattice: float,
    dt: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Verlet 반스텝: v(n) → v(n+1/2), x(n) → x(n+1).

    속도는 격자 단위, 위치는 도메인 좌표이므로 dx 변환 필요:
      v(n+1/2) = v(n) + (dt/2) * F_total / m       [격자 단위]
      x(n+1) = x(n) + dt * v(n+1/2) * dx            [도메인 좌표]

    Returns:
        pos_new: (2,) 새 위치 (도메인 좌표)
        vel_half: (2,) 반스텝 속도 (격자 단위)
    """
    F_gravity = compute_net_gravity(rho_ratio, r_lattice, g_lattice)
    F_total = force_hydro + F_gravity
    vel_half = vel + 0.5 * dt * F_total / mass
    pos_new = pos + dt * vel_half * dx
    return pos_new, vel_half


def verlet_full_step(
    vel_half: np.ndarray,         # (2,) 반스텝 속도
    force_hydro_new: np.ndarray,  # (2,) 새 유체력
    mass: float,
    rho_ratio: float,
    r_lattice: float,
    g_lattice: float,
    dt: float,
) -> np.ndarray:
    """Verlet 전스텝: v(n+1/2) → v(n+1).

    F_total_new = force_hydro_new + F_gravity
    v(n+1) = v(n+1/2) + (dt/2) * F_total_new / m

    Returns:
        vel_new: (2,) 완전 속도
    """
    F_gravity = compute_net_gravity(rho_ratio, r_lattice, g_lattice)
    F_total_new = force_hydro_new + F_gravity
    vel_new = vel_half + 0.5 * dt * F_total_new / mass
    return vel_new


# =====================================================================
# 마커 갱신
# =====================================================================

def update_markers(
    Lx_c: np.ndarray,    # (Lb,) 초기 마커 x좌표
    Ly_c: np.ndarray,    # (Lb,) 초기 마커 y좌표
    cx_init: float,       # 초기 중심 x
    cy_init: float,       # 초기 중심 y
    pos_new: np.ndarray,  # (2,) 새 중심 위치
) -> tuple[np.ndarray, np.ndarray]:
    """입자 이동에 따른 마커 좌표 갱신.

    강체 병진: 새 마커 = 초기 상대좌표 + 새 중심
    기존 oscillating 시나리오의 패턴과 동일.

    Returns:
        Lx_new, Ly_new: 갱신된 마커 좌표
    """
    dx = pos_new[0] - cx_init
    dy = pos_new[1] - cy_init
    return Lx_c + dx, Ly_c + dy


def compute_desired_velocity(
    vel: np.ndarray,  # (2,) 입자 속도
    Lb: int,
) -> np.ndarray:
    """강체 병진 속도를 마커별 desired_velocity로 변환.

    강체이므로 모든 마커가 동일한 속도를 가진다.

    Returns:
        desired_vel: (Lb, 2) 마커별 목표 속도
    """
    desired_vel = np.empty((Lb, 2))
    desired_vel[:, 0] = vel[0]
    desired_vel[:, 1] = vel[1]
    return desired_vel


# =====================================================================
# 도메인 경계 검사
# =====================================================================

def check_domain_bounds(
    pos: np.ndarray,    # (2,) 입자 위치
    r: float,           # 도메인 좌표 반지름
    xmax: float,
    ymax: float,
    safety: float = 2.0,   # delta support 반경 (격자 단위)
    dx: float = 1.0,
) -> str | None:
    """입자가 도메인 경계에 근접했는지 검사.

    delta function 지지 영역이 벽면과 겹치면 힘 계산이 부정확해진다.
    safety margin = safety * dx (도메인 좌표).

    Returns:
        None이면 안전, str이면 경고 메시지
    """
    margin = safety * dx
    x, y = pos

    if y - r < margin:
        return f"입자 하단이 하벽에 근접 (y-r={y-r:.4f} < margin={margin:.4f})"
    if y + r > ymax - margin:
        return f"입자 상단이 상벽에 근접 (y+r={y+r:.4f} > ymax-margin={ymax-margin:.4f})"
    if x - r < margin:
        return f"입자 좌측이 좌벽에 근접"
    if x + r > xmax - margin:
        return f"입자 우측이 우벽에 근접"

    return None


# =====================================================================
# SimState 초기화
# =====================================================================

def init_sedimentation_state(
    s,  # SimState (순환 import 방지를 위해 타입 힌트 생략)
    cfg: SimConfig,
    lattice_r: float,
) -> None:
    """침강 SimState 필드를 in-place 초기화.

    initialize() 완료 후 호출. SimState에 침강 전용 필드를 채운다.
    """
    cx, cy = cfg.cylinder_center
    s.particle_pos = np.array([cx, cy], dtype=float)
    s.particle_vel = np.array([0.0, 0.0], dtype=float)
    s.particle_force = np.array([0.0, 0.0], dtype=float)
    s.particle_mass = cfg.rho_ratio * np.pi * lattice_r**2
    s.gravity_lattice = cfg.gravity

    # 입력 검증
    if cfg.gravity <= 0.0:
        raise ValueError(
            f"침강에는 gravity > 0 필요 (현재: {cfg.gravity}). "
            f"시나리오에서 물리 단위를 변환하여 전달하세요."
        )
    if cfg.rho_ratio <= 1.0:
        raise ValueError(
            f"침강에는 rho_ratio > 1.0 필요 (현재: {cfg.rho_ratio})."
        )
