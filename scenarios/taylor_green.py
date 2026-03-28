"""Taylor-Green 감쇠 와류 시나리오 (D1: 공간 수렴 차수 검증).

Kang & Hassan (2011) §4.1 설정:
  - 도메인: [-L, L]² = [-1, 1]²
  - Re = u0 * L / nu = 10 (Kang 기준)
  - 측정 시점: T = L / u0 (1 convective time)
  - 해석해가 존재 → 격자 세분화로 L2 오차 감소 → 수렴 차수 ≥ 2차 증명

4개 해상도 (D = 실린더 직경 = L in lattice units):
  D = {10, 20, 40, 80} Δx
"""

from iblbm.config import SimConfig


def make_tg_config(D_over_dx: int, use_ibm: bool = False) -> SimConfig:
    """TG 시나리오 설정 생성.

    Args:
        D_over_dx: 실린더 직경 해상도 (격자점 수). D = L in TG 설정.
        use_ibm: IBM 포함 수렴 테스트 여부 (Kang Table I "with IB")

    Returns:
        SimConfig
    """
    # L = 1.0 (물리 반폭)
    L = 1.0
    u0 = 0.04  # 격자 속도 진폭 (Ma = u0 * sqrt(3) ≈ 0.07, 압축성 오차 최소)
    Re_tg = 10.0

    # D = L → cylinder_D_ratio = D / domain_size = 1 / 2  (도메인 = 2L = 2)
    # NN: 도메인 [0, 2L]을 NN-1 등분. D = L = (NN-1) * D_ratio * dx
    #   D_ratio = 0.5 → lattice_D = 0.5 * (NN-1)
    #   D_over_dx = lattice_D = 0.5 * (NN-1) → NN = 2*D_over_dx + 1
    NN = 2 * D_over_dx + 1

    # nu = u0 * lattice_D / Re → tau = 3*nu + 0.5
    lattice_D = 0.5 * (NN - 1)  # = D_over_dx
    nu = u0 * lattice_D / Re_tg
    # dt = 1.0 (lattice), dx = 1/(NN-1)
    # 물리 시간 T = L / u0 → lattice 스텝 = T / dt_phys
    # dt_phys = dx / (u0 / u0_lattice) ... 하지만 LBM에서 dt=1
    # convective time: T_conv = L / u0.
    # 격자 단위: lattice_D / u0 = D_over_dx / u0
    T_lattice = lattice_D / u0  # 1 convective time in lattice steps

    max_steps = int(round(T_lattice))  # 정확히 1 convective time에서 측정

    return SimConfig(
        Re=Re_tg,
        NN=NN,
        xmax=1.0,  # 도메인 [0, 1] (비율, 내부에서 [-L,L]로 매핑)
        ymax=1.0,
        cylinder_center=(0.5, 0.5),
        cylinder_D_ratio=0.5,  # D = L = 도메인/2
        lattice_u=u0,
        inflow_u=u0,
        bc_type="velocity_inlet",  # apply_bc_analytical이 오버라이드
        ibm_method="DF",
        delta_type="peskin4pt",
        max_steps=max_steps,
        convergence_threshold=1e-10,
        convergence_start=max_steps,
        check_interval=max(1, int(T_lattice) // 10),
        use_convergence=False,
        marker_spacing_factor=0.5,
        nx_formula="standard",
        scenario_type="taylor_green",
        tg_L=L,
        tg_u0=u0,
        tg_T_end=T_lattice,  # lattice 스텝 단위 측정 시점
        tg_with_ibm=use_ibm,
    )


# 기본 해상도 목록
RESOLUTIONS = [10, 20, 40, 80]
