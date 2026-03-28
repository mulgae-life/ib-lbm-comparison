"""시뮬레이션 설정 (SimConfig).

각 시나리오(정상류/진동/이동/회전)의 물리/수치 파라미터를 dataclass로 정의.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """IB-LBM 시뮬레이션 설정.

    Attributes:
        Re: Reynolds 수 (실린더 직경 기준, 회전은 r^2*omega/nu)
        NN: dx 해상도 파라미터 (dx = 1/(NN-1)). ny는 ymax/dx에서 유도.
        xmax: x 방향 도메인 크기 [0, xmax]
        ymax: y 방향 도메인 크기 [0, ymax]
        cylinder_center: 실린더 중심 좌표 (x, y) — 도메인 비율
        cylinder_D_ratio: 실린더 직경/물리 길이 비율
        lattice_u: 격자 단위 속도
        inflow_u: 유입 속도 (격자 단위)
        bc_type: 경계 조건 패턴 ('velocity_inlet' | 'open_boundary' | 'settling_channel')
        ibm_method: IBM 방법론 ('DF' | 'MDF' | 'DFC')
        mdf_iterations: MDF 반복 횟수
        max_steps: 최대 시간 스텝 수
        convergence_threshold: 수렴 판정 임계값
        convergence_start: 수렴 판정 시작 스텝 (I3: CPU 기준 100,000)
        check_interval: 수렴 체크 간격 (스텝)
        use_convergence: 수렴 판정 사용 여부
        motion_type: 운동 유형 (None | 'oscillating' | 'translating' | 'rotating' | 'sedimentation')
        KC: Keulegan-Carpenter 수 (진동 시나리오)
        omega: 각속도 (회전 시나리오, 격자 단위)
        marker_spacing_factor: 라그랑주 마커 간격 = dx * factor (T5)
            정상류: 0.5, 진동: 2/3
        nx_formula: 격자 nx 계산 방식 (T4)
            'standard': nx = NN*xmax - (xmax-1) — 정상류/회전 (dx=dy 보장)
            'simple': nx = NN*xmax — 진동 ((nx-1)*dx ≠ xmax 이산화 부산물, ~0.08% 차이)
    """

    Re: float
    NN: int
    xmax: float = 1.0
    ymax: float = 1.0
    cylinder_center: tuple[float, float] = (0.4, 0.5)  # I1 수정
    cylinder_D_ratio: float = 1 / 40
    lattice_u: float = 0.1
    inflow_u: float = 0.1
    bc_type: str = "velocity_inlet"
    delta_type: str = "peskin4pt"  # "peskin4pt" | "hat"
    ibm_method: str = "DF"   # "DF" | "MDF" | "DFC"
    mdf_iterations: int = 10
    max_steps: int = 800_000
    convergence_threshold: float = 1e-5
    convergence_start: int = 100_000  # I3: CPU 기준
    check_interval: int = 100
    use_convergence: bool = True
    motion_type: str | None = None
    KC: float = 5.0
    # --- 침강 (Sedimentation) ---
    rho_ratio: float = 1.0        # rho_s / rho_f (고체/유체 밀도비). 1.0 = 중성 부력
    gravity: float = 0.0           # 격자 단위 중력 가속도 (양수, -y 방향 적용)
    omega: float = 0.0
    marker_spacing_factor: float = 0.5  # T5: Δs = factor * dx
    nx_formula: str = "standard"  # T4: 'standard' | 'simple'
    use_gpu: bool = False  # CuPy GPU 가속 사용 여부

    # Taylor-Green 감쇠 와류 (D1)
    scenario_type: str | None = None  # "taylor_green" 또는 None
    tg_L: float = 1.0      # 도메인 반폭 (도메인 = [-L, L]²)
    tg_u0: float = 0.1     # 초기 속도 진폭
    tg_T_end: float | None = None  # 측정 시점 (물리 시간)
    tg_with_ibm: bool = False      # True: IBM 포함 수렴 테스트

    # 물리 상수 (단위 변환용)
    phy_u: float = 1e-3       # m/s
    phy_l: float = 1e-3       # m
    phy_density: float = 1000  # kg/m^3

    def warn_if_unstable(self) -> str | None:
        """tau가 안정 한계에 가까운지 경고.

        BGK-LBM은 tau → 0.5에서 불안정. 경험적으로 tau ≥ 0.6 필요.
        tau = 3*nu + 0.5, nu = lattice_u * lattice_D / Re
        lattice_D = (phy_l * cylinder_D_ratio) / (phy_l / (NN-1))
                  = cylinder_D_ratio * (NN-1)
        """
        lattice_D = self.cylinder_D_ratio * (self.NN - 1)
        lattice_nu = self.lattice_u * lattice_D / self.Re
        tau = 3.0 * lattice_nu + 0.5
        if tau < 0.55:
            return (f"tau={tau:.4f} < 0.55: BGK-LBM이 불안정할 수 있습니다. "
                    f"NN을 늘리거나 Re를 줄이세요 (현재 NN={self.NN}, Re={self.Re}). "
                    f"NN ≥ {int(0.6 * self.Re / (3.0 * self.lattice_u * self.cylinder_D_ratio) + 1)} 권장.")
        return None
