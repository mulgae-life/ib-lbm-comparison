"""초기화 모듈.

격자 생성, 단위 변환, Re→tau 계산, 라그랑주 경계점 초기화,
경계 인덱스, 초기 분포 함수를 생성한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from .backend import xp as np

from .config import SimConfig
from .diagnostics import tg_analytical_velocity_field
from .lbm import compute_feq
from .lbm import D2Q9, make_d2q9


@dataclass
class BoundaryIndices:
    """1D 배열 기준 경계 노드 인덱스 (0-base).

    left   = [0, nx, 2*nx, ...]
    right  = [nx-1, 2*nx-1, ...]
    bottom = [1, 2, ..., nx-2]  (velocity_inlet: 코너 제외)
    top    = [nx*(ny-1)+1, ..., nodenums-2]
    """

    left: np.ndarray
    right: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    nx: int
    ny: int


@dataclass
class SimState:
    """시뮬레이션 상태. initialize()의 반환값."""

    # 격자
    nx: int
    ny: int
    nodenums: int
    dx: float
    dy: float
    dt: float
    tau: float

    # 물리 (격자 단위)
    r: float               # 실린더 반지름 (도메인 비율)
    lattice_r: float        # 실린더 반지름 (격자 단위, 실수)
    in_u: float             # 유입 속도
    dens: float             # 격자 밀도 (= 1.0)

    # 분포 함수
    fstar: np.ndarray       # (nodenums, 9) — post-streaming
    feq: np.ndarray         # (nodenums, 9) — 평형

    # 거시 변수
    ro: np.ndarray          # (nodenums,) — 밀도
    ro_initial: np.ndarray  # (nodenums,) — 초기 밀도 (C2: IBM에 전달)
    U: np.ndarray           # (nodenums, 2) — 속도

    # IB 체적력
    fib: np.ndarray         # (nodenums, 2)

    # 라그랑주 경계
    Lx: np.ndarray          # (Lb,) — x 좌표
    Ly: np.ndarray          # (Lb,) — y 좌표
    Lb: int                 # 경계점 수
    Larea: float            # 각 점의 호 길이 (Δs)
    desired_velocity: np.ndarray  # (Lb, 2) — I5: 배열로 통일

    # 경계 인덱스
    idx: BoundaryIndices

    # 격자
    lattice: D2Q9

    # DFC 캐시 (ibm_method="DFC" 전용)
    lambda_cache: np.ndarray | None = None       # lambda(k) 캐시 (고정 경계)
    dfc_force_lagr: np.ndarray | None = None     # 최근 DFC 유체력 (Lb, 2)

    # 침강 전용 (motion_type="sedimentation")
    particle_pos: np.ndarray | None = None     # (2,) 입자 중심 [x, y] (도메인 좌표)
    particle_vel: np.ndarray | None = None     # (2,) 입자 속도 [vx, vy] (격자 단위)
    particle_force: np.ndarray | None = None   # (2,) 현재 스텝 유체력 [Fx, Fy]
    particle_mass: float = 0.0                 # 격자 단위 입자 질량
    gravity_lattice: float = 0.0               # 격자 단위 중력 가속도 (양수)


def initialize(cfg: SimConfig) -> SimState:
    """시뮬레이션 초기화.

    Args:
        cfg: SimConfig 설정

    Returns:
        SimState: 초기화된 시뮬레이션 상태
    """
    lattice = make_d2q9()

    # --- 격자 생성 ---
    # T4: 시나리오별 nx 공식
    if cfg.nx_formula == "standard":
        # 정상류/회전: nx = NN*xmax - (xmax-1) → dx=dy 보장
        nx = int(cfg.NN * cfg.xmax - (cfg.xmax - 1))
    else:
        # 진동: nx = NN*xmax → dx ≈ dy (0.08% 차이)
        nx = int(cfg.NN * cfg.xmax)

    # NN은 dx 해상도 파라미터: dx = 1/(NN-1)
    dx = 1.0 / (cfg.NN - 1)

    # ny를 ymax에서 유도 — 정사각 격자(dx=dy) 보장
    ny = round(cfg.ymax / dx) + 1
    dy = cfg.ymax / (ny - 1)

    # 정사각 격자 검증
    if abs(dx - dy) / dx > 1e-6:
        raise ValueError(
            f"정사각 격자 불일치: dx={dx:.8f}, dy={dy:.8f}. "
            f"ymax={cfg.ymax}가 dx와 호환되지 않습니다."
        )

    nodenums = nx * ny
    dt = 1.0  # Lattice_Unit_dt

    # --- 단위 변환 ---
    phy_D = cfg.phy_l * cfg.cylinder_D_ratio  # 물리 실린더 직경
    phy_dx = cfg.phy_l / (cfg.NN - 1)

    Cx = phy_dx / 1.0  # Lattice_Unit_dx = 1
    # Cu = cfg.phy_u / cfg.lattice_u
    # Ct = Cx / Cu

    r = phy_D / 2.0 / cfg.phy_l  # 도메인 비율 반지름
    lattice_D = phy_D / Cx
    lattice_r = lattice_D / 2.0

    # Re → nu → tau
    if cfg.motion_type == "rotating":
        # 회전: nu = r^2 * omega / Re
        lattice_nu = lattice_r**2 * cfg.omega / cfg.Re
    else:
        # 기본: nu = U * D / Re
        lattice_nu = cfg.lattice_u * 2.0 * lattice_r / cfg.Re

    tau = 3.0 * lattice_nu + 0.5

    in_u = cfg.inflow_u
    dens = 1.0  # Lattice_Unit_dens

    # --- 경계 인덱스 (0-base) ---
    if cfg.bc_type in ("open_boundary", "settling_channel"):
        # 개방 경계 / 침강 채널: 4면 모두 코너 제외 → corner closure가 단독 처리
        left = np.arange(nx, nodenums - nx, nx)         # y=1..ny-2 at x=0
        right = np.arange(2 * nx - 1, nodenums - nx, nx)  # y=1..ny-2 at x=nx-1
        bottom = np.arange(1, nx - 1)                   # x=1..nx-2 at y=0
        top = np.arange(nx * (ny - 1) + 1, nodenums - 1)  # x=1..nx-2 at y=ny-1
    else:
        # 정상류/이동/회전: left/right에 코너 포함, top/bottom에서 코너 제외
        left = np.arange(0, nodenums, nx)
        right = np.arange(nx - 1, nodenums, nx)
        bottom = np.arange(1, nx - 1)
        top = np.arange(nx * (ny - 1) + 1, nodenums - 1)
    idx = BoundaryIndices(left=left, right=right, top=top, bottom=bottom,
                          nx=nx, ny=ny)

    # --- 초기 거시 변수 ---
    U = np.zeros((nodenums, 2))
    if cfg.scenario_type == "taylor_green":
        # TG: 격자 인덱스를 중심 기준 격자 좌표로 변환 (lattice units)
        # 격자 [0, nx-1] → [-L_lat, L_lat], L_lat = (NN-1)/2
        L_lat = 0.5 * (cfg.NN - 1)
        X_c = np.arange(nx, dtype=float) - (nx - 1) / 2.0
        Y_c = np.arange(ny, dtype=float) - (ny - 1) / 2.0
        XX, YY = np.meshgrid(X_c, Y_c)  # (ny, nx) — 격자 좌표
        # t=0 해석해로 초기 속도 설정 (모든 변수 lattice units)
        lattice_D_tg = cfg.cylinder_D_ratio * (cfg.NN - 1)
        nu_tg = cfg.lattice_u * lattice_D_tg / cfg.Re
        ux_ana, uy_ana = tg_analytical_velocity_field(
            XX.ravel(), YY.ravel(), 0.0, cfg.tg_u0, L_lat, nu_tg,
        )
        U[:, 0] = ux_ana
        U[:, 1] = uy_ana
    elif cfg.motion_type == "oscillating":
        pass  # U = 0 (정지 유체)
    elif cfg.motion_type == "sedimentation":
        pass  # U = 0 (정지 유체, oscillating과 동일)
    else:
        U[:, 0] = in_u  # u = inflow_u

    ro = dens * np.ones(nodenums)
    ro_initial = dens * np.ones(nodenums)  # C2: IBM용 초기 밀도

    # 초기 분포 함수
    feq = compute_feq(ro, U, lattice)
    fstar = feq.copy()

    # IB 체적력
    fib = np.zeros((nodenums, 2))

    # --- 라그랑주 경계점 ---
    # T5: marker_spacing_factor = Δs (격자 단위 마커 간격)
    dd = 2.0 * np.pi * lattice_r / cfg.marker_spacing_factor
    step = 1.0 / dd
    Ldx = np.arange(0, 1.0 + step * 0.5, step)  # 매개변수 [0, ~1)
    # 1.0 이상인 점 제거
    Ldx = Ldx[Ldx <= 1.0 + 1e-12]
    # 마지막 점이 첫 점과 겹치면 제거 (원 닫힘)
    if len(Ldx) > 1 and np.abs(Ldx[-1] - 1.0) < step * 0.5:
        Ldx = Ldx[:-1]

    # 좌표는 도메인 비율 반지름(r) 사용, 격자 단위가 아님
    cx, cy = cfg.cylinder_center

    # 실린더 경계 검사
    if cx - r < 0 or cx + r > cfg.xmax or cy - r < 0 or cy + r > cfg.ymax:
        raise ValueError(
            f"실린더(cx={cx}, cy={cy}, r={r})가 도메인 밖: "
            f"[0,{cfg.xmax}]×[0,{cfg.ymax}]"
        )

    theta = 2.0 * np.pi * Ldx
    Lx = r * np.cos(theta) + cx
    Ly = r * np.sin(theta) + cy
    Lb = len(Lx)
    Larea = 2.0 * np.pi * lattice_r / Lb  # 격자 단위 호 길이

    # I5: desired_velocity를 (Lb, 2) 배열로 초기화
    desired_velocity = np.zeros((Lb, 2))

    state = SimState(
        nx=nx, ny=ny, nodenums=nodenums,
        dx=dx, dy=dy, dt=dt, tau=tau,
        r=r, lattice_r=lattice_r, in_u=in_u, dens=dens,
        fstar=fstar, feq=feq,
        ro=ro, ro_initial=ro_initial, U=U,
        fib=fib,
        Lx=Lx, Ly=Ly, Lb=Lb, Larea=Larea,
        desired_velocity=desired_velocity,
        idx=idx, lattice=lattice,
    )

    if cfg.motion_type == "sedimentation":
        from .physics.sedimentation import init_sedimentation_state
        init_sedimentation_state(state, cfg, lattice_r)

    return state
