"""진단: Cd/Cl 계산, 수렴 판정.

MATLAB 참조: IBLBM_steady_DF_CPU.m lines 206-241

Cd = |sum(fib[:,0])| / (mean(ro) * u^2 * r)  — x 방향 (항력)
Cl =  sum(fib[:,1])  / (mean(ro) * u^2 * r)  — y 방향 (양력, 부호 유지)

수렴 판정:
  L2 상대 오차 = sqrt(sum((U_new - U_old)^2) / sum(U_old^2))
  convergence_start 이후, check_interval 간격으로 체크, threshold 미만이면 수렴.
"""

from __future__ import annotations

import numpy as np


def compute_cd_cl(
    fib: np.ndarray, ro: np.ndarray,
    u_ref: float, r_lattice: float,
    motion_type: str | None = None,
) -> tuple[float, float]:
    """항력/양력 계수 계산.

    MATLAB 시나리오별 부호 규약:
      정상류/진동: Cd = abs(sum(fib[:,0])) / denom,  Cl = sum(fib[:,1]) / denom (부호 유지)
      이동/회전:   Cd = -sum(fib[:,0]) / denom,      Cl = sum(fib[:,1]) / denom

    Cd는 항상 양수 (abs 또는 부호 반전)로 반환.
    Cl은 부호를 유지하여 와류 방출에 의한 부호 교대를 보존한다.
    이동/회전에서 Cd 부호 반전 이유: 실린더 운동 방향과 항력 방향이 반대이므로
    fib의 합이 음수가 되고, -를 붙여 양의 Cd를 얻음.

    Args:
        fib: IB 체적력, (nodenums, 2)
        ro: 밀도, (nodenums,)
        u_ref: 참조 속도 (inflow_u, lattice 단위)
        r_lattice: 실린더 반지름 (lattice 단위)
        motion_type: 운동 유형 (None | 'oscillating' | 'translating' | 'rotating')

    Returns:
        Cd, Cl
    """
    denom = np.mean(ro) * u_ref**2 * r_lattice
    if motion_type in ("translating", "rotating"):
        # MATLAB: CdE = -(sum(fib(:,1))) / denom
        Cd = -np.sum(fib[:, 0]) / denom
        Cl = np.sum(fib[:, 1]) / denom
    else:
        # MATLAB: CdE = abs(sum(fib(:,1))) / denom
        Cd = np.abs(np.sum(fib[:, 0])) / denom
        Cl = np.sum(fib[:, 1]) / denom  # 부호 유지 (와류 방출 패턴 보존)
    return Cd, Cl


def compute_cd_cl_dfc(
    dfc_force_lagr: np.ndarray, ro_mean: float,
    u_ref: float, r_lattice: float,
    motion_type: str | None = None,
) -> tuple[float, float]:
    """DFC 유체력 기반 Cd/Cl 계산.

    기존 compute_cd_cl() 재사용 불가:
      DF: fib = Eulerian spread 힘, sum(fib) = Sigma_k F_lagr(k)*dS
      DFC: F_s(k) = per-point 힘, lambda에 1/dS 포함 -> Sigma_k F_s(k) = 총 물리 힘
    spread->sum 하면 dS 이중 적용 오류.

    Args:
        dfc_force_lagr: (Lb, 2) -- Eq.25 per-point 유체력 (dS 불필요)
        ro_mean: float -- 평균 밀도
        u_ref: float -- 참조 속도 (격자 단위)
        r_lattice: float -- 실린더 반지름 (격자 단위)
        motion_type: 운동 유형

    Returns:
        Cd, Cl
    """
    total_force = np.sum(dfc_force_lagr, axis=0)  # (2,)
    denom = ro_mean * u_ref**2 * r_lattice

    if motion_type in ("translating", "rotating"):
        Cd = -float(total_force[0]) / denom
        Cl = float(total_force[1]) / denom
    else:
        Cd = float(np.abs(total_force[0])) / denom
        Cl = float(total_force[1]) / denom

    return Cd, Cl


def check_convergence(
    Eux_new: np.ndarray, Euy_new: np.ndarray,
    Eux_old: np.ndarray, Euy_old: np.ndarray,
) -> float:
    """L2 상대 오차 계산 (수렴 판정).

    MATLAB lines 237-238:
      error = sqrt(sum(sum((Eux-pre_ux).^2 + (Euy-pre_uy).^2))
                   / sum(sum(pre_ux.^2 + pre_uy.^2)));

    Args:
        Eux_new, Euy_new: 현재 스텝 2D 속도장
        Eux_old, Euy_old: 이전 체크 시점 2D 속도장

    Returns:
        L2 상대 오차 (무차원)
    """
    numer = np.sum((Eux_new - Eux_old)**2 + (Euy_new - Euy_old)**2)
    denom = np.sum(Eux_old**2 + Euy_old**2)
    if denom == 0.0:
        return 1.0
    return np.sqrt(numer / denom)


def compute_strouhal(
    Cl_history: np.ndarray,
    steps: np.ndarray,
    D_lattice: float,
    u_ref: float,
    check_interval: int,
) -> float:
    """Re≥100 비정상류에서 Cl 시계열로부터 Strouhal 수 계산.

    알고리즘:
    1. 후반 50% Cl 시계열 추출
    2. 평균 제거 (디트렌드)
    3. np.fft.rfft → 파워 스펙트럼
    4. DC 제외 최대 파워 피크 주파수 추출
    5. St = f_shed * D_lattice / u_ref

    |Cl| (항상 양수) 데이터 처리:
      compute_cd_cl()이 정상류 케이스에서 np.abs()를 적용하므로
      Cl_history가 항상 ≥ 0일 수 있다. |Cl|의 주파수(= 2 × f_shed)가
      Nyquist 주파수를 초과하면 앨리어싱이 발생한다.
      두 해석(앨리어싱 O/X) 모두 계산 후, 물리적 범위로 자동 판별:
        - 앨리어싱: f_shed = (f_sampling - f_peak) / 2
        - 비앨리어싱: f_shed = f_peak / 2
      임계 St = D / (4 × check_interval × u_ref). 이 값 이상이면 앨리어싱 발생.

    Args:
        Cl_history: Cl 시계열 (배열, signed 또는 |Cl|)
        steps: 스텝 번호 배열 (check_interval 간격)
        D_lattice: 실린더 격자 직경 (= D_ratio × (NN-1))
        u_ref: 참조 속도 (격자 단위, inflow_u)
        check_interval: 데이터 샘플링 간격 (스텝 수)

    Returns:
        Strouhal 수
    """
    Cl_arr = np.asarray(Cl_history, dtype=float)
    N = len(Cl_arr)
    half = Cl_arr[N // 2:]

    is_rectified = bool(np.all(Cl_arr >= -1e-10))

    half_detrended = half - np.mean(half)
    fft_vals = np.fft.rfft(half_detrended)
    power = np.abs(fft_vals) ** 2
    power[0] = 0  # DC 제거

    peak_idx = int(np.argmax(power))
    N_half = len(half)
    f_peak = peak_idx / (N_half * check_interval)

    if is_rectified:
        # |Cl| 데이터: FFT 피크는 2×f_shed(직접) 또는 앨리어싱 주파수
        # 두 해석 모두 계산 후 자기일관성으로 판별
        f_sampling = 1.0 / check_interval
        St_dealiased = ((f_sampling - f_peak) / 2) * D_lattice / u_ref
        St_direct = (f_peak / 2) * D_lattice / u_ref

        # 앨리어싱 임계: St_threshold = D / (4 × ci × u)
        # St > threshold → 2f_shed > f_Nyquist → 앨리어싱 발생
        # St_direct는 항상 < threshold, St_dealiased는 항상 ≥ threshold
        # St_direct가 와류 방출 물리 범위(≥ 0.1)이면 앨리어싱 미발생으로 판단
        if St_direct >= 0.1:
            f_shed = f_peak / 2
        else:
            f_shed = (f_sampling - f_peak) / 2
    else:
        # signed Cl → FFT 피크가 직접 f_shed
        f_shed = f_peak

    return f_shed * D_lattice / u_ref


def tg_analytical_velocity_field(
    X: np.ndarray, Y: np.ndarray, t: float,
    u0: float, L: float, nu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Taylor-Green 감쇠 와류 해석해 속도장.

    u_x = -u0 * cos(π x/L) * sin(π y/L) * exp(-2ν(π/L)² t)
    u_y =  u0 * sin(π x/L) * cos(π y/L) * exp(-2ν(π/L)² t)

    Args:
        X, Y: 물리 좌표 배열 ([-L, L] 범위)
        t: 물리 시간
        u0: 초기 속도 진폭
        L: 도메인 반폭
        nu: 동점성 계수

    Returns:
        (ux, uy): 해석해 속도 성분
    """
    decay = np.exp(-2.0 * nu * (np.pi / L) ** 2 * t)
    ux = -u0 * np.cos(np.pi * X / L) * np.sin(np.pi * Y / L) * decay
    uy = u0 * np.sin(np.pi * X / L) * np.cos(np.pi * Y / L) * decay
    return ux, uy


def compute_l2_error(
    U_num: np.ndarray, U_ana: np.ndarray,
) -> float:
    """L2 상대 오차 = ||U_num - U_ana||₂ / ||U_ana||₂.

    Args:
        U_num: 수치 속도, (N, 2) 또는 (ny, nx) 등
        U_ana: 해석해 속도, 같은 shape

    Returns:
        L2 상대 오차 (무차원)
    """
    diff = U_num - U_ana
    numer = np.sqrt(np.sum(diff ** 2))
    denom = np.sqrt(np.sum(U_ana ** 2))
    if denom == 0.0:
        return float('inf')
    return float(numer / denom)


def compute_recirculation_length(
    Eux: np.ndarray,
    dx: float,
    dy: float,
    cx: float,
    cy: float,
    D: float,
) -> float:
    """실린더 하류 재순환 길이 Lr = L/D 계산.

    BENCHMARK_SUITE.md §A1:
      centerline에서 u_x < 0 → u_x > 0 전환점을 선형 보간하여
      실린더 후면으로부터의 거리를 직경으로 무차원화.

    Args:
        Eux: 2D x-속도장, shape (ny, nx)
        dx, dy: 격자 간격 (도메인 좌표)
        cx, cy: 실린더 중심 (도메인 좌표)
        D: 실린더 직경 (도메인 좌표)

    Returns:
        Lr = L/D (재순환 길이). 전환점이 없으면 NaN 반환.
    """
    ny, nx = Eux.shape
    j_center = int(round(cy / dy))
    ux_line = Eux[j_center, :]

    x_rear = cx + D / 2.0
    # 실린더 후면 바로 뒤부터 탐색 (1격자 여유)
    i_start = int(np.ceil(x_rear / dx)) + 1

    for i in range(i_start, nx - 1):
        if ux_line[i] < 0 and ux_line[i + 1] >= 0:
            # 선형 보간
            x0 = i * dx
            x1 = (i + 1) * dx
            u0 = ux_line[i]
            u1 = ux_line[i + 1]
            x_cross = x0 + (0.0 - u0) / (u1 - u0) * (x1 - x0)
            return (x_cross - x_rear) / D

    return float('nan')


# =====================================================================
# Boundary Fidelity 유틸리티 — delta 함수 NumPy 구현
# ibm.py의 backend 의존성 없이 순수 numpy로 보간/측정
# =====================================================================

def _delta_hat_np(r: np.ndarray) -> np.ndarray:
    """Hat delta function (numpy)."""
    return np.maximum(1.0 - np.abs(r), 0.0)


def _delta_peskin4pt_np(r: np.ndarray) -> np.ndarray:
    """Peskin 4-point delta function (numpy)."""
    ar = np.abs(r)
    result = np.zeros_like(ar)
    mask1 = ar <= 1.0
    mask2 = (~mask1) & (ar <= 2.0)
    r1 = ar[mask1]
    result[mask1] = (1.0 / 8.0) * (
        3.0 - 2.0 * r1 + np.sqrt(np.maximum(1.0 + 4.0 * r1 - 4.0 * r1**2, 0.0))
    )
    r2 = ar[mask2]
    result[mask2] = (1.0 / 8.0) * (
        5.0 - 2.0 * r2 - np.sqrt(np.maximum(-7.0 + 12.0 * r2 - 4.0 * r2**2, 0.0))
    )
    return result


_DELTA_NP = {
    "hat": (_delta_hat_np, 1),
    "peskin4pt": (_delta_peskin4pt_np, 2),
}


def _generate_cylinder_markers(
    cx: float, cy: float, r: float, NN: int,
    marker_spacing_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """실린더 표면 라그랑주 마커 생성 (init.py 로직 재현).

    Args:
        cx, cy: 실린더 중심 (도메인 좌표)
        r: 반지름 (도메인 좌표)
        NN: 격자 해상도 파라미터
        marker_spacing_factor: 마커 간격 (격자 단위, 기본 1.0)

    Returns:
        Lx, Ly: 마커 좌표 (도메인 좌표), theta: 각도 배열
    """
    lattice_r = r * (NN - 1)
    dd = 2.0 * np.pi * lattice_r / marker_spacing_factor
    step = 1.0 / dd
    Ldx = np.arange(0, 1.0 + step * 0.5, step)
    Ldx = Ldx[Ldx <= 1.0 + 1e-12]
    if len(Ldx) > 1 and np.abs(Ldx[-1] - 1.0) < step * 0.5:
        Ldx = Ldx[:-1]
    theta = 2.0 * np.pi * Ldx
    Lx = r * np.cos(theta) + cx
    Ly = r * np.sin(theta) + cy
    return Lx, Ly, theta


def _interpolate_to_markers(
    Eux: np.ndarray, Euy: np.ndarray,
    Lx: np.ndarray, Ly: np.ndarray,
    dx: float, dy: float, delta_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """오일러 속도를 라그랑주 마커에 보간 (ibm.py 보간 로직 재현).

    Args:
        Eux, Euy: 2D 속도장 (ny, nx)
        Lx, Ly: 마커 좌표 (도메인 좌표)
        dx, dy: 격자 간격
        delta_type: "hat" or "peskin4pt"

    Returns:
        Lux, Luy: 보간된 속도 (격자 단위)
    """
    ny, nx = Eux.shape
    delta_func, a = _DELTA_NP[delta_type]
    Lb = len(Lx)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)
    offsets = np.arange(-a, a + 1)

    Lux = np.zeros(Lb)
    Luy = np.zeros(Lb)

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy
            Lux += Eux[ej_c, ei_c] * w
            Luy += Euy[ej_c, ei_c] * w

    return Lux, Luy


def compute_slip_error(
    Eux: np.ndarray, Euy: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, D: float, NN: int,
    delta_type: str,
    u_target_x: np.ndarray | None = None,
    u_target_y: np.ndarray | None = None,
    u_ref: float = 0.1,
) -> dict:
    """경계면 슬립 오차 ε_slip 계산.

    ε_slip_mean = Σ‖u_interp(X_m) - u_target(X_m)‖ / (N_m · U_∞)
    ε_slip_max = max‖u_interp(X_m) - u_target(X_m)‖ / U_∞

    Args:
        Eux, Euy: 2D 속도장 (ny, nx), 격자 단위
        dx, dy: 격자 간격 (도메인 좌표)
        cx, cy: 실린더 중심 (도메인 좌표)
        D: 실린더 직경 (도메인 좌표)
        NN: 격자 해상도 파라미터
        delta_type: "hat" or "peskin4pt"
        u_target_x, u_target_y: 마커별 목표 속도 (None → 0, 고정 실린더)
        u_ref: 참조 속도 (격자 단위, 기본 0.1)

    Returns:
        dict: {'mean', 'max', 'per_marker' (array), 'theta' (array)}
    """
    r = D / 2.0
    Lx, Ly, theta = _generate_cylinder_markers(cx, cy, r, NN)
    Lb = len(Lx)

    Lux, Luy = _interpolate_to_markers(Eux, Euy, Lx, Ly, dx, dy, delta_type)

    if u_target_x is None:
        u_target_x = np.zeros(Lb)
    if u_target_y is None:
        u_target_y = np.zeros(Lb)

    slip_x = Lux - u_target_x
    slip_y = Luy - u_target_y
    slip_mag = np.sqrt(slip_x**2 + slip_y**2)

    return {
        'mean': float(np.mean(slip_mag) / u_ref),
        'max': float(np.max(slip_mag) / u_ref),
        'per_marker': slip_mag / u_ref,
        'theta': theta,
    }


def compute_leakage_flux(
    Eux: np.ndarray, Euy: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, D: float, NN: int,
    delta_type: str,
    u_target_x: np.ndarray | None = None,
    u_target_y: np.ndarray | None = None,
    u_ref: float = 0.1,
) -> dict:
    """경계면 관통 유량 Φ_leak 계산.

    Φ_leak = ∮ ((u_interp − u_target) · n) ds
    원형 실린더: n_m = (cos θ_m, sin θ_m), ds = 2πr / N_m

    정규화: U_∞ · π · D (특성 관통 유량)

    Args:
        (compute_slip_error와 동일)

    Returns:
        dict: {'total' (정규화), 'total_raw', 'per_marker', 'theta'}
    """
    r = D / 2.0
    Lx, Ly, theta = _generate_cylinder_markers(cx, cy, r, NN)
    Lb = len(Lx)

    Lux, Luy = _interpolate_to_markers(Eux, Euy, Lx, Ly, dx, dy, delta_type)

    if u_target_x is None:
        u_target_x = np.zeros(Lb)
    if u_target_y is None:
        u_target_y = np.zeros(Lb)

    slip_x = Lux - u_target_x
    slip_y = Luy - u_target_y

    # 외향 법선: n = (cos θ, sin θ)
    nx_vec = np.cos(theta)
    ny_vec = np.sin(theta)

    # 마커별 법선 유속
    un = slip_x * nx_vec + slip_y * ny_vec
    ds = 2.0 * np.pi * r / Lb

    flux_raw = float(np.sum(un * ds))
    flux_norm = flux_raw / (u_ref * np.pi * D)

    return {
        'total': flux_norm,
        'total_raw': flux_raw,
        'per_marker': un / u_ref,
        'theta': theta,
    }


def compute_inside_residual(
    Eux: np.ndarray, Euy: np.ndarray,
    dx: float, dy: float,
    cx: float, cy: float, D: float,
    kappa: float = 2.0,
    u_ref: float = 0.1,
) -> dict:
    """고체 내부 잔류 유동 ‖u‖_inside 계산.

    signed distance < -κ·dx 인 격자점에서의 속도 크기.
    κ≥2로 설정하여 delta smearing 영역을 제외.

    Args:
        Eux, Euy: 2D 속도장 (ny, nx), 격자 단위
        dx, dy: 격자 간격 (도메인 좌표)
        cx, cy: 실린더 중심 (도메인 좌표)
        D: 실린더 직경 (도메인 좌표)
        kappa: 경계 버퍼 (격자 단위, 기본 2.0)
        u_ref: 참조 속도 (격자 단위, 기본 0.1)

    Returns:
        dict: {'mean', 'max', 'n_points',
               'mean_normalized' (u_ref 기준), 'max_normalized'}
    """
    ny, nx = Eux.shape
    r = D / 2.0

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    XX, YY = np.meshgrid(x, y)

    dist = np.sqrt((XX - cx)**2 + (YY - cy)**2) - r
    mask = dist < -kappa * dx

    n_pts = int(np.sum(mask))
    if n_pts == 0:
        return {
            'mean': 0.0, 'max': 0.0, 'n_points': 0,
            'mean_normalized': 0.0, 'max_normalized': 0.0,
        }

    u_mag = np.sqrt(Eux[mask]**2 + Euy[mask]**2)

    return {
        'mean': float(np.mean(u_mag)),
        'max': float(np.max(u_mag)),
        'n_points': n_pts,
        'mean_normalized': float(np.mean(u_mag) / u_ref),
        'max_normalized': float(np.max(u_mag) / u_ref),
    }


def compute_cl_amplitude(Cl_history: np.ndarray) -> float:
    """Cl 시계열 후반 50%에서 진폭 계산.

    |Cl| (항상 양수) 데이터: max(후반50%) 반환 — |Cl|=A|sin(ωt)|의 최대값이 진폭.
    signed Cl 데이터: (max - min) / 2 반환.

    Args:
        Cl_history: Cl 시계열 (배열, signed 또는 |Cl|)

    Returns:
        Cl 진폭
    """
    Cl_arr = np.asarray(Cl_history, dtype=float)
    half = Cl_arr[len(Cl_arr) // 2:]

    if np.all(Cl_arr >= -1e-10):
        # |Cl| 데이터: 진폭 = max(|Cl|)
        return float(np.max(half))
    else:
        # signed Cl: 진폭 = (max - min) / 2
        return float((np.max(half) - np.min(half)) / 2.0)


def record_sedimentation_state(
    particle_pos: np.ndarray,  # (2,) 도메인 좌표
    particle_vel: np.ndarray,  # (2,) 격자 단위
    t: int,                     # 스텝 번호
    d_lattice: float,          # 입자 직경 (격자 단위)
    g_lattice: float,          # 격자 중력
    rho_ratio: float,
    y0: float,                 # 초기 y 위치 (도메인 좌표)
    dx: float = 1.0,           # 격자 간격 (도메인 좌표)
) -> dict:
    """침강 상태 무차원화 기록.

    단위 체계:
        pos: 도메인 좌표 (xmax, ymax 기준)
        vel: 격자 단위 (dx_lattice/dt_lattice)
        d_lattice: 격자 단위 직경
        D_domain = d_lattice * dx: 도메인 좌표 직경

    무차원화 관례 (sedimentation_benchmark_references.md §4):
        u_g = sqrt(|rho_ratio - 1| * g * d)     중력 속도 스케일 (격자 단위)
        t* = t * u_g / d_lattice                  무차원 시간 (격자 단위끼리)
        y* = (y0 - pos[1]) / D_domain             무차원 침강 거리 (도메인 좌표끼리)
        vy* = -vel[1] / u_g                       무차원 침강 속도 (격자 단위끼리)
        vx* = vel[0] / u_g                        무차원 횡방향 속도 (격자 단위끼리)

    Returns:
        dict: 무차원화 + raw 값
    """
    u_g = np.sqrt(abs(rho_ratio - 1.0) * g_lattice * d_lattice)
    D_domain = d_lattice * dx  # 도메인 좌표 직경

    if u_g < 1e-15:
        return {
            't_star': 0.0, 'y_star': 0.0, 'vy_star': 0.0, 'vx_star': 0.0,
            'x': float(particle_pos[0]), 'y': float(particle_pos[1]),
            'vx': float(particle_vel[0]), 'vy': float(particle_vel[1]),
        }

    return {
        't_star': float(t * u_g / d_lattice),
        'y_star': float((y0 - particle_pos[1]) / D_domain),
        'vy_star': float(-particle_vel[1] / u_g),
        'vx_star': float(particle_vel[0] / u_g),
        'x': float(particle_pos[0]),
        'y': float(particle_pos[1]),
        'vx': float(particle_vel[0]),
        'vy': float(particle_vel[1]),
    }
