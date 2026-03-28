"""IBM Distribution Function Correction (DFC).

Tao et al. (2019), Applied Mathematical Modelling 76, pp.362-379.
Non-iterative: f_i 직접 보정으로 no-slip 경계 조건 강제.
Guo forcing 불필요, velocity correction 불필요.
"""

from __future__ import annotations

from ..backend import xp as np, add_at, _use_gpu
from .common import get_delta


def interpolate_f(Lx, Ly, fstar, dx, dy, ny, nx, delta_type):
    """Lagrangian 점에서 9개 분포함수 보간 (Eq.15).

    f*_i(X_k) = Sigma_x f_i(x) * W(x - X_k) * (dx)^2

    기존 df.py 보간과 동일한 스텐실 구조. 차이: U,rho 대신 f_i 9개 동시 보간.
    (dx)^2 = 1 in lattice units.

    Args:
        Lx, Ly: (Lb,) -- Lagrangian 점 좌표
        fstar: (ny*nx, 9) -- post-streaming 분포함수
        dx, dy: 격자 간격
        ny, nx: 격자 크기
        delta_type: "hat" or "peskin4pt"

    Returns:
        f_interp: (Lb, 9)
    """
    fstar_3d = fstar.reshape(ny, nx, 9)
    delta_func, a = get_delta(delta_type)
    offsets = np.arange(-a, a + 1)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    f_interp = np.zeros((len(Lx), 9))

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy  # (Lb,)
            # C1: arr[j, i] -- fstar_3d[y, x, q]
            f_interp += fstar_3d[ej_c, ei_c, :] * w[:, None]

    return f_interp


def bounce_back_fi(f_interp, rho_f, u_wall, lattice):
    """Lagrangian bounce-back (Eq.16).

    f_bar_i(X_k) = f*_{opp[i]}(X_k) + 2 * w_i * rho_f * (e_i . u_wall) / cs^2

    u_wall=0 -> f_bar_i = f*_{opp[i]} (표준 bounce-back).

    Args:
        f_interp: (Lb, 9) -- 보간된 분포함수
        rho_f: float -- 전역 기준밀도 (= 1.0)
        u_wall: (Lb, 2) -- 경계 desired velocity
        lattice: D2Q9

    Returns:
        f_bb: (Lb, 9) -- desired 분포함수
    """
    cs2 = 1.0 / 3.0
    # f_opp[:, i] = f_interp[:, opp[i]]
    f_opp = f_interp[:, lattice.opp]            # (Lb, 9)
    # e_dot_u[:, i] = e_i . u_wall
    e_dot_u = u_wall @ lattice.e.T              # (Lb, 9)
    f_bb = f_opp + 2.0 * lattice.w[None, :] * rho_f * e_dot_u / cs2
    return f_bb


def compute_lambda(Lx, Ly, rho_f, Larea, dx, dy, ny, nx, delta_type):
    """lambda(k) 해석적 계산 (Eq.23).

    lambda(k) = 1 / [2 * rho_f * dS * (dx)^2 * W_sum(k)]

    spread -> interpolate 패턴으로 이중 합산:
      1) W_total(x) = Sigma_{k'} W(x-X_{k'})           -- spread
      2) W_sum(k)   = Sigma_x W_total(x) * W(x-X_k)   -- interpolate
      3) lambda(k)  = 1 / (2 * rho_f * dS * W_sum)

    dS만 최종 단계에서 1회 적용. (dx)^2 = 1 in lattice units.

    Args:
        Lx, Ly: (Lb,) -- Lagrangian 점 좌표
        rho_f: float -- 전역 밀도 (= 1.0)
        Larea: float -- 호 길이 (ds)
        dx, dy: 격자 간격
        ny, nx: 격자 크기
        delta_type: "hat" or "peskin4pt"

    Returns:
        lambda_k: (Lb,)
    """
    delta_func, a = get_delta(delta_type)
    offsets = np.arange(-a, a + 1)
    Lb = len(Lx)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    # Step 1: spread -- W_total(x) = Sigma_{k'} W(x - X_{k'})
    W_total = np.zeros((ny, nx))
    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy
            add_at(W_total, (ej_c, ei_c), w)

    # Step 2: interpolate -- W_sum(k) = Sigma_x W_total(x) * W(x - X_k)
    W_sum = np.zeros(Lb)
    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy
            W_sum += W_total[ej_c, ei_c] * w

    # (dx)^2 = 1 in lattice units
    lambda_k = 1.0 / (2.0 * rho_f * Larea * W_sum)

    return lambda_k


def spread_delta_f(delta_f, Lx, Ly, Larea, dx, dy, ny, nx, delta_type):
    """delta_f_i를 Eulerian 격자에 분산 (Eq.18).

    delta_f_i(x) = Sigma_k delta_f_i(X_k) * W(x - X_k) * dS

    기존 df.py spread_force와 동일 구조. 차이: 2성분 -> 9방향.

    Args:
        delta_f: (Lb, 9) -- Lagrangian delta_f_i
        Lx, Ly: (Lb,) -- Lagrangian 점 좌표
        Larea: float -- 호 길이
        dx, dy: 격자 간격
        ny, nx: 격자 크기
        delta_type: "hat" or "peskin4pt"

    Returns:
        Ef_corr: (ny*nx, 9) -- Eulerian 분포함수 보정
    """
    delta_func, a = get_delta(delta_type)
    offsets = np.arange(-a, a + 1)

    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)

    Ef_corr = np.zeros((ny, nx, 9))

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)
            wx = delta_func((Lx - ei * dx) / dx)
            wy = delta_func((Ly - ej * dy) / dy)
            w = wx * wy * Larea  # (Lb,)
            weighted = delta_f * w[:, None]  # (Lb, 9)
            for q in range(9):
                add_at(Ef_corr[:, :, q], (ej_c, ei_c), weighted[:, q])

    return Ef_corr.reshape(-1, 9)


def compute_dfc_fluid_force(deviation, lambda_k, Larea, lattice):
    """DFC 유체력 계산 (Eq.25).

    F_s(X_k) = -dS * Sigma_i e_i * lambda(k) * [f_bar_i - f*_i]

    lambda에 1/dS가 포함되어 dS와 상쇄:
      dS * lambda(k) = dS / (2*rho_f*dS*W_sum) = 1/(2*rho_f*W_sum)
    따라서 F_s는 마커 간격(dS)에 무관하며, Sigma_k F_s(k) = 총 물리 힘.

    Args:
        deviation: (Lb, 9) -- f_bar - f_interp (lambda 미적용)
        lambda_k: (Lb,)
        Larea: float -- 호 길이 (적분 가중치)
        lattice: D2Q9

    Returns:
        force: (Lb, 2) -- Lagrangian 점별 유체력 (dS 포함, 직접 합산 가능)
    """
    scaled_dev = lambda_k[:, None] * deviation * Larea  # (Lb, 9)
    force = -scaled_dev @ lattice.e                     # (Lb, 2)
    return force


def apply_dfc_correction(Lx, Ly, desired_vel, fstar,
                          dx, dy, Larea, ny, nx,
                          delta_type, lattice, lambda_cache=None):
    """DFC 분포함수 보정 + 유체력 계산.

    알고리즘 (Tao 2019 Eq.15-25):
      (a) f_i 보간: Euler -> Lagrange (9개 분포함수)
      (b) Bounce-back: desired 분포함수 계산
      (c) lambda(k) 계산 또는 캐시 사용
      (d) 편차 + 스케일링: delta_f = lambda * (f_bb - f_interp)
      (e) 분산: Lagrange -> Euler
      (f) 보정: fstar += delta_f
      (g) 유체력 계산 (Eq.25)

    Args:
        Lx, Ly: (Lb,) -- Lagrangian 점 좌표
        desired_vel: (Lb, 2) -- 경계 desired velocity
        fstar: (ny*nx, 9) -- post-streaming 분포함수
        dx, dy: 격자 간격
        Larea: float -- 호 길이
        ny, nx: 격자 크기
        delta_type: "hat" or "peskin4pt"
        lattice: D2Q9
        lambda_cache: (Lb,) or None -- lambda 캐시 (고정 경계 전용)

    Returns:
        fstar_corrected: (ny*nx, 9)
        dfc_force: (Lb, 2) -- Lagrangian 점별 유체력
        lambda_k: (Lb,) -- lambda 값 (캐싱용)
    """
    # (c) lambda(k)
    if lambda_cache is not None:
        lambda_k = lambda_cache
    elif _use_gpu:
        from ..gpu_kernels import compute_lambda_gpu
        lambda_k = compute_lambda_gpu(Lx, Ly, rho_f=1.0, Larea=Larea,
                                       dx=dx, dy=dy, ny=ny, nx=nx,
                                       delta_type=delta_type)
    else:
        lambda_k = compute_lambda(Lx, Ly, rho_f=1.0, Larea=Larea,
                                   dx=dx, dy=dy, ny=ny, nx=nx,
                                   delta_type=delta_type)

    # GPU: 전용 CUDA 커널 (interp + bb_lambda + spread 3개)
    if _use_gpu:
        from ..gpu_kernels import dfc_correction_gpu
        delta_f_euler, dfc_force = dfc_correction_gpu(
            Lx, Ly, desired_vel, fstar, lambda_k,
            dx, dy, Larea, ny, nx,
            delta_type=delta_type, lattice=lattice,
        )
        fstar_corrected = fstar + delta_f_euler
        return fstar_corrected, dfc_force, lambda_k

    # --- CPU 경로 (변경 없음) ---

    # (a) f_i 보간: Euler -> Lagrange
    f_interp = interpolate_f(Lx, Ly, fstar, dx, dy, ny, nx, delta_type)

    # (b) Bounce-back: desired 분포함수
    f_bb = bounce_back_fi(f_interp, rho_f=1.0, u_wall=desired_vel,
                           lattice=lattice)

    # (d) 편차 + 스케일링
    deviation = f_bb - f_interp                    # (Lb, 9)
    delta_f_lagr = lambda_k[:, None] * deviation   # (Lb, 9)

    # (e) 분산: Lagrange -> Euler
    delta_f_euler = spread_delta_f(delta_f_lagr, Lx, Ly, Larea,
                                    dx, dy, ny, nx, delta_type)

    # (f) 보정
    fstar_corrected = fstar + delta_f_euler

    # (g) 유체력 (Eq.25)
    dfc_force = compute_dfc_fluid_force(deviation, lambda_k, Larea, lattice)

    return fstar_corrected, dfc_force, lambda_k
