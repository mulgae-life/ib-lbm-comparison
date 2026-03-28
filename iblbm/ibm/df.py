"""IBM Direct Forcing (DF).

Guo et al. (2002) forcing scheme 기반.

리뷰 이슈 반영:
  C1: 2D 배열 (ny, nx) 형상 통일. arr[j, i] 접근, .ravel() C-order.
  T2: F_L = 2*R_interp * (U_desired - U_interp) / dt (밀도 보간 방식).
"""

from __future__ import annotations

from ..backend import xp as np, add_at, _use_gpu
from .common import get_delta


def ibm_direct_forcing(
    Lx: np.ndarray, Ly: np.ndarray,
    desired_vel: np.ndarray,
    Eux: np.ndarray, Euy: np.ndarray, Ro: np.ndarray,
    dx: float, dy: float, dt: float,
    Larea: float, ny: int, nx: int,
    delta_type: str = "peskin4pt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """IBM Direct Forcing (CPU 버전, 2*R 밀도 보간).

    알고리즘:
      1) 밀도/속도 보간: Eulerian grid → Lagrangian points (delta 함수)
      2) 힘 계산: F_L = 2*R * (U_desired - U_interp) / dt (T2: CPU 방식)
      3) 힘 분산: Lagrangian forces → Eulerian grid (np.add.at)

    2D 배열 규칙 (C1):
      Eux, Euy, Ro: shape (ny, nx), 접근 arr[j, i] (j=y행, i=x열)

    Args:
        Lx, Ly: 라그랑주 점 좌표, (Lb,) — 도메인 좌표계
        desired_vel: 목표 속도, (Lb, 2) — I5 통일
        Eux, Euy: 오일러 속도장, (ny, nx) — C1 통일
        Ro: 밀도장, (ny, nx)
        dx, dy: 격자 간격
        dt: 시간 스텝
        Larea: 라그랑주 점 호 길이 (Δs, 격자 단위)
        ny, nx: 격자 크기

    Returns:
        fib: IB 체적력, (nodenums, 2)
        Lux, Luy: 보간된 속도, (Lb,)
        Lfx, Lfy: 라그랑주 힘, (Lb,)
        R: 보간된 밀도, (Lb,)
    """
    Lb = len(Lx)
    nodenums = ny * nx

    delta_func, a = get_delta(delta_type)

    if _use_gpu:
        from ..gpu_kernels import ibm_direct_forcing_gpu
        fib = ibm_direct_forcing_gpu(
            Lx, Ly, desired_vel, Eux, Euy, Ro,
            dx, dy, dt, Larea, ny, nx,
            delta_type=delta_type,
        )
        return fib, None, None, None, None, None

    # 라그랑주 점의 가장 가까운 격자 인덱스 (0-base)
    # np.floor(x + 0.5): half-away-from-zero 반올림
    ix0 = np.floor(Lx / dx + 0.5).astype(np.int64)  # x-인덱스, (Lb,)
    iy0 = np.floor(Ly / dy + 0.5).astype(np.int64)  # y-인덱스, (Lb,)

    offsets = np.arange(-a, a + 1)

    # --- 보간: Eulerian → Lagrangian ---
    Lux = np.zeros(Lb)
    Luy = np.zeros(Lb)
    R = np.zeros(Lb)

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di  # x-인덱스, (Lb,)
            ej = iy0 + dj  # y-인덱스, (Lb,)

            # 경계 클램핑 (실린더가 도메인 내부에 있으면 불필요하나 안전)
            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)

            # delta 가중치 (정규화된 거리)
            wx = delta_func((Lx - ei * dx) / dx)  # (Lb,)
            wy = delta_func((Ly - ej * dy) / dy)  # (Lb,)
            w = wx * wy

            # C1: 2D 배열 접근 arr[j, i] — j=y(행), i=x(열)
            R += Ro[ej_c, ei_c] * w
            Lux += Eux[ej_c, ei_c] * w
            Luy += Euy[ej_c, ei_c] * w

    # --- 힘 계산 (T2: 2*R, 밀도 보간 방식) ---
    Lfx = 2.0 * R * (desired_vel[:, 0] - Lux) / dt
    Lfy = 2.0 * R * (desired_vel[:, 1] - Luy) / dt

    # --- 분산: Lagrangian → Eulerian ---
    Efx = np.zeros((ny, nx))  # C1: (ny, nx)
    Efy = np.zeros((ny, nx))

    for di in offsets:
        for dj in offsets:
            ei = ix0 + di
            ej = iy0 + dj

            ei_c = np.clip(ei, 0, nx - 1)
            ej_c = np.clip(ej, 0, ny - 1)

            # |r| = |-r|이므로 부호 반전해도 동일
            wx = delta_func((ei * dx - Lx) / dx)
            wy = delta_func((ej * dy - Ly) / dy)
            w = wx * wy * Larea

            # 중복 인덱스 누적 (일반 += 은 중복 무시)
            # GPU: cupyx.scatter_add, CPU: np.add.at (backend.add_at 래퍼)
            add_at(Efx, (ej_c, ei_c), Lfx * w)
            add_at(Efy, (ej_c, ei_c), Lfy * w)

    # C1: .ravel() C-order → 1D
    fib = np.zeros((nodenums, 2))
    fib[:, 0] = Efx.ravel()
    fib[:, 1] = Efy.ravel()

    return fib, Lux, Luy, Lfx, Lfy, R
