"""IBM Multi-Direct Forcing (MDF).

DF를 n_iter회 반복하여 경계 조건 만족도를 향상.
수렴 모니터링 + 발산 감지로 반복 안정성 보장.

Relaxation parameter ω (Bao et al. 2025, arXiv:2507.04986):
  최적 ω = C_s⁻¹. peskin4pt: C₄=0.375 → ω≈2.667, hat: C₃=0.5 → ω=2.0.
  ω=1(기본 MDF)은 이동 경계 + 낮은 밀도비에서 발산 가능.
  향후 ω 최적화는 Q1 확장판에서 검증 예정.
"""

from __future__ import annotations

from ..backend import xp as np
from .df import ibm_direct_forcing

# 최적 relaxation parameter (Bao et al. 2025) — 향후 사용
# C_s = delta function의 보간-분산 행렬 최대 고유값
OMEGA_OPTIMAL = {
    "peskin4pt": 1.0 / 0.375,  # ≈ 2.667
    "hat": 1.0 / 0.5,          # = 2.0
}


def ibm_multi_direct_forcing(
    Lx: np.ndarray, Ly: np.ndarray,
    desired_vel: np.ndarray,
    U: np.ndarray, ro: np.ndarray,
    dx: float, dy: float, dt: float,
    Larea: float, ny: int, nx: int,
    n_iter: int = 10,
    delta_type: str = "peskin4pt",
    omega: float = 1.0,
) -> np.ndarray:
    """IBM Multi-Direct Forcing (MDF).

    DF를 n_iter회 반복하여 경계 조건 만족도를 향상.
    수렴 모니터링으로 조기 종료 + 발산 감지로 안정성 보장.

    Args:
        Lx, Ly: 라그랑주 점 좌표, (Lb,)
        desired_vel: 목표 속도, (Lb, 2)
        U: 거시 속도, (nodenums, 2)
        ro: 현재 밀도, (nodenums,)
        dx, dy, dt: 격자 간격/시간 스텝
        Larea: 호 길이
        ny, nx: 격자 크기
        n_iter: 최대 반복 횟수 (기본 10, 수렴/발산 시 조기 종료)
        delta_type: delta 함수 유형
        omega: relaxation parameter (기본 1.0 = 기존 MDF)

    Returns:
        fib_total: 누적 IB 체적력, (nodenums, 2)
    """
    nodenums = ny * nx
    fib_total = np.zeros((nodenums, 2))
    U_work = U.copy()
    Ro = ro.reshape(ny, nx)

    tol = 1e-6
    prev_residual = float('inf')

    for i in range(n_iter):
        Eux = U_work[:, 0].reshape(ny, nx)
        Euy = U_work[:, 1].reshape(ny, nx)

        fib, _, _, _, _, _ = ibm_direct_forcing(
            Lx, Ly, desired_vel, Eux, Euy, Ro,
            dx, dy, dt, Larea, ny, nx,
            delta_type=delta_type,
        )

        residual = float(np.max(np.abs(fib)))

        # 발산 감지: 잔여 힘이 이전보다 커지면 반복 발산 → 즉시 중단
        if i > 0 and residual > prev_residual * 1.5:
            break

        fib_relaxed = omega * fib
        fib_total += fib_relaxed
        U_work = U_work + fib_relaxed * dt / (2.0 * ro[:, None])
        prev_residual = residual

        # 수렴: 잔여 힘이 충분히 작으면 조기 종료
        if residual < tol:
            break

    return fib_total
