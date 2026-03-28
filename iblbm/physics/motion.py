"""실린더 운동 업데이트.

시나리오별 운동:
  진동: x(t) = A*sin(2*pi*f0*t), desired_vel += (x(t) - x(t-1))
  이동: Lx += u*dx/10, desired_vel = -u
  회전: desired_vel = (-omega*(Ly - cy), omega*(Lx - cx))
"""

from __future__ import annotations

from ..backend import xp as np


def update_oscillating(
    t: int, Lx_c: np.ndarray, Ly_c: np.ndarray,
    desired_vel: np.ndarray,
    lattice_u: float, KC: float, r: float, lattice_r: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """진동 실린더 위치/속도 업데이트.

    Args:
        t: 현재 시간 스텝
        Lx_c, Ly_c: 초기 라그랑주 점 좌표, (Lb,)
        desired_vel: 목표 속도, (Lb, 2) — in-place 수정
        lattice_u: 격자 단위 속도
        KC: Keulegan-Carpenter 수
        r: 실린더 반지름 (도메인 비율)
        lattice_r: 실린더 반지름 (격자 단위)

    Returns:
        Lx, Ly: 갱신된 라그랑주 점 좌표
        desired_vel: 갱신된 목표 속도
    """
    f0 = lattice_u / (KC * 2.0 * lattice_r)
    A = KC * 2.0 * r / (2.0 * np.pi)

    xt = A * np.sin(f0 * 2.0 * np.pi * t)
    xtt = A * np.sin(f0 * 2.0 * np.pi * (t - 1))

    Lx = Lx_c - xt
    Ly = Ly_c.copy()
    desired_vel[:, 0] += (xt - xtt)

    return Lx, Ly, desired_vel


def update_translating(
    Lx: np.ndarray, Ly_c: np.ndarray,
    desired_vel: np.ndarray,
    lattice_u: float, dx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이동 실린더 위치/속도 업데이트.

    Args:
        Lx: 현재 x 좌표, (Lb,)
        Ly_c: y 좌표 (고정), (Lb,)
        desired_vel: 목표 속도, (Lb, 2) — in-place 수정
        lattice_u: 격자 단위 속도
        dx: 격자 간격

    Returns:
        Lx, Ly, desired_vel
    """
    Lx = Lx + lattice_u * dx / 10.0
    desired_vel[:, 0] = -lattice_u
    return Lx, Ly_c, desired_vel


def update_rotating(
    Lx: np.ndarray, Ly: np.ndarray,
    center: tuple[float, float], omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """회전 실린더 좌표/속도 갱신.

    매 스텝 라그랑주 점을 omega만큼 회전시킨 후,
    회전된 위치에서의 접선 속도를 desired_velocity로 설정.

    Args:
        Lx, Ly: 라그랑주 점 좌표, (Lb,)
        center: 실린더 중심 (cx, cy)
        omega: 각속도 (rad/timestep)

    Returns:
        Lx_new, Ly_new: 회전된 좌표
        desired_vel: (Lb, 2)
    """
    cx, cy = center

    # 중심 기준 상대 좌표
    dx = Lx - cx
    dy = Ly - cy

    # 회전 행렬 적용
    cos_w = np.cos(omega)
    sin_w = np.sin(omega)
    dx_r = cos_w * dx - sin_w * dy
    dy_r = sin_w * dx + cos_w * dy

    # 전역 좌표 복원
    Lx_new = dx_r + cx
    Ly_new = dy_r + cy

    # 회전된 위치에서의 접선 속도: v = omega × r
    desired_vel = np.column_stack([-dy_r * omega, dx_r * omega])

    return Lx_new, Ly_new, desired_vel
