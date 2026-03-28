"""Zou-He 경계 조건 (Zou & He, 1997).

패턴 A (velocity_inlet): 정상류/이동/회전
  좌: Zou-He 속도 입구 (u = u_in, v = 0)
  우: Zou-He 압력 출구 (rho = rho_0)
  상/하: 벽면 (v = 0) + 2차 외삽 (du/dy = 0)

패턴 B (open_boundary): 진동
  4면 모두: Zou-He 압력 개방 (rho = rho_0)
  접선 속도: 2차 외삽 (T3: 3개 점 사용, 2차 정확도)

패턴 D (settling_channel): 침강
  4면 모두: Zou-He 속도 no-slip (u = 0, v = 0) — 밀폐 도메인
  Feng (1994), Feng & Michaelides (2004) 표준 설정

리뷰 이슈 반영:
  T1: 우측 출구 u_x 계산에서 (전체)/ro 적용 (이론적으로 올바른 괄호).
  I4: 벽면 BC에서 v(normal)=0이므로 1/6*ro*v 항이 자연 소거.
      개방 경계(패턴 B)에서는 v ≠ 0이므로 항 존재.
  T3: 2차 외삽 — u_wall = (4*u_1 - u_2)/3 + O(h^2).

D2Q9 방향 인덱스:
  0(rest) 1(E) 2(N) 3(W) 4(S) 5(NE) 6(NW) 7(SW) 8(SE)
"""

from __future__ import annotations

from .backend import xp as np

from .diagnostics import tg_analytical_velocity_field
from .lbm import compute_feq
from .init import BoundaryIndices


def apply_bc_velocity_inlet(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    inflow_u: float, dens: float, idx: BoundaryIndices,
    lattice=None,
) -> None:
    """패턴 A — 속도 입구/압력 출구/벽면 BC (in-place).

    Args:
        fstar: post-streaming 분포, (nodenums, 9) — in-place 수정
        ro: 밀도, (nodenums,) — in-place 수정
        U: 속도, (nodenums, 2) — in-place 수정
        inflow_u: 유입 x-속도 (lattice 단위)
        dens: 격자 밀도 (= 1.0)
        idx: 경계 인덱스
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx

    # === 좌: Zou-He 속도 입구 (u_x = inflow_u, u_y = 0) ===
    U[left, 0] = inflow_u
    U[left, 1] = 0.0
    ro[left] = (1.0 / (1.0 - U[left, 0])) * (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 6] + fstar[left, 7])
    )
    fstar[left, 1] = fstar[left, 3] + (2.0 / 3.0) * ro[left] * U[left, 0]
    fstar[left, 5] = (
        fstar[left, 7]
        - 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        + 0.5 * ro[left] * U[left, 1]
    )
    fstar[left, 8] = (
        fstar[left, 6]
        + 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        - 0.5 * ro[left] * U[left, 1]
    )

    # === 우: Zou-He 압력 출구 (rho = rho_0) ===
    # T1: (전체)/ro — 이론적으로 올바른 괄호 적용
    ro[right] = dens
    U[right, 0] = -1.0 + (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    ) / ro[right]
    # 2차 외삽 (T3): dv/dx = 0 → v_wall = (4*v_1 - v_2) / 3
    U[right, 1] = (4.0 * U[right - 1, 1] - U[right - 2, 1]) / 3.0
    fstar[right, 3] = fstar[right, 1] - (2.0 / 3.0) * ro[right] * U[right, 0]
    fstar[right, 7] = (
        fstar[right, 5]
        + 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        - 0.5 * ro[right] * U[right, 1]
    )
    fstar[right, 6] = (
        fstar[right, 8]
        - 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        + 0.5 * ro[right] * U[right, 1]
    )

    # === 상: 벽면 (v = 0) + 2차 외삽 (du/dy = 0) ===
    # I4: 벽면 BC에서 normal 속도 성분(v = U[top,1])이 0이므로
    #     1/6*ro*v 항이 자연스럽게 사라진다. 개방 경계에서는 v ≠ 0이므로 항이 존재.
    U[top, 0] = (4.0 * U[top - nx, 0] - U[top - 2 * nx, 0]) / 3.0
    U[top, 1] = 0.0
    ro[top] = (1.0 / (1.0 + U[top, 1])) * (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    )
    fstar[top, 4] = fstar[top, 2] - (2.0 / 3.0) * ro[top] * U[top, 1]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
        - 0.5 * ro[top] * U[top, 0]
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
        + 0.5 * ro[top] * U[top, 0]
    )

    # === 하: 벽면 (v = 0) + 2차 외삽 (du/dy = 0) ===
    U[bottom, 0] = (4.0 * U[bottom + nx, 0] - U[bottom + 2 * nx, 0]) / 3.0
    U[bottom, 1] = 0.0
    ro[bottom] = (1.0 / (1.0 - U[bottom, 1])) * (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    )
    fstar[bottom, 2] = fstar[bottom, 4] + (2.0 / 3.0) * ro[bottom] * U[bottom, 1]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + 0.5 * ro[bottom] * U[bottom, 0]
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        - 0.5 * ro[bottom] * U[bottom, 0]
    )

    # === 코너 closure (P1-2) ===
    # 코너는 face BC 하나로 모든 unknown 분포를 결정할 수 없다.
    # (ρ, U)를 분포함수 없이 직접 규정 후 feq 마감 — 인공 closure.
    if lattice is not None:
        # inlet 코너 (좌하, 좌상)
        # ro=dens: closure 선택 (face에서는 ρ를 계산하지만, 코너는 stale 분포로 불가)
        inlet_corners = np.array([left[0], left[-1]])
        ro[inlet_corners] = dens
        U[inlet_corners, 0] = inflow_u
        U[inlet_corners, 1] = 0.0
        fstar[inlet_corners] = compute_feq(
            ro[inlet_corners], U[inlet_corners], lattice,
        )

        # outlet 코너 (우하, 우상)
        # ci-1, ci-2: top/bottom 경계 노드 (face BC에서 외삽된 값 → 체인 외삽)
        outlet_corners = np.array([right[0], right[-1]])
        ro[outlet_corners] = dens
        for ci in outlet_corners:
            U[ci, 0] = (4.0 * U[ci - 1, 0] - U[ci - 2, 0]) / 3.0
        U[outlet_corners, 1] = 0.0
        fstar[outlet_corners] = compute_feq(
            ro[outlet_corners], U[outlet_corners], lattice,
        )


def apply_bc_open_boundary(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    dens: float, idx: BoundaryIndices,
    lattice=None,
) -> None:
    """패턴 B — 4면 압력 개방 경계 BC (in-place).

    진동 시나리오: 파동 반사 방지를 위해 4면 모두 rho = rho_0 고정.

    Args:
        fstar: post-streaming 분포, (nodenums, 9) — in-place 수정
        ro: 밀도, (nodenums,) — in-place 수정
        U: 속도, (nodenums, 2) — in-place 수정
        dens: 격자 밀도 (= 1.0)
        idx: 경계 인덱스
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx

    # === 좌: 압력 개방 (rho = rho_0, u_x 계산) ===
    ro[left] = dens
    U[left, 0] = 1.0 - (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 7] + fstar[left, 6])
    ) / ro[left]
    U[left, 1] = (4.0 * U[left + 1, 1] - U[left + 2, 1]) / 3.0
    fstar[left, 1] = fstar[left, 3] + (2.0 / 3.0) * ro[left] * U[left, 0]
    fstar[left, 5] = (
        fstar[left, 7]
        - 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        + 0.5 * ro[left] * U[left, 1]
    )
    fstar[left, 8] = (
        fstar[left, 6]
        + 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        - 0.5 * ro[left] * U[left, 1]
    )

    # === 우: 압력 개방 (rho = rho_0, u_x 계산) ===
    ro[right] = dens
    U[right, 0] = -1.0 + (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    ) / ro[right]
    U[right, 1] = (4.0 * U[right - 1, 1] - U[right - 2, 1]) / 3.0
    fstar[right, 3] = fstar[right, 1] - (2.0 / 3.0) * ro[right] * U[right, 0]
    fstar[right, 7] = (
        fstar[right, 5]
        + 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        - 0.5 * ro[right] * U[right, 1]
    )
    fstar[right, 6] = (
        fstar[right, 8]
        - 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        + 0.5 * ro[right] * U[right, 1]
    )

    # === 상: 압력 개방 (rho = rho_0, u_y 계산) ===
    # I4: 개방 경계에서는 normal 속도(v)가 0이 아니므로 1/6*ro*v 항이 존재
    ro[top] = dens
    U[top, 1] = -1.0 + (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    ) / ro[top]
    U[top, 0] = (4.0 * U[top - nx, 0] - U[top - 2 * nx, 0]) / 3.0
    fstar[top, 4] = fstar[top, 2] - (2.0 / 3.0) * ro[top] * U[top, 1]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
        - (1.0 / 6.0) * ro[top] * U[top, 1]
        - 0.5 * ro[top] * U[top, 0]
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
        - (1.0 / 6.0) * ro[top] * U[top, 1]
        + 0.5 * ro[top] * U[top, 0]
    )

    # === 하: 압력 개방 (rho = rho_0, u_y 계산) ===
    ro[bottom] = dens
    U[bottom, 1] = 1.0 - (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    ) / ro[bottom]
    U[bottom, 0] = (4.0 * U[bottom + nx, 0] - U[bottom + 2 * nx, 0]) / 3.0
    fstar[bottom, 2] = fstar[bottom, 4] + (2.0 / 3.0) * ro[bottom] * U[bottom, 1]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + (1.0 / 6.0) * ro[bottom] * U[bottom, 1]
        + 0.5 * ro[bottom] * U[bottom, 0]
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + (1.0 / 6.0) * ro[bottom] * U[bottom, 1]
        - 0.5 * ro[bottom] * U[bottom, 0]
    )

    # === 코너 closure ===
    # 코너는 두 면이 만나는 특이점: Zou-He(단일면 가정)가 부정정.
    # face BC에서 코너를 제외(init.py)하고, 여기서 단독 처리.
    # 대각 내부 노드에서 0차 외삽 (두 면 경계의 streaming 영향이
    # diag1에 유입되므로, 2차 외삽의 smoothness 가정이 성립하지 않음).
    if lattice is not None:
        ny = idx.ny
        corners = np.array([
            0,                          # 좌하 (0, 0)
            nx - 1,                     # 우하 (nx-1, 0)
            nx * (ny - 1),              # 좌상 (0, ny-1)
            nx * ny - 1,                # 우상 (nx-1, ny-1)
        ])
        # 대각 내부 노드: (1,1), (nx-2,1), (1,ny-2), (nx-2,ny-2)
        diag1 = np.array([
            nx + 1,                     # 좌하 → (1,1)
            2 * nx - 2,                 # 우하 → (nx-2,1)
            nx * (ny - 2) + 1,          # 좌상 → (1,ny-2)
            nx * (ny - 1) - 2,          # 우상 → (nx-2,ny-2)
        ])
        ro[corners] = dens
        U[corners] = U[diag1]
        fstar[corners] = compute_feq(ro[corners], U[corners], lattice)


def apply_bc_settling_channel(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    dens: float, idx: BoundaryIndices,
    lattice=None,
) -> None:
    """패턴 D — 침강 채널 BC (in-place).

    4면 모두 Zou-He 속도 BC (u=0, v=0) → no-slip wall (밀폐 도메인).

    Feng (1994), Feng & Michaelides (2004), Glowinski (2001) 표준 설정.
    밀폐 도메인에서 질량 보존 자동 충족, 압력파 반사 없음.

    Args:
        fstar: post-streaming 분포, (nodenums, 9) — in-place 수정
        ro: 밀도, (nodenums,) — in-place 수정
        U: 속도, (nodenums, 2) — in-place 수정
        dens: 격자 밀도 (= 1.0)
        idx: 경계 인덱스
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx = idx.nx

    # === 좌: Zou-He 속도 no-slip (u=0, v=0) ===
    U[left, 0] = 0.0
    U[left, 1] = 0.0
    ro[left] = (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 6] + fstar[left, 7])
    )
    fstar[left, 1] = fstar[left, 3]
    fstar[left, 5] = (
        fstar[left, 7]
        - 0.5 * (fstar[left, 2] - fstar[left, 4])
    )
    fstar[left, 8] = (
        fstar[left, 6]
        + 0.5 * (fstar[left, 2] - fstar[left, 4])
    )

    # === 우: Zou-He 속도 no-slip (u=0, v=0) ===
    U[right, 0] = 0.0
    U[right, 1] = 0.0
    ro[right] = (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    )
    fstar[right, 3] = fstar[right, 1]
    fstar[right, 7] = (
        fstar[right, 5]
        + 0.5 * (fstar[right, 2] - fstar[right, 4])
    )
    fstar[right, 6] = (
        fstar[right, 8]
        - 0.5 * (fstar[right, 2] - fstar[right, 4])
    )

    # === 상: Zou-He 속도 no-slip (u=0, v=0) ===
    U[top, 0] = 0.0
    U[top, 1] = 0.0
    ro[top] = (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    )
    fstar[top, 4] = fstar[top, 2]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
    )

    # === 하: Zou-He 속도 no-slip (u=0, v=0) ===
    U[bottom, 0] = 0.0
    U[bottom, 1] = 0.0
    ro[bottom] = (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    )
    fstar[bottom, 2] = fstar[bottom, 4]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
    )

    # === 코너 closure ===
    # 4면 모두 no-slip wall → 코너는 두 벽면의 교차점.
    # 대각 내부 노드에서 0차 외삽 + feq 마감.
    if lattice is not None:
        ny = idx.ny
        corners = np.array([
            0,                          # 좌하 (0, 0)
            nx - 1,                     # 우하 (nx-1, 0)
            nx * (ny - 1),              # 좌상 (0, ny-1)
            nx * ny - 1,                # 우상 (nx-1, ny-1)
        ])
        diag1 = np.array([
            nx + 1,                     # 좌하 → (1,1)
            2 * nx - 2,                 # 우하 → (nx-2,1)
            nx * (ny - 2) + 1,          # 좌상 → (1,ny-2)
            nx * (ny - 1) - 2,          # 우상 → (nx-2,ny-2)
        ])
        ro[corners] = dens
        U[corners] = U[diag1]
        fstar[corners] = compute_feq(ro[corners], U[corners], lattice)


def apply_bc_analytical(
    fstar: np.ndarray, ro: np.ndarray, U: np.ndarray,
    t_physical: float, cfg, idx: BoundaryIndices, lattice,
) -> None:
    """패턴 C — 4면 시간 의존 Dirichlet BC (Taylor-Green 와류용, in-place).

    각 경계면에서 해석해 속도를 목표값으로 Zou-He를 적용한다.
    D2Q9 방향: 0(rest) 1(E) 2(N) 3(W) 4(S) 5(NE) 6(NW) 7(SW) 8(SE)

    Args:
        fstar: post-streaming 분포, (nodenums, 9)
        ro: 밀도, (nodenums,)
        U: 속도, (nodenums, 2)
        t_physical: 현재 물리 시간
        cfg: SimConfig (tg_L, tg_u0, Re 등)
        idx: 경계 인덱스
        lattice: D2Q9 격자
    """
    left, right = idx.left, idx.right
    top, bottom = idx.top, idx.bottom
    nx, ny = idx.nx, idx.ny

    # 격자 좌표 (lattice units): [0, nx-1] → [-L_lat, L_lat]
    L_lat = 0.5 * (cfg.NN - 1)
    lattice_D = cfg.cylinder_D_ratio * (cfg.NN - 1)
    nu = cfg.lattice_u * lattice_D / cfg.Re

    X_lat = np.arange(nx, dtype=float) - (nx - 1) / 2.0
    Y_lat = np.arange(ny, dtype=float) - (ny - 1) / 2.0

    # === Left (i=0): 미지 f1, f5, f8 ===
    # left 인덱스 = [0, nx, 2*nx, ...] → j = left // nx
    j_left = left // nx
    x_left = np.full_like(j_left, X_lat[0], dtype=float)
    y_left = Y_lat[j_left]
    ux_t, uy_t = tg_analytical_velocity_field(x_left, y_left, t_physical,
                                               cfg.tg_u0, L_lat, nu)
    U[left, 0] = ux_t
    U[left, 1] = uy_t
    ro[left] = (1.0 / (1.0 - U[left, 0])) * (
        fstar[left, 0] + fstar[left, 2] + fstar[left, 4]
        + 2.0 * (fstar[left, 3] + fstar[left, 6] + fstar[left, 7])
    )
    fstar[left, 1] = fstar[left, 3] + (2.0 / 3.0) * ro[left] * U[left, 0]
    fstar[left, 5] = (
        fstar[left, 7]
        - 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        + 0.5 * ro[left] * U[left, 1]
    )
    fstar[left, 8] = (
        fstar[left, 6]
        + 0.5 * (fstar[left, 2] - fstar[left, 4])
        + (1.0 / 6.0) * ro[left] * U[left, 0]
        - 0.5 * ro[left] * U[left, 1]
    )

    # === Right (i=nx-1): 미지 f3, f6, f7 ===
    j_right = right // nx
    x_right = np.full_like(j_right, X_lat[nx - 1], dtype=float)
    y_right = Y_lat[j_right]
    ux_t, uy_t = tg_analytical_velocity_field(x_right, y_right, t_physical,
                                               cfg.tg_u0, L_lat, nu)
    U[right, 0] = ux_t
    U[right, 1] = uy_t
    ro[right] = (1.0 / (1.0 + U[right, 0])) * (
        fstar[right, 0] + fstar[right, 2] + fstar[right, 4]
        + 2.0 * (fstar[right, 1] + fstar[right, 5] + fstar[right, 8])
    )
    fstar[right, 3] = fstar[right, 1] - (2.0 / 3.0) * ro[right] * U[right, 0]
    fstar[right, 7] = (
        fstar[right, 5]
        + 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        - 0.5 * ro[right] * U[right, 1]
    )
    fstar[right, 6] = (
        fstar[right, 8]
        - 0.5 * (fstar[right, 2] - fstar[right, 4])
        - (1.0 / 6.0) * ro[right] * U[right, 0]
        + 0.5 * ro[right] * U[right, 1]
    )

    # === Top (j=ny-1): 미지 f4, f7, f8 ===
    i_top = top % nx
    x_top = X_lat[i_top]
    y_top = np.full_like(i_top, Y_lat[ny - 1], dtype=float)
    ux_t, uy_t = tg_analytical_velocity_field(x_top, y_top, t_physical,
                                               cfg.tg_u0, L_lat, nu)
    U[top, 0] = ux_t
    U[top, 1] = uy_t
    ro[top] = (1.0 / (1.0 + U[top, 1])) * (
        fstar[top, 0] + fstar[top, 1] + fstar[top, 3]
        + 2.0 * (fstar[top, 2] + fstar[top, 5] + fstar[top, 6])
    )
    fstar[top, 4] = fstar[top, 2] - (2.0 / 3.0) * ro[top] * U[top, 1]
    fstar[top, 7] = (
        fstar[top, 5]
        + 0.5 * (fstar[top, 1] - fstar[top, 3])
        - (1.0 / 6.0) * ro[top] * U[top, 1]
        - 0.5 * ro[top] * U[top, 0]
    )
    fstar[top, 8] = (
        fstar[top, 6]
        - 0.5 * (fstar[top, 1] - fstar[top, 3])
        - (1.0 / 6.0) * ro[top] * U[top, 1]
        + 0.5 * ro[top] * U[top, 0]
    )

    # === Bottom (j=0): 미지 f2, f5, f6 ===
    i_bottom = bottom % nx
    x_bottom = X_lat[i_bottom]
    y_bottom = np.full_like(i_bottom, Y_lat[0], dtype=float)
    ux_t, uy_t = tg_analytical_velocity_field(x_bottom, y_bottom, t_physical,
                                               cfg.tg_u0, L_lat, nu)
    U[bottom, 0] = ux_t
    U[bottom, 1] = uy_t
    ro[bottom] = (1.0 / (1.0 - U[bottom, 1])) * (
        fstar[bottom, 0] + fstar[bottom, 1] + fstar[bottom, 3]
        + 2.0 * (fstar[bottom, 4] + fstar[bottom, 7] + fstar[bottom, 8])
    )
    fstar[bottom, 2] = fstar[bottom, 4] + (2.0 / 3.0) * ro[bottom] * U[bottom, 1]
    fstar[bottom, 5] = (
        fstar[bottom, 7]
        - 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + (1.0 / 6.0) * ro[bottom] * U[bottom, 1]
        + 0.5 * ro[bottom] * U[bottom, 0]
    )
    fstar[bottom, 6] = (
        fstar[bottom, 8]
        + 0.5 * (fstar[bottom, 1] - fstar[bottom, 3])
        + (1.0 / 6.0) * ro[bottom] * U[bottom, 1]
        - 0.5 * ro[bottom] * U[bottom, 0]
    )

    # === 코너 closure: feq 마감 ===
    corners = np.array([
        0,                          # 좌하 (0, 0)
        nx - 1,                     # 우하 (nx-1, 0)
        nx * (ny - 1),              # 좌상 (0, ny-1)
        nx * ny - 1,                # 우상 (nx-1, ny-1)
    ])
    corner_i = np.array([0, nx - 1, 0, nx - 1])
    corner_j = np.array([0, 0, ny - 1, ny - 1])
    x_corner = X_lat[corner_i]
    y_corner = Y_lat[corner_j]
    ux_c, uy_c = tg_analytical_velocity_field(x_corner, y_corner, t_physical,
                                               cfg.tg_u0, L_lat, nu)
    ro[corners] = 1.0
    U[corners, 0] = ux_c
    U[corners, 1] = uy_c
    fstar[corners] = compute_feq(ro[corners], U[corners], lattice)
