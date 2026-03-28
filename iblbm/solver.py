"""메인 시간 루프 (solver).

공통 시간 루프. BC/IBM/운동을 config로 분기.

알고리즘 (매 시간 스텝):
  0. [침강] Verlet 반스텝: v(n+½), x(n+1) 갱신 + 마커/desired_vel 설정
  1. 충돌:     f = fstar - (fstar - feq)/tau + F*dt
  2. 스트리밍:  fstar = Stream(f)
  3. 경계조건:  Zou-He BC (경계 fstar 보정)
  4. 거시변수:  rho = sum(fstar), U = (fstar @ e) / rho
  5. IBM:      보간 → 힘 계산 → 분산 (DF 또는 MDF)
  6. 속도보정:  U = U + fib*dt/(2*rho)
  6b.[침강] Verlet 전스텝: F_hydro 추출 → v(n+1) 완성
  7. feq 갱신:  feq = rho*w*(1 + 3*eU + 9/2*(eU)^2 - 3/2*U^2)
  8. 운동:     oscillating/translating/rotating (침강은 0,6b에서 처리)
  9. 진단:     Cd/Cl 또는 침강 궤적/종단속도
"""

from __future__ import annotations

import time

from .backend import xp as np, _use_gpu

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _macroscopic_nb(fstar, e):
        """거시변수 복원 numba 커널. ro = sum(fstar), U = fstar@e / ro."""
        N = fstar.shape[0]
        ro = np.empty(N)
        U = np.empty((N, 2))
        for n in prange(N):
            s = 0.0
            ux = 0.0
            uy = 0.0
            for i in range(9):
                s += fstar[n, i]
                ux += fstar[n, i] * e[i, 0]
                uy += fstar[n, i] * e[i, 1]
            ro[n] = s
            U[n, 0] = ux / s
            U[n, 1] = uy / s
        return ro, U


def _macroscopic(fstar, lattice):
    """거시변수 복원. GPU: CUDA 커널, CPU: numba."""
    if _use_gpu:
        from .gpu_kernels import macroscopic_gpu
        return macroscopic_gpu(fstar)
    return _macroscopic_nb(fstar, lattice.e)


# --- GPU↔CPU 전송 헬퍼 ---
def _to_cpu(arr):
    """GPU 배열을 CPU(numpy)로 전송. CPU 모드에서는 no-op."""
    if _use_gpu:
        return arr.get()
    return arr


from .boundary import (
    apply_bc_analytical, apply_bc_open_boundary,
    apply_bc_settling_channel, apply_bc_velocity_inlet,
)
from .lbm import collision_step
from .config import SimConfig
from .diagnostics import (
    check_convergence, compute_cd_cl, compute_cd_cl_dfc,
    record_sedimentation_state, tg_analytical_velocity_field,
)
from .lbm import compute_feq
from .ibm import ibm_direct_forcing, ibm_multi_direct_forcing, apply_dfc_correction
from .init import SimState, initialize
from .physics.motion import update_oscillating, update_rotating, update_translating
from .physics.sedimentation import (
    check_domain_bounds, compute_desired_velocity,
    extract_hydro_force, update_markers,
    verlet_full_step, verlet_half_step,
)
from .lbm import streaming_step


# --- IBM 단계 private 함수 ---

def _update_tg_desired_velocity(s, cfg, ttt):
    """TG with IBM: desired_velocity를 해석해로 매 스텝 갱신."""
    t_physical = ttt * s.dt
    L_lat = 0.5 * (cfg.NN - 1)
    lattice_D = cfg.cylinder_D_ratio * (cfg.NN - 1)
    nu = cfg.lattice_u * lattice_D / cfg.Re

    Lx_centered = s.Lx / s.dx - (s.nx - 1) / 2.0
    Ly_centered = s.Ly / s.dy - (s.ny - 1) / 2.0
    ux_ana, uy_ana = tg_analytical_velocity_field(
        Lx_centered, Ly_centered, t_physical, cfg.tg_u0, L_lat, nu)
    s.desired_velocity[:, 0] = ux_ana
    s.desired_velocity[:, 1] = uy_ana


def _ibm_df(s, cfg):
    """DF: 보간 → 힘 → 분산."""
    Eux = s.U[:, 0].reshape(s.ny, s.nx)
    Euy = s.U[:, 1].reshape(s.ny, s.nx)
    Ro = np.clip(s.ro, 0.9, 1.1).reshape(s.ny, s.nx)
    s.fib, _, _, _, _, _ = ibm_direct_forcing(
        s.Lx, s.Ly, s.desired_velocity,
        Eux, Euy, Ro,
        s.dx, s.dy, s.dt, s.Larea, s.ny, s.nx,
        delta_type=cfg.delta_type,
    )


def _ibm_mdf(s, cfg):
    """MDF: DF를 n_iter회 반복."""
    s.fib = ibm_multi_direct_forcing(
        s.Lx, s.Ly, s.desired_velocity,
        s.U, s.ro,
        s.dx, s.dy, s.dt, s.Larea, s.ny, s.nx,
        n_iter=cfg.mdf_iterations,
        delta_type=cfg.delta_type,
    )


def _velocity_correction(s):
    """속도 보정: U += fib·dt/(2ρ). GPU/CPU 분기 포함."""
    if _use_gpu:
        from .gpu_kernels import velocity_correction_gpu
        velocity_correction_gpu(s.U, s.fib, s.ro, s.dt)
    else:
        s.U = s.U + s.fib * s.dt / (2.0 * s.ro[:, None])


def _ibm_dfc(s, cfg):
    """DFC: 분포함수 직접 보정. velocity correction 불필요."""
    # 이동 경계: lambda 캐시 무효화 (매 스텝 재계산)
    cache = s.lambda_cache if cfg.motion_type is None else None

    s.fstar, s.dfc_force_lagr, lambda_new = apply_dfc_correction(
        s.Lx, s.Ly, s.desired_velocity,
        s.fstar,
        s.dx, s.dy, s.Larea, s.ny, s.nx,
        delta_type=cfg.delta_type,
        lattice=s.lattice,
        lambda_cache=cache,
    )

    # 고정 경계만 캐시 유지
    if cfg.motion_type is None:
        s.lambda_cache = lambda_new

    # Macro 재계산 (보정된 f_i에서)
    s.ro, s.U = _macroscopic(s.fstar, s.lattice)


def _ibm_step(s, cfg, ttt):
    """IBM 통합 디스패치. cfg.ibm_method에 따라 DF/MDF/DFC 분기."""
    if cfg.scenario_type == "taylor_green" and not cfg.tg_with_ibm:
        return
    if cfg.scenario_type == "taylor_green" and cfg.tg_with_ibm:
        _update_tg_desired_velocity(s, cfg, ttt)

    if cfg.ibm_method == "DFC":
        _ibm_dfc(s, cfg)
        # DFC: velocity correction 불필요, macro는 _ibm_dfc 내에서 재계산
    elif cfg.ibm_method == "MDF":
        _ibm_mdf(s, cfg)
        _velocity_correction(s)
    else:  # "DF"
        _ibm_df(s, cfg)
        _velocity_correction(s)


def run(cfg: SimConfig, verbose: bool = True, callback=None) -> dict:
    """IB-LBM 시뮬레이션 실행.

    Args:
        cfg: 시뮬레이션 설정
        verbose: 진행 상황 출력 여부
        callback: check_interval마다 호출되는 콜백 함수 (모니터링용)

    Returns:
        결과 딕셔너리:
          'Cd_history': 항력 계수 시계열
          'Cl_history': 양력 계수 시계열
          'state': 최종 SimState
          'converged': 수렴 여부
          'final_step': 최종 스텝 번호
    """
    s = initialize(cfg)

    Cd_history = []
    Cl_history = []

    # 초기 마커 좌표 보존 (진동/침강에서 재사용)
    Lx_c = s.Lx.copy()
    Ly_c = s.Ly.copy()

    # 침강 전용 초기화
    if cfg.motion_type == "sedimentation":
        cx_init, cy_init = cfg.cylinder_center
        y0_init = cy_init  # 초기 y 위치 (무차원화 기준)
        d_lattice = 2.0 * s.lattice_r
        sedimentation_history = []
        vel_half_cache = None  # Verlet 반스텝 속도 임시 저장

    # 수렴 판정용
    pre_Eux = None
    pre_Euy = None
    converged = False
    error = 1.0

    t_start = time.time()
    ttt = 1  # 1-base 스텝 카운터

    while True:
        # --- 종료 조건 ---
        if cfg.use_convergence and converged:
            break
        if ttt > cfg.max_steps:
            break

        # === 침강: Verlet 반스텝 (IBM 전 위치 갱신) ===
        if cfg.motion_type == "sedimentation":
            pos_new, vel_half = verlet_half_step(
                s.particle_pos, s.particle_vel, s.particle_force,
                s.particle_mass, cfg.rho_ratio, s.lattice_r,
                s.gravity_lattice, s.dt, s.dx,
            )
            s.particle_pos = pos_new

            # 마커 갱신 (초기 상대좌표 + 새 중심)
            s.Lx, s.Ly = update_markers(
                Lx_c, Ly_c, cx_init, cy_init, pos_new,
            )

            # desired_velocity = 반스텝 속도 v(n+½)를 IBM 경계 속도로 사용
            s.desired_velocity = compute_desired_velocity(vel_half, s.Lb)
            vel_half_cache = vel_half

            # GPU 모드: 침강 물리는 CPU numpy로 계산하므로, IBM 커널에 전달 전 GPU 변환
            if _use_gpu:
                import cupy as cp
                s.Lx = cp.asarray(s.Lx)
                s.Ly = cp.asarray(s.Ly)
                s.desired_velocity = cp.asarray(s.desired_velocity)

        # === Step 1: 충돌 (BGK + Guo forcing) ===
        f = collision_step(s.fstar, s.feq, s.U, s.fib, s.tau, s.dt, s.lattice)

        # === Step 2: 스트리밍 ===
        s.fstar = streaming_step(s.fstar, f, s.nx, s.ny)

        # === Step 3: 경계 조건 ===
        # BC를 macroscopic보다 먼저 실행 (표준 LBM 순서):
        # 스트리밍 직후 경계 fstar를 보정한 뒤 거시변수를 계산해야 경계 밀도/속도 안정.
        if cfg.scenario_type == "taylor_green":
            t_physical = ttt * s.dt
            apply_bc_analytical(
                s.fstar, s.ro, s.U, t_physical, cfg, s.idx, s.lattice,
            )
        elif cfg.bc_type == "open_boundary":
            apply_bc_open_boundary(
                s.fstar, s.ro, s.U, s.dens, s.idx, s.lattice,
            )
        elif cfg.bc_type == "settling_channel":
            apply_bc_settling_channel(
                s.fstar, s.ro, s.U, s.dens, s.idx, s.lattice,
            )
        else:
            apply_bc_velocity_inlet(
                s.fstar, s.ro, s.U, s.in_u, s.dens, s.idx, s.lattice,
            )

        # === Step 4: 거시 변수 복원 ===
        s.ro, s.U = _macroscopic(s.fstar, s.lattice)

        # === Step 5-6: IBM + 속도 보정 ===
        _ibm_step(s, cfg, ttt)

        # === 침강: Verlet 전스텝 (IBM 후 속도 완성) ===
        if cfg.motion_type == "sedimentation":
            # GPU 최적화: 전체 배열 전송 대신 GPU에서 합산 후 스칼라만 CPU로 전송.
            # _to_cpu(s.fib) 제거 → .sum()이 GPU에서 실행, float()이 스칼라만 전송.
            # DF/MDF: fib (nodenums×2) 전송 제거 → 스칼라 2개만. (NN=641: 19MB → 16B)
            # DFC: dfc_force_lagr (Lb×2) 자체가 작아 효과 미미하나 일관성 유지.
            F_hydro = extract_hydro_force(
                fib=s.fib if cfg.ibm_method != "DFC" else None,
                dfc_force_lagr=s.dfc_force_lagr if cfg.ibm_method == "DFC" else None,
                ibm_method=cfg.ibm_method,
                use_gpu=_use_gpu,
            )
            vel_new = verlet_full_step(
                vel_half_cache, F_hydro,
                s.particle_mass, cfg.rho_ratio, s.lattice_r,
                s.gravity_lattice, s.dt,
            )
            s.particle_vel = vel_new
            s.particle_force = F_hydro  # 다음 스텝의 반스텝에 사용

        # === Step 7: feq 갱신 ===
        s.feq = compute_feq(s.ro, s.U, s.lattice)

        # === Step 8: 운동 업데이트 ===
        if cfg.motion_type == "oscillating":
            s.Lx, s.Ly, s.desired_velocity = update_oscillating(
                ttt, Lx_c, Ly_c, s.desired_velocity,
                cfg.lattice_u, cfg.KC, s.r, s.lattice_r,
            )
        elif cfg.motion_type == "translating":
            s.Lx, s.Ly, s.desired_velocity = update_translating(
                s.Lx, s.Ly, s.desired_velocity,
                cfg.lattice_u, s.dx,
            )
        elif cfg.motion_type == "rotating":
            s.Lx, s.Ly, s.desired_velocity = update_rotating(
                s.Lx, s.Ly, cfg.cylinder_center, cfg.omega,
            )
        # 침강: Verlet 반스텝/전스텝에서 이미 처리됨 → skip

        # === Step 9: 진단 ===
        if ttt % cfg.check_interval == 0:
            # GPU→CPU 전송: 진단 계산은 CPU에서 수행
            ro_cpu = _to_cpu(s.ro)

            # === 침강: Cd/Cl 대신 침강 전용 진단 ===
            if cfg.motion_type == "sedimentation":
                record = record_sedimentation_state(
                    s.particle_pos, s.particle_vel, ttt,
                    d_lattice, s.gravity_lattice, cfg.rho_ratio,
                    y0_init, dx=s.dx,
                )
                sedimentation_history.append(record)
                Cd_history.append(0.0)
                Cl_history.append(0.0)

                # NaN 감지 → 조기 종료
                import math
                if not math.isfinite(record['vy_star']):
                    if verbose:
                        print(f"[STOP] NaN detected at step {ttt}")
                    break

                # 도메인 경계 검사
                bound_warn = check_domain_bounds(
                    s.particle_pos, s.r, cfg.xmax, cfg.ymax,
                    safety=2.0, dx=s.dx,
                )
                if bound_warn:
                    if verbose:
                        print(f"[STOP] {bound_warn}")
                    break

                if verbose:
                    elapsed = time.time() - t_start
                    print(f"step {ttt:>7d} | y*={record['y_star']:.4f} "
                          f"vy*={record['vy_star']:.6f} "
                          f"vx*={record['vx_star']:.6f} | {elapsed:.1f}s")

                if callback is not None:
                    callback(step=ttt, Cd=0.0, Cl=0.0, error=error,
                             state=s, elapsed=time.time() - t_start,
                             converged=converged)
            else:
                # === 기존 Cd/Cl 계산 (변경 없음) ===
                u_ref = cfg.lattice_u if cfg.motion_type == "oscillating" else s.in_u

                if cfg.ibm_method == "DFC":
                    dfc_force_cpu = _to_cpu(s.dfc_force_lagr)
                    Cd, Cl = compute_cd_cl_dfc(
                        dfc_force_cpu, float(ro_cpu.mean()),
                        u_ref, s.lattice_r,
                        motion_type=cfg.motion_type)
                else:
                    fib_cpu = _to_cpu(s.fib)
                    Cd, Cl = compute_cd_cl(fib_cpu, ro_cpu, u_ref, s.lattice_r,
                                            motion_type=cfg.motion_type)
                Cd_history.append(Cd)
                Cl_history.append(Cl)

                if verbose:
                    elapsed = time.time() - t_start
                    print(f"step {ttt:>7d} | Cd={Cd:.6f} Cl={Cl:.6f} "
                          f"| err={error:.2e} | {elapsed:.1f}s")

                if callback is not None:
                    callback(step=ttt, Cd=Cd, Cl=Cl, error=error,
                             state=s, elapsed=time.time() - t_start,
                             converged=converged)

            # 수렴 판정
            Eux_now = _to_cpu(s.U[:, 0]).reshape(s.ny, s.nx)
            Euy_now = _to_cpu(s.U[:, 1]).reshape(s.ny, s.nx)

            if cfg.use_convergence:
                if ttt == cfg.convergence_start:
                    pre_Eux = Eux_now.copy()
                    pre_Euy = Euy_now.copy()
                elif ttt > cfg.convergence_start and pre_Eux is not None:
                    error = check_convergence(Eux_now, Euy_now, pre_Eux, pre_Euy)
                    pre_Eux = Eux_now.copy()
                    pre_Euy = Euy_now.copy()
                    if error < cfg.convergence_threshold:
                        converged = True

        ttt += 1

    elapsed = time.time() - t_start
    if verbose:
        status = "CONVERGED" if converged else "MAX_STEPS"
        print(f"\n{status} at step {ttt-1}, elapsed={elapsed:.1f}s")

    import numpy as _np
    result = {
        "Cd_history": _np.array(Cd_history),
        "Cl_history": _np.array(Cl_history),
        "state": s,
        "converged": converged,
        "final_step": ttt - 1,
    }

    if cfg.motion_type == "sedimentation":
        result["sedimentation_history"] = sedimentation_history

    return result
