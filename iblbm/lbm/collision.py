"""BGK 충돌 + Guo 강제항.

MATLAB 참조:
  ForceDistributionFunc.m — Guo 강제항
  IBLBM_steady_DF_CPU.m line 137-138 — 충돌 스텝

Guo forcing (Guo et al., Physical Review E, 2002):
  F_i = (1 - 1/(2*tau)) * w_i * [3*(e_i - u) + 9*(e_i · u)*e_i] · f_IB

  cs^2 = 1/3 대입 결과:
    3*(e_i - u) = (e_i - u) / cs^2
    9*(e_i · u)*e_i = (e_i · u)*e_i / cs^4

BGK 충돌:
  f = fstar - (1/tau)*(fstar - feq) + F*dt
"""

from ..backend import xp as np, _use_gpu
from .lattice import D2Q9

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _collision_step_nb(fstar, feq, U, fib, tau, dt, e, w):
        """충돌 + Guo forcing 융합 커널. 임시 배열 없이 원소별 계산.

        (N, 9, 2) 텐서 임시 배열을 제거하고 prange로 12코어 병렬 실행.
        """
        N = fstar.shape[0]
        f = np.empty_like(fstar)
        inv_tau = 1.0 / tau
        guo_pref = 1.0 - 0.5 / tau
        for n in prange(N):
            for i in range(9):
                # Guo forcing inline
                ax = e[i, 0] - U[n, 0]
                ay = e[i, 1] - U[n, 1]
                eU = e[i, 0] * U[n, 0] + e[i, 1] * U[n, 1]
                term = ((3.0 * ax + 9.0 * eU * e[i, 0]) * fib[n, 0]
                        + (3.0 * ay + 9.0 * eU * e[i, 1]) * fib[n, 1])
                F_ni = guo_pref * w[i] * term
                # BGK collision
                f[n, i] = fstar[n, i] - inv_tau * (fstar[n, i] - feq[n, i]) + F_ni * dt
        return f


def guo_forcing(U, fib, tau, lattice):
    """Guo 강제항 계산 (진단용, 시간 루프에서는 collision_step에 융합).

    Args:
        U: 거시 속도, (N, 2)
        fib: IB 체적력, (N, 2)
        tau: BGK 완화 시간
        lattice: D2Q9 격자 상수

    Returns:
        F: 강제항 분포, (N, 9)
    """
    e = lattice.e   # (9, 2)
    w = lattice.w   # (9,)
    a = e[None, :, :] - U[:, None, :]         # (N, 9, 2)
    eU = np.sum(e[None, :, :] * U[:, None, :], axis=2)  # (N, 9)
    b = eU[:, :, None] * e[None, :, :]        # (N, 9, 2)
    term = np.sum((3 * a + 9 * b) * fib[:, None, :], axis=2)
    return (1.0 - 0.5 / tau) * (w[None, :] * term)


def collision_step(fstar, feq, U, fib, tau, dt, lattice):
    """BGK 충돌 + Guo 강제항 적용.

    GPU: 벡터화, CPU: numba 커널.
    """
    if _use_gpu:
        from ..gpu_kernels import collision_gpu
        return collision_gpu(fstar, feq, U, fib, tau, dt)
    return _collision_step_nb(fstar, feq, U, fib, tau, dt, lattice.e, lattice.w)
