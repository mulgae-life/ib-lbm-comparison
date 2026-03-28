"""평형 분포 함수 (feq) 계산.

MATLAB 참조: IBLBM_steady_DF_CPU.m lines 90-95

D2Q9 평형 분포:
  feq_i = w_i * rho * (1 + (e_i · u)/cs^2 + (e_i · u)^2/(2*cs^4) - u^2/(2*cs^2))

cs^2 = 1/3, c = 1 적용 (M2, M3):
  feq_i = w_i * rho * (1 + 3*(e_i · u) + 9/2*(e_i · u)^2 - 3/2*u^2)
"""

from ..backend import xp as np, _use_gpu
from .lattice import D2Q9

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _compute_feq_nb(ro, U, e, w):
        """feq 계산 numba 커널. prange로 병렬 실행."""
        N = ro.shape[0]
        feq = np.empty((N, 9))
        for n in prange(N):
            u2 = U[n, 0] ** 2 + U[n, 1] ** 2
            for i in range(9):
                eU = e[i, 0] * U[n, 0] + e[i, 1] * U[n, 1]
                feq[n, i] = ro[n] * w[i] * (1.0 + 3.0 * eU + 4.5 * eU * eU - 1.5 * u2)
        return feq


def compute_feq(ro, U, lattice):
    """평형 분포 함수 계산.

    GPU: 벡터화, CPU: numba 커널.
    """
    if _use_gpu:
        from ..gpu_kernels import feq_gpu
        return feq_gpu(ro, U)
    return _compute_feq_nb(ro, U, lattice.e, lattice.w)
