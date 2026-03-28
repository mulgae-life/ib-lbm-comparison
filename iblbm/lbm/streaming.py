"""D2Q9 스트리밍 (인덱스 시프트 방식).

MATLAB 참조: Stream.m

MATLAB과 동일한 비주기적(non-periodic) 인덱스 시프트 방식 사용 (I2).
경계 행/열은 이전 fstar 값을 보존하고, BC에서 덮어씀.
np.roll 대비 MATLAB과의 수치 동일성 보장.

2D 배열 규칙 (C1):
  fstar[:, i].reshape(ny, nx) → (ny, nx), 행=y(j), 열=x(i)
  MATLAB: reshape(fstar(:,i), nx, ny)' → 전치 후 (ny, nx)
"""

from ..backend import xp as np, _use_gpu

if not _use_gpu:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _streaming_step_nb(fstar, f, nx, ny):
        """D2Q9 스트리밍 numba 커널. 명시적 인덱스 산술로 prange 병렬화."""
        fstar_new = fstar.copy()
        # i=0: 정지 — fstar_new = f
        for n in prange(ny * nx):
            fstar_new[n, 0] = f[n, 0]
        # i=1: 우 — dst[:, 1:nx] = src[:, 0:nx-1]
        for j in prange(ny):
            for i in range(1, nx):
                fstar_new[j * nx + i, 1] = f[j * nx + (i - 1), 1]
        # i=2: 상 — dst[1:ny, :] = src[0:ny-1, :]
        for j in prange(1, ny):
            for i in range(nx):
                fstar_new[j * nx + i, 2] = f[(j - 1) * nx + i, 2]
        # i=3: 좌 — dst[:, 0:nx-1] = src[:, 1:nx]
        for j in prange(ny):
            for i in range(nx - 1):
                fstar_new[j * nx + i, 3] = f[j * nx + (i + 1), 3]
        # i=4: 하 — dst[0:ny-1, :] = src[1:ny, :]
        for j in prange(ny - 1):
            for i in range(nx):
                fstar_new[j * nx + i, 4] = f[(j + 1) * nx + i, 4]
        # i=5: 우상 — dst[1:,1:] = src[:ny-1,:nx-1]
        for j in prange(1, ny):
            for i in range(1, nx):
                fstar_new[j * nx + i, 5] = f[(j - 1) * nx + (i - 1), 5]
        # i=6: 좌상 — dst[1:,:nx-1] = src[:ny-1,1:]
        for j in prange(1, ny):
            for i in range(nx - 1):
                fstar_new[j * nx + i, 6] = f[(j - 1) * nx + (i + 1), 6]
        # i=7: 좌하 — dst[:ny-1,:nx-1] = src[1:,1:]
        for j in prange(ny - 1):
            for i in range(nx - 1):
                fstar_new[j * nx + i, 7] = f[(j + 1) * nx + (i + 1), 7]
        # i=8: 우하 — dst[:ny-1,1:] = src[1:,:nx-1]
        for j in prange(ny - 1):
            for i in range(1, nx):
                fstar_new[j * nx + i, 8] = f[(j + 1) * nx + (i - 1), 8]
        return fstar_new


def streaming_step(fstar, f, nx, ny):
    """D2Q9 스트리밍 단계.

    GPU: 벡터화 슬라이싱, CPU: numba 커널.
    """
    if _use_gpu:
        from ..gpu_kernels import streaming_gpu
        return streaming_gpu(fstar, f, nx, ny)
    return _streaming_step_nb(fstar, f, nx, ny)
