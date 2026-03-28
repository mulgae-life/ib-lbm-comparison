"""GPU 커널 (CuPy RawModule).

CPU 경로에서는 import되지 않음. _use_gpu=True일 때만 사용.
각 커널은 RawModule로 한 번 컴파일되고 이후 캐시 재사용.

커널 목록:
  - collision_gpu: BGK 충돌 + Guo forcing
  - streaming_gpu: D2Q9 pull-based 스트리밍
  - macroscopic_gpu: 거시변수 복원 (ro, U)
  - feq_gpu: 평형 분포 함수
  - velocity_correction_gpu: U += fib*dt/(2*ro) (in-place)
  - ibm_direct_forcing_gpu: IBM DF (보간+힘+분산)
  - dfc_correction_gpu: DFC (f_i 보간+BB+deviation+분산+유체력)
"""

import cupy as cp

# =============================================================================
# CUDA 소스 코드
# =============================================================================

_CUDA_SOURCE = r"""
// D2Q9 격자 상수
__device__ const double D_E[9][2] = {
    {0,0}, {1,0}, {0,1}, {-1,0}, {0,-1},
    {1,1}, {-1,1}, {-1,-1}, {1,-1}
};
__device__ const double D_W[9] = {
    4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};

// Delta function: delta_type_id=0 → Peskin 4-point, delta_type_id=1 → hat
__device__ double delta_func(double r, int delta_type_id) {
    double ar = fabs(r);
    if (delta_type_id == 0) {  // peskin4pt
        if (ar <= 1.0) {
            return 0.125 * (3.0 - 2.0*ar + sqrt(fmax(1.0 + 4.0*ar - 4.0*ar*ar, 0.0)));
        } else if (ar <= 2.0) {
            return 0.125 * (5.0 - 2.0*ar - sqrt(fmax(-7.0 + 12.0*ar - 4.0*ar*ar, 0.0)));
        }
        return 0.0;
    } else {  // hat
        return fmax(1.0 - ar, 0.0);
    }
}

// =========================================================================
// 1. 충돌 (BGK + Guo forcing) — 1 thread per (n, i)
// =========================================================================
extern "C" __global__ void collision_kernel(
    const double* __restrict__ fstar,
    const double* __restrict__ feq,
    const double* __restrict__ U,
    const double* __restrict__ fib,
    double inv_tau, double guo_pref, double dt,
    int N,
    double* __restrict__ f_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * 9) return;

    int n = tid / 9;
    int i = tid % 9;

    double ux = U[n*2], uy = U[n*2+1];
    double ex = D_E[i][0], ey = D_E[i][1];
    double eU = ex*ux + ey*uy;

    double term = (3.0*(ex - ux) + 9.0*eU*ex) * fib[n*2]
                + (3.0*(ey - uy) + 9.0*eU*ey) * fib[n*2+1];
    double F_ni = guo_pref * D_W[i] * term;

    f_out[tid] = fstar[tid] - inv_tau*(fstar[tid] - feq[tid]) + F_ni*dt;
}

// =========================================================================
// 2. 스트리밍 (pull-based) — 1 thread per node
// =========================================================================
extern "C" __global__ void streaming_kernel(
    const double* __restrict__ f,
    const double* __restrict__ fstar_old,
    int nx, int ny,
    double* __restrict__ fstar_out
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nx * ny) return;

    int j = tid / nx;
    int i = tid % nx;

    // D2Q9 방향 벡터 (ex, ey)
    const int ex[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int ey[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

    // d=0: 정지 — 항상 복사
    fstar_out[tid*9] = f[tid*9];

    // d=1..8: pull from neighbor
    for (int d = 1; d < 9; d++) {
        int sj = j - ey[d];
        int si = i - ex[d];
        if (sj >= 0 && sj < ny && si >= 0 && si < nx) {
            fstar_out[tid*9 + d] = f[(sj*nx + si)*9 + d];
        } else {
            fstar_out[tid*9 + d] = fstar_old[tid*9 + d];
        }
    }
}

// =========================================================================
// 3. 거시변수 복원 — 1 thread per node
// =========================================================================
extern "C" __global__ void macroscopic_kernel(
    const double* __restrict__ fstar,
    int N,
    double* __restrict__ ro,
    double* __restrict__ U
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double s = 0.0, ux = 0.0, uy = 0.0;
    for (int i = 0; i < 9; i++) {
        double fi = fstar[n*9 + i];
        s += fi;
        ux += fi * D_E[i][0];
        uy += fi * D_E[i][1];
    }
    ro[n] = s;
    U[n*2]   = ux / s;
    U[n*2+1] = uy / s;
}

// =========================================================================
// 4. 평형 분포 (feq) — 1 thread per (n, i)
// =========================================================================
extern "C" __global__ void feq_kernel(
    const double* __restrict__ ro,
    const double* __restrict__ U,
    int N,
    double* __restrict__ feq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * 9) return;

    int n = tid / 9;
    int i = tid % 9;

    double ux = U[n*2], uy = U[n*2+1];
    double eU = D_E[i][0]*ux + D_E[i][1]*uy;
    double u2 = ux*ux + uy*uy;
    feq[tid] = ro[n] * D_W[i] * (1.0 + 3.0*eU + 4.5*eU*eU - 1.5*u2);
}

// =========================================================================
// 5. 속도 보정 (in-place) — 1 thread per node
// =========================================================================
extern "C" __global__ void velocity_correction_kernel(
    double* __restrict__ U,
    const double* __restrict__ fib,
    const double* __restrict__ ro,
    double dt, int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    double factor = dt / (2.0 * ro[n]);
    U[n*2]   += fib[n*2]   * factor;
    U[n*2+1] += fib[n*2+1] * factor;
}

// =========================================================================
// 6. IBM 보간 + 힘 계산 — 1 thread per Lagrangian point
// =========================================================================
extern "C" __global__ void ibm_interp_force_kernel(
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    const double* __restrict__ desired_vel,
    const double* __restrict__ Eux,
    const double* __restrict__ Euy,
    const double* __restrict__ Ro,
    double dx, double dy, double dt,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ Lfx,
    double* __restrict__ Lfy
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double lux = 0.0, luy = 0.0, r_interp = 0.0;

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            int idx = ej_c * nx + ei_c;
            r_interp += Ro[idx] * w;
            lux += Eux[idx] * w;
            luy += Euy[idx] * w;
        }
    }

    // T2: 2*R method
    Lfx[k] = 2.0 * r_interp * (desired_vel[k*2]   - lux) / dt;
    Lfy[k] = 2.0 * r_interp * (desired_vel[k*2+1] - luy) / dt;
}

// =========================================================================
// 7. IBM 힘 분산 (atomicAdd) — 1 thread per Lagrangian point
// =========================================================================
extern "C" __global__ void ibm_spread_kernel(
    const double* __restrict__ Lfx,
    const double* __restrict__ Lfy,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy, double Larea,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ Efx,
    double* __restrict__ Efy
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double lfx = Lfx[k], lfy = Lfy[k];

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((ei * dx - lx) / dx, delta_type_id);
            double wy = delta_func((ej * dy - ly) / dy, delta_type_id);
            double w = wx * wy * Larea;

            int idx = ej_c * nx + ei_c;
            atomicAdd(&Efx[idx], lfx * w);
            atomicAdd(&Efy[idx], lfy * w);
        }
    }
}

// =========================================================================
// D2Q9 반대 방향 인덱스 (DFC bounce-back용)
// =========================================================================
__device__ const int D_OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

// =========================================================================
// 8. DFC f_i 보간 (Eq.15) — 1 thread per Lagrangian point
//    fstar(ny*nx, 9) → f_interp(Lb, 9)
// =========================================================================
extern "C" __global__ void dfc_interp_kernel(
    const double* __restrict__ fstar,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ f_interp
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double fi[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            int idx = ej_c * nx + ei_c;
            for (int q = 0; q < 9; q++) {
                fi[q] += fstar[idx * 9 + q] * w;
            }
        }
    }

    for (int q = 0; q < 9; q++) {
        f_interp[k * 9 + q] = fi[q];
    }
}

// =========================================================================
// 9. DFC BB + deviation·λ + force 융합 (Eq.16,17,25)
//    1 thread per Lagrangian point
// =========================================================================
extern "C" __global__ void dfc_bb_lambda_kernel(
    const double* __restrict__ f_interp,
    const double* __restrict__ desired_vel,
    const double* __restrict__ lambda_k,
    double rho_f, double Larea,
    int Lb,
    double* __restrict__ delta_f,
    double* __restrict__ force
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double ux_wall = desired_vel[k * 2];
    double uy_wall = desired_vel[k * 2 + 1];
    double lam = lambda_k[k];
    double cs2 = 1.0 / 3.0;

    double fx = 0.0, fy = 0.0;

    for (int q = 0; q < 9; q++) {
        double fi_star = f_interp[k * 9 + q];
        double fi_opp  = f_interp[k * 9 + D_OPP[q]];

        // BB (Eq.16): f_bar = f_opp + 2*w*rho_f*(e·u_wall)/cs2
        double e_dot_u = D_E[q][0] * ux_wall + D_E[q][1] * uy_wall;
        double f_bb = fi_opp + 2.0 * D_W[q] * rho_f * e_dot_u / cs2;

        // deviation + scaling: delta_f = lambda * (f_bb - f_star)
        double dev = f_bb - fi_star;
        double df = lam * dev;
        delta_f[k * 9 + q] = df;

        // force (Eq.25): F = -dS * sum_i e_i * lambda * dev
        fx += -df * Larea * D_E[q][0];
        fy += -df * Larea * D_E[q][1];
    }

    force[k * 2]     = fx;
    force[k * 2 + 1] = fy;
}

// =========================================================================
// 10. DFC delta_f 분산 (Eq.18, atomicAdd) — 1 thread per Lagrangian point
//     delta_f(Lb, 9) → Ef_corr(ny*nx*9)
// =========================================================================
extern "C" __global__ void dfc_spread_kernel(
    const double* __restrict__ delta_f,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy, double Larea,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ Ef_corr
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double df[9];
    for (int q = 0; q < 9; q++) {
        df[q] = delta_f[k * 9 + q];
    }

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy * Larea;

            int idx = ej_c * nx + ei_c;
            for (int q = 0; q < 9; q++) {
                atomicAdd(&Ef_corr[idx * 9 + q], df[q] * w);
            }
        }
    }
}

// =========================================================================
// 11. DFC lambda spread (Eq.23 Step 1) — 1 thread per Lagrangian point
//     W_total(x) = Sigma_k W(x - X_k)  (atomicAdd)
// =========================================================================
extern "C" __global__ void dfc_lambda_spread_kernel(
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ W_total
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            atomicAdd(&W_total[ej_c * nx + ei_c], w);
        }
    }
}

// =========================================================================
// 12. DFC lambda interpolate + compute (Eq.23 Step 2+3)
//     W_sum(k) = Sigma_x W_total(x) * W(x - X_k)
//     lambda(k) = 1 / (2 * rho_f * dS * W_sum)
// =========================================================================
extern "C" __global__ void dfc_lambda_interp_kernel(
    const double* __restrict__ W_total,
    const double* __restrict__ Lx,
    const double* __restrict__ Ly,
    double dx, double dy,
    double rho_f, double Larea,
    int nx, int ny, int Lb,
    int delta_type_id, int stencil_a,
    double* __restrict__ lambda_k
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= Lb) return;

    double lx = Lx[k], ly = Ly[k];
    int ix0 = (int)floor(lx / dx + 0.5);
    int iy0 = (int)floor(ly / dy + 0.5);

    double W_sum = 0.0;

    for (int dj = -stencil_a; dj <= stencil_a; dj++) {
        for (int di = -stencil_a; di <= stencil_a; di++) {
            int ei = ix0 + di;
            int ej = iy0 + dj;
            int ei_c = min(max(ei, 0), nx - 1);
            int ej_c = min(max(ej, 0), ny - 1);

            double wx = delta_func((lx - ei * dx) / dx, delta_type_id);
            double wy = delta_func((ly - ej * dy) / dy, delta_type_id);
            double w = wx * wy;

            W_sum += W_total[ej_c * nx + ei_c] * w;
        }
    }

    lambda_k[k] = 1.0 / (2.0 * rho_f * Larea * W_sum);
}
"""

# =============================================================================
# 모듈 컴파일 + 커널 참조
# =============================================================================

_module = cp.RawModule(code=_CUDA_SOURCE)

_collision_kern = _module.get_function("collision_kernel")
_streaming_kern = _module.get_function("streaming_kernel")
_macroscopic_kern = _module.get_function("macroscopic_kernel")
_feq_kern = _module.get_function("feq_kernel")
_vel_correction_kern = _module.get_function("velocity_correction_kernel")
_ibm_interp_force_kern = _module.get_function("ibm_interp_force_kernel")
_ibm_spread_kern = _module.get_function("ibm_spread_kernel")
_dfc_interp_kern = _module.get_function("dfc_interp_kernel")
_dfc_bb_lambda_kern = _module.get_function("dfc_bb_lambda_kernel")
_dfc_spread_kern = _module.get_function("dfc_spread_kernel")
_dfc_lambda_spread_kern = _module.get_function("dfc_lambda_spread_kernel")
_dfc_lambda_interp_kern = _module.get_function("dfc_lambda_interp_kernel")

_BLOCK = 256


def _grid(n):
    """n개 스레드에 필요한 1D grid 크기."""
    return ((n + _BLOCK - 1) // _BLOCK,)


# =============================================================================
# Python 래퍼
# =============================================================================

def collision_gpu(fstar, feq, U, fib, tau, dt):
    """GPU BGK 충돌 + Guo forcing."""
    N = fstar.shape[0]
    f = cp.empty_like(fstar)
    inv_tau = 1.0 / tau
    guo_pref = 1.0 - 0.5 * inv_tau
    _collision_kern(
        _grid(N * 9), (_BLOCK,),
        (fstar, feq, U, fib,
         cp.float64(inv_tau), cp.float64(guo_pref), cp.float64(dt),
         cp.int32(N), f),
    )
    return f


def streaming_gpu(fstar_old, f, nx, ny):
    """GPU D2Q9 pull-based 스트리밍."""
    N = nx * ny
    fstar_out = cp.empty_like(fstar_old)
    _streaming_kern(
        _grid(N), (_BLOCK,),
        (f, fstar_old, cp.int32(nx), cp.int32(ny), fstar_out),
    )
    return fstar_out


def macroscopic_gpu(fstar):
    """GPU 거시변수 복원. Returns (ro, U)."""
    N = fstar.shape[0]
    ro = cp.empty(N, dtype=cp.float64)
    U = cp.empty((N, 2), dtype=cp.float64)
    _macroscopic_kern(
        _grid(N), (_BLOCK,),
        (fstar, cp.int32(N), ro, U),
    )
    return ro, U


def feq_gpu(ro, U):
    """GPU 평형 분포 함수."""
    N = ro.shape[0]
    feq = cp.empty((N, 9), dtype=cp.float64)
    _feq_kern(
        _grid(N * 9), (_BLOCK,),
        (ro, U, cp.int32(N), feq),
    )
    return feq


def velocity_correction_gpu(U, fib, ro, dt):
    """GPU 속도 보정 (in-place). U += fib*dt/(2*ro)."""
    N = U.shape[0]
    _vel_correction_kern(
        _grid(N), (_BLOCK,),
        (U, fib, ro, cp.float64(dt), cp.int32(N)),
    )


_DELTA_TYPE_IDS = {"peskin4pt": 0, "hat": 1}
_DELTA_STENCIL_A = {"peskin4pt": 2, "hat": 1}


def ibm_direct_forcing_gpu(Lx, Ly, desired_vel, Eux, Euy, Ro,
                            dx, dy, dt, Larea, ny, nx,
                            delta_type="peskin4pt"):
    """GPU IBM Direct Forcing (보간 + 힘 + 분산).

    Returns:
        fib: IB 체적력, (nodenums, 2)
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]

    # CUDA 커널은 C-contiguous 메모리 레이아웃을 가정
    Eux = cp.ascontiguousarray(Eux)
    Euy = cp.ascontiguousarray(Euy)
    Ro = cp.ascontiguousarray(Ro)
    desired_vel = cp.ascontiguousarray(desired_vel)
    Lb = len(Lx)
    nodenums = ny * nx

    # --- 보간 + 힘 계산 (1 kernel launch) ---
    Lfx = cp.empty(Lb, dtype=cp.float64)
    Lfy = cp.empty(Lb, dtype=cp.float64)
    _ibm_interp_force_kern(
        _grid(Lb), (_BLOCK,),
        (Lx, Ly, desired_vel, Eux, Euy, Ro,
         cp.float64(dx), cp.float64(dy), cp.float64(dt),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         Lfx, Lfy),
    )

    # --- 분산 (1 kernel launch, atomicAdd) ---
    Efx = cp.zeros((ny, nx), dtype=cp.float64)
    Efy = cp.zeros((ny, nx), dtype=cp.float64)
    _ibm_spread_kern(
        _grid(Lb), (_BLOCK,),
        (Lfx, Lfy, Lx, Ly,
         cp.float64(dx), cp.float64(dy), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         Efx, Efy),
    )

    # fib (nodenums, 2) 패킹
    fib = cp.empty((nodenums, 2), dtype=cp.float64)
    fib[:, 0] = Efx.ravel()
    fib[:, 1] = Efy.ravel()

    return fib


def compute_lambda_gpu(Lx, Ly, rho_f, Larea, dx, dy, ny, nx, delta_type):
    """GPU lambda(k) 계산 (Eq.23).

    CUDA 커널 2개 순차 호출:
      1) dfc_lambda_spread_kernel: W_total(x) = Σ_k W(x - X_k)
      2) dfc_lambda_interp_kernel: W_sum(k) + lambda(k) = 1/(2·ρ·dS·W_sum)

    Args:
        Lx, Ly: (Lb,) — Lagrangian 점 좌표
        rho_f: float — 기준 밀도
        Larea: float — 호 길이
        dx, dy: 격자 간격
        ny, nx: 격자 크기
        delta_type: "hat" or "peskin4pt"

    Returns:
        lambda_k: (Lb,)
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]
    Lb = len(Lx)

    # Step 1: spread — W_total(x)
    W_total = cp.zeros((ny * nx,), dtype=cp.float64)
    _dfc_lambda_spread_kern(
        _grid(Lb), (_BLOCK,),
        (Lx, Ly,
         cp.float64(dx), cp.float64(dy),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         W_total),
    )

    # Step 2: interpolate + lambda 계산
    lambda_k = cp.empty(Lb, dtype=cp.float64)
    _dfc_lambda_interp_kern(
        _grid(Lb), (_BLOCK,),
        (W_total, Lx, Ly,
         cp.float64(dx), cp.float64(dy),
         cp.float64(rho_f), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         lambda_k),
    )

    return lambda_k


def dfc_correction_gpu(Lx, Ly, desired_vel, fstar, lambda_k,
                        dx, dy, Larea, ny, nx,
                        delta_type, lattice):
    """GPU DFC 분포함수 보정 (interp + bb_lambda + spread).

    3개 CUDA 커널 순차 호출:
      1) dfc_interp_kernel: f_i 9개 보간 (Eq.15)
      2) dfc_bb_lambda_kernel: BB + deviation·λ + force 융합 (Eq.16,17,25)
      3) dfc_spread_kernel: delta_f 분산 (Eq.18, atomicAdd)

    Args:
        Lx, Ly: (Lb,) — Lagrangian 점 좌표
        desired_vel: (Lb, 2) — 경계 desired velocity
        fstar: (ny*nx, 9) — post-streaming 분포함수
        lambda_k: (Lb,) — lambda 값
        dx, dy: 격자 간격
        Larea: 호 길이
        ny, nx: 격자 크기
        delta_type: "hat" or "peskin4pt"
        lattice: D2Q9 (opp, e, w 참조용 — CUDA 상수로 하드코딩)

    Returns:
        delta_f_euler: (ny*nx, 9) — Eulerian 분포함수 보정
        dfc_force: (Lb, 2) — Lagrangian 점별 유체력
    """
    delta_type_id = _DELTA_TYPE_IDS[delta_type]
    stencil_a = _DELTA_STENCIL_A[delta_type]
    Lb = len(Lx)

    fstar = cp.ascontiguousarray(fstar)
    desired_vel = cp.ascontiguousarray(desired_vel)

    # --- 1) f_i 보간 ---
    f_interp = cp.empty((Lb, 9), dtype=cp.float64)
    _dfc_interp_kern(
        _grid(Lb), (_BLOCK,),
        (fstar, Lx, Ly,
         cp.float64(dx), cp.float64(dy),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         f_interp),
    )

    # --- 2) BB + deviation·λ + force ---
    delta_f = cp.empty((Lb, 9), dtype=cp.float64)
    dfc_force = cp.empty((Lb, 2), dtype=cp.float64)
    _dfc_bb_lambda_kern(
        _grid(Lb), (_BLOCK,),
        (f_interp, desired_vel, lambda_k,
         cp.float64(1.0), cp.float64(Larea),
         cp.int32(Lb),
         delta_f, dfc_force),
    )

    # --- 3) delta_f 분산 (atomicAdd) ---
    Ef_corr = cp.zeros((ny * nx, 9), dtype=cp.float64)
    _dfc_spread_kern(
        _grid(Lb), (_BLOCK,),
        (delta_f, Lx, Ly,
         cp.float64(dx), cp.float64(dy), cp.float64(Larea),
         cp.int32(nx), cp.int32(ny), cp.int32(Lb),
         cp.int32(delta_type_id), cp.int32(stencil_a),
         Ef_corr),
    )

    return Ef_corr, dfc_force
