"""침강 시나리오 -- Feng et al. (1994) 표준 구성.

Ma-constrained 격자 파라미터 결정:
  물리 파라미터(CGS)에서 Archimedes 수를 계산하고,
  Ma < Ma_max 제약으로 ν_lat → τ → g_lattice를 역산한다.
  NN이 부족하면 자동 상향 또는 에러 발생.

도메인 프리셋:
  "paper0"        — L=8cm, 초기 (1,7)cm. 본 논문 기본 설정.
  "glowinski2001" — L=6cm, 초기 (1,4)cm. Glowinski 2001 / Wang 2008 canonical.
"""

from iblbm.config import SimConfig
from iblbm.solver import run
import numpy as np


# 도메인 프리셋 (좌표는 W=2cm 기준 무차원)
DOMAIN_PRESETS = {
    "paper0": {
        "ymax": 4.0,              # 4W = 8cm
        "cylinder_center": (0.5, 3.5),  # (1cm, 7cm) / W
    },
    "glowinski2001": {
        "ymax": 3.0,              # 3W = 6cm
        "cylinder_center": (0.5, 2.0),  # (1cm, 4cm) / W
    },
    "tall80D": {
        "ymax": 10.0,             # 10W = 20cm = 80D
        "cylinder_center": (0.5, 9.0),  # (1cm, 18cm) / W — 상단 1W 여유
    },
}


def make_sedimentation_config(
    rho_ratio: float = 1.01,
    ibm_method: str = "DF",
    delta_type: str = "peskin4pt",
    NN: int | None = None,
    Ma_max: float = 0.08,
    tau_min: float = 0.52,
    verbose: bool = True,
    case_name: str = "paper0",
    center_x_offset_dx: float = 0.0,
) -> SimConfig:
    """침강 벤치마크 SimConfig 생성 — Ma-constrained 방식.

    Feng 1994 물리 파라미터 (CGS):
        W = 2 cm, d = 0.25 cm, nu = 0.01 cm²/s, g = 981 cm/s²

    격자 파라미터는 Ma 제약으로 자동 결정:
        Ma = √(3·Ar) × ν_lat / D_lat < Ma_max
        ν_lat ≥ (τ_min - 0.5) / 3  (안정성)
        → NN이 부족하면 에러

    Args:
        rho_ratio: rho_s / rho_f
        ibm_method: "DF" | "MDF" | "DFC"
        delta_type: "peskin4pt" | "hat"
        NN: 격자 해상도 (None → 자동 결정)
        Ma_max: 최대 허용 Mach 수 (기본 0.08, 안전 여유 포함)
        tau_min: 최소 τ (기본 0.52, 안정성)
        case_name: 도메인 프리셋 ("paper0" | "glowinski2001")
        center_x_offset_dx: 입자 초기 x위치 오프셋 (격자 단위, 0=중심)

    Returns:
        SimConfig
    """
    # 물리 파라미터 (CGS) — Feng 1994
    W_phys = 2.0       # cm
    d_phys = 0.25       # cm  → W/d = 8
    nu_phys = 0.01      # cm²/s
    g_phys = 981.0      # cm/s²

    # 도메인 프리셋
    if case_name not in DOMAIN_PRESETS:
        raise ValueError(
            f"Unknown case_name '{case_name}'. "
            f"Available: {list(DOMAIN_PRESETS.keys())}"
        )
    preset = DOMAIN_PRESETS[case_name]
    xmax = 1.0           # W (채널 폭)
    ymax = preset["ymax"]
    D_ratio = d_phys / W_phys   # = 0.125

    # Archimedes 수 (무차원 — rho_ratio 의존)
    delta_rho = abs(rho_ratio - 1.0)
    Ar = delta_rho * g_phys * d_phys**3 / nu_phys**2
    nu_lat_min = (tau_min - 0.5) / 3.0

    # NN 결정: Ma 제약으로 최소 D_lat 계산
    # safety_factor=4: BGK 침강 안정성을 위해 충분한 해상도 확보.
    if delta_rho > 1e-15 and Ar > 0:
        safety_factor = 4.0
        D_lat_min = safety_factor * np.sqrt(3.0 * Ar) * nu_lat_min / Ma_max
        NN_min = int(np.ceil(D_lat_min / D_ratio)) + 1
    else:
        NN_min = 81

    if NN is None:
        for candidate in [81, 161, 321, 641, 1281]:
            if candidate >= NN_min:
                NN = candidate
                break
        else:
            NN = NN_min
    elif NN < NN_min:
        import warnings
        warnings.warn(
            f"NN={NN} < NN_min={NN_min} (SF={safety_factor}). "
            f"안정성 미보장 — 격자 수렴 연구 등 의도적 사용만 권장.",
            stacklevel=2,
        )

    # 격자 파라미터
    dx = 1.0 / (NN - 1)
    lattice_D = D_ratio * (NN - 1)
    lattice_r = lattice_D / 2.0

    # ν_lat: Ma 제약의 80%로 설정 (안전 여유)
    if delta_rho > 1e-15 and Ar > 0:
        nu_lat_max = Ma_max * lattice_D / np.sqrt(3.0 * Ar)
        nu_lat = max(nu_lat_max * 0.8, nu_lat_min)
    else:
        nu_lat = 0.05

    tau = 3.0 * nu_lat + 0.5

    # g_lattice = Ar × ν_lat² / (Δρ × D_lat³)
    if delta_rho > 1e-15:
        g_lattice = Ar * nu_lat**2 / (delta_rho * lattice_D**3)
    else:
        g_lattice = 1e-6

    # 종단 속도 추정 + Ma 검증
    u_t_est = np.sqrt(delta_rho * g_lattice * lattice_D) if delta_rho > 0 else 0.0
    Ma_est = u_t_est * np.sqrt(3.0)

    if Ma_est > Ma_max:
        import warnings
        warnings.warn(
            f"Ma_est={Ma_est:.4f} > Ma_max={Ma_max}. "
            f"NN을 높이거나 Ma_max를 완화하세요.",
            stacklevel=2,
        )

    # lattice_u: 침강에서 참조 속도. u_t 추정값 사용 (config Re 계산용)
    lattice_u = max(u_t_est, 0.01)
    Re_input = lattice_u * lattice_D / nu_lat

    # max_steps: 채널 횡단 시간 × 2 (여유)
    channel_height_lattice = ymax / dx
    if u_t_est > 1e-10:
        max_steps = int(2.0 * channel_height_lattice / u_t_est)
    else:
        max_steps = 500_000

    # 초기 위치: 프리셋 + x 오프셋 (격자 단위)
    cx_base, cy = preset["cylinder_center"]
    cx = cx_base + center_x_offset_dx * dx

    # MDF 반복 횟수
    mdf_iter = 5

    cfg = SimConfig(
        Re=Re_input,
        NN=NN,
        xmax=xmax,
        ymax=ymax,
        cylinder_center=(cx, cy),
        cylinder_D_ratio=D_ratio,
        lattice_u=lattice_u,
        inflow_u=0.0,
        bc_type="settling_channel",
        delta_type=delta_type,
        ibm_method=ibm_method,
        mdf_iterations=mdf_iter,
        max_steps=max_steps,
        use_convergence=False,
        check_interval=100,
        motion_type="sedimentation",
        rho_ratio=rho_ratio,
        gravity=g_lattice,
        marker_spacing_factor=0.5,
        nx_formula="standard",
    )

    if verbose:
        print(f"[Sedimentation] rho_ratio={rho_ratio}, Ar={Ar:.0f}, "
              f"case={case_name}")
        print(f"  NN={NN}, D_lat={lattice_D:.0f}, tau={tau:.4f}")
        print(f"  g_lat={g_lattice:.6e}, u_t_est={u_t_est:.4f}, "
              f"Ma_est={Ma_est:.4f}")
        print(f"  domain: xmax={xmax}, ymax={ymax}, "
              f"center=({cx:.6f}, {cy})")
        if center_x_offset_dx != 0.0:
            print(f"  x_offset: {center_x_offset_dx:.2f} dx "
                  f"= {center_x_offset_dx * dx:.6e} domain units")

    return cfg


# === CLI ===
if __name__ == "__main__":
    import sys

    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 1.01
    method = sys.argv[2] if len(sys.argv) > 2 else "DF"
    delta = sys.argv[3] if len(sys.argv) > 3 else "peskin4pt"

    cfg = make_sedimentation_config(
        rho_ratio=rho, ibm_method=method, delta_type=delta,
    )

    result = run(cfg, verbose=True)

    history = result.get("sedimentation_history", [])
    if history:
        last = history[-1]
        print(f"\nFinal: vy*={last['vy_star']:.6f}, y*={last['y_star']:.4f}")
