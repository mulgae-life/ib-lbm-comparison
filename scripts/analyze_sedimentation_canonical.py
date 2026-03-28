"""침강 벤치마크 정밀 분석 — Glowinski 2001 canonical 도메인 (2×6cm).

실험 완료 후 실행:
    cd <public-repo-root>
    python scripts/analyze_sedimentation_canonical.py

출력:
    figures/fig8_sed_velocity.png
    figures/fig9_sed_trajectory.png
    figures/fig10_sed_3way.png
    참고: 최종 composite figure는 패키지에 포함된 `figures/fig11_v29.png` 사용
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

# =====================================================================
# 경로 설정
# =====================================================================

DATA_ROOT = Path("data/sedimentation_canonical")
FIG_DIR = Path("figures")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 실험 매트릭스
# =====================================================================

# (run_id, rho, method, delta, subdir)
EXPERIMENTS = [
    ("A01", 1.01, "DF",  "peskin4pt", "peskin4pt/rho1.01/df"),
    ("A02", 1.01, "MDF", "peskin4pt", "peskin4pt/rho1.01/mdf"),
    ("A03", 1.01, "DFC", "peskin4pt", "peskin4pt/rho1.01/dfc"),
    ("A04", 1.01, "DF",  "hat",       "hat/rho1.01/df"),
    ("A05", 1.01, "MDF", "hat",       "hat/rho1.01/mdf"),
    ("A06", 1.01, "DFC", "hat",       "hat/rho1.01/dfc"),
    ("A07", 1.1,  "DF",  "peskin4pt", "peskin4pt/rho1.1/df"),
    ("A08", 1.1,  "MDF", "peskin4pt", "peskin4pt/rho1.1/mdf"),
    ("A09", 1.1,  "DFC", "peskin4pt", "peskin4pt/rho1.1/dfc"),
    ("A10", 1.1,  "DF",  "hat",       "hat/rho1.1/df"),
    ("A11", 1.1,  "MDF", "hat",       "hat/rho1.1/mdf"),
    ("A12", 1.1,  "DFC", "hat",       "hat/rho1.1/dfc"),
    ("A13", 1.5,  "DF",  "peskin4pt", "peskin4pt/rho1.5/df"),
    ("A14", 1.5,  "MDF", "peskin4pt", "peskin4pt/rho1.5/mdf"),
    ("A15", 1.5,  "DFC", "peskin4pt", "peskin4pt/rho1.5/dfc"),
    ("A16", 1.5,  "DF",  "hat",       "hat/rho1.5/df"),
    ("A17", 1.5,  "MDF", "hat",       "hat/rho1.5/mdf"),
    ("A18", 1.5,  "DFC", "hat",       "hat/rho1.5/dfc"),
]

RHOS = [1.01, 1.1, 1.5]

# 물리 상수 (Feng 1994 표준)
D_PHYS = 0.25       # cm
NU_PHYS = 0.01      # cm²/s
G_PHYS = 981.0      # cm/s²
W_PHYS = 2.0        # cm

# 문헌 레퍼런스 Re_t 값
# Re_t = U_t × d / ν
# 2D 원형 실린더 침강 Re_t 문헌값 (Feng 1994 도메인 파라미터 기준)
# rho=1.01/1.1: 저 Re 레짐 — 기존 paper0 실험에서 Re≈34/137 확인, 정량 비교 가능한 문헌 부재
# rho=1.5: 고 Re 레짐 — 복수 문헌 정량 비교 가능
LITERATURE = {
    1.01: {
        "regime": "I (steady, Re~34)",
    },
    1.1: {
        "regime": "II (transition, Re~137)",
    },
    1.5: {
        "Glowinski2001_FEM_low": 438.6,
        "Glowinski2001_FEM_high": 466.0,
        "Uhlmann2005_DF_IBM": 495.0,
        "Wang2008_NF1_DF": 484.75,
        "Wang2008_NF20_MDF": 503.38,
        "regime": "III (strong wake, Re~440-503)",
    },
}


def compute_reference_scales(rho_ratio):
    """무차원화 기준 속도/레이놀즈 수 계산."""
    delta_rho = abs(rho_ratio - 1.0)
    u_g = np.sqrt(delta_rho * G_PHYS * D_PHYS)
    Ar = delta_rho * G_PHYS * D_PHYS**3 / NU_PHYS**2
    Ga = np.sqrt(Ar)
    return {"u_g": u_g, "Ar": Ar, "Ga": Ga, "delta_rho": delta_rho}


# =====================================================================
# 데이터 로딩
# =====================================================================

def load_history(subdir):
    path = DATA_ROOT / subdir / "sedimentation_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_status(subdir):
    path = DATA_ROOT / subdir / "status.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_terminal_velocity(history, min_points=20):
    """종단 속도 추출 — plateau 자동 탐색 (rolling std 기반)."""
    if not history or len(history) < min_points:
        return None

    vy_all = np.array([r["vy_star"] for r in history])
    vx_all = np.array([r["vx_star"] for r in history])

    if not np.all(np.isfinite(vy_all)):
        return None

    n = len(vy_all)

    # Rolling std로 가장 안정적인 구간 탐색
    window = max(n // 10, 5)
    rolling_std = np.full(n, np.inf)
    for i in range(window, n - window):
        seg = vy_all[i - window:i + window]
        rolling_std[i] = np.std(seg)

    # 벽면 감속 제거: 마지막 10% 제외
    cutoff = int(n * 0.9)
    search_region = rolling_std[:cutoff]
    valid = np.isfinite(search_region)

    if not np.any(valid):
        start, end = int(n * 0.5), int(n * 0.8)
    else:
        best_center = np.argmin(search_region)
        start = max(0, best_center - window)
        end = min(cutoff, best_center + window)

    if end - start < min_points:
        start = max(0, int(n * 0.5))
        end = min(n, int(n * 0.85))

    plateau_vy = vy_all[start:end]
    plateau_vx = vx_all[start:end]

    vy_terminal = float(np.mean(plateau_vy))
    t_all = np.array([r["t_star"] for r in history])

    # t*_99: 종단 속도 99% 도달 시간
    threshold_99 = 0.99 * vy_terminal
    t99_idx = np.where(vy_all >= threshold_99)[0]
    t_star_99 = float(t_all[t99_idx[0]]) if len(t99_idx) > 0 else float(t_all[-1])

    # 오버슈트
    vy_peak = float(np.max(vy_all))
    peak_idx = int(np.argmax(vy_all))
    t_star_peak = float(t_all[peak_idx])
    overshoot_pct = (vy_peak - vy_terminal) / vy_terminal * 100

    # 횡방향 드리프트
    vx_max = float(np.max(np.abs(vx_all)))
    x_all = np.array([r["x"] for r in history])
    x_drift = float(np.abs(x_all[-1] - x_all[0]))

    # 벽면 감속
    tail_start = int(n * 0.95)
    if tail_start < n - 2:
        wall_decel = float((vy_all[tail_start] - vy_all[-1]) / vy_all[tail_start] * 100)
    else:
        wall_decel = 0.0

    return {
        "vy_star_mean": vy_terminal,
        "vy_star_std": float(np.std(plateau_vy)),
        "vx_star_mean": float(np.mean(np.abs(plateau_vx))),
        "t_star_final": history[-1]["t_star"],
        "y_star_final": history[-1]["y_star"],
        "plateau_range": f"{start/n*100:.0f}-{end/n*100:.0f}%",
        "n_points": len(plateau_vy),
        "t_star_99": t_star_99,
        "vy_peak": vy_peak,
        "t_star_peak": t_star_peak,
        "overshoot_pct": overshoot_pct,
        "vx_max": vx_max,
        "x_drift": x_drift,
        "wall_decel_pct": wall_decel,
    }


# =====================================================================
# 논문 그림 생성
# =====================================================================

COLORS = {"DF": "#000000", "MDF": "#CC0000", "DFC": "#0044CC"}
MARKERS = {"DF": "o", "MDF": "^", "DFC": "D"}
FILLS = {"DF": True, "MDF": False, "DFC": True}
LINESTYLES = {"peskin4pt": "-", "hat": "--"}
MS = 8


def fig_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.8,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_fig8_velocity(all_data):
    """Fig 8: vy*(t*) 3-way 비교 (rho별 3 패널)."""
    fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_idx, rho in enumerate(RHOS):
        ax = axes[ax_idx]
        for run_id, r, method, delta, subdir in EXPERIMENTS:
            if r != rho:
                continue
            hist = all_data.get(run_id)
            if hist is None:
                continue

            t = [h["t_star"] for h in hist]
            vy = [h["vy_star"] for h in hist]

            label = f"{method} ({delta[:3]})"
            ax.plot(t, vy,
                    color=COLORS[method],
                    linestyle=LINESTYLES[delta],
                    linewidth=1.2,
                    label=label)

        ax.set_xlabel("$t^*$")
        ax.set_ylabel("$v_y^*$")
        ax.set_title(f"$\\rho_s/\\rho_f = {rho}$")
        ax.legend(loc="lower right", ncol=2, fontsize=8)
        ax.grid(True, ls=":", lw=0.4, color="#cccccc", alpha=0.7)
        ax.text(0.02, 0.95, f"({chr(97 + ax_idx)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout()
    out = FIG_DIR / "fig8_sed_velocity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig9_trajectory(all_data):
    """Fig 9: y*(t*) 3-way 비교 (rho별 3 패널)."""
    fig_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_idx, rho in enumerate(RHOS):
        ax = axes[ax_idx]
        for run_id, r, method, delta, subdir in EXPERIMENTS:
            if r != rho:
                continue
            hist = all_data.get(run_id)
            if hist is None:
                continue

            t = [h["t_star"] for h in hist]
            y = [h["y_star"] for h in hist]

            label = f"{method} ({delta[:3]})"
            ax.plot(t, y,
                    color=COLORS[method],
                    linestyle=LINESTYLES[delta],
                    linewidth=1.2,
                    label=label)

        ax.set_xlabel("$t^*$")
        ax.set_ylabel("$y^*$")
        ax.set_title(f"$\\rho_s/\\rho_f = {rho}$")
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.grid(True, ls=":", lw=0.4, color="#cccccc", alpha=0.7)
        ax.text(0.02, 0.95, f"({chr(97 + ax_idx)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout()
    out = FIG_DIR / "fig9_sed_trajectory.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig10_3way_bar(terminal_data):
    """Fig 10: 종단 속도 dot plot (method × delta × rho)."""
    fig_style()

    DELTA_MARKERS = {"peskin4pt": "o", "hat": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax_idx, rho in enumerate(RHOS):
        ax = axes[ax_idx]
        methods = ["DF", "MDF", "DFC"]
        x_pos = np.arange(len(methods))
        all_vals = []

        for d_idx, delta in enumerate(["peskin4pt", "hat"]):
            offset = (d_idx - 0.5) * 0.2
            for m_idx, m in enumerate(methods):
                key = f"{m}_{delta}_{rho}"
                td = terminal_data.get(key)
                if not td:
                    continue
                val = td["vy_star_mean"]
                err = td["vy_star_std"]
                all_vals.append(val)

                fc = COLORS[m] if delta == "peskin4pt" else "none"
                ax.errorbar(
                    x_pos[m_idx] + offset, val, yerr=err,
                    fmt=DELTA_MARKERS[delta],
                    color=COLORS[m],
                    markerfacecolor=fc,
                    markeredgecolor=COLORS[m],
                    markeredgewidth=1.2,
                    markersize=8, capsize=4, capthick=0.8,
                    linewidth=0, elinewidth=0.8,
                )

            # 같은 delta 내 methods 연결선
            line_vals = []
            for m in methods:
                td = terminal_data.get(f"{m}_{delta}_{rho}")
                line_vals.append(td["vy_star_mean"] if td else None)
            valid = [(x_pos[i] + offset, v) for i, v in enumerate(line_vals) if v]
            if len(valid) > 1:
                ax.plot([p[0] for p in valid], [p[1] for p in valid],
                        color="#cccccc", lw=0.6, zorder=0)

        if all_vals:
            mean_v = np.mean(all_vals)
            ax.axhline(mean_v, color="#bbbbbb", ls="--", lw=0.8, zorder=0)

        if all_vals:
            v_min, v_max = min(all_vals), max(all_vals)
            spread = v_max - v_min
            margin = max(spread * 2.5, np.mean(all_vals) * 0.005)
            ax.set_ylim(np.mean(all_vals) - margin, np.mean(all_vals) + margin)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_xlabel("IBM method")
        ax.set_ylabel("$v_{y,t}^*$")
        ax.set_title(f"$\\rho_s/\\rho_f = {rho}$")
        ax.grid(True, ls=":", lw=0.4, color="#cccccc", alpha=0.7, axis="y")
        ax.text(0.02, 0.95, f"({chr(97 + ax_idx)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    legend_elements = [
        Line2D([0], [0], marker="o", color="#555555", markerfacecolor="#555555",
               markersize=7, label="peskin 4-pt", linewidth=0),
        Line2D([0], [0], marker="s", color="#555555", markerfacecolor="none",
               markeredgecolor="#555555", markeredgewidth=1.2,
               markersize=7, label="hat (2-pt)", linewidth=0),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "fig10_sed_3way.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_fig11_sedimentation_snapshots(all_data):
    """Fig 11: 침강 시각화 — canonical 도메인 (2×6cm → W/d=8, L/d=24)."""
    fig_style()

    W_D = 8       # W / d
    L_D = 24      # 6cm / 0.25cm = 24
    y0_D = 16     # 초기 y=4cm → (4/0.25)=16D, 상단에서 = L_D - 16 = 8D 아래

    fig, axes = plt.subplots(1, 3, figsize=(8, 7), sharey=True)

    for ax_idx, (rho, run_id) in enumerate([(1.01, "A01"), (1.1, "A07"), (1.5, "A13")]):
        ax = axes[ax_idx]
        hist = all_data.get(run_id)
        if hist is None:
            continue

        # 벽면 해칭
        wall_t = 0.6
        hatch_kw = dict(facecolor="white", edgecolor="#aaaaaa",
                        linewidth=0.4, hatch="///", zorder=0)
        ax.add_patch(plt.Rectangle((-wall_t, -wall_t), wall_t,
                                    L_D + 2 * wall_t, **hatch_kw))
        ax.add_patch(plt.Rectangle((W_D, -wall_t), wall_t,
                                    L_D + 2 * wall_t, **hatch_kw))
        ax.add_patch(plt.Rectangle((0, L_D), W_D, wall_t, **hatch_kw))
        ax.add_patch(plt.Rectangle((0, -wall_t), W_D, wall_t, **hatch_kw))

        # 채널 경계
        ax.plot([0, 0, W_D, W_D], [0, L_D, L_D, 0], "k-", lw=0.9)
        ax.plot([0, W_D], [0, 0], "k-", lw=0.9)

        # 스냅샷 7개
        n_snap = 7
        indices = np.linspace(0, len(hist) - 1, n_snap, dtype=int)
        x_c = W_D / 2

        # 궤적선
        y_pts = [y0_D - hist[idx]["y_star"] for idx in indices]
        ax.plot([x_c] * len(y_pts), y_pts,
                color="#dddddd", lw=1.0, zorder=2)

        # 흑백 그라디언트
        grays = np.linspace(0.75, 0.10, n_snap)

        for i, idx in enumerate(indices):
            h = hist[idx]
            y_pos = y0_D - h["y_star"]
            gc = str(grays[i])

            ax.add_patch(plt.Circle(
                (x_c, y_pos), 0.5,
                facecolor=gc, edgecolor="black", linewidth=0.6, zorder=5,
            ))

            if i == 0:
                ax.annotate(
                    f"$t^*=0$", xy=(x_c + 0.6, y_pos),
                    xytext=(x_c + 1.8, y_pos),
                    fontsize=7.5, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888"),
                )
            elif i == n_snap - 1:
                ax.annotate(
                    f"$t^*={h['t_star']:.0f}$", xy=(x_c + 0.6, y_pos),
                    xytext=(x_c + 1.8, y_pos),
                    fontsize=7.5, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888"),
                )

        # 시간 진행 화살표
        ax.annotate(
            "", xy=(W_D + 0.8, y_pts[-1] + 0.5),
            xytext=(W_D + 0.8, y_pts[0] - 0.5),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#666666"),
        )
        ax.text(W_D + 1.3, (y_pts[0] + y_pts[-1]) / 2, "$t$",
                fontsize=9, color="#666666", ha="left", va="center")

        # 중력 화살표
        if ax_idx == 0:
            ax.annotate(
                "", xy=(-1.2, L_D * 0.35),
                xytext=(-1.2, L_D * 0.48),
                arrowprops=dict(arrowstyle="-|>", lw=1.2, fc="black"),
            )
            ax.text(-1.2, L_D * 0.50, "$\\mathbf{g}$", ha="center",
                    fontsize=10, fontweight="bold")

        ax.set_xlim(-1.8, W_D + 2.2)
        ax.set_ylim(-1.2, L_D + 1.0)
        ax.set_aspect("equal")
        ax.set_xlabel("$x/D$")
        if ax_idx == 0:
            ax.set_ylabel("$y/D$")
        ax.set_title(f"$\\rho_s/\\rho_f = {rho}$", pad=6)
        ax.text(0.04, 0.97, f"({chr(97 + ax_idx)})", transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top")

    fig.tight_layout()
    out = FIG_DIR / "fig11_sed_snapshots.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# =====================================================================
# Re_t 계산 + 문헌 비교
# =====================================================================

def compute_re_t(vy_star, rho_ratio):
    """vy* → Re_t 변환. Re_t = U_t × d / ν, U_t = vy* × u_g."""
    ref = compute_reference_scales(rho_ratio)
    u_t_phys = vy_star * ref["u_g"]  # cm/s
    return u_t_phys * D_PHYS / NU_PHYS


# =====================================================================
# 데이터 리포트 생성
# =====================================================================

def generate_report(all_data, terminal_data, statuses):
    """10_sedimentation_canonical.md 생성."""

    lines = []
    lines.append("# 침강 벤치마크 분석 — Canonical 도메인 (Sedimentation Validation)")
    lines.append("")
    lines.append(f"- 날짜: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("- 데이터: `data/sedimentation_canonical/`")
    lines.append("- 도메인: Glowinski (2001) canonical — (0,2)×(0,6) cm, 초기 위치 (1, 4) cm")
    lines.append("- 물리 파라미터: Feng et al. (1994) 표준 — W=2cm, d=0.25cm, W/d=8, ν=0.01cm²/s, g=981cm/s²")
    lines.append("- 경계 조건: 4벽 no-slip (Zou-He, settling_channel)")
    lines.append("- 충돌 모델: BGK/SRT")
    lines.append("")
    lines.append("> Reduced Pressure 방식: 유체 방정식에 중력 없음.")
    lines.append("> 입자에 순 중력 (ρ_s - ρ_f)·V·g 직접 적용. Velocity Verlet 시간 적분.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # § 1. 격자 파라미터
    lines.append("## 1. 격자 파라미터 요약")
    lines.append("")
    lines.append("Ma-constrained 자동 격자 선택 (safety_factor=4, Ma_max=0.08, τ_min=0.52).")
    lines.append("")
    lines.append("| ρ_s/ρ_f | Ar | NN | D_lat | τ | g_lat | max_steps |")
    lines.append("|:-------:|----:|----:|------:|-----:|------:|----------:|")

    for rho in RHOS:
        ref = compute_reference_scales(rho)
        # status에서 실제 값 가져오기
        for run_id, r, method, delta, subdir in EXPERIMENTS:
            if r == rho and method == "DF" and delta == "peskin4pt":
                st = statuses.get(run_id, {})
                cfg = st.get("config", {})
                nn = cfg.get("NN", "?")
                g_lat = cfg.get("gravity", 0)
                max_s = cfg.get("max_steps", "?")
                tau = cfg.get("tau", 0)
                break
        else:
            nn = g_lat = max_s = tau = "?"

        if isinstance(nn, int):
            d_lat = 0.125 * (nn - 1)
        else:
            d_lat = "?"

        if isinstance(g_lat, (int, float)):
            g_lat_str = f"{g_lat:.6e}"
        else:
            g_lat_str = "?"

        if isinstance(tau, (int, float)) and tau > 0:
            tau_str = f"{tau:.4f}"
        else:
            tau_str = "?"

        lines.append(f"| {rho} | {ref['Ar']:.0f} | {nn} | {d_lat:.0f} | {tau_str} | {g_lat_str} | {max_s} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # § 2. 종단 속도 3-way 비교
    lines.append("## 2. 종단 속도 3-Way 비교")
    lines.append("")
    lines.append("종단 속도: rolling std 최소 plateau 구간 평균.")
    lines.append("$v_y^* = |v_y| / u_g$, $u_g = \\sqrt{|\\Delta\\rho/\\rho_f| \\cdot g \\cdot d}$")
    lines.append("")
    lines.append("| ρ_s/ρ_f | Delta | DF | MDF | DFC | Δ%(DF→MDF) | Δ%(DF→DFC) | Max spread |")
    lines.append("|:-------:|-------|-----:|-----:|-----:|----------:|----------:|----------:|")

    for rho in RHOS:
        for delta in ["peskin4pt", "hat"]:
            vals = {}
            for m in ["DF", "MDF", "DFC"]:
                key = f"{m}_{delta}_{rho}"
                td = terminal_data.get(key)
                vals[m] = td["vy_star_mean"] if td else None

            if all(v is not None for v in vals.values()):
                df_v = vals["DF"]
                mdf_pct = (vals["MDF"] - df_v) / df_v * 100
                dfc_pct = (vals["DFC"] - df_v) / df_v * 100
                all_v = list(vals.values())
                spread = (max(all_v) - min(all_v)) / np.mean(all_v) * 100
                lines.append(
                    f"| {rho} | {delta} | {vals['DF']:.6f} | {vals['MDF']:.6f} | "
                    f"{vals['DFC']:.6f} | {mdf_pct:+.2f}% | {dfc_pct:+.2f}% | {spread:.2f}% |"
                )

    lines.append("")
    lines.append("---")
    lines.append("")

    # § 3. Re_t 문헌 비교 (핵심 테이블)
    lines.append("## 3. 종단 Reynolds 수 (Re_t) — 문헌 비교")
    lines.append("")
    lines.append("$Re_t = U_t \\times d / \\nu$, $U_t = v_y^* \\times u_g$.")
    lines.append("")

    for rho in RHOS:
        ref = compute_reference_scales(rho)
        lit = LITERATURE[rho]
        lines.append(f"### 3.{RHOS.index(rho)+1}. ρ_s/ρ_f = {rho} — 레짐 {lit['regime']}")
        lines.append("")

        # 문헌 값 목록
        lit_vals = []
        lit_labels = []
        for key, val in lit.items():
            if key == "regime" or val is None:
                continue
            lit_vals.append(val)
            lit_labels.append(key)

        lines.append("| Source | Re_t |")
        lines.append("|--------|-----:|")
        for label, val in zip(lit_labels, lit_vals):
            lines.append(f"| {label} | {val:.2f} |")

        # 본 연구 결과
        lines.append("| --- **본 연구 (BGK)** --- | --- |")
        for delta in ["peskin4pt", "hat"]:
            for m in ["DF", "MDF", "DFC"]:
                key = f"{m}_{delta}_{rho}"
                td = terminal_data.get(key)
                if td:
                    re_t = compute_re_t(td["vy_star_mean"], rho)
                    lines.append(f"| {m} ({delta[:3]}) | {re_t:.2f} |")

        # 평균 편차 계산
        if lit_vals:
            lit_mean = np.mean(lit_vals)
            our_vals = []
            for delta in ["peskin4pt", "hat"]:
                for m in ["DF", "MDF", "DFC"]:
                    key = f"{m}_{delta}_{rho}"
                    td = terminal_data.get(key)
                    if td:
                        our_vals.append(compute_re_t(td["vy_star_mean"], rho))
            if our_vals:
                our_mean = np.mean(our_vals)
                deviation = (our_mean - lit_mean) / lit_mean * 100
                lines.append(f"| **본 연구 평균** | **{our_mean:.2f}** |")
                lines.append(f"| **문헌 평균** | **{lit_mean:.2f}** |")
                lines.append(f"| **편차** | **{deviation:+.1f}%** |")

        lines.append("")

    lines.append("---")
    lines.append("")

    # § 4. Delta 함수 민감도
    lines.append("## 4. Delta 함수 민감도")
    lines.append("")
    lines.append("| ρ_s/ρ_f | Method | peskin4pt | hat | Δ%(P4→hat) |")
    lines.append("|:-------:|--------|----------:|----:|-----------:|")

    for rho in RHOS:
        for m in ["DF", "MDF", "DFC"]:
            p4_key = f"{m}_peskin4pt_{rho}"
            hat_key = f"{m}_hat_{rho}"
            p4 = terminal_data.get(p4_key)
            hat = terminal_data.get(hat_key)
            if p4 and hat:
                pct = (hat["vy_star_mean"] - p4["vy_star_mean"]) / p4["vy_star_mean"] * 100
                lines.append(
                    f"| {rho} | {m} | {p4['vy_star_mean']:.6f} | "
                    f"{hat['vy_star_mean']:.6f} | {pct:+.2f}% |"
                )

    lines.append("")
    lines.append("---")
    lines.append("")

    # § 5. 과도 역학
    lines.append("## 5. 과도 역학 (Transient Dynamics)")
    lines.append("")
    lines.append("| ρ_s/ρ_f | Delta | Method | t*_99 | vy_peak | overshoot(%) | t*_peak |")
    lines.append("|:-------:|-------|--------|------:|--------:|------------:|--------:|")

    for rho in RHOS:
        for delta in ["peskin4pt", "hat"]:
            for m in ["DF", "MDF", "DFC"]:
                key = f"{m}_{delta}_{rho}"
                td = terminal_data.get(key)
                if td and "t_star_99" in td:
                    lines.append(
                        f"| {rho} | {delta} | {m} | {td['t_star_99']:.2f} | "
                        f"{td['vy_peak']:.4f} | {td['overshoot_pct']:.2f} | {td['t_star_peak']:.2f} |"
                    )

    lines.append("")
    lines.append("---")
    lines.append("")

    # § 6. 횡방향 안정성
    lines.append("## 6. 횡방향 안정성 (Lateral Stability)")
    lines.append("")
    lines.append("| ρ_s/ρ_f | Delta | Method | |vx*|_max | x_drift |")
    lines.append("|:-------:|-------|--------|--------:|-----------:|")

    for rho in RHOS:
        for delta in ["peskin4pt", "hat"]:
            for m in ["DF", "MDF", "DFC"]:
                key = f"{m}_{delta}_{rho}"
                td = terminal_data.get(key)
                if td and "vx_max" in td:
                    lines.append(
                        f"| {rho} | {delta} | {m} | {td['vx_max']:.2e} | {td['x_drift']:.2e} |"
                    )

    lines.append("")
    lines.append("---")
    lines.append("")

    # § 7. 벽면 감속
    lines.append("## 7. 벽면 근접 감속")
    lines.append("")
    lines.append("| ρ_s/ρ_f | Delta | Method | 벽면 감속(%) |")
    lines.append("|:-------:|-------|--------|----------:|")

    for rho in RHOS:
        for delta in ["peskin4pt", "hat"]:
            for m in ["DF", "MDF", "DFC"]:
                key = f"{m}_{delta}_{rho}"
                td = terminal_data.get(key)
                if td and "wall_decel_pct" in td:
                    lines.append(f"| {rho} | {delta} | {m} | {td['wall_decel_pct']:.1f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # § 8. 핵심 발견 + BGK 한계 분석
    lines.append("## 8. 핵심 발견 및 BGK/SRT 한계 분석")
    lines.append("")

    # 8.1 방법론 비교 요약
    lines.append("### 8.1. 3-Way 방법론 비교 요약")
    lines.append("")
    for rho in RHOS:
        lines.append(f"**ρ = {rho}:**")
        for delta in ["peskin4pt", "hat"]:
            vals = {}
            for m in ["DF", "MDF", "DFC"]:
                td = terminal_data.get(f"{m}_{delta}_{rho}")
                if td:
                    vals[m] = td["vy_star_mean"]
            if len(vals) == 3:
                all_v = list(vals.values())
                spread = (max(all_v) - min(all_v)) / np.mean(all_v) * 100
                fastest = max(vals, key=vals.get)
                slowest = min(vals, key=vals.get)
                lines.append(f"- {delta}: spread={spread:.2f}%, 최고={fastest} ({vals[fastest]:.4f}), 최저={slowest} ({vals[slowest]:.4f})")
        lines.append("")

    # 8.2 Re gap 분석
    lines.append("### 8.2. Re_t 문헌 비교 + BGK 한계")
    lines.append("")
    for rho in RHOS:
        ref = compute_reference_scales(rho)
        lit = LITERATURE[rho]
        lit_vals = [v for k, v in lit.items() if k != "regime" and v is not None]
        lit_mean = np.mean(lit_vals) if lit_vals else 0

        our_vals = []
        for delta in ["peskin4pt", "hat"]:
            for m in ["DF", "MDF", "DFC"]:
                td = terminal_data.get(f"{m}_{delta}_{rho}")
                if td:
                    our_vals.append(compute_re_t(td["vy_star_mean"], rho))
        our_mean = np.mean(our_vals) if our_vals else 0

        if lit_mean > 0 and our_mean > 0:
            deviation = (our_mean - lit_mean) / lit_mean * 100
            lines.append(f"**ρ = {rho}:** 본 연구 Re_t = {our_mean:.2f}, 문헌 평균 = {lit_mean:.2f}, 편차 = {deviation:+.1f}%")
            if abs(deviation) > 10:
                lines.append(f"  → 유의미한 편차. BGK/SRT 충돌 연산자의 한계가 주요 원인으로 추정.")
            elif abs(deviation) > 5:
                lines.append(f"  → 중간 수준 편차. 2차 정확도 + IBM delta smearing 영향 가능.")
            else:
                lines.append(f"  → 양호한 일치. BGK 한계가 이 레짐에서는 두드러지지 않음.")
        else:
            lines.append(f"**ρ = {rho}:** 문헌값 부재로 비교 불가.")
        lines.append("")

    lines.append("### 8.3. BGK 한계의 3-tier 인과 구조")
    lines.append("")
    lines.append("1. **BGK/SRT 충돌 연산자**: 단일 이완 시간 → 높은 수치 점도 → wake 불안정 억제")
    lines.append("   - Re > 200에서 과도한 감쇠 → 대칭 유지 → 종단 속도 과소 예측")
    lines.append("   - TRT/MRT 전환으로 해결 가능 (Q1 확장 계획)")
    lines.append("2. **2차 공간 정확도**: LBM 고유 제약. 경계층 해상도 제한")
    lines.append("   - 격자 수렴으로 부분 완화 가능하나 근본 한계 존재")
    lines.append("3. **IBM delta smearing**: peskin 4-pt 지지 폭 = 4dx")
    lines.append("   - 경계 두께 ≈ 4dx → 유효 직경 변화")
    lines.append("   - hat (2-pt)과 비교하여 영향 정량화 완료")
    lines.append("")
    lines.append("---")
    lines.append("")

    # § 9. 생성된 그림
    lines.append("## 9. 생성된 그림")
    lines.append("")
    lines.append("| 그림 | 파일 | 내용 |")
    lines.append("|------|------|------|")
    lines.append("| Fig. 8 | `fig8_sed_velocity.png` | vy*(t*) — 종단 속도 시계열 3-way 비교 |")
    lines.append("| Fig. 9 | `fig9_sed_trajectory.png` | y*(t*) — 침강 궤적 3-way 비교 |")
    lines.append("| Fig. 10 | `fig10_sed_3way.png` | 종단 속도 dot plot (method × delta) |")
    lines.append("| Fig. 11 | `fig11_sed_snapshots.png` | 침강 시각화 — 입자 위치 시계열 스냅샷 |")
    lines.append("")

    return "\n".join(lines)


# =====================================================================
# 메인
# =====================================================================

def main():
    print("=" * 60)
    print("침강 벤치마크 정밀 분석 — Canonical 도메인 (2×6cm)")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1/4] 데이터 로딩...")
    all_data = {}
    statuses = {}
    missing = []

    for run_id, rho, method, delta, subdir in EXPERIMENTS:
        hist = load_history(subdir)
        st = load_status(subdir)
        if hist is None:
            missing.append(run_id)
            print(f"  {run_id}: history 없음")
        else:
            all_data[run_id] = hist
            print(f"  {run_id}: {len(hist)} records")
        if st:
            statuses[run_id] = st

    if not all_data:
        print("\n  오류: 분석할 데이터가 없습니다.")
        sys.exit(1)

    # 2. 종단 속도 추출
    print("\n[2/4] 종단 속도 추출...")
    terminal_data = {}
    for run_id, rho, method, delta, subdir in EXPERIMENTS:
        hist = all_data.get(run_id)
        if hist is None:
            continue
        td = extract_terminal_velocity(hist)
        if td:
            key = f"{method}_{delta}_{rho}"
            terminal_data[key] = td
            re_t = compute_re_t(td["vy_star_mean"], rho)
            print(f"  {run_id} ({key}): vy* = {td['vy_star_mean']:.6f}, Re_t = {re_t:.2f}")

    # 3. 논문 그림 생성
    print("\n[3/3] 논문 그림 생성...")
    plot_fig8_velocity(all_data)
    plot_fig9_trajectory(all_data)
    plot_fig10_3way_bar(terminal_data)

    # 요약 출력
    print("\n" + "=" * 60)
    print("Re_t 문헌 비교 요약")
    print("=" * 60)
    for rho in RHOS:
        lit = LITERATURE[rho]
        lit_vals = [v for k, v in lit.items() if k != "regime" and v is not None]
        lit_mean = np.mean(lit_vals) if lit_vals else 0

        our_vals = []
        for delta in ["peskin4pt", "hat"]:
            for m in ["DF", "MDF", "DFC"]:
                td = terminal_data.get(f"{m}_{delta}_{rho}")
                if td:
                    our_vals.append(compute_re_t(td["vy_star_mean"], rho))
        our_mean = np.mean(our_vals) if our_vals else 0

        if lit_mean > 0:
            dev = (our_mean - lit_mean) / lit_mean * 100
            print(f"  ρ={rho}: 본 연구 Re_t={our_mean:.2f}, 문헌={lit_mean:.2f}, 편차={dev:+.1f}%")
        else:
            print(f"  ρ={rho}: 본 연구 Re_t={our_mean:.2f}, 문헌값 부재")

    print("\n분석 완료!")


if __name__ == "__main__":
    main()
