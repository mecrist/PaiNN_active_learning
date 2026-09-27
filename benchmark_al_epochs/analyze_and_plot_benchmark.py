#!/usr/bin/env python3
"""
================================================================================
   BENCHMARK ANALYSIS & PUBLICATION PLOT: AL EPOCH VARIATION & CONVERGENCE
================================================================================
Parses JSON metrics from PaiNN and MACE replay runs with 1, 2, 3, 5, and 10 epochs
per cycle, plus final post-AL convergence sessions.
Generates publication-quality head-to-head figures (300 DPI PNG).
================================================================================
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

BENCHMARK_DIR = Path("/home/maria.crist/dft_mlip/sep_pax/benchmark_al_epochs")
OUT_DIR = BENCHMARK_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette matching figures_new
PURPLE = "#6A4C93"         # PaiNN
ORANGE = "#E85D04"         # MACE
TEAL = "#2A9D8F"           # Accent / Converged
RED = "#D90429"            # Overfitting / Outliers
LIGHT_BLUE = "#4EA8DE"     # Secondary
INK = "#14140F"            # Text & ticks
MUTED = "#8A887F"          # Subtle lines

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

def style_axes(ax, logy=False):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, direction="out", color=INK)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, color=MUTED)
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())

def panel_label(ax, letter):
    ax.text(
        -0.12, 1.05, letter, transform=ax.transAxes,
        fontsize=11.5, fontweight="bold", color=INK, va="bottom", ha="left",
    )

def col_title(ax, text):
    ax.set_title(text, fontsize=10.5, fontweight="medium", color=INK, pad=8)


def load_metrics(pattern):
    results = {}
    for d in sorted(BENCHMARK_DIR.glob(pattern)):
        m_file = d / "replay_metrics.json"
        if m_file.exists():
            with open(m_file, "r") as f:
                results[d.name] = json.load(f)
    return results


def main():
    painn_results = load_metrics("results_painn_*")
    mace_results = load_metrics("results_mace_*")

    print(f"Loaded {len(painn_results)} PaiNN benchmark runs, {len(mace_results)} MACE benchmark runs.")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    fig.subplots_adjust(hspace=0.35, wspace=0.30)

    # Panel A: Test Energy RMSE vs AL Cycle
    ax = axes[0, 0]
    style_axes(ax)
    panel_label(ax, "(a)")
    col_title(ax, "Test Energy RMSE Progression vs. AL Cycle")
    ax.set_xlabel("Active Learning Cycle (Unified Pool)")
    ax.set_ylabel("Energy RMSE (meV/atom)")

    epoch_styles = {
        2: ("-", 1.5, 0.7),
        3: ("--", 1.6, 0.8),
        5: ("-.", 1.8, 0.9),
        10: (":", 2.0, 1.0),
    }

    # Plot PaiNN curves
    for name, data in painn_results.items():
        ep = data["config"]["epochs_per_cycle"]
        ls, lw, alpha = epoch_styles.get(ep, ("-", 1.5, 0.8))
        cycles = [c["cycle"] for c in data["cycles"] if c.get("test_eval")]
        e_rmse = [c["test_eval"]["energy_rmse_mev_per_atom"] for c in data["cycles"] if c.get("test_eval")]
        if len(cycles) > 0:
            ax.plot(cycles, e_rmse, color=PURPLE, linestyle=ls, lw=lw, alpha=alpha, label=f"PaiNN ({ep} ep/cycle)")

    # Plot MACE curves
    for name, data in mace_results.items():
        ep = data["config"]["epochs_per_cycle"]
        ls, lw, alpha = epoch_styles.get(ep, ("-", 1.5, 0.8))
        cycles = [c["cycle"] for c in data["cycles"] if c.get("test_eval")]
        e_rmse = [c["test_eval"]["energy_rmse_mev_per_atom"] for c in data["cycles"] if c.get("test_eval")]
        if len(cycles) > 0:
            ax.plot(cycles, e_rmse, color=ORANGE, linestyle=ls, lw=lw, alpha=alpha, label=f"MACE ({ep} ep/cycle)")

    ax.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor=MUTED)

    # Panel B: Test Force RMSE vs AL Cycle
    ax = axes[0, 1]
    style_axes(ax)
    panel_label(ax, "(b)")
    col_title(ax, "Test Force RMSE Progression vs. AL Cycle")
    ax.set_xlabel("Active Learning Cycle (Unified Pool)")
    ax.set_ylabel("Force RMSE (meV/Å)")

    for name, data in painn_results.items():
        ep = data["config"]["epochs_per_cycle"]
        ls, lw, alpha = epoch_styles.get(ep, ("-", 1.5, 0.8))
        cycles = [c["cycle"] for c in data["cycles"] if c.get("test_eval")]
        f_rmse = [c["test_eval"]["force_rmse_mev_per_A"] for c in data["cycles"] if c.get("test_eval")]
        if len(cycles) > 0:
            ax.plot(cycles, f_rmse, color=PURPLE, linestyle=ls, lw=lw, alpha=alpha, label=f"PaiNN ({ep} ep)")

    for name, data in mace_results.items():
        ep = data["config"]["epochs_per_cycle"]
        ls, lw, alpha = epoch_styles.get(ep, ("-", 1.5, 0.8))
        cycles = [c["cycle"] for c in data["cycles"] if c.get("test_eval")]
        f_rmse = [c["test_eval"]["force_rmse_mev_per_A"] for c in data["cycles"] if c.get("test_eval")]
        if len(cycles) > 0:
            ax.plot(cycles, f_rmse, color=ORANGE, linestyle=ls, lw=lw, alpha=alpha, label=f"MACE ({ep} ep)")

    ax.legend(loc="upper right", fontsize=8.0, frameon=True, facecolor="white", edgecolor=MUTED)

    # Panel C: Final Post-AL Converged Energy RMSE Comparison
    ax = axes[1, 0]
    style_axes(ax)
    panel_label(ax, "(c)")
    col_title(ax, "Post-AL Final Converged Energy RMSE (50 Ep)")
    ax.set_ylabel("Final Converged Energy RMSE (meV/atom)")

    epochs_list = [2, 3, 5, 10]
    x = np.arange(len(epochs_list))
    width = 0.35

    painn_conv_e = []
    mace_conv_e = []

    for ep in epochs_list:
        p_val = np.nan
        for name, data in painn_results.items():
            if data["config"]["epochs_per_cycle"] == ep and data.get("final_convergence"):
                p_val = data["final_convergence"]["energy_rmse_mev_per_atom"]
        painn_conv_e.append(p_val)

        m_val = np.nan
        for name, data in mace_results.items():
            if data["config"]["epochs_per_cycle"] == ep and data.get("final_convergence"):
                m_val = data["final_convergence"]["energy_rmse_mev_per_atom"]
        mace_conv_e.append(m_val)

    b1 = ax.bar(x - width/2, [0 if np.isnan(v) else v for v in painn_conv_e], width, color=PURPLE, label="PaiNN Converged")
    b2 = ax.bar(x + width/2, [0 if np.isnan(v) else v for v in mace_conv_e], width, color=ORANGE, label="MACE Converged")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ep} ep/cyc" for ep in epochs_list])
    ax.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor="white", edgecolor=MUTED)

    # Panel D: Total Computational Overhead vs. Final Error
    ax = axes[1, 1]
    style_axes(ax)
    panel_label(ax, "(d)")
    col_title(ax, "Training Wall-Clock vs. Final Accuracy (Pareto Trade-off)")
    ax.set_xlabel("Total Training Wall-Clock Time (minutes)")
    ax.set_ylabel("Final Converged Energy RMSE (meV/atom)")

    for name, data in painn_results.items():
        ep = data["config"]["epochs_per_cycle"]
        if data.get("final_convergence"):
            t_total = sum(c["train_wallclock_sec"] for c in data["cycles"]) + data["final_convergence"]["conv_wallclock_sec"]
            e_final = data["final_convergence"]["energy_rmse_mev_per_atom"]
            ax.scatter(t_total / 60.0, e_final, color=PURPLE, s=80, marker="o")
            ax.annotate(f"PaiNN {ep}ep", xy=(t_total / 60.0, e_final), xytext=(5, 5), textcoords="offset points", fontsize=8)

    for name, data in mace_results.items():
        ep = data["config"]["epochs_per_cycle"]
        if data.get("final_convergence"):
            t_total = sum(c["train_wallclock_sec"] for c in data["cycles"]) + data["final_convergence"]["conv_wallclock_sec"]
            e_final = data["final_convergence"]["energy_rmse_mev_per_atom"]
            ax.scatter(t_total / 60.0, e_final, color=ORANGE, s=80, marker="s")
            ax.annotate(f"MACE {ep}ep", xy=(t_total / 60.0, e_final), xytext=(5, 5), textcoords="offset points", fontsize=8)

    out_png = OUT_DIR / "fig_benchmark_epoch_variation_and_convergence.png"
    fig.suptitle("Active Learning Replay Benchmark: Impact of Intermediate Epochs & Final Convergence", fontsize=12.0, fontweight="bold", color=INK, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved publication figure: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
