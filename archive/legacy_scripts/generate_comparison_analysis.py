#!/usr/bin/env python3
"""
================================================================================
   COMPREHENSIVE COMPARATIVE ANALYSIS: AIMS-PAX (MACE) vs. PAINN-AL
================================================================================
Generates publication-quality figures and detailed diagnostic tables comparing:
  1. Active learning progression, loss drops, and force/energy accuracy.
  2. MD simulation stability, thermal behavior, and trajectory explosion diagnostics.
  3. Uncertainty metrics, adaptive threshold dynamics, and trigger distributions.
  4. DFT reference calculation execution, failure root causes, and sampling statistics.
Aesthetics strictly follow dft_mlip/my_dataset/figures_new.
================================================================================
"""

import os
import re
import json
import shutil
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator

# Paths
BASE_DIR = Path("/home/maria.crist/dft_mlip/sep_pax")
MACE_AL_DIR = BASE_DIR / "al_5A_mace"
PAINN_AL_DIR = BASE_DIR / "al_5A_painn"
COMP_RES_DIR = BASE_DIR / "comparison_results"
OUT_FIG_DIR = BASE_DIR / "comparison_figures"
ARTIFACTS_DIR = Path("/home/maria.crist/.gemini/antigravity-cli/brain/254c8bf2-5ded-4d7c-82e2-20a87332976c")

OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Aesthetic palette matching dft_mlip/my_dataset/figures_new
PURPLE = "#6A4C93"         # PaiNN / Primary baseline
ORANGE = "#E85D04"         # MACE / Active Learning focus
PLUM = "#8E4585"           # Referenced / Model variants
LIGHT_BLUE = "#4EA8DE"     # Validation / Secondary curve
PURPLE_LIGHT = "#9D4EDD"   # Secondary purple
ORANGE_LIGHT = "#F48C06"   # Secondary orange
TEAL = "#2A9D8F"           # Stable reference / PET
RED = "#D90429"            # Unstable / Explosive failure
INK = "#14140F"            # Deep charcoal black for text
MUTED = "#8A887F"          # Subtle gray for secondary annotations

# Matplotlib styling parameters
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
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

def style_axes(ax, logy=False):
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, direction="out", color=INK)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color=MUTED)
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

def save_fig(fig, name):
    png = OUT_FIG_DIR / f"{name}.png"
    pdf = OUT_FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    # Also copy to artifacts dir for interactive embedding
    shutil.copyfile(png, ARTIFACTS_DIR / f"{name}.png")
    shutil.copyfile(pdf, ARTIFACTS_DIR / f"{name}.pdf")
    print(f"Saved: {png} and copied to {ARTIFACTS_DIR / f'{name}.png'}")
    plt.close(fig)


# ==============================================================================
# DATA EXTRACTION
# ==============================================================================

# 1. Parse MACE AL Job 162797 log
mace_log_file = MACE_AL_DIR / "mace_al_5A_162797.out"
mace_text = mace_log_file.read_text() if mace_log_file.exists() else ""

epochs_raw = re.findall(
    r"Epoch 0: head: Default, loss=([0-9\.]+), MAE_E_per_atom=\s*([0-9\.]+) meV, MAE_F=\s*([0-9\.]+) meV / A",
    mace_text,
)
mace_al_cycles = []
for i, ep in enumerate(epochs_raw):
    mace_al_cycles.append({
        "cycle": i + 1,
        "loss": float(ep[0]),
        "energy_mae": float(ep[1]),
        "force_mae": float(ep[2]),
    })

triggers_raw = re.findall(
    r"Uncertainty of point is beyond threshold ([0-9\.]+) at worker (\d+): ([0-9\.]+)",
    mace_text,
)
mace_triggers = []
for tr in triggers_raw:
    u_str = tr[2].rstrip(".")
    th_str = tr[0].rstrip(".")
    mace_triggers.append({
        "threshold": float(th_str) * 1000.0,  # convert eV/A to meV/A
        "worker": int(tr[1]),
        "uncertainty": float(u_str) * 1000.0,
    })

# 2. Parse 500-step MD benchmark summary
sum_path = COMP_RES_DIR / "mace_vs_painn_summary.json"
with open(sum_path, "r") as f:
    md_comp_data = json.load(f)

# 3. Parse PaiNN offline 500-epoch baseline log
painn_log_path = Path("/home/maria.crist/dft_mlip/my_dataset/mine/model_2_ep500/silica_Painn_model_finetuned/painn_ensemble/model_0/log.csv")
painn_epochs = []
painn_train_loss = []
painn_val_loss = []
painn_val_e_mae = []
painn_val_f_mae = []

if painn_log_path.exists():
    import csv
    with open(painn_log_path, "r") as f:
        reader = csv.DictReader(f)
        for ep, row in enumerate(reader):
            painn_epochs.append(ep + 1)
            painn_train_loss.append(float(row["Train loss"]))
            painn_val_loss.append(float(row["Validation loss"]))
            painn_val_e_mae.append(float(row["MAE_energy"]))
            painn_val_f_mae.append(float(row["MAE_energy_grad"]))


# ==============================================================================
# FIGURE 1: ACTIVE LEARNING LOSS, ACCURACY & THRESHOLD DYNAMICS (aims-PAX MACE)
# ==============================================================================
fig1, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(2, 2, figsize=(9.5, 7.5))
fig1.subplots_adjust(hspace=0.35, wspace=0.30)

# (a) Model Validation Loss across AL Retraining Cycles
cycles = [d["cycle"] for d in mace_al_cycles]
val_losses = [d["loss"] for d in mace_al_cycles]

ax1a.plot(cycles, val_losses, marker="o", markersize=5, color=ORANGE, linewidth=1.8, label="MACE Validation Loss")
ax1a.scatter([7], [val_losses[6]], color=TEAL, s=70, zorder=5, label=f"Best Model (Cycle 7: {val_losses[6]:.4f})")
ax1a.set_xlabel("Active Learning Retraining Cycle")
ax1a.set_ylabel("Validation Loss")
ax1a.set_xticks(cycles)
col_title(ax1a, "Online Retraining Loss Drop")
panel_label(ax1a, "(a)")
style_axes(ax1a)
ax1a.legend(loc="upper right")

# (b) Validation Energy & Force MAE across AL Cycles
ax1b_f = ax1b
ax1b_e = ax1b.twinx()

e_maes = [d["energy_mae"] for d in mace_al_cycles]
f_maes = [d["force_mae"] for d in mace_al_cycles]

line_f = ax1b_f.plot(cycles, f_maes, marker="s", markersize=4.5, color=PURPLE, linewidth=1.6, label="Force MAE (meV/Å)")
line_e = ax1b_e.plot(cycles, e_maes, marker="^", markersize=4.5, color=LIGHT_BLUE, linewidth=1.6, linestyle="--", label="Energy MAE (meV/atom)")

ax1b_f.set_xlabel("Active Learning Retraining Cycle")
ax1b_f.set_ylabel("Validation Force MAE (meV/Å)", color=PURPLE)
ax1b_e.set_ylabel("Validation Energy MAE (meV/atom)", color=LIGHT_BLUE)
ax1b_f.tick_params(axis="y", labelcolor=PURPLE)
ax1b_e.tick_params(axis="y", labelcolor=LIGHT_BLUE)
ax1b_f.set_xticks(cycles)

# Combine legends
lines = line_f + line_e
labels = [l.get_label() for l in lines]
ax1b_f.legend(lines, labels, loc="upper right")
col_title(ax1b, "Validation Accuracy Improvement")
panel_label(ax1b, "(b)")
style_axes(ax1b_f)
ax1b_e.spines["top"].set_visible(False)

# (c) Adaptive Threshold Tightening vs Triggered Uncertainty
tr_idx = np.arange(1, len(mace_triggers) + 1)
thresholds = [t["threshold"] for t in mace_triggers]
uncerts = [t["uncertainty"] for t in mace_triggers]

ax1c.plot(tr_idx, thresholds, marker="o", markersize=4.5, color=INK, linewidth=1.5, label=r"Adaptive Threshold $U_{\rm thresh}$")
ax1c.scatter(tr_idx, uncerts, color=ORANGE, marker="x", s=50, linewidths=1.8, label="Triggered Uncertainty")
ax1c.fill_between(tr_idx, thresholds, uncerts, color=ORANGE, alpha=0.15)
ax1c.set_xlabel("Acquisition Trigger Event")
ax1c.set_ylabel("Force Uncertainty (meV/Å)")
ax1c.set_xticks(tr_idx)
col_title(ax1c, "Dynamic Threshold Tightening")
panel_label(ax1c, "(c)")
style_axes(ax1c)
ax1c.legend(loc="upper right")

# (d) Dataset Accumulation (Training & Validation Sets)
# Initial: 490 train, 140 val. Final: 509 train (+19), 142 val (+2)
ax1d.bar([0], [490], color=MUTED, width=0.45, label="Initial Base Dataset (490)")
ax1d.bar([1], [490], color=MUTED, width=0.45)
ax1d.bar([1], [19], bottom=[490], color=ORANGE, width=0.45, label="New Train (+19 DFT queries)")
ax1d.bar([1], [2], bottom=[509], color=LIGHT_BLUE, width=0.45, label="New Val (+2 DFT queries)")
ax1d.set_xticks([0, 1])
ax1d.set_xticklabels(["Initial (Cycle 0)", "Post-AL (Cycle 12)"])
ax1d.set_ylabel("Dataset Size (Configurations)")
ax1d.set_ylim(0, 680)
ax1d.text(0, 505, "490 frames", ha="center", fontsize=8.5, fontweight="medium")
ax1d.text(1, 525, "511 frames\n(+21 DFT queries)", ha="center", fontsize=8.5, fontweight="bold", color=INK)
col_title(ax1d, "Dataset Expansion via DFT Queries")
panel_label(ax1d, "(d)")
style_axes(ax1d)
ax1d.legend(loc="upper left", fontsize=8.0)

save_fig(fig1, "fig1_active_learning_dynamics")


# ==============================================================================
# FIGURE 2: HEAD-TO-HEAD MD STABILITY & THERMAL TRAJECTORIES (500 STEPS)
# ==============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex=True)
fig2.subplots_adjust(hspace=0.25, wspace=0.28)

geom_keys = ["geometry_alpha_5", "geometry_amor_5", "geometry_beta_5"]
geom_titles = ["Alpha-Quartz (348 atoms)", "Amorphous Silica (336 atoms)", "Beta-Cristobalite (321 atoms)"]

for col_idx, (g_key, g_title) in enumerate(zip(geom_keys, geom_titles)):
    ax_top = axes2[0, col_idx]
    ax_bot = axes2[1, col_idx]

    p_hist = md_comp_data[g_key]["painn_history"]
    m_hist = md_comp_data[g_key]["mace_history"]

    # Extract time and metrics
    p_time = [h["time_fs"] for h in p_hist]
    p_temp = [h["temperature_K"] for h in p_hist]
    p_epot = [h["potential_energy_eV"] for h in p_hist]

    m_time = [h["time_fs"] for h in m_hist]
    m_temp = [h["temperature_K"] for h in m_hist]
    m_epot = [h["potential_energy_eV"] for h in m_hist]

    # TOP: Temperature
    ax_top.axhline(300, color=MUTED, linestyle=":", linewidth=1.0, label="Target T (300 K)")
    ax_top.plot(m_time, m_temp, color=ORANGE, linewidth=1.4, label="MACE (Stable)")

    # For PaiNN, handle NaN / explosion
    p_temp_clean = [t if not np.isnan(t) else None for t in p_temp]
    if g_key == "geometry_beta_5":
        valid_len = 23
        ax_top.plot(p_time[:valid_len], p_temp[:valid_len], color=RED, linewidth=1.5, label="PaiNN (Exploded)")
        ax_top.scatter([p_time[22]], [p_temp[22]], color=RED, marker="X", s=80, zorder=5, label="T = 2,217 K -> NaN")
        ax_top.annotate(
            "Explosion (2,217 K)\nNaN at t=115 fs",
            xy=(p_time[22], p_temp[22]),
            xytext=(p_time[22] - 80, 1600),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
            fontsize=7.5, color=RED, fontweight="bold",
        )
        ax_top.set_ylim(200, 2400)
    else:
        ax_top.plot(p_time, p_temp, color=PURPLE, linewidth=1.4, label="PaiNN (Stable)")
        ax_top.set_ylim(250, 700)

    col_title(ax_top, g_title)
    if col_idx == 0:
        ax_top.set_ylabel("Temperature (K)")
        ax_top.legend(loc="upper left")
    style_axes(ax_top)

    # BOTTOM: Potential Energy (relative shift in eV for clarity)
    m_epot_rel = np.array(m_epot) - m_epot[0]
    p_epot_rel = np.array(p_epot) - p_epot[0]

    ax_bot.plot(m_time, m_epot_rel, color=ORANGE, linewidth=1.4, label="MACE")
    if g_key == "geometry_beta_5":
        ax_bot.plot(p_time[:valid_len], p_epot_rel[:valid_len], color=RED, linewidth=1.5, label="PaiNN")
        ax_bot.scatter([p_time[22]], [p_epot_rel[22]], color=RED, marker="X", s=80, zorder=5)
        ax_bot.set_ylim(-30, 5)
    else:
        ax_bot.plot(p_time, p_epot_rel, color=PURPLE, linewidth=1.4, label="PaiNN")
        ax_bot.set_ylim(-30, 5)

    ax_bot.set_xlabel("MD Time (fs)")
    if col_idx == 0:
        ax_bot.set_ylabel(r"$\Delta E_{\rm pot}$ (eV)")
        ax_bot.legend(loc="lower left")
    style_axes(ax_bot)

panel_label(axes2[0, 0], "(a)")
panel_label(axes2[1, 0], "(b)")

save_fig(fig2, "fig2_md_stability_head_to_head")


# ==============================================================================
# FIGURE 3: UNCERTAINTY QUANTIFICATION, PEAKS & COMPUTATIONAL EFFICIENCY
# ==============================================================================
fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, figsize=(10, 7.5))
fig3.subplots_adjust(hspace=0.35, wspace=0.30)

# (a) Max Force Uncertainty Profile vs Time (Alpha-quartz & Amorphous)
p_hist_alpha = md_comp_data["geometry_alpha_5"]["painn_history"]
m_hist_alpha = md_comp_data["geometry_alpha_5"]["mace_history"]

t_fs = [h["time_fs"] for h in m_hist_alpha]
p_u_alpha = [h["max_uncertainty_meV_A"] for h in p_hist_alpha]
m_u_alpha = [h["max_uncertainty_meV_A"] for h in m_hist_alpha]

ax3a.plot(t_fs, p_u_alpha, color=PURPLE, linewidth=1.4, label="PaiNN (Alpha)")
ax3a.plot(t_fs, m_u_alpha, color=ORANGE, linewidth=1.4, label="MACE (Alpha)")
ax3a.axhline(25.0, color=MUTED, linestyle=":", linewidth=1.0, label="Nominal AL Thresh (25 meV/Å)")
ax3a.set_xlabel("MD Time (fs)")
ax3a.set_ylabel(r"Max Atomic Disagreement $\sigma_{\rm max}$ (meV/Å)")
col_title(ax3a, "Uncertainty Trajectory (Alpha-Quartz)")
panel_label(ax3a, "(a)")
style_axes(ax3a, logy=True)
ax3a.legend(loc="upper right")

# (b) Beta-Cristobalite Extreme Disagreement Divergence
p_hist_beta = md_comp_data["geometry_beta_5"]["painn_history"]
m_hist_beta = md_comp_data["geometry_beta_5"]["mace_history"]
p_u_beta = [h["max_uncertainty_meV_A"] for h in p_hist_beta[:23]]
m_u_beta = [h["max_uncertainty_meV_A"] for h in m_hist_beta]

ax3b.plot(t_fs, m_u_beta, color=ORANGE, linewidth=1.4, label="MACE (Stable, <1,700 meV/Å)")
ax3b.plot(t_fs[:23], p_u_beta, color=RED, linewidth=1.5, label="PaiNN (Divergent, >230,000 meV/Å)")
ax3b.scatter([t_fs[21]], [p_u_beta[21]], color=RED, marker="X", s=70, zorder=5)
ax3b.set_xlabel("MD Time (fs)")
ax3b.set_ylabel(r"Max Atomic Disagreement $\sigma_{\rm max}$ (meV/Å)")
col_title(ax3b, "Uncertainty Explosion (Beta-Cristobalite)")
panel_label(ax3b, "(b)")
style_axes(ax3b, logy=True)
ax3b.legend(loc="upper right")

# (c) Peak Uncertainty Comparison Across Systems (Bar Chart)
labels = ["Alpha-Quartz", "Amorphous", "Beta-Cristobalite"]
painn_peaks = [9183.13, 8390.58, 230928.38]
mace_peaks = [1475.83, 1365.41, 1688.77]

x = np.arange(len(labels))
width = 0.35

ax3c.bar(x - width/2, mace_peaks, width, color=ORANGE, label="MACE Peak")
ax3c.bar(x + width/2, painn_peaks, width, color=PURPLE, label="PaiNN Peak")
ax3c.set_xticks(x)
ax3c.set_xticklabels(labels)
ax3c.set_ylabel("Peak Uncertainty (meV/Å)")
col_title(ax3c, "Peak Force Disagreement Comparison")
panel_label(ax3c, "(c)")
style_axes(ax3c, logy=True)
ax3c.legend(loc="upper left")

# (d) Computational Speed (ms/step)
painn_speeds = [315.90, 314.27, 1755.47]
mace_speeds = [582.90, 529.70, 455.24]

ax3d.bar(x - width/2, mace_speeds, width, color=ORANGE, label="MACE Speed")
ax3d.bar(x + width/2, painn_speeds, width, color=PURPLE, label="PaiNN Speed")
# Add explosion annotation on beta
ax3d.text(2 + width/2, 1800, "NaN / blowup", ha="center", fontsize=7.5, color=RED, fontweight="bold")
ax3d.set_xticks(x)
ax3d.set_xticklabels(labels)
ax3d.set_ylabel("Inference Speed (ms / step)")
col_title(ax3d, "Computational Throughput (1 GPU)")
panel_label(ax3d, "(d)")
style_axes(ax3d)
ax3d.legend(loc="upper left")

save_fig(fig3, "fig3_uncertainty_and_efficiency")


# ==============================================================================
# FIGURE 4: PRE-AL MODEL BASELINES & MULTI-MODEL ACCURACY BENCHMARK
# ==============================================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 4.2))
fig4.subplots_adjust(wspace=0.32)

# (a) PaiNN Offline 500-Epoch Fine-Tuning Loss Curve (Mine Dataset)
if painn_epochs:
    ax4a.plot(painn_epochs, painn_train_loss, color=PURPLE, linewidth=1.2, label="Train Loss")
    ax4a.plot(painn_epochs, painn_val_loss, color=LIGHT_BLUE, linewidth=1.2, linestyle="--", label="Validation Loss")
    ax4a.scatter([263], [painn_val_loss[262]], color=TEAL, marker="o", s=40, zorder=5, label="Best Checkpoint (Ep 263)")
    ax4a.annotate(
        "Early Stopping\n(Ep 263)",
        xy=(263, painn_val_loss[262]),
        xytext=(180, 120),
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.0),
        fontsize=7.5, color=TEAL, fontweight="bold",
    )
    col_title(ax4a, "PaiNN Pre-AL Fine-Tuning (500 Epochs)")
    ax4a.set_xlabel("Training Epoch")
    ax4a.set_ylabel("Loss")
    panel_label(ax4a, "(a)")
    style_axes(ax4a, logy=True)
    ax4a.legend(loc="upper right")

# (b) Test Set Error Comparison Across MLIP Architectures
# Data from test_performance_table.txt on mine 70-structure test set
models = ["MACE\n(mine ep500)", "PET\n(mine ep500)", "PaiNN\n(mine ep500)"]
e_rmse = [26.30, 6.94, 2.71]       # meV / atom
f_rmse = [317.60, 323.51, 318.20]  # meV / A

x = np.arange(len(models))
width = 0.35

ax4b_f = ax4b
ax4b_e = ax4b.twinx()

rects1 = ax4b_f.bar(x - width/2, f_rmse, width, color=ORANGE, label="Force RMSE (meV/Å)")
rects2 = ax4b_e.bar(x + width/2, e_rmse, width, color=PURPLE, label="Energy RMSE (meV/at)")

ax4b_f.set_xticks(x)
ax4b_f.set_xticklabels(models)
ax4b_f.set_ylabel("Force RMSE (meV/Å)", color=ORANGE)
ax4b_e.set_ylabel("Energy RMSE (meV/atom)", color=PURPLE)
ax4b_f.tick_params(axis="y", labelcolor=ORANGE)
ax4b_e.tick_params(axis="y", labelcolor=PURPLE)
ax4b_f.set_ylim(250, 350)
ax4b_e.set_ylim(0, 32)

col_title(ax4b, "MLIP Pre-AL Test Set Errors (Mine Data)")
panel_label(ax4b, "(b)")
style_axes(ax4b_f)
ax4b_e.spines["top"].set_visible(False)

save_fig(fig4, "fig4_pre_al_baselines_accuracy")

print("\nAll figures generated successfully matching figures_new aesthetic!")
