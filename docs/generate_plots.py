#!/usr/bin/env python3
"""Generate individual plots for the README — dark theme, one per file."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#C9D1D9",
    "text.color": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
    "grid.alpha": 0.8,
    "font.size": 12,
})

ACCENT, RED, GREEN, YELLOW, ORANGE, PURPLE = "#58A6FF", "#F85149", "#3FB950", "#D29922", "#F0883E", "#BC8CFF"
OUT = os.path.dirname(os.path.abspath(__file__))
versions = ["v1\nOriginal\n(56 feat.)", "v2\nPlatt calib.\n(21 feat.)", "v3\n+ Market\n(25 feat.)", "v4\n+ xG\n(28 feat.)", "v5\n+ H2H/Ref\n(53 feat.)"]

# 1. AUC
fig, ax = plt.subplots(figsize=(11, 5))
auc = [0.548, 0.535, 0.554, 0.535, 0.5601]
colors_auc = [RED, ACCENT, ACCENT, ACCENT, GREEN]
ax.bar(range(5), auc, color=colors_auc, edgecolor="#30363D", width=0.55, alpha=0.9)
ax.axhline(0.50, color="#8B949E", ls="--", lw=1, alpha=0.6)
ax.axhspan(0.58, 0.62, color=GREEN, alpha=0.07)
ax.text(4.4, 0.595, "Profitable zone (AUC >= 0.58)", fontsize=9, color=GREEN, alpha=0.6, ha="right")
ax.text(4.4, 0.503, "Random", fontsize=9, color="#8B949E", alpha=0.6, ha="right")
for i, v in enumerate(auc):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=13, fontweight="bold", color=colors_auc[i])
ax.set_xticks(range(5)); ax.set_xticklabels(versions, fontsize=10)
ax.set_ylim(0.48, 0.62); ax.set_ylabel("AUC (out-of-sample)")
ax.set_title("Discriminative Power Across 5 Iterations", fontsize=14, fontweight="bold", color="#F0F6FC", pad=15)
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/01_auc.png", dpi=150, facecolor="#0D1117", bbox_inches="tight"); plt.close()
print("01_auc.png")

# 2. ROI
fig, ax = plt.subplots(figsize=(11, 5))
roi = [-7.91, -8.68, -6.56, -4.49, -3.2]
ax.bar(range(5), roi, color=RED, edgecolor="#30363D", width=0.55, alpha=0.9)
ax.axhline(0, color=GREEN, ls="-", lw=1.5, alpha=0.5)
for i, v in enumerate(roi):
    ax.text(i, v - 0.4, f"{v:+.1f}%", ha="center", fontsize=13, fontweight="bold", color=RED)
ax.annotate("", xy=(4, -3.2), xytext=(0, -7.91), arrowprops=dict(arrowstyle="->", color=YELLOW, lw=2, ls="--"))
ax.text(2.0, -2.2, "Improving but never profitable", fontsize=10, color=YELLOW, ha="center", style="italic", alpha=0.8)
ax.set_xticks(range(5)); ax.set_xticklabels(versions, fontsize=10)
ax.set_ylabel("ROI % (flat stake)")
ax.set_title("Return on Investment — All Versions Negative", fontsize=14, fontweight="bold", color="#F0F6FC", pad=15)
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/02_roi.png", dpi=150, facecolor="#0D1117", bbox_inches="tight"); plt.close()
print("02_roi.png")

# 3. Calibration
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cp, ca, cn = [0.471, 0.553, 0.630], [0.468, 0.534, 0.621], [1037, 5124, 1005]
ax1.plot([0.35, 0.75], [0.35, 0.75], "w--", lw=1, alpha=0.3)
ax1.scatter(cp, ca, s=[n/max(cn)*500 for n in cn], color=GREEN, alpha=0.8, edgecolors="white", linewidth=0.8, zorder=5)
for p, a, n in zip(cp, ca, cn):
    ax1.annotate(f"n={n}\n\u0394={a-p:+.1%}", (p, a), textcoords="offset points", xytext=(12, -8), fontsize=9, color="#8B949E")
ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
ax1.set_title("Global Calibration  \u2714", fontsize=13, fontweight="bold", color=GREEN)
ax1.set_xlim(0.35, 0.75); ax1.set_ylim(0.35, 0.75); ax1.grid(alpha=0.3)
vp = [0.474, 0.528, 0.571, 0.650, 0.728]
va = [0.383, 0.421, 0.483, 0.535, 0.514]
vn = [282, 1595, 1378, 628, 70]
ax2.plot([0.35, 0.85], [0.35, 0.85], "w--", lw=1, alpha=0.3)
ax2.scatter(vp, va, s=[n/max(vn)*500 for n in vn], color=RED, alpha=0.8, edgecolors="white", linewidth=0.8, zorder=5)
ax2.set_xlabel("Predicted"); ax2.set_ylabel("Actual")
ax2.set_title("On Value Bets: Overconfidence  \u2718", fontsize=13, fontweight="bold", color=RED)
ax2.set_xlim(0.35, 0.85); ax2.set_ylim(0.35, 0.85); ax2.grid(alpha=0.3)
ax2.text(0.7, 0.42, "Predicts 70%\nReality: 51%", fontsize=10, color=RED, ha="center", style="italic", alpha=0.8)
plt.suptitle("The Calibration Paradox", fontsize=15, fontweight="bold", color="#F0F6FC", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/03_calibration.png", dpi=150, facecolor="#0D1117", bbox_inches="tight"); plt.close()
print("03_calibration.png")

# 4. Edge vs ROI
fig, ax = plt.subplots(figsize=(10, 5))
el = ["1%", "2%", "3%", "4%", "5%", "7%", "10%"]
er = [-6.2, -5.6, -6.6, -7.1, -7.9, -6.3, -6.0]
ax.bar(range(7), er, color=RED, edgecolor="#30363D", width=0.55, alpha=0.9)
ax.axhline(0, color=GREEN, ls="-", lw=1.5, alpha=0.5)
ax.plot(range(7), [-5, -3, -1, 1, 3, 5, 8], color=GREEN, ls="--", lw=1.5, alpha=0.4, marker="o", markersize=4)
ax.text(5.5, 6, "Expected if\nedge was real", fontsize=9, color=GREEN, alpha=0.5, ha="center")
ax.set_xticks(range(7)); ax.set_xticklabels(el)
ax.set_xlabel("Minimum edge threshold"); ax.set_ylabel("ROI (%)")
ax.set_title("Higher Confidence \u2260 Better Results", fontsize=14, fontweight="bold", color="#F0F6FC", pad=15)
ax.grid(axis="y", alpha=0.3)
ax.text(3, -9.5, "A real edge shows increasing ROI with higher thresholds.\nHere: flat negative regardless of confidence.", fontsize=10, color=YELLOW, alpha=0.8, ha="center", style="italic")
plt.tight_layout()
plt.savefig(f"{OUT}/04_edge_vs_roi.png", dpi=150, facecolor="#0D1117", bbox_inches="tight"); plt.close()
print("04_edge_vs_roi.png")

# 5. Feature importance
fig, ax = plt.subplots(figsize=(10, 5))
feats = ["h_avg_goals_scored", "odds_spread_under", "h_avg_shots", "delta_goals_scored", "combined_avg_goals", "combined_xg_proxy", "combined_low_score", "h_form_5", "combined_under_rate_5", "a_avg_low_score_rate"]
imp = [0.0276, 0.0232, 0.0215, 0.0201, 0.0201, 0.0197, 0.0197, 0.0196, 0.0196, 0.0196]
cf = [ORANGE if "odds_spread" in f else ACCENT for f in feats]
ax.barh(range(9, -1, -1), imp, color=cf, edgecolor="#30363D", height=0.6, alpha=0.9)
ax.set_yticks(range(9, -1, -1)); ax.set_yticklabels(feats, fontsize=10)
ax.set_xlabel("XGBoost Importance")
ax.set_title("Market Spread Signal Ranks 2nd Among All Features", fontsize=14, fontweight="bold", color="#F0F6FC", pad=15)
ax.grid(axis="x", alpha=0.3)
ax.legend(handles=[Patch(facecolor=ORANGE, edgecolor="#30363D", label="Market signal"), Patch(facecolor=ACCENT, edgecolor="#30363D", label="Statistical signal")], loc="lower right", fontsize=10, framealpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/05_features.png", dpi=150, facecolor="#0D1117", bbox_inches="tight"); plt.close()
print("05_features.png")

# 6. League breakdown
fig, ax = plt.subplots(figsize=(10, 4.5))
lg = ["Championship (E1)", "Ligue 2 (F2)", "Serie B (I2)", "Segunda (SP2)"]
rl = [-4.3, 1.4, -9.9, -19.9]
cl = [GREEN if r >= 0 else RED for r in rl]
ax.barh(range(4), rl, color=cl, edgecolor="#30363D", height=0.5, alpha=0.9)
ax.axvline(0, color="#8B949E", ls="-", lw=1)
for i, v in enumerate(rl):
    if v >= 0:
        ax.text(v + 0.5, i, f"{v:+.1f}%", va="center", fontsize=11, fontweight="bold", ha="left", color=cl[i])
    elif v < -8:
        ax.text(v + 0.5, i, f"{v:+.1f}%", va="center", fontsize=11, fontweight="bold", ha="left", color="#0D1117")
    else:
        ax.text(v - 0.5, i, f"{v:+.1f}%", va="center", fontsize=11, fontweight="bold", ha="right", color=cl[i])
ax.set_yticks(range(4)); ax.set_yticklabels(lg, fontsize=11)
ax.set_xlabel("ROI (%)"); ax.set_title("Under 2.5 Market ROI by League (v5)", fontsize=14, fontweight="bold", color="#F0F6FC", pad=15)
ax.grid(axis="x", alpha=0.3); plt.tight_layout()
plt.savefig(f"{OUT}/06_leagues_v2.png", dpi=150, facecolor="#0D1117", bbox_inches="tight"); plt.close()
print("06_leagues_v2.png")

print("\nAll plots done")
