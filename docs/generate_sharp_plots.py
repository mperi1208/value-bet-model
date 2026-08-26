#!/usr/bin/env python3
"""Generate the sharp-anchor strategy plots for the README — dark theme.

Inputs
------
  src/value_bets_sharp.csv   portfolio at EV > 0.02 (produced by
                             `python src/value_bet_sharp.py --kelly`)
  --all-ev                   optional export at a permissive threshold
                             (`--ev -0.10`) used for the EV-gradient plot
  src/csv/                   raw football-data files, for the per-book plot

Usage
-----
  python src/value_bet_sharp.py --kelly --out src/value_bets_sharp.csv
  python src/value_bet_sharp.py --ev -0.10 --out /tmp/all_ev.csv
  python docs/generate_sharp_plots.py --all-ev /tmp/all_ev.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

ACCENT, RED, GREEN, YELLOW, ORANGE, PURPLE = (
    "#58A6FF", "#F85149", "#3FB950", "#D29922", "#F0883E", "#BC8CFF")
GREY = "#8B949E"
OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

TITLE_KW = dict(fontsize=14, fontweight="bold", color="#F0F6FC", pad=15)


def boot_ci(profit, n_boot=3000, seed=0):
    """Bootstrap 95% CI of the mean ROI, in percent."""
    p = np.asarray(profit, dtype=float)
    if len(p) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(p, size=(n_boot, len(p)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5) * 100, np.percentile(means, 97.5) * 100


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=150, facecolor="#0D1117",
                bbox_inches="tight")
    plt.close(fig)
    print(name)


# ═══════════════════════════════════════════════════════════════
# 07 — EV gradient (the placebo test)
# ═══════════════════════════════════════════════════════════════

def plot_ev_gradient(all_ev_path):
    d = pd.read_csv(all_ev_path)
    edges = [-0.10, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08, 1.0]
    labels = ["-10 to -6%", "-6 to -4%", "-4 to -2%", "-2 to 0%",
              "0 to +2%", "+2 to +4%", "+4 to +8%", "> +8%"]
    d["bucket"] = pd.cut(d["ev_pre"], bins=edges, labels=labels,
                         right=False, include_lowest=True)
    g = d.groupby("bucket", observed=True)["profit"].agg(["mean", "count"])
    g = g.reindex(labels).dropna()
    roi = g["mean"] * 100
    baseline = d["profit"].mean() * 100

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [RED if v < 0 else GREEN for v in roi]
    ax.bar(range(len(g)), roi, color=colors, edgecolor="#30363D",
           width=0.6, alpha=0.9)
    ax.axhline(0, color=GREY, lw=1.2, alpha=0.7)
    ax.axhline(baseline, color=YELLOW, ls="--", lw=1.3, alpha=0.8)
    ax.text(len(g) - 0.4, baseline - 0.55,
            f"All Max-odds bets: {baseline:+.1f}%", fontsize=9, color=YELLOW,
            ha="right", alpha=0.85, style="italic")

    lo, hi = roi.min() - 3.2, roi.max() + 1.8
    ax.set_ylim(lo, hi)
    for i, (v, n) in enumerate(zip(roi, g["count"])):
        off = 0.35 if v >= 0 else -0.95
        ax.text(i, v + off, f"{v:+.1f}%", ha="center", fontsize=12,
                fontweight="bold", color=colors[i])
        ax.text(i, lo + 0.5, f"n={int(n):,}".replace(",", " "),
                ha="center", fontsize=8, color=GREY, alpha=0.8)

    ax.axvspan(4.5, len(g) - 0.4, color=GREEN, alpha=0.06)
    ax.text(5.5, hi - 1.0, "strategy zone (EV > +2%)", fontsize=9.5,
            color=GREEN, alpha=0.8, ha="center")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Pre-match EV vs power-devigged Pinnacle")
    ax.set_ylabel("ROI % (flat stake)")
    ax.set_title("Placebo Test — ROI Rises Monotonically With Estimated Edge",
                 **TITLE_KW)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "07_sharp_ev_gradient.png")


# ═══════════════════════════════════════════════════════════════
# 08 — ROI by league, with bootstrap CI
# ═══════════════════════════════════════════════════════════════

NAMES = {"E0": "Premier League", "E1": "Championship", "D1": "Bundesliga",
         "D2": "2. Bundesliga", "F1": "Ligue 1", "F2": "Ligue 2",
         "I1": "Serie A", "I2": "Serie B", "SP1": "La Liga",
         "SP2": "Segunda"}


def plot_leagues(vb):
    rows = []
    for div, g in vb.groupby("Div"):
        lo, hi = boot_ci(g["profit"])
        rows.append((div, g["profit"].mean() * 100, lo, hi, len(g)))
    r = pd.DataFrame(rows, columns=["Div", "roi", "lo", "hi", "n"])
    r = r.sort_values("roi").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [RED if v < 0 else GREEN for v in r["roi"]]
    y = range(len(r))
    ax.barh(y, r["roi"], color=colors, edgecolor="#30363D", height=0.6,
            alpha=0.9)
    ax.errorbar(r["roi"], y, xerr=[r["roi"] - r["lo"], r["hi"] - r["roi"]],
                fmt="none", ecolor="#C9D1D9", elinewidth=1.3, capsize=4,
                alpha=0.75)
    ax.axvline(0, color=GREY, lw=1.2, alpha=0.7)
    port = vb["profit"].mean() * 100
    ax.axvline(port, color=ACCENT, ls="--", lw=1.4, alpha=0.85)
    ax.text(port + 0.3, -0.85, f"portfolio {port:+.2f}%", fontsize=9,
            color=ACCENT, alpha=0.9)

    for i, row in r.iterrows():
        ax.text(row["hi"] + 0.6, i, f"{row['roi']:+.1f}%  (n={int(row['n']):,})"
                .replace(",", " "), va="center", fontsize=9.5,
                color=colors[i], fontweight="bold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{d}  {NAMES.get(d, '')}" for d in r["Div"]],
                       fontsize=10)
    ax.set_xlabel("ROI % (flat stake, EV > 2%) — bars show bootstrap 95% CI")
    ax.set_xlim(r["lo"].min() - 2, r["hi"].max() + 7)
    ax.set_title("Positive in 9 of 10 Leagues — No Single League Carries It",
                 **TITLE_KW)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "08_sharp_leagues.png")


# ═══════════════════════════════════════════════════════════════
# 09 — CLV by season: the real evidence
# ═══════════════════════════════════════════════════════════════

def plot_clv(vb):
    d = vb.dropna(subset=["clv_ev"])
    g = d.groupby("season_year")["clv_ev"].agg(
        mean="mean", pos=lambda s: (s > 0).mean() * 100, n="count")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.bar(g.index, g["mean"] * 100, color=GREEN, edgecolor="#30363D",
           width=0.62, alpha=0.9)
    ax.axhline(0, color=GREY, lw=1.2, alpha=0.7)
    overall = d["clv_ev"].mean() * 100
    ax.axhline(overall, color=ACCENT, ls="--", lw=1.4, alpha=0.85)
    ax.text(g.index[-1] + 0.35, overall, f" mean\n {overall:+.2f}%",
            fontsize=9, color=ACCENT, va="center")
    for x, v in zip(g.index, g["mean"] * 100):
        ax.text(x, v + 0.09, f"{v:+.1f}", ha="center", fontsize=9,
                color=GREEN, fontweight="bold")
    ax.set_ylabel("Mean CLV % vs Pinnacle close")
    ax.set_ylim(0, (g["mean"] * 100).max() * 1.28)
    ax.set_title("Closing Line Value — Positive in Every Single Season",
                 **TITLE_KW)
    ax.grid(axis="y", alpha=0.3)

    ax2.plot(g.index, g["pos"], color=PURPLE, marker="o", lw=2, ms=5)
    ax2.axhline(50, color=RED, ls="--", lw=1.2, alpha=0.7)
    ax2.text(g.index[0], 51, "coin flip", fontsize=8.5, color=RED, alpha=0.8)
    ax2.set_ylim(45, 80)
    ax2.set_ylabel("% of bets\nbeating the close", fontsize=10)
    ax2.set_xlabel("Season")
    ax2.set_xticks(list(g.index))
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "09_sharp_clv.png")


# ═══════════════════════════════════════════════════════════════
# 10 — Equity curve and drawdown (flat stake, 1 unit)
# ═══════════════════════════════════════════════════════════════

def plot_equity(vb):
    d = vb.sort_values("Date").reset_index(drop=True)
    eq = d["profit"].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    x = pd.to_datetime(d["Date"])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(x, eq, color=GREEN, lw=1.6)
    ax.fill_between(x, 0, eq, color=GREEN, alpha=0.10)
    ax.axhline(0, color=GREY, lw=1.2, alpha=0.7)
    ax.set_ylabel("Cumulative profit (units)")
    ax.set_title("Equity Curve — 20 676 Bets, 1 Unit Flat Stake", **TITLE_KW)
    ax.grid(alpha=0.3)
    ax.text(x.iloc[len(x) // 22], eq.max() * 0.92,
            f"final {eq.iloc[-1]:+,.0f} units".replace(",", " ") +
            f"   |   ROI {d['profit'].mean() * 100:+.2f}%",
            fontsize=10.5, color=GREEN, fontweight="bold")

    ax2.fill_between(x, dd, 0, color=RED, alpha=0.55)
    ax2.set_ylabel("Drawdown\n(units)", fontsize=10)
    ax2.set_xlabel("Season")
    ax2.grid(alpha=0.3)
    ax2.text(x.iloc[len(x) // 22], dd.min() * 0.82,
             f"max drawdown {dd.min():,.0f} units".replace(",", " "),
             fontsize=9.5, color=RED)
    fig.tight_layout()
    save(fig, "10_sharp_equity.png")


# ═══════════════════════════════════════════════════════════════
# 11 — How many bookmaker accounts does the edge need?
# ═══════════════════════════════════════════════════════════════

BOOKS = [("Max", "Best of panel"), ("B365", "Bet365"), ("WH", "William Hill"),
         ("VC", "BetVictor"), ("LB", "Ladbrokes"), ("BW", "bwin"),
         ("IW", "Interwetten")]


def plot_books(csv_root, ev_min=0.02):
    import value_bet_sharp as vbs
    extra = [f"{b}{s}" for b, _ in BOOKS for s in ("H", "D", "A")]
    vbs.ODDS_COLS = list(dict.fromkeys(vbs.ODDS_COLS + extra))
    od = vbs.load_odds(csv_root)
    od = od.dropna(subset=["PSH", "PSD", "PSA"])
    fair = vbs.novig_power([od["PSH"], od["PSD"], od["PSA"]])
    has_c = od[["PSCH", "PSCD", "PSCA"]].notna().all(axis=1).values
    fair_c = np.full_like(fair, np.nan)
    if has_c.any():
        fair_c[np.where(has_c)[0]] = vbs.novig_power(
            [od.loc[has_c, "PSCH"], od.loc[has_c, "PSCD"],
             od.loc[has_c, "PSCA"]])

    rows = []
    for prefix, label in BOOKS:
        cols = [f"{prefix}{s}" for s in ("H", "D", "A")]
        if not all(c in od.columns for c in cols):
            continue
        prof, clv = [], []
        for j, side in enumerate(("H", "D", "A")):
            o = od[cols[j]].values
            ev = fair[:, j] * o - 1
            m = np.isfinite(o) & (ev > ev_min)
            if not m.any():
                continue
            prof.append(np.where(od["FTR"].values[m] == side, o[m] - 1, -1.0))
            cm = m & has_c
            if cm.any():
                clv.append(fair_c[cm, j] * od[cols[j]].values[cm] - 1)
        if not prof:
            continue
        p = np.concatenate(prof)
        c = np.concatenate(clv) if clv else np.array([np.nan])
        rows.append((label, p.mean() * 100, len(p), np.nanmean(c) * 100,
                     p.sum()))

    r = pd.DataFrame(rows, columns=["book", "roi", "n", "clv", "units"])
    r = r.sort_values("roi", ascending=False).reset_index(drop=True)

    def cols_for(vals):
        return [ACCENT if b == "Best of panel" else (GREEN if v > 0 else RED)
                for b, v in zip(r["book"], vals)]

    panels = [
        ("roi", "ROI % per bet (1X2, EV > 2%)", "Per-Bet Edge", "{:+.1f}%"),
        ("clv", "CLV % vs Pinnacle close", "Closing Line Value", "{:+.1f}%"),
        ("units", "Total profit (units, 1 per bet)", "Total Profit Generated",
         "{:+,.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, (col, ylab, sub, fmt) in zip(axes, panels):
        c = cols_for(r[col])
        ax.bar(range(len(r)), r[col], color=c, edgecolor="#30363D",
               width=0.62, alpha=0.9)
        ax.axhline(0, color=GREY, lw=1.2, alpha=0.7)
        span = r[col].max() - min(r[col].min(), 0)
        ax.set_ylim(min(r[col].min(), 0) - span * 0.22,
                    r[col].max() + span * 0.16)
        for i, v in enumerate(r[col]):
            off = span * 0.03 if v >= 0 else -span * 0.09
            ax.text(i, v + off, fmt.format(v).replace(",", " "), ha="center",
                    fontsize=10.5, fontweight="bold", color=c[i])
        if col == "roi":
            for i, n in enumerate(r["n"]):
                ax.text(i, ax.get_ylim()[0] + span * 0.03,
                        f"n={int(n):,}".replace(",", " "), ha="center",
                        fontsize=8, color=GREY, alpha=0.85)
        ax.set_xticks(range(len(r)))
        ax.set_xticklabels(r["book"], rotation=28, ha="right", fontsize=9)
        ax.set_ylabel(ylab, fontsize=10.5)
        ax.set_title(sub, fontsize=12, fontweight="bold", color="#F0F6FC",
                     pad=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("One Account or the Whole Panel? Per-Bet Edge Is Not "
                 "Where the Money Is", fontsize=14.5, fontweight="bold",
                 color="#F0F6FC", y=1.02)
    fig.tight_layout()
    save(fig, "11_sharp_books.png")


# ═══════════════════════════════════════════════════════════════
# 12 — The window is closing
# ═══════════════════════════════════════════════════════════════

def plot_decay(vb):
    """1X2 only: it is the sole market covered across all 13 seasons.

    Pooling markets here would be misleading — O/U and AH odds only enter
    the football-data files in 2019, so the portfolio's raw bet count jumps
    for a coverage reason, not a market one.
    """
    x = vb[vb["market"] == "1X2"]
    g = x.groupby("season_year")["profit"].agg(n="count", roi="mean")
    c = x.dropna(subset=["clv_ev"]).groupby("season_year")["clv_ev"].mean()

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [1.5, 1]})
    ax.bar(g.index, g["n"], color=ACCENT, edgecolor="#30363D", width=0.62,
           alpha=0.6)
    e0, e1 = g["n"].loc[2012:2014].mean(), g["n"].loc[2022:2024].mean()
    ax.annotate("", xy=(2023.4, e1), xytext=(2012.6, e0),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2.2,
                                ls="--", alpha=0.95))
    ax.text(2018, e0 * 1.06, f"{e0:.0f} -> {e1:.0f} per season", fontsize=10.5,
            color=ORANGE, ha="center", style="italic", fontweight="bold")
    ax.set_ylabel("Mispricings detected\n(1X2, EV > 2%)")
    ax.set_ylim(0, g["n"].max() * 1.12)
    ax.set_title("Half as Many Mispricings — But the Ones Left Are Just as "
                 "Good", **TITLE_KW)
    ax.grid(axis="y", alpha=0.25)

    ax2.plot(g.index, g["roi"] * 100, color=GREEN, marker="o", lw=2, ms=5,
             label="ROI % (noisy: 95% CI spans ~13 pts on a single season)")
    ax2.plot(c.index, c * 100, color=PURPLE, marker="s", lw=2, ms=4.5,
             label="CLV % vs Pinnacle close (stable)")
    ax2.axhline(0, color=GREY, ls=":", lw=1.2, alpha=0.7)
    ax2.set_ylabel("%")
    ax2.set_xlabel("Season")
    ax2.set_xticks(list(g.index))
    ax2.set_ylim((g["roi"] * 100).min() - 4.5, (g["roi"] * 100).max() + 1.5)
    ax2.legend(loc="lower center", ncol=2, fontsize=8.5, facecolor="#161B22",
               edgecolor="#30363D", labelcolor="#C9D1D9")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "12_sharp_decay.png")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bets", default=os.path.join(SRC, "value_bets_sharp.csv"))
    ap.add_argument("--all-ev", default=None,
                    help="export at a permissive EV threshold, for plot 07")
    ap.add_argument("--csv", default=os.path.join(SRC, "csv"),
                    help="raw football-data root, for plot 11")
    a = ap.parse_args()

    vb = pd.read_csv(a.bets)
    vb["Date"] = pd.to_datetime(vb["Date"])

    if a.all_ev:
        plot_ev_gradient(a.all_ev)
    else:
        print("07 skipped (pass --all-ev)")
    plot_leagues(vb)
    plot_clv(vb)
    plot_equity(vb)
    if os.path.isdir(a.csv):
        plot_books(a.csv)
    else:
        print("11 skipped (raw csv/ not found)")
    plot_decay(vb)
