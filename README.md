# Value Bet Model — Can ML Beat the Bookmakers?

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![XGBoost](https://img.shields.io/badge/XGBoost-3.0-orange)

A rigorous, iterative machine learning investigation into whether publicly available data can generate profitable betting signals on European football markets.

**Short answer: no.** After 5 model iterations, 4 feature enrichment strategies, and 25 seasons of out-of-sample testing across 10 leagues, the bookmaker closing line remains unbeatable with public data. This repository documents the complete scientific process — from false positive to confirmed null result.

---

## The Approach

Each version adds new information to the model while keeping the same rigorous walk-forward validation:

| Version | What changed | Features | Market |
|---------|-------------|----------|--------|
| **v1** | Baseline — XGBoost + isotonic calibration | 56 rolling stats | Under 2.5 (Div2) |
| **v2** | Fixed calibration (Platt), reduced features | 21 lean features | Under 2.5 (Div2) |
| **v3** | Added inter-bookmaker disagreement signals | 25 features | Under 2.5 (Div2) |
| **v4** | Added expected goals from Understat | 28 features | Draw (Div1) |
| **v5** | H2H history, fixture congestion, referee stats, odds spread + league filter | 53 features | Under 2.5 (E1+F2) |

The current codebase is the **v5 pipeline** incorporating all lessons learned.

---

## Result 1 — The AUC ceiling

The model's ability to discriminate between outcomes barely improves across iterations, and never reaches the profitable threshold (~0.58 AUC).

![AUC across versions](docs/01_auc_v2.png)

With an AUC ranging from 0.535 to 0.5601, the model cannot generate enough *separation* between value bets and non-value bets to overcome the 5–8% bookmaker margin. Adding market features (v3) gave the best single-step improvement (+0.019), xG (v4) added nothing, but the richer v5 feature set (H2H history, fixture congestion, referee stats, odds spread) pushed AUC to its highest point at 0.5601.

---

## Result 2 — Consistently negative ROI

Every version loses money. The trend improves slightly, but never crosses zero.

![ROI across versions](docs/02_roi_v2.png)

The improvement from v1 (−7.9%) to v5 (−3.2%) is mostly due to better calibration and smarter league selection — not because the model found a real edge. Restricting to E1 and F2 (the two leagues where the model's signal is most consistent) accounts for the final gain.

---

## Result 3 — The calibration paradox

The model is well-calibrated *globally* (left panel), but systematically overconfident *on the bets it selects* (right panel). This is the core issue.

![Calibration paradox](docs/03_calibration.png)

When the model predicts 55% probability and the bookie implies 48%, the *actual* frequency is ~48% — the bookie was right. The model's confidence comes from the noisy tail of its distribution, where it's least reliable.

---

## Result 4 — No monotonic edge

If a model has a genuine edge, higher-confidence bets should produce higher returns. Instead, ROI is flat negative regardless of the edge threshold — the signature of a model with no real predictive advantage.

![Edge vs ROI](docs/04_edge_vs_roi.png)

The green dashed line shows what we'd expect from a model with a real edge. The red bars show reality.

---

## Result 5 — Goals-based features dominate, market spread confirms the signal

In v5, `h_avg_goals_scored` ranks first and `odds_spread_under` (Max<2.5 / Avg<2.5) ranks second — confirming that both the statistical signal and the market's own uncertainty are informative, yet insufficient.

![Feature importance](docs/05_features_v2.png)

The odds spread measures how much sharp money has moved the under line relative to the average bookmaker. When this ratio is high, the market is signalling genuine uncertainty — a useful but not sufficient discriminator. All other top features are goal and shot-based rolling averages, consistent across all versions.

---

## Result 6 — No consistent league-level edge

Across the four Div2 leagues tested on the Under 2.5 market, only Ligue 2 shows a positive ROI (+1.4%) on 197 bets — statistically indistinguishable from noise (p-value > 0.5). Serie B (I2) and Segunda División (SP2) are strongly negative, dragging the overall result down. The final pipeline restricts betting to E1 and F2.

![League breakdown](docs/06_leagues_v2.png)

---

## Root Cause — Why v1 showed a false +3.8% ROI

The original model used **isotonic calibration** (`CalibratedClassifierCV(method='isotonic', cv='prefit')`) fitted on validation sets of ~300–500 matches. Isotonic calibration is non-parametric with as many parameters as unique predictions — it memorised the validation set's noise, creating systematic overconfidence that inflated the detected "edge."

Switching to **Platt calibration** (logistic sigmoid, only 2 parameters) eliminated the artefact and revealed the true ROI: **−8.7%**.

---

## Methodology

### Walk-Forward Validation

```
Season N-k → N-2      Season N-1      Season N
┌────────────────┐   ┌────────────┐  ┌────────────┐
│     TRAIN       │   │    VAL     │  │    TEST    │
│  (fit model)    │   │ (calibrate)│  │  (evaluate) │
└────────────────┘   └────────────┘  └────────────┘
```

- Model retrained from scratch at each fold — no information leakage
- Calibration fitted on validation set only (never on test)
- Statistical significance: t-test + bootstrap 95% CI on every backtest

### Model

- **XGBoost** (conservative: `max_depth=4`, `min_child_weight=8`) with **Platt calibration** (sigmoid)
- Features: rolling team stats (goals, shots, under-rate, variance), dynamic league rankings, shot accuracy xG proxy, head-to-head under rate, fixture congestion (days rest), referee under-rate history, bookmaker odds spread (Max/Avg), no-vig bookmaker probabilities — 53 features total
- Edge = `P(model) − P(no-vig bookie)`

### Data

| Source | Coverage |
|--------|----------|
| [football-data.co.uk](https://www.football-data.co.uk) | 25 seasons × 10 leagues — match results, shots, cards, odds from 6+ bookmakers |
| [Understat](https://understat.com) | 8 seasons × 5 leagues — expected goals (xG) per match |

---

## What Would Actually Beat the Market

| Strategy | Why it works | Why we can't backtest it |
|----------|-------------|--------------------------|
| **Bet opening lines** | Lines move 2–5% before closing | football-data.co.uk only has closing odds |
| **React to team news** | Injuries shift true probability | No historical real-time news data |
| **Exotic markets** | Corners/cards/props have wider margins + less modelling effort from bookmakers | Not available in historical datasets |
| **Multi-sport volume** | 1–2% edge × 50,000 bets/year | Requires infrastructure, not ML research |

---

## Project Structure

```
value-bet-model/
├── src/
│   ├── download.py             # Auto-download from football-data.co.uk (25 seasons × 10 leagues)
│   ├── load.py                 # Data loading & cleaning
│   ├── features.py             # Feature engineering (rolling stats, H2H, congestion, referee, odds spread)
│   ├── model.py                # XGBoost + walk-forward + Platt calibration (53 features)
│   ├── backtest.py             # ROI simulation, significance tests, edge optimisation
│   ├── main.py                 # Under 2.5 pipeline (E1 + F2)
│   ├── draw_pipeline.py        # Draw pipeline (Div1)
│   └── scrape_understat.py     # Selenium-based xG scraper
├── docs/
│   ├── generate_plots.py       # Regenerate all diagnostic plots
│   └── *.png                   # AUC, ROI, calibration, edge, feature importance, league breakdown
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Usage

```bash
pip install -r requirements.txt

# Download match data (25 seasons × 10 leagues)
python src/download.py --seasons 25

# Scrape xG data (requires Chrome + chromedriver)
python src/scrape_understat.py --seasons 2017 2025

# Run Under 2.5 pipeline
python src/main.py --edge 0.05

# Run Draw pipeline
python src/draw_pipeline.py --data-dir ./src/csv --edge 0.05

# Update current season only
python src/main.py --download --update
```

---

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Selenium` · `SciPy`

---

**Marc'Andria Peri** — CPES 3A (Paris-Saclay × HEC × IP Paris), Data Science track

*Data: [football-data.co.uk](https://www.football-data.co.uk) · [Understat](https://understat.com)*
