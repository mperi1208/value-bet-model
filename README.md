# Value Bet Model — Can ML Beat the Bookmakers?

A rigorous, iterative machine learning investigation into whether publicly available data can generate profitable betting signals on European football markets.

**Short answer: no.** After 4 model iterations, 3 feature enrichment strategies, and 25 seasons of out-of-sample testing across 10 leagues, the bookmaker closing line remains unbeatable with public data. This repository documents the complete scientific process — from false positive to confirmed null result.

---

## The Approach

Each version adds new information to the model while keeping the same rigorous walk-forward validation:

| Version | What changed | Features | Market |
|---------|-------------|----------|--------|
| **v1** | Baseline — XGBoost + isotonic calibration | 56 rolling stats | Under 2.5 (Div2) |
| **v2** | Fixed calibration (Platt), reduced features | 21 lean features | Under 2.5 (Div2) |
| **v3** | Added inter-bookmaker disagreement signals | 25 features | Under 2.5 (Div2) |
| **v4** | Added expected goals from Understat | 28 features | Draw (Div1) |

The current codebase is the **final refactored pipeline** incorporating all lessons learned.

---

## Result 1 — The AUC ceiling

The model's ability to discriminate between outcomes barely improves across iterations, and never reaches the profitable threshold (~0.58 AUC).

![AUC across versions](docs/01_auc.png)

With an AUC stuck around 0.535–0.554, the model cannot generate enough *separation* between value bets and non-value bets to overcome the 5–8% bookmaker margin. Adding market features (v3) gave the best improvement (+0.02), but xG (v4) added nothing — the information was already captured by goals scored/conceded.

---

## Result 2 — Consistently negative ROI

Every version loses money. The trend improves slightly, but never crosses zero.

![ROI across versions](docs/02_roi.png)

The improvement from v1 (−7.9%) to v4 (−4.5%) is mostly due to better calibration reducing the number of false value bets, not because the model found a real edge.

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

## Result 5 — Market features dominate (v3)

In v3, inter-bookmaker disagreement (`mkt_dispersion`, `max_avg_ratio`) ranked above all statistical features. The market's own uncertainty signal is more informative than rolling averages, rankings, or xG.

![Feature importance](docs/05_features.png)

This makes sense: bookmaker odds already *encode* all public statistics. The only marginal information comes from measuring how much bookmakers *disagree* with each other — but even this isn't enough to generate a profitable edge. The final pipeline drops market features as model inputs to keep the model independent from real-time odds data.

---

## Result 6 — No consistent league-level edge

Only Serie A shows a positive ROI (+4.2%) on the draw market, but with 353 bets and a p-value of 0.31, this is indistinguishable from noise.

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
- Features: rolling team stats (goals, shots, under-rate, variance), dynamic league rankings, shot accuracy xG proxy, bookmaker no-vig probabilities
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
│   ├── features.py             # Feature engineering (rolling stats, rankings, xG proxy)
│   ├── model.py                # XGBoost + walk-forward + Platt calibration
│   ├── backtest.py             # ROI simulation, significance tests, edge optimisation
│   ├── main.py                 # Under 2.5 pipeline (Div2)
│   ├── draw_pipeline.py        # Draw pipeline (Div1)
│   └── scrape_understat.py     # Selenium-based xG scraper
├── docs/                       # Diagnostic plots (AUC, ROI, calibration, feature importance)
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
