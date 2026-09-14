# Value Bet Model

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![XGBoost](https://img.shields.io/badge/XGBoost-3.0-orange)

Can publicly available data generate profitable betting signals on European
football? Two answers, and they point in opposite directions.

| | Approach | Result |
|---|---|---|
| **1** | Machine learning on public data — 5 iterations, 53 features, walk-forward across 25 seasons and 10 leagues | **−6.7% ROI**, AUC ceiling ~0.56 |
| **2** | Sharp anchor + line shopping — bet outlier odds against the power-devigged Pinnacle price | **+4.86% ROI**, n=20 676, CLV +3.05% |

The model never beats the closing line. A profitable edge does exist in the same
data, but it comes from bookmakers pricing slowly, not from predicting football.

![EV gradient](docs/07_sharp_ev_gradient.png)

*The strategy's placebo test: the worst-EV bucket loses 7.3%, the best makes
10.9%, and the ordering never breaks between them. Betting the best available
price indiscriminately loses 1.2%, so the price premium alone explains none of
it.*

## Read more

| | |
|---|---|
| [**STRATEGY.md**](STRATEGY.md) | The profitable strategy — how the audit uncovered it, the evidence, robustness tests, staking, and why it does not survive on an exchange |
| [**RESULTS_ML.md**](RESULTS_ML.md) | The ML null result — the AUC ceiling, the calibration paradox, and the false positive that started it |
| [**AUDIT.md**](AUDIT.md) | The July 2026 audit that corrected this repository's own headline numbers |

## Data

| Source | Coverage |
|--------|----------|
| [football-data.co.uk](https://www.football-data.co.uk) | 25 seasons × 10 leagues — results, shots, cards, odds from 6+ bookmakers |
| [Understat](https://understat.com) | 8 seasons × 5 leagues — expected goals per match |

## Usage

```bash
pip install -r requirements.txt

# Download match data
python src/download.py --seasons 25

# The profitable strategy
python src/value_bet_sharp.py --ev 0.02 --kelly

# Detect value bets on upcoming fixtures, then settle them later
python src/value_bet_sharp.py --predict --kelly
python src/value_bet_sharp.py --evaluate

# The ML pipeline
python src/main.py --edge 0.05
```

## Structure

```
src/
  value_bet_sharp.py   Sharp-anchor strategy, Kelly staking, paper trading
  main.py              ML pipeline (Under 2.5)
  draw_pipeline.py     ML pipeline (Draw)
  features.py          Feature engineering
  model.py             XGBoost + walk-forward + Platt calibration
  backtest.py          ROI simulation, significance tests
  download.py          football-data.co.uk fetcher
docs/
  generate_plots.py        Charts 01–06 (ML results)
  generate_sharp_plots.py  Charts 07–12 (strategy)
```

## Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `Pandas` · `NumPy` · `Matplotlib` · `Selenium` · `SciPy`

---

**Marc'Andria Peri** — CPES 3A (Paris-Saclay × HEC × IP Paris), Data Science track

*Data: [football-data.co.uk](https://www.football-data.co.uk) · [Understat](https://understat.com)*
