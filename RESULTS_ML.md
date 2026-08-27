# The Machine Learning Investigation — A Null Result

Full detail of the ML side of [value-bet-model](README.md). **Conclusion: public
data cannot beat the bookmaker's closing line.** Five iterations, a −6.7% ROI,
and two false positives that I detected and corrected in my own work.

For the strategy that *does* make money, see [STRATEGY.md](STRATEGY.md). For the
audit that produced the corrected numbers, see [AUDIT.md](AUDIT.md).

---

## The five iterations

Each version adds new information while keeping the same walk-forward validation:

| Version | What changed | Features | Market |
|---------|-------------|----------|--------|
| **v1** | Baseline — XGBoost + isotonic calibration | 56 rolling stats | Under 2.5 (Div2) |
| **v2** | Fixed calibration (Platt), reduced features | 21 lean features | Under 2.5 (Div2) |
| **v3** | Added inter-bookmaker disagreement signals | 25 features | Under 2.5 (Div2) |
| **v4** | Added expected goals from Understat | 28 features | Draw (Div1) |
| **v5** | H2H history, fixture congestion, referee stats, odds spread | 53 features | Under 2.5 |

---

## Result 1 — The AUC ceiling

![AUC across versions](docs/01_auc.png)

Discriminative power barely moves across iterations and never reaches the
profitable threshold (~0.58). With AUC between 0.535 and 0.5601, the model cannot
separate value bets from non-value bets well enough to overcome a 5–8% margin.
Market features (v3) gave the best single step (+0.019); xG (v4) added nothing.

## Result 2 — Consistently negative ROI

![ROI across versions](docs/02_roi.png)

Every version loses money, and there is no trend toward profitability. The v5
figure shown is the **audited −6.7%**, not the −3.2% this repository displayed
before July 2026 — that number was inflated by roughly 3.5 points by a global
isotonic recalibration fitted on the full out-of-sample set (leakage) and by
excluding leagues after seeing the results.

## Result 3 — The calibration paradox

![Calibration paradox](docs/03_calibration.png)

The model is well calibrated *globally* but systematically overconfident *on the
bets it selects*. When it predicts 55% and the bookmaker implies 48%, the actual
frequency is ~48% — the bookmaker was right. The model's confidence comes from
the noisy tail of its distribution, exactly where it is least reliable. This is
the core issue, and no amount of extra features addressed it.

## Result 4 — No monotonic edge

![Edge vs ROI](docs/04_edge_vs_roi.png)

A model with a genuine edge returns more on its higher-confidence bets. Here ROI
is flat negative at every threshold — the signature of no real advantage. Worth
contrasting with the same test applied to the sharp-anchor strategy in
[STRATEGY.md](STRATEGY.md), where the gradient is clean and monotone.

## Result 5 — Which features mattered

![Feature importance](docs/05_features.png)

`h_avg_goals_scored` ranks first and `odds_spread_under` (Max<2.5 / Avg<2.5)
second — both the statistical signal and the market's own uncertainty are
informative, yet insufficient. All other top features are goal and shot-based
rolling averages, consistent across every version.

## Result 6 — No consistent league-level edge

![League breakdown](docs/06_leagues_v2.png)

Across the Div2 leagues tested on Under 2.5, only Ligue 2 is positive (+1.4% on
197 bets), statistically indistinguishable from noise. Serie B and Segunda are
strongly negative.

> **Note.** Earlier versions of this repository concluded from this chart that
> the pipeline should restrict betting to E1 and F2. The audit removed that
> restriction: choosing leagues after seeing their results is exactly the
> a-posteriori selection that inflated the headline ROI. `EXCLUDED_DIVS` is now
> empty and the −6.7% figure is measured across all leagues.

---

## Root cause — why v1 showed a false +3.8%

The original model used **isotonic calibration**
(`CalibratedClassifierCV(method='isotonic', cv='prefit')`) fitted on validation
sets of 300–500 matches. Isotonic calibration is non-parametric with as many
parameters as unique predictions — it memorised the validation set's noise,
producing systematic overconfidence that manufactured an apparent edge.

Switching to **Platt calibration** (logistic sigmoid, two parameters) eliminated
the artefact and revealed the true ROI of −8.7%.

The same class of error recurred at a larger scale and was caught by the July
2026 audit: a global isotonic recalibration, this time fitted on the whole
out-of-sample set. Both are documented in [AUDIT.md](AUDIT.md). Detecting and
publishing them is the part of this project I consider the actual contribution.

---

## Methodology

### Walk-forward validation

```
Season N-k → N-2      Season N-1      Season N
┌────────────────┐   ┌────────────┐  ┌────────────┐
│     TRAIN      │   │    VAL     │  │    TEST    │
│  (fit model)   │   │ (calibrate)│  │ (evaluate) │
└────────────────┘   └────────────┘  └────────────┘
```

- Model retrained from scratch at each fold — no information leakage
- Calibration fitted on the validation set only, never on test
- t-test + bootstrap 95% CI on every backtest

### Model

**XGBoost** (conservative: `max_depth=4`, `min_child_weight=8`) with **Platt
calibration**. 53 features: rolling team stats (goals, shots, under-rate,
variance), dynamic league rankings, shot-accuracy xG proxy, head-to-head under
rate, fixture congestion, referee under-rate history, bookmaker odds spread, and
no-vig bookmaker probabilities. Edge = `P(model) − P(no-vig bookmaker)`.

### Reproducing

```bash
python src/main.py --edge 0.05              # Under 2.5 pipeline
python src/draw_pipeline.py --data-dir ./src/csv --edge 0.05
python docs/generate_plots.py               # regenerate charts 01–06
```
