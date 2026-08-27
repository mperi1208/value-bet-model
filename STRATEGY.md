# The Sharp-Anchor Strategy — The Edge That Does Exist

The machine learning search in this repository failed (see
[RESULTS_ML.md](RESULTS_ML.md)). The same dataset nevertheless contains a
profitable strategy — one that predicts nothing at all about football.

**Anchor on the sharp price, bet the slow book.** Take Pinnacle's odds as the
market's best estimate, remove its margin with a *power* devig, and bet whenever
some bookmaker in the panel prices an outcome more than 2% above that fair value.

| Market | Period | n | ROI | 95% CI | CLV |
|---|---|---|---|---|---|
| 1X2 | 2012–2024 | 17 890 | +4.8% | [+2.2, +7.4] | +3.2% |
| O/U 2.5 | 2019–2024 | 1 311 | +5.9% | — | +4.2% |
| Asian Handicap | 2019–2024 | 1 475 | +5.0% | — | +1.6% |
| **Portfolio** | 2012–2024 | **20 676** | **+4.86%** | **[+2.5, +7.2]** | **+3.05%** |

Implementation: [`src/value_bet_sharp.py`](src/value_bet_sharp.py).

---

## The bets behave exactly as a real edge should

![EV gradient](docs/07_sharp_ev_gradient.png)

The strategy's own placebo test: every Max-odds bet in the dataset, bucketed by
the edge estimated *before* the match. Bets the method calls bad lose 7.3%; bets
it calls good win 10.9%; the ordering never breaks. A backtest artefact would not
produce a monotone gradient through zero. Note also that betting Max odds
indiscriminately loses 1.2% — the premium of the best available price explains
none of this.

## No single league carries the result

![League breakdown](docs/08_sharp_leagues.png)

Nine of ten leagues are positive. Individual league CIs are wide and none is
independently conclusive, but the result does not depend on any one of them:
leave any league out and the rest still returns between +4.4% and +5.4%.

## Closing line value is the real evidence

![CLV by season](docs/09_sharp_clv.png)

ROI is noisy; CLV is not. In all thirteen seasons the selected bets were priced
better than Pinnacle's own closing line, and roughly two thirds of individual
bets beat the close — against 50% for a coin flip. This is the standard proof
that a selection captures genuine mispricing rather than variance, and it is the
metric to watch in any forward test.

## Equity curve

![Equity curve](docs/10_sharp_equity.png)

Flat 1-unit stakes, no compounding: +1 004 units over thirteen seasons, worst
drawdown 98 units. The slope visibly flattens after 2022.

## The edge is line shopping, not stock picking

![Per-book comparison](docs/11_sharp_books.png)

A counter-intuitive result worth stating plainly: the best *per-bet* return comes
from Interwetten alone (+7.7%), not from the panel maximum (+4.8%). But per-bet
edge is not where the money is — the panel finds 5.5× more opportunities and
generates **854 units against 248**. One book is also reliably *un*profitable to
bet into (BetVictor, −6.4%). The edge lives in having somewhere to shop.

## The window is closing

![Decay](docs/12_sharp_decay.png)

Restricted to 1X2, the only market covered across all thirteen seasons:
qualifying mispricings have **halved**, from ~1 810 per season in 2012–2014 to
~845 in 2022–2024. ROI over 2022–2024 is +1.75% with a 95% CI of [−4.4, +8.1] —
too wide to claim the edge has died, but no longer enough to claim it is intact.
CLV over the same span holds at +2.96% with 71% of bets beating the close, which
suggests the selection still works and the opportunities are simply rarer.

---

## Robustness

- **Leave-one-league-out**: removing any league leaves the rest at +4.4% to +5.4%
- **Placebo**: bets at EV < −0.02 lose 3.8%, the neutral zone loses 0.5%, the
  strategy makes +4.8% — a monotone gradient
- **Random Max selection** on the same matches returns −1.0%: the best-price
  premium alone is not the edge
- **Walk-forward threshold selection** (each season uses only the past) converges
  to the low thresholds it should
- **Worst-case execution** — betting only at Bet365 instead of the theoretical
  maximum — still returns +1.6% with +1.4% CLV
- **Slippage**: at Max −2% the 0.02 threshold dominates; at −3%, thresholds below
  0.01 turn losing

## Staking

Fractional Kelly (`--kelly`, default ¼): stake = min(¼ × EV/(odds−1), 2% of
bankroll), daily exposure capped at 25%, same-day bets sized off the morning
bankroll. Across the 20 676 bets, quarter-Kelly produces a 22.6% max drawdown
against >60% at full Kelly. The fraction being ≤ ¼ is not negotiable given the
noise in estimated EV. Any bankroll *multiple* reported by the simulator is
theoretical compounding — only the ranking and the drawdown are informative.

## What this does and does not mean

The edge comes from soft bookmakers updating prices more slowly than the sharp
market. It does not survive contact with a fast one: executed on Betfair
Exchange, only 51% of bets beat the close — a coin flip — and realistic
commission turns the return negative. Exchange odds match the soft-book maximum
in level, but the exchange moves *with* Pinnacle, so an outlier there is
information rather than latency.

The binding constraints are operational, not statistical:

- soft books limit winning accounts within weeks — this is the real ceiling
- historical odds are snapshots, not executable prices
- capturing the panel maximum assumes accounts almost everywhere
- an adaptation to French (ANJ) bookmakers was tested and abandoned in August
  2026: measured French odds sit at 0.929 × Pinnacle, a discount that leaves no
  profit

## Forward testing

The judge of this strategy is not the backtest.

```bash
python src/value_bet_sharp.py --predict --kelly   # log upcoming value bets
python src/value_bet_sharp.py --evaluate          # settle them, compute realised CLV
```

Three months of positive CLV in paper trading is worth more than thirteen
seasons of backtest.
