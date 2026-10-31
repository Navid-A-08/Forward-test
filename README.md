# Forward Test: Can a Strategy Call Patternless Markets?

A live, pre-registered forward test of simple trading strategies on ten markets. It accompanies my research on the false strategy problem: backtests can be tuned until they fit the past by chance, so the honest test is a record of calls made **before** the outcome exists.

Every Thursday an automated run:

1. downloads daily closing prices (Yahoo Finance) up to Wednesday's close,
2. scores every earlier call whose holding period has ended,
3. makes this week's calls and appends them to the log,
4. commits everything to this repository and sends me a report.

The git history is the timestamp: each week's calls are committed before their holding period starts.

## Protocol (version v1, frozen from the first run)

| Item | Rule |
| --- | --- |
| Decision time | Thursday about 08:20 UTC (11:50 Tehran) |
| Data used | Daily closes strictly before the decision date |
| Entry | Close of the decision Thursday (first close on or after it) |
| Exit | Close of the following Wednesday (last close before the next Thursday) |
| Position | +1 long, -1 short, 0 flat |
| Return | Position × (exit / entry − 1), gross of costs |
| Weekly score | Equal-weight average over the ten markets |

### Markets

Gold (GC=F), Silver (SI=F), WTI oil (CL=F), Natural gas (NG=F), Copper (HG=F), EUR/USD (EURUSD=X), USD/JPY (JPY=X), S&P 500 (^GSPC), Nasdaq 100 (^NDX), Bitcoin (BTC-USD).

### Strategies

| Strategy | Rule |
| --- | --- |
| Elastic net | Predicts the next 5-day log return from lagged returns (1, 5, 10, 20, 60 days), 20-day volatility and distance from the 50-day average. Retrained every week on the last ~3 years, penalty chosen by 5-fold time-series cross-validation. Long above a dead band of 0.1 × the training target's standard deviation, short below its negative, flat inside it. |
| Buy & hold | Always long. The minimum bar. |
| Momentum 12w | Long if the close is above its level 60 trading days ago, otherwise short. |
| Mean reversion 1w | Short if the close is above its level 5 trading days ago, otherwise long. |
| Coin flip | Random long or short from a hash of the date and market. The control: a strategy with no edge should not beat it. |

## Integrity rules

- `data/signals.csv` and `data/results.csv` are **append-only**. Rows are never edited or deleted.
- Any change to markets, features, strategies or timing creates a new version (v2, ...). Old versions keep running in the record as they were.
- A missed week is recorded as missing, never filled in afterwards.
- With weekly data, results under about 26 weeks are mostly noise. The scoreboard reports a t-statistic so this stays visible.

## Files

- `forward_test.py`: the whole pipeline.
- `data/signals.csv`: every call, with the last date of data it used.
- `data/results.csv`: every scored call, with entry and exit prices and dates.
- `reports/YYYY-MM-DD.md`: the weekly report.

## Run it yourself

```bash
pip install -r requirements.txt
python forward_test.py               # today's run
python forward_test.py --date 2026-11-05
```

Author: Navid Abdollahzadeh
