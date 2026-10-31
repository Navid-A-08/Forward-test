"""Weekly live forward test of simple trading strategies.

Each Thursday run:
  1. downloads daily closes (Yahoo Finance) up to the last completed day,
  2. scores every earlier signal whose holding period has ended,
  3. issues this week's signals (+1 long, 0 hold/flat, -1 short),
  4. appends them to append-only CSV logs and writes a weekly report.

Holding period for a signal made on Thursday D:
  enter at the close of D (first close on/after D), exit at the close of
  the following Wednesday (last close before D + 7). The signal only uses
  closes strictly before D, so it is fixed before the entry price exists.

Strategies and features are frozen per VERSION. Changing anything means a
new VERSION; old rows are never edited or deleted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")  # log of negative WTI prices (Apr 2020) and solver notes

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "v1"
ROOT = Path(__file__).resolve().parent
SIGNALS = ROOT / "data" / "signals.csv"
RESULTS = ROOT / "data" / "results.csv"
REPORTS = ROOT / "reports"

MARKETS = {  # name -> Yahoo ticker
    "Gold": "GC=F",
    "Silver": "SI=F",
    "WTI oil": "CL=F",
    "Natural gas": "NG=F",
    "Copper": "HG=F",
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Bitcoin": "BTC-USD",
}
STRATEGIES = ["elastic_net", "buy_hold", "momentum_12w", "mean_reversion_1w", "coin_flip"]
LABELS = {
    "elastic_net": "Elastic net",
    "buy_hold": "Buy & hold",
    "momentum_12w": "Momentum 12w",
    "mean_reversion_1w": "Mean rev 1w",
    "coin_flip": "Coin flip",
}

SIGNAL_COLS = ["decision_date", "created_utc", "version", "market", "ticker",
               "strategy", "signal", "data_through", "last_close", "en_pred"]
RESULT_COLS = ["decision_date", "version", "market", "ticker", "strategy", "signal",
               "entry_date", "entry_price", "exit_date", "exit_price",
               "market_return", "strategy_return", "scored_utc"]


# ----------------------------------------------------------------- data
def fetch(ticker: str) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(ticker)}"
    r = requests.get(url, params={"range": "10y", "interval": "1d"},
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"].get("exchangeTimezoneName") or "UTC"
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    close = res["indicators"]["quote"][0]["close"]
    s = pd.Series(close, index=pd.DatetimeIndex(ts.date), name=ticker, dtype="float64")
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s.sort_index()


# ----------------------------------------------------------- strategies
def features(close: pd.Series) -> pd.DataFrame:
    lr = np.log(close).diff()
    f = pd.DataFrame(index=close.index)
    for k in (1, 5, 10, 20, 60):
        f[f"ret_{k}"] = np.log(close / close.shift(k))
    f["vol_20"] = lr.rolling(20).std()
    f["dist_ma50"] = np.log(close / close.rolling(50).mean())
    return f


def elastic_net_signal(close: pd.Series) -> tuple[int, float]:
    """Predict the next 5-day log return from lagged features.

    Trains on the last ~3 years of closes available, labels use only
    closes that are already known. Long if the prediction is above a
    dead band, short if below, otherwise hold.
    """
    X = features(close)
    y = np.log(close.shift(-5) / close)
    train = X.join(y.rename("y")).dropna().iloc[-756:]
    x_now = X.iloc[[-1]]
    if len(train) < 300 or x_now.isna().any(axis=None):
        return 0, float("nan")
    model = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=TimeSeriesSplit(5),
                     max_iter=20000, random_state=0),
    )
    model.fit(train.drop(columns="y"), train["y"])
    pred = float(model.predict(x_now)[0])
    band = 0.1 * float(train["y"].std())
    sig = 1 if pred > band else (-1 if pred < -band else 0)
    return sig, pred


def momentum_signal(close: pd.Series) -> int:
    if len(close) < 61:
        return 0
    return 1 if close.iloc[-1] > close.iloc[-61] else -1


def mean_reversion_signal(close: pd.Series) -> int:
    if len(close) < 6:
        return 0
    return -1 if close.iloc[-1] > close.iloc[-6] else 1


def coin_flip_signal(decision: str, market: str) -> int:
    h = hashlib.sha256(f"{VERSION}|{decision}|{market}".encode()).digest()
    return 1 if h[0] % 2 == 0 else -1


# ------------------------------------------------------------- ledgers
def load(path: Path, cols: list[str]) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, dtype={"decision_date": str})
    return pd.DataFrame(columns=cols)


def append(path: Path, rows: list[dict], cols: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=cols)
    header = not path.exists() or path.stat().st_size == 0
    new.to_csv(path, mode="a", header=header, index=False)


# ---------------------------------------------------------------- run
def run(today: dt.date) -> dict:
    decision = today.isoformat()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    signals = load(SIGNALS, SIGNAL_COLS)
    results = load(RESULTS, RESULT_COLS)

    prices, errors = {}, []
    for name, tk in MARKETS.items():
        try:
            prices[name] = fetch(tk)
        except Exception as e:  # keep going with the other markets
            errors.append(f"{name}: {type(e).__name__}")

    # 2. score signals whose holding period has ended
    done = set(zip(results["decision_date"], results["market"], results["strategy"]))
    new_results = []
    for _, s in signals.iterrows():
        key = (s["decision_date"], s["market"], s["strategy"])
        if key in done or s["market"] not in prices:
            continue
        d = dt.date.fromisoformat(s["decision_date"])
        exit_cutoff = d + dt.timedelta(days=7)        # next Thursday
        if today < exit_cutoff:
            continue                                  # period not over yet
        px = prices[s["market"]]
        entry = px[px.index >= pd.Timestamp(d)]
        exit_ = px[px.index < pd.Timestamp(exit_cutoff)]
        if entry.empty or exit_.empty or entry.index[0] >= exit_.index[-1]:
            mret, ed, ep, xd, xp = 0.0, "", float("nan"), "", float("nan")
        else:
            ed, ep = entry.index[0].date().isoformat(), float(entry.iloc[0])
            xd, xp = exit_.index[-1].date().isoformat(), float(exit_.iloc[-1])
            mret = xp / ep - 1
        new_results.append({
            "decision_date": s["decision_date"], "version": s["version"],
            "market": s["market"], "ticker": s["ticker"], "strategy": s["strategy"],
            "signal": int(s["signal"]), "entry_date": ed, "entry_price": ep,
            "exit_date": xd, "exit_price": xp, "market_return": mret,
            "strategy_return": int(s["signal"]) * mret, "scored_utc": now,
        })
    append(RESULTS, new_results, RESULT_COLS)

    # 3. this week's signals (once per decision date)
    new_signals = []
    if not (signals["decision_date"] == decision).any():
        for name, px in prices.items():
            hist = px[px.index < pd.Timestamp(today)]
            if len(hist) < 70:
                errors.append(f"{name}: not enough history")
                continue
            en_sig, en_pred = elastic_net_signal(hist)
            sigs = {
                "elastic_net": en_sig,
                "buy_hold": 1,
                "momentum_12w": momentum_signal(hist),
                "mean_reversion_1w": mean_reversion_signal(hist),
                "coin_flip": coin_flip_signal(decision, name),
            }
            for strat in STRATEGIES:
                new_signals.append({
                    "decision_date": decision, "created_utc": now, "version": VERSION,
                    "market": name, "ticker": MARKETS[name], "strategy": strat,
                    "signal": sigs[strat], "data_through": hist.index[-1].date().isoformat(),
                    "last_close": float(hist.iloc[-1]),
                    "en_pred": en_pred if strat == "elastic_net" else float("nan"),
                })
        append(SIGNALS, new_signals, SIGNAL_COLS)

    report = build_report(decision, new_results, new_signals, errors)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{decision}.md").write_text(report["markdown"], encoding="utf-8")
    (ROOT / "data" / "last_report.txt").write_text(report["telegram"], encoding="utf-8")
    return report


# ------------------------------------------------------------- report
def scoreboard(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat in STRATEGIES:
        r = results[results["strategy"] == strat]
        if r.empty:
            continue
        weekly = r.groupby("decision_date")["strategy_return"].mean().sort_index()
        n = len(weekly)
        active = r[r["signal"] != 0]
        hits = (np.sign(active["market_return"]) == active["signal"]).mean() if len(active) else float("nan")
        sd = weekly.std(ddof=1) if n > 1 else float("nan")
        sharpe = weekly.mean() / sd * math.sqrt(52) if n >= 2 and sd > 0 else float("nan")
        tstat = weekly.mean() / (sd / math.sqrt(n)) if n >= 2 and sd > 0 else float("nan")
        rows.append({"strategy": strat, "weeks": n,
                     "total": float((1 + weekly).prod() - 1), "hit": hits,
                     "sharpe": sharpe, "t": tstat})
    return pd.DataFrame(rows)


def pct(x: float, d: int = 1) -> str:
    return "  n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x*100:+.{d}f}%"


def num(x: float) -> str:
    return " n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.2f}"


def arrow(s: int) -> str:
    return {1: "BUY ", -1: "SELL", 0: "hold"}[int(s)]


def build_report(decision: str, new_results: list, new_signals: list, errors: list) -> dict:
    results = load(RESULTS, RESULT_COLS)
    lines = [f"Forward test {VERSION} | Thursday {decision}"]

    if new_results:
        nr = pd.DataFrame(new_results)
        wk = nr["decision_date"].max()
        lines.append(f"\nLast week's result (signals of {wk}):")
        lines.append("Strategy        Avg return  Hits")
        for strat in STRATEGIES:
            r = nr[(nr["strategy"] == strat) & (nr["decision_date"] == wk)]
            if r.empty:
                continue
            act = r[r["signal"] != 0]
            hit = f"{int((np.sign(act['market_return']) == act['signal']).sum())}/{len(act)}"
            lines.append(f"{LABELS[strat]:<15} {pct(r['strategy_return'].mean(), 2):>9}  {hit:>5}")
    else:
        lines.append("\nNo finished week to score yet.")

    sb = scoreboard(results)
    if not sb.empty:
        lines.append(f"\nScoreboard (all weeks so far, avg of {len(MARKETS)} markets):")
        lines.append("Strategy        Wks  Total   Hit%  Sharpe  t")
        for _, r in sb.iterrows():
            hit = "  n/a" if math.isnan(r["hit"]) else f"{r['hit']*100:4.0f}%"
            lines.append(f"{LABELS[r['strategy']]:<15} {r['weeks']:>3} {pct(r['total']):>7} {hit:>5} {num(r['sharpe']):>6} {num(r['t']):>5}")
        if sb["weeks"].max() < 26:
            lines.append("(Under 26 weeks: differences are mostly noise.)")

    if new_signals:
        ns = pd.DataFrame(new_signals)
        lines.append(f"\nThis week's calls (enter Thu close, exit Wed close):")
        lines.append("Market       EN   Mom  MRev Coin")
        for m in MARKETS:
            r = ns[ns["market"] == m].set_index("strategy")["signal"]
            if r.empty:
                continue
            lines.append(f"{m:<12} {arrow(r['elastic_net'])} {arrow(r['momentum_12w'])} "
                         f"{arrow(r['mean_reversion_1w'])} {arrow(r['coin_flip'])}")
        lines.append("Buy & hold: BUY everything.")
    else:
        lines.append("\nSignals for this date already existed; none added.")

    if errors:
        lines.append("\nProblems: " + "; ".join(errors))

    body = "\n".join(lines)
    md = f"# Forward test report, {decision}\n\n```\n{body}\n```\n"
    return {"telegram": body, "markdown": md}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="decision date YYYY-MM-DD (default: today, UTC)")
    a = ap.parse_args()
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(dt.timezone.utc).date()
    rep = run(day)
    print(rep["telegram"])
