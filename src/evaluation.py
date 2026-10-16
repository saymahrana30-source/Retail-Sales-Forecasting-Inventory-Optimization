"""
evaluation.py
-------------
Honest accuracy testing with ROLLING-ORIGIN BACKTESTS.

We pretend it is the past several times. Each time we train ONLY on data
that existed then, forecast the next 28 days, and compare with what really
happened. Stockout days are skipped when scoring because true demand was
hidden on those days.

Metrics
  MAE   : average absolute error, in units per day
  RMSE  : like MAE but punishes big misses more
  MASE  : error relative to a simple naive forecast (below 1.0 beats naive)
  WAPE  : total absolute error divided by total actual units (a % error)
"""

import numpy as np
import pandas as pd

import config
from src.features import add_features, SERIES_KEYS
from src.models import train_ml_model, generate_forecasts

METHOD_COLUMNS = {
    "Seasonal naive": "pred_naive",
    "Croston/SBA": "pred_croston",
    "ML model only": "pred_ml",
    "Hybrid (final)": "forecast",
}


def naive_scale(history, lag=7):
    """Average size of a weekly naive change. Used to make MASE scale-free."""
    rows = []
    for key, g in history.groupby(SERIES_KEYS):
        values = g.sort_values("date")["qty"].to_numpy(dtype=float)
        diffs = np.abs(values[lag:] - values[:-lag])
        diffs = diffs[~np.isnan(diffs)]
        scale = diffs.mean() if diffs.size and diffs.mean() > 0 else np.nan
        rows.append((key[0], key[1], scale))
    return pd.DataFrame(rows, columns=SERIES_KEYS + ["scale"])


def run_backtest(panel, horizon=config.HORIZON_DAYS, n_folds=config.N_BACKTEST_FOLDS):
    """Train and forecast at several past origins. Returns predictions, metrics, residuals."""
    last_date = panel["date"].max()
    folds = []
    for fold_number, k in enumerate(range(n_folds, 0, -1), start=1):
        origin = last_date - pd.Timedelta(days=horizon * k)
        history = panel[panel["date"] <= origin]
        future = panel[(panel["date"] > origin) & (panel["date"] <= origin + pd.Timedelta(days=horizon))]

        train = add_features(history)
        model = train_ml_model(train[train["qty"].notna()])
        forecasts = generate_forecasts(model, history, future)

        actual = future[SERIES_KEYS + ["date", "category", "qty"]]
        scored = forecasts.merge(actual, on=SERIES_KEYS + ["date"], how="inner")
        scored = scored[scored["qty"].notna()].copy()      # only days with real sales
        scored = scored.merge(naive_scale(history), on=SERIES_KEYS, how="left")
        scored["fold"] = fold_number
        scored["origin"] = origin
        folds.append(scored)

    backtest = pd.concat(folds, ignore_index=True)
    return backtest, summarise_backtest(backtest), residual_stats(backtest)


def residual_stats(backtest):
    """Typical daily forecast error per series (RMSE). Feeds the safety stock formula."""
    squared = (backtest["qty"] - backtest["forecast"]) ** 2
    stats = backtest.assign(sq_error=squared).groupby(SERIES_KEYS).agg(
        sigma_sq=("sq_error", "mean"),
        n_days=("sq_error", "size"),
    ).reset_index()
    stats["sigma_daily"] = np.sqrt(stats["sigma_sq"])
    return stats.drop(columns="sigma_sq")


def summarise_backtest(backtest):
    """Accuracy of every method, overall, by demand pattern and by category."""
    long = backtest.melt(
        id_vars=SERIES_KEYS + ["date", "qty", "scale", "pattern", "category"],
        value_vars=list(METHOD_COLUMNS.values()),
        var_name="method_col",
        value_name="pred",
    )
    long["method"] = long["method_col"].map({v: k for k, v in METHOD_COLUMNS.items()})
    long["abs_err"] = (long["qty"] - long["pred"]).abs()
    long["sq_err"] = (long["qty"] - long["pred"]) ** 2
    long["scaled_err"] = long["abs_err"] / long["scale"]

    stacked = pd.concat([
        long.assign(segment_type="overall", segment="All series"),
        long.assign(segment_type="pattern", segment=long["pattern"]),
        long.assign(segment_type="category", segment=long["category"]),
    ], ignore_index=True)

    table = stacked.groupby(["segment_type", "segment", "method"]).agg(
        rows=("qty", "size"),
        MAE=("abs_err", "mean"),
        RMSE=("sq_err", "mean"),
        MASE=("scaled_err", "mean"),
        abs_err_total=("abs_err", "sum"),
        actual_total=("qty", "sum"),
    ).reset_index()
    table["RMSE"] = np.sqrt(table["RMSE"])
    table["WAPE_pct"] = 100 * table["abs_err_total"] / table["actual_total"].replace(0, np.nan)
    return table.drop(columns=["abs_err_total", "actual_total"]).round(3)
