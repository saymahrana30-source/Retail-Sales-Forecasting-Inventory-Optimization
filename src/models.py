"""
models.py
---------
Forecasting methods and the HYBRID forecast.

Three methods
  1. Seasonal naive : "same weekday as last week". The baseline any model must beat.
  2. Croston / SBA  : designed for intermittent demand (many zero-sales days).
  3. ML model       : ONE gradient-boosting model learned across ALL series.
                      It forecasts 28 days ahead one day at a time (recursive).

Hybrid rule
  intermittent series -> Croston / SBA
  regular series      -> 70% ML model + 30% seasonal naive
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from src.features import FEATURE_COLUMNS, CATEGORICAL_FEATURES, SERIES_KEYS, add_features


def train_ml_model(train_df):
    """Fit the gradient-boosting model. Only rows with a real target value are used."""
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES]
    model = HistGradientBoostingRegressor(
        loss="poisson",            # demand is a count of units, so Poisson loss suits it
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        categorical_features=cat_idx,
        random_state=config.RANDOM_SEED,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["qty"])
    return model


def recursive_forecast(model, history, future, tail_days=70):
    """Predict the ML model's value for every future day, one day at a time.

    Day 1 is predicted from real history. Day 2 uses the predicted day 1 as
    its 'yesterday', and so on. Returns: store_id, item_id, date, pred_ml.
    """
    future = future.assign(qty=np.nan)   # never use actual future sales here
    cutoff = history["date"].max() - pd.Timedelta(days=tail_days)
    combined = pd.concat([history[history["date"] > cutoff], future], ignore_index=True)

    future_days = sorted(future["date"].unique())
    for day in future_days:
        feats = add_features(combined)
        todo = feats["date"] == day
        preds = np.clip(model.predict(feats.loc[todo, FEATURE_COLUMNS]), 0, None)

        pred_by_key = pd.Series(preds, index=pd.MultiIndex.from_frame(feats.loc[todo, SERIES_KEYS]))
        target = combined["date"] == day
        target_keys = pd.MultiIndex.from_frame(combined.loc[target, SERIES_KEYS])
        combined.loc[target, "qty"] = pred_by_key.reindex(target_keys).to_numpy()

    result = combined[combined["date"].isin(future_days)][SERIES_KEYS + ["date", "qty"]]
    return result.rename(columns={"qty": "pred_ml"})


def share_of_zero_days(observed, window=config.CLASSIFY_WINDOW_DAYS):
    """Share of zero-sales days in the most recent window (used to classify demand)."""
    recent = np.asarray(observed, dtype=float)[-window:]
    return float(np.mean(recent == 0)) if recent.size else 1.0


def seasonal_naive(hist_dates, hist_qty, future_dates, max_weeks_back=8, fallback_days=28):
    """Forecast each future day with the most recent observed value for the same weekday."""
    hist_qty = np.asarray(hist_qty, dtype=float)
    lookup = {pd.Timestamp(d): q for d, q in zip(hist_dates, hist_qty) if not np.isnan(q)}
    observed = hist_qty[~np.isnan(hist_qty)]
    fallback = float(observed[-fallback_days:].mean()) if observed.size else 0.0

    preds = []
    for d in pd.to_datetime(future_dates):
        value = fallback
        for weeks in range(1, max_weeks_back + 1):
            candidate = d - pd.Timedelta(weeks=weeks)
            if candidate in lookup:
                value = lookup[candidate]
                break
        preds.append(value)
    return np.array(preds, dtype=float)


def croston_sba(demand, horizon, alpha=config.CROSTON_ALPHA):
    """Croston's method with the Syntetos-Boylan (SBA) bias correction.

    Intermittent demand is forecast as:  average size of a sale / average gap between sales.
    The same value is returned for every future day.
    """
    demand = np.asarray(demand, dtype=float)
    nonzero = np.flatnonzero(demand > 0)
    if nonzero.size == 0:
        return np.zeros(horizon)

    size_hat = demand[nonzero[0]]              # smoothed size of a sale
    interval_hat = float(nonzero[0] + 1)       # smoothed gap between sales
    periods_since_sale = 1
    for t in range(nonzero[0] + 1, len(demand)):
        if demand[t] > 0:
            size_hat = alpha * demand[t] + (1 - alpha) * size_hat
            interval_hat = alpha * periods_since_sale + (1 - alpha) * interval_hat
            periods_since_sale = 1
        else:
            periods_since_sale += 1

    forecast = (size_hat / interval_hat) * (1 - alpha / 2)   # SBA correction
    return np.full(horizon, forecast)


def generate_forecasts(model, history, future):
    """Forecast every store x item for the dates in `future`.

    history : panel rows up to the forecast start (qty is NaN on stockout days)
    future  : rows for the days to forecast; its qty column is ignored
    """
    future = future.assign(qty=np.nan)
    ml = recursive_forecast(model, history, future)

    history_by_series = {key: g.sort_values("date") for key, g in history.groupby(SERIES_KEYS)}
    parts = []
    for key, fut in future.sort_values("date").groupby(SERIES_KEYS):
        hist = history_by_series[key]
        observed = hist["qty"].dropna().to_numpy()
        p_zero = share_of_zero_days(observed)
        pattern = "intermittent" if p_zero > config.INTERMITTENT_THRESHOLD else "regular"
        dates = fut["date"].to_numpy()
        parts.append(pd.DataFrame({
            "store_id": key[0],
            "item_id": key[1],
            "date": dates,
            "pattern": pattern,
            "p_zero": round(p_zero, 3),
            "pred_naive": seasonal_naive(hist["date"].to_numpy(), hist["qty"].to_numpy(), dates),
            "pred_croston": croston_sba(observed, len(dates)),
        }))

    base = pd.concat(parts, ignore_index=True)
    base = base.merge(ml, on=SERIES_KEYS + ["date"], how="left")
    regular = config.ML_WEIGHT * base["pred_ml"] + (1 - config.ML_WEIGHT) * base["pred_naive"]
    base["forecast"] = np.where(base["pattern"] == "intermittent", base["pred_croston"], regular)
    base["forecast"] = base["forecast"].clip(lower=0)
    return base
