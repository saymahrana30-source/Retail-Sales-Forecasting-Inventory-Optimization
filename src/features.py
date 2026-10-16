"""
features.py
-----------
Turns cleaned sales into a DAILY PANEL and builds the model inputs.

Key ideas
- The panel has one row per store x item x day, including every calendar day.
  That way "7 days ago" really means 7 days ago.
- On stockout days true demand is hidden, so the target 'qty' is NaN there
  (not zero!). The row still exists, so lag features stay correct.
- Every feature for a given day uses ONLY earlier days. This prevents
  "data leakage", the most common beginner mistake in forecasting.
"""

import numpy as np
import pandas as pd

import config

SERIES_KEYS = ["store_id", "item_id"]

FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_14",
    "dow", "week_of_year", "month", "is_weekend",
    "holiday_flag", "on_promo", "discount_pct", "price",
    "store_code", "item_code",
]
CATEGORICAL_FEATURES = ["store_code", "item_code"]


def holiday_flags(dates):
    """Return 1 on festival days from config.FESTIVAL_DATES, otherwise 0."""
    dates = pd.DatetimeIndex(dates)
    flags = np.zeros(len(dates), dtype=int)
    for month, day in config.FESTIVAL_DATES:
        flags |= ((dates.month == month) & (dates.day == day)).astype(int)
    return flags


def build_daily_panel(clean):
    """Create one row per store x item x day, with gaps filled as NaN."""
    all_dates = pd.date_range(clean["date"].min(), clean["date"].max(), freq="D")
    keys = clean[SERIES_KEYS].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"date": all_dates}), how="cross")
    panel = grid.merge(clean, on=SERIES_KEYS + ["date"], how="left")

    # Demand is only trusted on days WITHOUT a stockout (and where a record exists)
    panel["qty"] = panel["qty_sold"].where(panel["stockout_flag"] == 0)
    panel["on_promo"] = panel["on_promo"].fillna(0).astype(int)
    panel["discount_pct"] = panel["discount_pct"].fillna(0.0)
    panel["holiday_flag"] = holiday_flags(panel["date"])
    panel["category"] = panel.groupby(SERIES_KEYS)["category"].transform("first")

    panel = panel.sort_values(SERIES_KEYS + ["date"]).reset_index(drop=True)
    panel["price"] = panel.groupby(SERIES_KEYS)["price"].transform(lambda s: s.ffill().bfill())

    # Numeric codes for the model (store and item as categories)
    panel["store_code"] = panel["store_id"].astype("category").cat.codes.astype(int)
    panel["item_code"] = panel["item_id"].astype("category").cat.codes.astype(int)

    return panel[SERIES_KEYS + ["date", "category", "qty", "store_code", "item_code",
                                "price", "on_promo", "discount_pct", "holiday_flag"]]


def _series_features(g):
    """Add lag, rolling and calendar features for ONE store x item series."""
    g = g.copy()
    qty = g["qty"]
    past = qty.shift(1)  # rolling windows look at yesterday and earlier only

    g["lag_1"] = qty.shift(1)
    g["lag_7"] = qty.shift(7)
    g["lag_14"] = qty.shift(14)
    g["lag_28"] = qty.shift(28)
    g["roll_mean_7"] = past.rolling(7, min_periods=3).mean()
    g["roll_mean_14"] = past.rolling(14, min_periods=7).mean()
    g["roll_mean_28"] = past.rolling(28, min_periods=14).mean()
    g["roll_std_14"] = past.rolling(14, min_periods=7).std()

    g["dow"] = g["date"].dt.dayofweek
    g["week_of_year"] = g["date"].dt.isocalendar().week.astype(int)
    g["month"] = g["date"].dt.month
    g["is_weekend"] = (g["dow"] >= 5).astype(int)
    return g


def add_features(panel):
    """Apply the feature logic to every store x item series."""
    parts = []
    for _, g in panel.groupby(SERIES_KEYS, sort=False):
        parts.append(_series_features(g.sort_values("date")))
    return pd.concat(parts, ignore_index=True)


def build_future_calendar(panel, horizon_days=config.HORIZON_DAYS):
    """Rows for the days we want to forecast.

    Only information we would KNOW in advance is filled in: the calendar,
    the festival calendar and the latest price. No promotion plan exists yet,
    so promotions are assumed to be 'none'.
    """
    last_known = panel.sort_values("date").groupby(SERIES_KEYS).tail(1)
    last_known = last_known[SERIES_KEYS + ["store_code", "item_code", "price"]]

    future_dates = pd.date_range(panel["date"].max() + pd.Timedelta(days=1),
                                 periods=horizon_days, freq="D")
    future = last_known.merge(pd.DataFrame({"date": future_dates}), how="cross")
    future["qty"] = np.nan
    future["on_promo"] = 0
    future["discount_pct"] = 0.0
    future["holiday_flag"] = holiday_flags(future["date"])
    return future
