"""
data_quality.py
---------------
Step 1 of the workflow: load the raw sales file, check it, and clean it.
Beginners often skip this step. Bad input gives bad forecasts.
"""

import pandas as pd

import config

REQUIRED_COLUMNS = ["date", "store_id", "item_id", "qty_sold"]


def load_sales(path=config.SALES_FILE):
    """Read the sales CSV and parse the date column."""
    return pd.read_csv(path, parse_dates=["date"])


def validate_schema(df):
    """Stop early with a clear message if a required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Sales file is missing required columns: {missing}")


def quality_report(df):
    """Count the problems we look for in raw data. Returned as a dictionary."""
    key = ["store_id", "item_id", "date"]
    has_stockout = "stockout_flag" in df.columns
    return {
        "rows": len(df),
        "store_item_series": df.groupby(["store_id", "item_id"]).ngroups,
        "date_from": str(df["date"].min().date()),
        "date_to": str(df["date"].max().date()),
        "duplicate_rows": int(df.duplicated(key).sum()),
        "missing_qty_rows": int(df["qty_sold"].isna().sum()),
        "negative_qty_rows": int((df["qty_sold"] < 0).sum()),
        "stockout_days": int(df["stockout_flag"].sum()) if has_stockout else 0,
        "stockout_rate_pct": round(100 * df["stockout_flag"].mean(), 2) if has_stockout else 0.0,
    }


def clean_sales(df):
    """Remove duplicates, fix impossible values and fill obvious gaps."""
    key = ["store_id", "item_id", "date"]
    out = df.drop_duplicates(key, keep="first")          # one row per series per day
    out = out.dropna(subset=["qty_sold"])                # cannot use a day with no quantity
    out = out.assign(qty_sold=out["qty_sold"].clip(lower=0).round().astype(int))

    if "stockout_flag" not in out.columns:
        out = out.assign(stockout_flag=0)
    out = out.assign(stockout_flag=out["stockout_flag"].fillna(0).astype(int))

    for col in ["on_promo", "holiday_flag"]:
        if col in out.columns:
            out = out.assign(**{col: out[col].fillna(0).astype(int)})
    if "discount_pct" in out.columns:
        out = out.assign(discount_pct=out["discount_pct"].fillna(0.0))
    if "price" in out.columns:
        out = out.sort_values(key)
        out = out.assign(price=out.groupby(["store_id", "item_id"])["price"].ffill())

    return out.sort_values(key).reset_index(drop=True)
