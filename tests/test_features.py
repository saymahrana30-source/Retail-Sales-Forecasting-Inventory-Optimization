"""Tests for the feature pipeline. Run with: python -m pytest"""
import numpy as np

from src.data_quality import clean_sales
from src.features import (add_features, build_daily_panel, FEATURE_COLUMNS, SERIES_KEYS)
from src.simulate_data import build_master_tables, simulate_sales


def _panel():
    items, stores = build_master_tables()
    sales = simulate_sales(items, stores, start_date="2024-01-01", end_date="2024-04-30", seed=2)
    return build_daily_panel(clean_sales(sales))


def test_panel_has_every_day_for_every_series():
    panel = _panel()
    lengths = panel.groupby(SERIES_KEYS).size()
    assert lengths.nunique() == 1


def test_features_do_not_leak_future_information():
    panel = _panel()
    base = add_features(panel)

    changed = panel.copy()
    last_day = changed["date"].max()
    changed.loc[changed["date"] == last_day, "qty"] = 9999   # change ONLY the final day
    alt = add_features(changed)

    before_last = (base["date"] < last_day).to_numpy()
    for col in FEATURE_COLUMNS:
        a = base.loc[before_last, col].to_numpy(dtype=float)
        b = alt.loc[before_last, col].to_numpy(dtype=float)
        assert np.allclose(a, b, equal_nan=True), f"feature {col} uses future data"
