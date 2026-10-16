"""Tests for the forecasting building blocks. Run with: python -m pytest"""
import numpy as np
import pandas as pd

from src.models import croston_sba, seasonal_naive, share_of_zero_days


def test_croston_returns_zero_for_all_zero_history():
    assert np.all(croston_sba(np.zeros(30), 5) == 0)


def test_croston_sporadic_sales_converges_to_average_rate():
    # One sale of 2 units every 5 days -> true average 0.4/day.
    # SBA correction shrinks it by (1 - alpha/2) = 0.95 -> about 0.38.
    demand = np.zeros(200)
    demand[::5] = 2
    f = croston_sba(demand, 7)
    assert f.shape == (7,)
    assert abs(f[0] - 0.38) < 0.01


def test_seasonal_naive_repeats_same_weekday_last_week():
    dates = pd.date_range("2024-01-01", periods=14)
    qty = np.tile([1, 2, 3, 4, 5, 6, 7], 2).astype(float)
    future = pd.date_range("2024-01-15", periods=7)
    pred = seasonal_naive(dates.to_numpy(), qty, future.to_numpy())
    assert np.array_equal(pred, np.array([1, 2, 3, 4, 5, 6, 7], dtype=float))


def test_seasonal_naive_skips_censored_days():
    dates = pd.date_range("2024-01-01", periods=14)
    qty = np.tile([1, 2, 3, 4, 5, 6, 7], 2).astype(float)
    qty[7] = np.nan                        # the Monday 2024-01-08 was a stockout
    pred = seasonal_naive(dates.to_numpy(), qty, pd.date_range("2024-01-15", periods=1).to_numpy())
    # The Monday 8 Jan is censored, so the forecast must use the Monday 1 Jan (value 1.0)
    assert pred[0] == 1.0


def test_share_of_zero_days():
    assert share_of_zero_days(np.array([0, 0, 1, 0])) == 0.75
