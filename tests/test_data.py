"""Tests for the synthetic data and its invariants. Run with: python -m pytest"""
import pandas as pd

from src.simulate_data import build_master_tables, simulate_sales


def _small_sales():
    items, stores = build_master_tables()
    sales = simulate_sales(items, stores, start_date="2024-01-01", end_date="2024-06-30", seed=1)
    return sales, items, stores


def test_sales_are_non_negative_integers():
    sales, _, _ = _small_sales()
    assert (sales["qty_sold"] >= 0).all()
    assert pd.api.types.is_integer_dtype(sales["qty_sold"])


def test_stockouts_leave_an_empty_shelf():
    sales, _, _ = _small_sales()
    stockouts = sales[sales["stockout_flag"] == 1]
    assert len(stockouts) > 0                 # the simulation really creates stockouts
    assert (stockouts["stock_on_hand"] == 0).all()


def test_exactly_one_row_per_store_item_day():
    sales, items, stores = _small_sales()
    assert not sales.duplicated(["store_id", "item_id", "date"]).any()
    n_days = len(pd.date_range("2024-01-01", "2024-06-30"))
    assert len(sales) == len(items) * len(stores) * n_days
