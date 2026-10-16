"""Tests for the inventory formulas. Run with: python -m pytest"""
import math

import numpy as np

from src.inventory import (safety_stock, economic_order_quantity,
                           lead_time_demand, compute_policy)


def test_safety_stock_formula():
    # 95% service -> z = 1.6449; sigma = 10; lead time 4 days -> z * 10 * 2
    assert math.isclose(safety_stock(0.95, 10, 4), 1.6448536 * 10 * 2, rel_tol=1e-6)


def test_eoq_formula():
    # D = 10,000, K = 500, H = 100 * 0.2 = 20  ->  sqrt(2*10000*500/20) = 707.1
    assert math.isclose(economic_order_quantity(10000, 500, 100, 0.2), 707.1068, rel_tol=1e-5)


def test_lead_time_demand_sums_first_days():
    f = [1, 2, 3, 4, 5]
    assert lead_time_demand(f, 3) == 6
    # lead time longer than the forecast: pad with the average (3)
    assert lead_time_demand(f, 7) == 15 + 2 * 3


def test_order_placed_when_position_below_reorder_point():
    p = compute_policy(np.full(28, 10.0), sigma_daily=2, on_hand=20, on_order=0,
                       lead_time_days=3, unit_cost=100, service_level=0.95)
    assert p["status"] == "REORDER NOW"
    assert p["inventory_position"] <= p["reorder_point"]
    assert p["recommended_order_qty"] >= p["eoq"]   # at least one EOQ


def test_no_order_when_stock_is_healthy():
    p = compute_policy(np.full(28, 10.0), sigma_daily=2, on_hand=500, on_order=0,
                       lead_time_days=3, unit_cost=100, service_level=0.95)
    assert p["recommended_order_qty"] == 0
    assert p["status"] == "OK"


def test_pending_orders_count_toward_position():
    base = dict(daily_forecast=np.full(28, 10.0), sigma_daily=2, on_hand=10,
                lead_time_days=3, unit_cost=100, service_level=0.95)
    without = compute_policy(on_order=0, **base)
    with_pending = compute_policy(on_order=500, **base)
    assert without["status"] == "REORDER NOW"
    assert with_pending["status"] == "OK"
    assert with_pending["recommended_order_qty"] == 0
