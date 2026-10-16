"""
inventory.py
------------
Turns forecasts into stocking decisions.

Formulas (every one is explained in docs/SIMULATION.md with a worked example)
  mu_L  = expected demand during the supplier lead time (sum of the next L forecasts)
  SS    = z * sigma_daily * sqrt(L)     safety stock: buffer for forecast error
  ROP   = mu_L + SS                      reorder point: when to place an order
  EOQ   = sqrt(2 * D * K / H)            economic order quantity: how much to order
  S     = ROP + EOQ                      order-up-to level
  Order = S - inventory position         placed only when position <= ROP

  z     = normal-distribution score for the service level (95% -> 1.645)
  D     = annual demand, K = ordering cost, H = unit cost * holding rate
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from src.features import SERIES_KEYS

OUTPUT_COLUMNS = [
    "store_id", "store_city", "item_id", "item_name", "category", "status",
    "on_hand", "on_order", "inventory_position", "avg_daily_forecast",
    "lead_time_days", "unit_cost", "service_level", "sigma_daily",
    "lead_time_demand", "safety_stock", "reorder_point", "eoq", "order_up_to",
    "recommended_order_qty", "days_of_cover",
]


def service_level_for(category):
    """Service level target for a product category (falls back to the default)."""
    return config.SERVICE_LEVEL_BY_CATEGORY.get(category, config.DEFAULT_SERVICE_LEVEL)


def lead_time_demand(daily_forecast, lead_time_days):
    """Forecast demand during the supplier lead time. If the lead time is longer than
    the forecast, the missing days are filled with the average forecast."""
    f = np.asarray(daily_forecast, dtype=float)
    lead = int(lead_time_days)
    if lead <= f.size:
        return float(f[:lead].sum())
    return float(f.sum() + (lead - f.size) * f.mean())


def safety_stock(service_level, sigma_daily, lead_time_days):
    """Buffer stock that protects against forecast error during the lead time."""
    z = float(norm.ppf(service_level))
    return z * sigma_daily * np.sqrt(lead_time_days)


def economic_order_quantity(annual_demand, ordering_cost, unit_cost, holding_rate):
    """Classic EOQ: the order size that balances ordering cost against holding cost."""
    annual_holding_cost = unit_cost * holding_rate
    if annual_demand <= 0 or annual_holding_cost <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * ordering_cost / annual_holding_cost))


def compute_policy(daily_forecast, sigma_daily, on_hand, on_order, lead_time_days,
                   unit_cost, service_level):
    """Stocking decision for ONE store x item. Returns a dictionary."""
    f = np.asarray(daily_forecast, dtype=float)
    avg_daily = float(f.mean())

    mu_lead = lead_time_demand(f, lead_time_days)
    ss = safety_stock(service_level, sigma_daily, lead_time_days)
    rop = mu_lead + ss
    eoq = economic_order_quantity(avg_daily * config.DAYS_PER_YEAR,
                                  config.ORDERING_COST, unit_cost, config.HOLDING_RATE)
    order_up_to = rop + eoq
    position = on_hand + on_order
    triggered = position <= rop

    # Order enough to get back up to the order-up-to level (always at least one EOQ)
    order_qty = int(np.ceil(order_up_to - position)) if triggered else 0
    days_of_cover = on_hand / avg_daily if avg_daily > 0 else np.inf

    if avg_daily == 0:
        status = "NO DEMAND FORECAST"
    elif triggered:
        status = "REORDER NOW"
    elif days_of_cover > config.OVERSTOCK_DAYS_OF_COVER:
        status = "OVERSTOCK RISK"
    else:
        status = "OK"

    return {
        "on_hand": on_hand,
        "on_order": on_order,
        "inventory_position": position,
        "avg_daily_forecast": avg_daily,
        "sigma_daily": sigma_daily,
        "lead_time_demand": mu_lead,
        "safety_stock": ss,
        "reorder_point": rop,
        "eoq": eoq,
        "order_up_to": order_up_to,
        "recommended_order_qty": order_qty,
        "days_of_cover": days_of_cover,
        "status": status,
    }


def recommend_replenishment(forecasts, residuals, snapshot, items, stores):
    """One stocking decision per store x item, sorted so urgent rows come first."""
    sigma_map = {(r.store_id, r.item_id): r.sigma_daily for r in residuals.itertuples(index=False)}
    item_info = items.set_index("item_id")
    store_info = stores.set_index("store_id")
    stock = snapshot.set_index(SERIES_KEYS)

    records = []
    for (store_id, item_id), g in forecasts.sort_values("date").groupby(SERIES_KEYS):
        item = item_info.loc[item_id]
        state = stock.loc[(store_id, item_id)]
        sl = service_level_for(item["category"])
        policy = compute_policy(
            daily_forecast=g["forecast"].to_numpy(),
            sigma_daily=sigma_map.get((store_id, item_id), 0.0),
            on_hand=int(state["stock_on_hand"]),
            on_order=int(state["on_order"]),
            lead_time_days=int(item["lead_time_days"]),
            unit_cost=float(item["unit_cost"]),
            service_level=sl,
        )
        records.append({
            "store_id": store_id,
            "store_city": store_info.loc[store_id, "store_city"],
            "item_id": item_id,
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_cost": float(item["unit_cost"]),
            "lead_time_days": int(item["lead_time_days"]),
            "service_level": sl,
            **policy,
        })

    recs = pd.DataFrame(records)
    urgency = {"REORDER NOW": 0, "OVERSTOCK RISK": 1, "OK": 2, "NO DEMAND FORECAST": 3}
    recs = recs.assign(_rank=recs["status"].map(urgency))
    recs = recs.sort_values(["_rank", "store_id", "item_id"]).drop(columns="_rank")
    return recs[OUTPUT_COLUMNS].round(2).reset_index(drop=True)


def build_po_export(recs, order_date):
    """Purchase orders to raise today: one row per series that needs stock."""
    po = recs[recs["recommended_order_qty"] > 0].copy()
    order_day = pd.Timestamp(order_date)
    po["order_date"] = order_day.date()
    po["expected_arrival_date"] = (order_day + pd.to_timedelta(po["lead_time_days"], unit="D")).dt.date
    po["order_value_inr"] = (po["recommended_order_qty"] * po["unit_cost"]).round(0)
    columns = ["store_id", "store_city", "item_id", "item_name", "recommended_order_qty",
               "unit_cost", "order_value_inr", "order_date", "expected_arrival_date"]
    return po[columns].sort_values("order_value_inr", ascending=False).reset_index(drop=True)
