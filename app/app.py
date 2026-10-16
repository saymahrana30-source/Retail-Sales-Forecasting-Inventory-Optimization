"""
app.py
------
Streamlit dashboard for planners.

Run from the project folder:
    streamlit run app/app.py

It reads the files written by `python main.py`. It does not retrain anything,
so it opens instantly.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402

FORECAST_FILE = config.OUTPUTS_DIR / "forecasts_next_28_days.csv"
RECS_FILE = config.OUTPUTS_DIR / "replenishment_recommendations.csv"

st.set_page_config(page_title="Retail Forecast & Replenishment", layout="wide")
st.title("Retail Sales Forecast & Inventory Optimization")
st.caption("Synthetic data created by the project simulator. Numbers come from `python main.py`.")

missing = [p for p in [config.SALES_FILE, FORECAST_FILE, RECS_FILE] if not p.exists()]
if missing:
    st.error("Output files not found. Run `python main.py` from the project folder first.")
    st.stop()


@st.cache_data
def load_tables():
    sales = pd.read_csv(config.SALES_FILE, parse_dates=["date"])
    forecasts = pd.read_csv(FORECAST_FILE, parse_dates=["date"])
    recs = pd.read_csv(RECS_FILE)
    return sales, forecasts, recs


sales, forecasts, recs = load_tables()

# ---- Sidebar: choose a store and an item -----------------------------------
store_names = dict(zip(recs["store_id"], recs["store_city"]))
item_names = dict(zip(recs["item_id"], recs["item_name"]))

store_id = st.sidebar.selectbox("Store", sorted(store_names), format_func=lambda s: f"{s} - {store_names[s]}")
item_id = st.sidebar.selectbox("Item", sorted(item_names), format_func=lambda i: item_names[i])

row = recs[(recs["store_id"] == store_id) & (recs["item_id"] == item_id)].iloc[0]

# ---- Key numbers for this store-item ----------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("On hand", int(row["on_hand"]))
c2.metric("Reorder point", f"{row['reorder_point']:.0f}")
c3.metric("Safety stock", f"{row['safety_stock']:.0f}")
c4.metric("Recommended order", int(row["recommended_order_qty"]))

if row["status"] == "REORDER NOW":
    st.error(f"REORDER NOW: raise a purchase order for {int(row['recommended_order_qty'])} units.")
elif row["status"] == "OVERSTOCK RISK":
    st.warning(f"Overstock risk: about {row['days_of_cover']:.0f} days of cover on hand.")
else:
    st.success("Stock level is healthy. No order needed today.")

# ---- History and forecast chart ----------------------------------------------
st.subheader("Last 90 days of sales and the next 28 days of forecast")
hist = sales[(sales["store_id"] == store_id) & (sales["item_id"] == item_id)].sort_values("date")
hist = hist.set_index("date")
actual = hist["qty_sold"].where(hist["stockout_flag"] == 0).tail(90).rename("Actual sales (stockout days hidden)")

fc = forecasts[(forecasts["store_id"] == store_id) & (forecasts["item_id"] == item_id)]
fc = fc.sort_values("date").set_index("date")["forecast"].rename("Forecast")

st.line_chart(pd.concat([actual, fc], axis=1))
pattern = forecasts.loc[(forecasts["store_id"] == store_id) & (forecasts["item_id"] == item_id), "pattern"].iloc[0]
st.caption(f"Demand pattern: {pattern}. Stockout days are hidden because true demand is unknown on those days.")

# ---- All recommendations, filterable, exportable -----------------------------
st.subheader("All stocking recommendations")
statuses = sorted(recs["status"].unique())
chosen = st.multiselect("Show status", statuses, default=statuses)
view = recs[recs["status"].isin(chosen)]
st.dataframe(view)

st.download_button(
    "Download recommendations as CSV",
    data=recs.to_csv(index=False),
    file_name="replenishment_recommendations.csv",
    mime="text/csv",
)
