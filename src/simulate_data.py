"""
simulate_data.py
----------------
Creates a realistic SYNTHETIC retail dataset, so the project runs with no
real company data.

What is simulated
- Stores: 4 stores with different footfall (how many customers walk in).
- Items: 12 products in 6 categories. Milk sells every day; a premium trimmer
  sells only a few times a month (this is "intermittent" demand).
- Calendar effects: weekend footfall, a summer peak for drinks, a festive
  peak for snacks, and 8 festival days a year when sales jump.
- Trend: about 8% growth per year. Prices rise 4% on 1 Jan 2025.
- Promotions: random 3-day discount events that lift demand.
- Inventory: each store reorders with a simple rule, and suppliers take
  several days to deliver. When the shelf runs empty we record only what was
  sold and set stockout_flag = 1. Real point-of-sale data looks exactly like
  this, which is why cleaning matters.
"""

import numpy as np
import pandas as pd

import config
from src.features import holiday_flags

ITEM_COLUMNS = [
    "item_id", "item_name", "category", "brand", "pack_size",
    "shelf_life_days", "base_price", "unit_cost", "lead_time_days",
    "base_demand_per_day",
]

# item_id, name, category, brand, pack size, shelf life (days), price (INR),
# unit cost (INR), supplier lead time (days), average units/day in an average store
ITEMS = [
    ("ITM001", "Full Cream Milk 1L", "Dairy", "FreshFarm", "1 L", 7, 66, 52, 2, 38.0),
    ("ITM002", "Fresh Curd 400g", "Dairy", "FreshFarm", "400 g", 10, 45, 32, 2, 16.0),
    ("ITM003", "White Bread 400g", "Bakery", "BakeHouse", "400 g", 5, 40, 28, 1, 22.0),
    ("ITM004", "Glucose Biscuits 150g", "Snacks", "CrunchCo", "150 g", 180, 25, 17, 4, 28.0),
    ("ITM005", "Masala Chips 90g", "Snacks", "CrunchCo", "90 g", 150, 20, 13, 4, 20.0),
    ("ITM006", "Instant Noodles 70g", "Snacks", "NoodleNest", "70 g", 240, 15, 10, 5, 30.0),
    ("ITM007", "Cola 750ml", "Beverages", "FizzCo", "750 ml", 270, 40, 27, 3, 20.0),
    ("ITM008", "Mixed Fruit Juice 1L", "Beverages", "JuiceJoy", "1 L", 180, 110, 78, 5, 6.0),
    ("ITM009", "Shampoo 340ml", "Personal Care", "SilkCare", "340 ml", 730, 240, 170, 7, 4.5),
    ("ITM010", "Face Wash 100g", "Personal Care", "GlowSkin", "100 g", 730, 180, 115, 10, 1.0),
    ("ITM011", "Detergent Powder 1kg", "Household", "CleanPro", "1 kg", 1095, 120, 85, 6, 5.0),
    ("ITM012", "Premium Trimmer", "Personal Care", "ShaveTech", "1 unit", 1095, 1499, 1000, 14, 0.25),
]

STORE_COLUMNS = ["store_id", "store_city", "cluster", "footfall_base"]
STORES = [
    ("S01", "Mumbai", "Metro", 1.20),
    ("S02", "Delhi", "Metro", 1.15),
    ("S03", "Bengaluru", "Metro", 1.00),
    ("S04", "Jaipur", "Tier-2", 0.75),
]

# Yearly seasonality per category: (strength, peak day-of-year, festival uplift)
CATEGORY_SEASONALITY = {
    "Dairy": (0.05, 90, 1.25),
    "Bakery": (0.03, 30, 1.15),
    "Snacks": (0.12, 300, 1.35),
    "Beverages": (0.35, 135, 1.10),     # summer peak around mid-May
    "Personal Care": (0.04, 20, 1.10),
    "Household": (0.06, 330, 1.20),
}

DOW_FOOTFALL = np.array([0.90, 0.90, 0.95, 1.00, 1.05, 1.20, 1.25])  # Mon ... Sun
PRICE_INCREASE_DATE = pd.Timestamp("2025-01-01")
PRICE_INCREASE_PCT = 0.04
TREND_PER_YEAR = 0.08
PROMO_WEEKLY_PROB = 0.10          # about one promotion every 10 weeks per store-item
PROMO_LENGTH_DAYS = 3
PROMO_DISCOUNTS = [0.10, 0.15, 0.20, 0.25]
PROMO_LIFT_PER_DISCOUNT = 3.5     # 20% off -> demand x 1.70
SUPPLIER_DELAY_PROB = 0.05        # 5% of orders arrive late
SUPPLIER_DELAY_DAYS = 2
GAMMA_SHAPE = 6.0                 # higher = less random noise around the expected demand


def build_master_tables():
    """Return the item and store master tables as DataFrames."""
    items = pd.DataFrame(ITEMS, columns=ITEM_COLUMNS)
    stores = pd.DataFrame(STORES, columns=STORE_COLUMNS)
    return items, stores


def make_promotions(n_days, rng):
    """Random 3-day discount events, roughly one every 10 weeks."""
    on_promo = np.zeros(n_days, dtype=int)
    discount = np.zeros(n_days)
    for week_start in range(0, n_days, 7):
        if rng.random() < PROMO_WEEKLY_PROB:
            start = week_start + rng.integers(0, 7)
            if start >= n_days:
                continue
            end = min(start + PROMO_LENGTH_DAYS, n_days)
            on_promo[start:end] = 1
            discount[start:end] = rng.choice(PROMO_DISCOUNTS)
    return on_promo, discount


def simulate_series(store, item, dates, footfall, rng):
    """Simulate demand, price and stock for one store x item over all dates."""
    n_days = len(dates)
    day = np.arange(n_days)
    doy = dates.dayofyear.to_numpy()

    # 1) Expected demand for each day, built from simple, explainable parts
    strength, peak_doy, festival_uplift = CATEGORY_SEASONALITY[item["category"]]
    season = 1 + strength * np.cos(2 * np.pi * (doy - peak_doy) / 365.25)
    trend = 1 + TREND_PER_YEAR * day / 365.0
    holidays = holiday_flags(dates)
    festival = np.where(holidays == 1, festival_uplift, 1.0)
    on_promo, discount = make_promotions(n_days, rng)
    promo_lift = 1 + PROMO_LIFT_PER_DISCOUNT * discount
    expected = (item["base_demand_per_day"] * (footfall / 100.0) * trend
                * season * festival * promo_lift)

    # 2) Real daily demand: random counts around the expected value
    demand = rng.poisson(expected * rng.gamma(GAMMA_SHAPE, 1.0 / GAMMA_SHAPE, n_days))

    # 3) Price: list price (with 2025 increase), minus any promotion discount
    list_price = item["base_price"] * np.where(dates >= PRICE_INCREASE_DATE, 1 + PRICE_INCREASE_PCT, 1.0)
    price = np.round(list_price * (1 - discount), 2)

    # 4) Store stock: a simple reorder rule with supplier lead time
    lead = int(item["lead_time_days"])
    planner_mean = item["base_demand_per_day"] * store["footfall_base"]
    reorder_point = max(1, int(round(planner_mean * (lead + 2))))
    order_up_to = max(reorder_point + 1, int(round(planner_mean * (lead + 10))))

    on_hand = order_up_to
    on_order = 0
    arrivals = np.zeros(n_days + 60, dtype=int)   # deliveries scheduled per day
    sold_arr = np.zeros(n_days, dtype=int)
    stockout_arr = np.zeros(n_days, dtype=int)
    stock_arr = np.zeros(n_days, dtype=int)
    order_arr = np.zeros(n_days, dtype=int)

    for t in range(n_days):
        received = int(arrivals[t])
        on_hand += received
        on_order -= received

        d = int(demand[t])
        stockout_arr[t] = int(d > on_hand)     # customers wanted more than we had
        sold = min(d, on_hand)
        on_hand -= sold
        sold_arr[t] = sold

        position = on_hand + on_order          # stock on shelf + stock on the way
        if position <= reorder_point:
            qty = order_up_to - position
            delay = SUPPLIER_DELAY_DAYS if rng.random() < SUPPLIER_DELAY_PROB else 0
            arrivals[t + lead + delay] += qty
            on_order += qty

        stock_arr[t] = on_hand
        order_arr[t] = on_order

    return pd.DataFrame({
        "date": dates,
        "store_id": store["store_id"],
        "store_city": store["store_city"],
        "item_id": item["item_id"],
        "item_name": item["item_name"],
        "category": item["category"],
        "qty_sold": sold_arr,
        "price": price,
        "discount_pct": np.round(discount * 100, 1),
        "on_promo": on_promo,
        "holiday_flag": holidays,
        "footfall_index": footfall,
        "stock_on_hand": stock_arr,
        "on_order": order_arr,
        "stockout_flag": stockout_arr,
    })


def simulate_sales(items, stores, start_date=config.SIM_START_DATE,
                   end_date=config.SIM_END_DATE, seed=config.RANDOM_SEED):
    """Simulate the full store x item x day table."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    dow = dates.dayofweek.to_numpy()

    frames = []
    for _, store in stores.iterrows():
        # Footfall is shared by every item sold in the same store on the same day
        footfall = np.round(100 * store["footfall_base"] * DOW_FOOTFALL[dow]
                            * rng.lognormal(0, 0.05, len(dates))).astype(int)
        for _, item in items.iterrows():
            frames.append(simulate_series(store, item, dates, footfall, rng))

    sales = pd.concat(frames, ignore_index=True)
    return sales.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)


def generate_dataset():
    """Create the synthetic sales, item and store files in data/raw/."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    items, stores = build_master_tables()
    sales = simulate_sales(items, stores)
    sales.to_csv(config.SALES_FILE, index=False)
    items.to_csv(config.ITEMS_FILE, index=False)
    stores.to_csv(config.STORES_FILE, index=False)
    print(f"Saved {len(sales):,} sales rows to {config.SALES_FILE}")
    return sales, items, stores


if __name__ == "__main__":
    generate_dataset()
