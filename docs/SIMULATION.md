# Virtual Simulation Guide

This project works without access to any company system. Instead, it **creates a realistic retail world in code**, then runs the full forecasting and replenishment pipeline on it. This guide explains how that world is built, how each part of the data is produced, and what proof to capture.

---

## 1. How the simulated retail world is represented

| Concept | How it is represented | Where to find it |
|---|---|---|
| **Stores** | 4 stores, each with a city, a cluster (Metro or Tier-2) and a footfall level | `data/raw/stores_master.csv` |
| **Products** | 12 items in 6 categories, each with price, unit cost, shelf life, supplier lead time and typical daily demand | `data/raw/items_master.csv` |
| **Dates** | Every calendar day from 1 Jan 2024 to 31 Dec 2025 (731 days) | `date` column |
| **Sales** | One row per store × item × day | `data/raw/retail_sales_daily.csv` |
| **Stock levels** | End-of-day shelf stock and stock on the way | `stock_on_hand`, `on_order` |
| **Stockouts** | Days when customers wanted more than was on the shelf | `stockout_flag` |

The simulation has **35,088 rows** in total (4 × 12 × 731).

---

## 2. How demand is generated

Each day's expected demand for a store-item is built from simple, explainable parts:

```
expected demand = base demand per day
                × (store footfall ÷ 100)
                × trend (about +8% a year)
                × seasonality (category-specific, yearly cycle)
                × festival uplift (on 8 fixed festival days a year)
                × promotion lift (1 + 3.5 × discount)
```

Then **real** daily demand is drawn at random around that expected value (Poisson counts with extra randomness). That is why the data is noisy, as real sales are.

| Pattern | How it is simulated | Why it matters for the model |
|---|---|---|
| **Weekly pattern** | Saturdays and Sundays are busiest (footfall up to +25%), Fridays +5% | The model must learn weekday effects |
| **Yearly seasonality** | Drinks peak around mid-May (summer). Snacks peak around late October (festive). Household peaks near the end of the year | Lag and calendar features must capture it |
| **Festivals** | 8 dates a year lift sales (for example, Snacks ×1.35) | A known calendar input |
| **Trend** | About 8% growth a year | Recent history is more relevant than old history |
| **Promotions** | About one 3-day promotion every 10 weeks per store-item, 10–25% off | Promotions raise demand, and the model should learn the lift |
| **Price change** | A 4% list-price increase on 1 January 2025 | Price is a feature |
| **Intermittent demand** | The premium trimmer sells about 0.25 units a day; face wash sells about 1 unit a day | Needs Croston/SBA, not a plain regression |
| **Supplier delays** | 5% of orders arrive 2 days late | Real-world uncertainty in lead times |

---

## 3. How stockouts are simulated

Each store follows a simple reorder rule every day:

1. Stock arriving today is added to the shelf.
2. Demand for the day is drawn. If it is more than the shelf holds, a **stockout** is recorded and only the units on the shelf are sold.
3. If (stock on shelf + stock on the way) falls to the reorder point, an order is placed. It arrives after the supplier's lead time, sometimes 2 days later.

Because store planners use an average demand figure, summer, festival and promotion peaks sometimes exceed what was ordered. The result is about **2.85%** of store-item-days with a stockout, which matches the kind of gaps seen in real point-of-sale records.

**Why this matters:** on a stockout day the sales figure is *lower than true demand*. If the model learned from these days as if they were normal, it would under-forecast. The project therefore keeps stockout days out of training and scoring.

---

## 4. How the forecast is generated

1. **Clean** the data and remove stockout days from the target.
2. **Build the daily panel:** every store-item series on every calendar day.
3. **Classify each series** using the last 90 days: if more than 30% of days had zero sales, it is *intermittent*.
4. **Forecast the next 28 days:**
   - Intermittent series use Croston/SBA.
   - Regular series use the gradient-boosting model, run recursively one day at a time.
5. **Score the method** with backtests: pretend to be at 3 past dates, forecast 28 days each time, and compare with what really happened.

---

## 5. How inventory decisions are made

For each store-item, using the forecast and the backtest error:

| Quantity | Meaning |
|---|---|
| **Lead-time demand** (μL) | Units expected to sell before a new order would arrive |
| **Safety stock** (SS) | Extra buffer for forecast error, sized to the service level |
| **Reorder point** (ROP) | μL + SS. When stock falls to this level, order |
| **EOQ** | The order size that balances ordering cost against holding cost |
| **Order-up-to** (S) | ROP + EOQ |
| **Recommended order** | S minus the current position, placed only when the position is at or below ROP |

Worked example from the real output (Delhi, Glucose Biscuits 150g):

| Step | Calculation | Result |
|---|---|---|
| Lead time | Supplier takes 4 days | L = 4 |
| Lead-time demand | Sum of forecasts for days 1–4 | 51.20 + 40.12 + 47.94 + 48.87 = **188.13** |
| Safety stock | 1.645 × 18.40 × √4 | **60.54** |
| Reorder point | 188.13 + 60.54 | **248.67** |
| EOQ | √(2 × 14,990 × 500 ÷ 3.40) | **≈ 2,100** |
| Order-up-to | 248.67 + 2,099.7 | **2,348.37** |
| Position | 214 on hand, 0 on order | 214 |
| Decision | 214 ≤ 248.67, so order | **Order 2,135 units** |

You can check this yourself. Open `outputs/replenishment_recommendations.csv`, find the row, and reproduce the numbers with a calculator.

---

## 6. How the output is visualised

| Output | Where | What it shows |
|---|---|---|
| EDA charts | `images/01–05` | Trends, categories, weekday pattern, promotions, intermittency |
| Forecast charts | `images/06–07` | History and forecast for a regular and an intermittent item |
| Backtest charts | `images/08–09` | Predictions against actuals, and error by method |
| Reorder chart | `images/10` | Position against reorder point for all 48 series |
| Dashboard | `streamlit run app/app.py` | Interactive view for a chosen store and item |
| Tables | `outputs/*.csv` | Forecasts, recommendations, purchase orders, metrics |

---

## 7. Step-by-step simulation workflow

Follow these steps in order. Each one ends with something you can check.

**Step 1: Set up the project.**
Create the virtual environment and install the libraries (see README). *Check:* `python -c "import pandas, sklearn, streamlit"` runs without error.

**Step 2: Read the settings.**
Open `config.py`. Note the horizon (28 days), service levels, holding rate and ordering cost. *Check:* you can explain what `ORDERING_COST = 500` means.

**Step 3: Create the synthetic dataset.**
Run `python src/simulate_data.py` or `python main.py --regenerate`. *Check:* `data/raw/retail_sales_daily.csv` has 35,088 rows.

**Step 4: Inspect the master tables.**
Open `items_master.csv` and `stores_master.csv`. Compare the base demand of milk (38/day) with the premium trimmer (0.25/day). *Check:* you can say which items are intermittent before running any model.

**Step 5: Look at one store-item's raw sales.**
Filter the sales file to `store_id = S01` and `item_id = ITM001`. Find a `stockout_flag = 1` row and confirm `stock_on_hand = 0` on that day. *Check:* the stockout rule holds.

**Step 6: Run the cleaning and data-quality checks.**
`python main.py` writes `reports/data_quality_report.md`. *Check:* duplicates 0, missing 0, negative 0, stockout rate 2.85%.

**Step 7: Run exploratory analysis.**
The EDA charts are in `images/01–05`. *Check:* the weekday chart peaks on weekends, and the beverages line rises in summer.

**Step 8: Build the daily panel and features.**
This happens automatically inside `main.py`. *Check:* `python tests/run_tests.py` passes the leakage and panel tests.

**Step 9: Run the honest backtest.**
Read `outputs/backtest_metrics.csv`. *Check:* the ML model and the hybrid beat the seasonal-naive baseline (MASE below 1.0).

**Step 10: Train the final model.**
The model is saved to `models/forecast_model.joblib` (not committed to GitHub, because it is large and regenerated each run).

**Step 11: Generate the 28-day forecast and recommendations.**
Open `outputs/forecasts_next_28_days.csv` and `outputs/replenishment_recommendations.csv`. *Check:* the forecast follows the weekly rhythm, and the recommended orders match the worked example in section 5.

**Step 12: Review the purchase orders.**
Open `outputs/po_export.csv`. The largest order is Shampoo at Delhi (262 units, INR 44,540, arriving 8 January 2026). *Check:* order value equals quantity × unit cost.

**Step 13: Open the dashboard.**
Run `streamlit run app/app.py`. Choose a store and item, and check that the status message matches the recommendations table. *Check:* the CSV download works.

**Step 14: Try a what-if.**
Change `ORDERING_COST` from 500 to 100 in `config.py`, then run `python main.py`. Notice how EOQ and the order quantities fall. Change it back. *Check:* you can explain why cheaper orders mean smaller batches.

---

## 8. Proof to capture (screenshots, graphs, tables, recordings)

**Proof images (in `images/screenshots/`):** most are already generated by `python scripts/make_screenshots.py`. The index is in `images/screenshots/README.md`.

| File | Status | What it shows |
|---|---|---|
| `01_dataset_preview.png` | generated | First 10 days of one store-item in the raw data |
| `02_data_quality_report.png` | generated | Data quality checks before modelling |
| `03_terminal_run.png` | generated | Terminal output of `python main.py` |
| `04_dashboard.png` | capture by hand | Streamlit dashboard with an item selected |
| `05_recommendations_table.png` | generated | REORDER NOW rows with the stocking numbers |
| `06_github_repo.png` | capture by hand, after upload | Your repository's main page on GitHub |
| `07_readme_preview.png` | capture by hand, after upload | Your README as rendered on GitHub |
| `08_tests_passing.png` | generated | Test suite output |
| `09_purchase_orders.png` | generated | Purchase orders to raise |
| `code_01` to `code_07` | generated | Key functions copied from the source files |

**Graphs (generated automatically, in `images/`):** sales trend, category sales, weekday pattern, promo lift, intermittency, forecast (regular and intermittent), actual vs predicted, model comparison, reorder alerts.

**Tables (in `outputs/`):** forecasts, recommendations, purchase orders, backtest metrics, residual statistics.

**Screen recordings (optional but strong):**
1. A 60-second recording of `python main.py` running from start to finish.
2. A 60-second recording of the dashboard: pick a store, pick a REORDER NOW item, read the recommended order, and download the CSV. Save as `docs/demo.gif` or upload to YouTube and link it in the README.

**Outputs to commit to GitHub:** all of `outputs/`, all of `images/` (charts), all of `reports/`, `data/raw/`, and the code. Do not commit `models/*.joblib` (already in `.gitignore`).
