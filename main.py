"""
main.py
-------
Runs the whole project from start to finish.

    python main.py                 run everything (creates the synthetic data if missing)
    python main.py --regenerate    create a brand-new synthetic dataset first

Steps
  0. (optional) simulate a retail sales dataset
  1. load, validate and clean the raw sales data
  2. exploratory data analysis (tables + charts)
  3. build the daily panel
  4. rolling-origin backtest (an honest accuracy test)
  5. train the final model and save it to models/
  6. forecast the next 28 days
  7. inventory policy: safety stock, reorder point, EOQ, order quantities
  8. save outputs, charts and reports
"""

import argparse
import time

import joblib
import pandas as pd

import config
from src.data_quality import load_sales, validate_schema, quality_report, clean_sales
from src.eda import run_eda
from src.evaluation import run_backtest
from src.features import (SERIES_KEYS, FEATURE_COLUMNS, build_daily_panel,
                          build_future_calendar, add_features)
from src.inventory import recommend_replenishment, build_po_export
from src.models import train_ml_model, generate_forecasts
from src.reports import write_data_quality_report, write_model_summary
from src.simulate_data import generate_dataset
from src.visualize import (plot_forecast_example, plot_actual_vs_predicted,
                           plot_model_comparison, plot_reorder_chart)


def ensure_folders():
    for folder in [config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR,
                   config.OUTPUTS_DIR, config.IMAGES_DIR, config.REPORTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Retail sales forecasting and inventory optimization")
    parser.add_argument("--regenerate", action="store_true", help="create a new synthetic dataset first")
    args = parser.parse_args()

    started = time.time()
    ensure_folders()
    if args.regenerate or not config.SALES_FILE.exists():
        print("[0] Creating synthetic retail dataset ...")
        generate_dataset()

    print("[1] Loading, validating and cleaning raw data ...")
    raw = load_sales()
    validate_schema(raw)
    write_data_quality_report(quality_report(raw))
    clean = clean_sales(raw)
    print(f"    {len(clean):,} clean rows across {clean.groupby(SERIES_KEYS).ngroups} store-item series")

    print("[2] Exploratory data analysis ...")
    run_eda(clean)

    print("[3] Building the daily panel ...")
    panel = build_daily_panel(clean)

    print("[4] Rolling-origin backtest (3 folds x 28 days) ...")
    backtest, metrics, residuals = run_backtest(panel)
    metrics.to_csv(config.OUTPUTS_DIR / "backtest_metrics.csv", index=False)
    backtest.to_csv(config.OUTPUTS_DIR / "backtest_predictions.csv", index=False)
    residuals.to_csv(config.OUTPUTS_DIR / "residual_stats.csv", index=False)

    print("[5] Training the final model ...")
    train_rows = add_features(panel)
    model = train_ml_model(train_rows[train_rows["qty"].notna()])
    joblib.dump({"model": model, "features": FEATURE_COLUMNS},
                config.MODELS_DIR / "forecast_model.joblib")

    print("[6] Forecasting the next 28 days ...")
    future = build_future_calendar(panel)
    forecasts = generate_forecasts(model, panel, future)
    forecasts.to_csv(config.OUTPUTS_DIR / "forecasts_next_28_days.csv", index=False)

    print("[7] Inventory policy ...")
    items = pd.read_csv(config.ITEMS_FILE)
    stores = pd.read_csv(config.STORES_FILE)
    last_date = clean["date"].max()
    snapshot = clean.loc[clean["date"] == last_date, SERIES_KEYS + ["stock_on_hand", "on_order"]]
    recs = recommend_replenishment(forecasts, residuals, snapshot, items, stores)
    recs.to_csv(config.OUTPUTS_DIR / "replenishment_recommendations.csv", index=False)
    po = build_po_export(recs, order_date=last_date + pd.Timedelta(days=1))
    po.to_csv(config.OUTPUTS_DIR / "po_export.csv", index=False)

    print("[8] Charts and reports ...")
    names = items.set_index("item_id")["item_name"]
    stores_city = stores.set_index("store_id")["store_city"]
    plot_forecast_example(panel, forecasts, residuals, "S01", "ITM001",
                          f"{names['ITM001']} in {stores_city['S01']}: 90 days + 28-day forecast",
                          "06_forecast_regular_item.png")
    plot_forecast_example(panel, forecasts, residuals, "S03", "ITM012",
                          f"{names['ITM012']} in {stores_city['S03']}: intermittent demand forecast",
                          "07_forecast_intermittent_item.png")
    plot_actual_vs_predicted(backtest, "S01", "ITM001",
                             f"Backtest: {names['ITM001']} in {stores_city['S01']}",
                             "08_actual_vs_predicted.png")
    plot_model_comparison(metrics, "09_model_comparison.png")
    plot_reorder_chart(recs, "10_reorder_alerts.png")
    write_model_summary(metrics, recs, po)

    overall = metrics[(metrics["segment_type"] == "overall")]
    print("\nBacktest accuracy (overall, lower is better):")
    print(overall[["method", "MAE", "RMSE", "MASE", "WAPE_pct"]].to_string(index=False))
    print("\nInventory status:")
    print(recs["status"].value_counts().to_string())
    print(f"\nDone in {time.time() - started:.0f} s. See outputs/, images/ and reports/.")


if __name__ == "__main__":
    main()
