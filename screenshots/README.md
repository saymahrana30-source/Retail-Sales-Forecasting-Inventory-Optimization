# Proof screenshots

This folder holds the proof images for the project: terminal output, source code and result tables.

## What is here

| File | What it shows | How it was made |
|---|---|---|
| `01_dataset_preview.png` | First 10 days of Mumbai milk sales from the raw data | Rendered from `data/raw/retail_sales_daily.csv` |
| `02_data_quality_report.png` | Data quality checks before modelling | Rendered from `reports/data_quality_report.md` |
| `03_terminal_run.png` | Full terminal output of `python main.py` | Captured from a real run |
| `05_recommendations_table.png` | The REORDER NOW rows, with safety stock, reorder point and order size | Rendered from `outputs/replenishment_recommendations.csv` |
| `08_tests_passing.png` | Test suite output (16 tests passing) | Captured from a real run |
| `09_purchase_orders.png` | Purchase orders to raise, with values and arrival dates | Rendered from `outputs/po_export.csv` |
| `code_01_inventory_policy.png` | `compute_policy`: safety stock, reorder point, EOQ and order size | Copied from `src/inventory.py` |
| `code_02_croston_sba.png` | `croston_sba`: intermittent demand forecasting | Copied from `src/models.py` |
| `code_03_recursive_forecast.png` | `recursive_forecast`: 28-day ML forecast, one day at a time | Copied from `src/models.py` |
| `code_04_features_no_leakage.png` | `_series_features`: features that use only past data | Copied from `src/features.py` |
| `code_05_clean_sales.png` | `clean_sales`: duplicates, negatives and stockout flags | Copied from `src/data_quality.py` |
| `code_06_simulate_promotions.png` | `make_promotions`: how promotions are simulated | Copied from `src/simulate_data.py` |
| `code_07_leakage_test.png` | The test that catches data leakage | Copied from `tests/test_features.py` |

## Still to capture by hand

These need a real screen, so they are not generated:

| File | What to capture |
|---|---|
| `04_dashboard.png` | `streamlit run app/app.py`, with one store and item selected |
| `06_github_repo.png` | Your repository's main page on GitHub, after you upload |
| `07_readme_preview.png` | Your README as it renders on GitHub |

The numbers skip 04, 06 and 07 because those are the three manual captures.

## Regenerating

After any code or data change, run this from the project root (takes about 40 seconds):

```bash
python scripts/make_screenshots.py
```

It re-runs the pipeline and tests, then rebuilds every generated image from the new results. The generated images are rendered from real output, not from a screen, so they always match the code.
