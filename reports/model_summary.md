# Model and inventory summary

Accuracy comes from rolling-origin backtests (3 folds of 28 days, stockout days excluded).
Lower is better for every metric. MASE below 1.0 means the method beats a naive forecast.

## Overall accuracy

| method | MAE | RMSE | MASE | WAPE_pct |
| --- | --- | --- | --- | --- |
| Croston/SBA | 7.689 | 12.748 | 0.79 | 39.08 |
| Hybrid (final) | 7.252 | 11.726 | 0.756 | 36.857 |
| ML model only | 7.253 | 11.726 | 0.762 | 36.864 |
| Seasonal naive | 10.21 | 16.654 | 1.018 | 51.891 |

## Accuracy by demand pattern

| segment | method | MAE | MASE | WAPE_pct |
| --- | --- | --- | --- | --- |
| intermittent | Croston/SBA | 0.58 | 0.941 | 99.798 |
| intermittent | Hybrid (final) | 0.58 | 0.941 | 99.798 |
| intermittent | ML model only | 0.59 | 0.983 | 101.554 |
| intermittent | Seasonal naive | 0.69 | 1.079 | 118.707 |
| regular | Croston/SBA | 8.757 | 0.768 | 38.845 |
| regular | Hybrid (final) | 8.254 | 0.729 | 36.614 |
| regular | ML model only | 8.254 | 0.729 | 36.614 |
| regular | Seasonal naive | 11.639 | 1.009 | 51.632 |

## Inventory status (next 28 days)

| status | series |
| --- | --- |
| OK | 39 |
| REORDER NOW | 9 |

## Ten largest purchase orders

| store_city | item_name | recommended_order_qty | order_value_inr | expected_arrival_date |
| --- | --- | --- | --- | --- |
| Delhi | Shampoo 340ml | 262 | 44540.0 | 2026-01-08 |
| Delhi | Glucose Biscuits 150g | 2135 | 36295.0 | 2026-01-05 |
| Delhi | Mixed Fruit Juice 1L | 401 | 31278.0 | 2026-01-06 |
| Delhi | Premium Trimmer | 27 | 27000.0 | 2026-01-15 |
| Jaipur | Detergent Powder 1kg | 307 | 26095.0 | 2026-01-07 |
| Bengaluru | Premium Trimmer | 25 | 25000.0 | 2026-01-15 |
| Jaipur | Premium Trimmer | 19 | 19000.0 | 2026-01-15 |
| Mumbai | Face Wash 100g | 147 | 16905.0 | 2026-01-11 |
| Bengaluru | Face Wash 100g | 140 | 16100.0 | 2026-01-11 |
