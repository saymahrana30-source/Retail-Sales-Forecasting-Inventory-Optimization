"""
reports.py
----------
Writes the plain-text reports in reports/ (Markdown, easy to read on GitHub).
"""

import config


def markdown_table(df):
    """Turn a DataFrame into a Markdown table."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def write_data_quality_report(report):
    """Save the raw-data checks as reports/data_quality_report.md."""
    lines = ["# Data quality report", "",
             "Checks run on the raw sales file before any modelling.", ""]
    for name, value in report.items():
        lines.append(f"- **{name.replace('_', ' ').capitalize()}:** {value}")
    path = config.REPORTS_DIR / "data_quality_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_model_summary(metrics_table, recs, po):
    """Save the model accuracy and inventory summary as reports/model_summary.md."""
    overall = metrics_table[metrics_table["segment_type"] == "overall"]
    by_pattern = metrics_table[metrics_table["segment_type"] == "pattern"]
    status_counts = recs["status"].value_counts().rename_axis("status").reset_index(name="series")
    top_orders = po.nlargest(10, "order_value_inr")[
        ["store_city", "item_name", "recommended_order_qty", "order_value_inr", "expected_arrival_date"]]

    cols = ["method", "MAE", "RMSE", "MASE", "WAPE_pct"]
    lines = [
        "# Model and inventory summary",
        "",
        "Accuracy comes from rolling-origin backtests (3 folds of 28 days, stockout days excluded).",
        "Lower is better for every metric. MASE below 1.0 means the method beats a naive forecast.",
        "",
        "## Overall accuracy",
        "",
        markdown_table(overall[cols]),
        "",
        "## Accuracy by demand pattern",
        "",
        markdown_table(by_pattern[["segment", "method", "MAE", "MASE", "WAPE_pct"]]),
        "",
        "## Inventory status (next 28 days)",
        "",
        markdown_table(status_counts),
        "",
        "## Ten largest purchase orders",
        "",
        markdown_table(top_orders),
        "",
    ]
    path = config.REPORTS_DIR / "model_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
