"""
visualize.py
------------
Charts that show forecast quality and reorder decisions.
All charts are saved to images/ so they can go straight into the README.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import config


def _save(fig, filename):
    path = config.IMAGES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_forecast_example(panel, forecasts, residuals, store_id, item_id, title, filename):
    """Last 90 days of actual sales followed by the 28-day forecast."""
    last = panel["date"].max()
    hist = panel[(panel["store_id"] == store_id) & (panel["item_id"] == item_id)
                 & (panel["date"] > last - pd.Timedelta(days=90))]
    fc = forecasts[(forecasts["store_id"] == store_id) & (forecasts["item_id"] == item_id)]
    fc = fc.sort_values("date")
    sigma = float(residuals.loc[(residuals["store_id"] == store_id)
                                & (residuals["item_id"] == item_id), "sigma_daily"].iloc[0])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hist["date"], hist["qty"], marker="o", markersize=3, linewidth=1,
            label="Actual sales (stockout days left out)")
    ax.plot(fc["date"], fc["forecast"], color="tab:orange", linewidth=2.2, label="Forecast (hybrid)")
    ax.fill_between(fc["date"], np.clip(fc["forecast"] - 1.96 * sigma, 0, None),
                    fc["forecast"] + 1.96 * sigma, color="tab:orange", alpha=0.2,
                    label="Approx. 95% range")
    ax.axvline(last, color="grey", linestyle="--", linewidth=1, label="Forecast starts")
    ax.set_title(title)
    ax.set_ylabel("Units per day")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_actual_vs_predicted(backtest, store_id, item_id, title, filename):
    """Backtest: what we predicted at each origin vs what actually happened."""
    s = backtest[(backtest["store_id"] == store_id) & (backtest["item_id"] == item_id)]
    s = s.sort_values("date")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(s["date"], s["qty"], "o", markersize=3, label="Actual sales")
    ax.plot(s["date"], s["forecast"], "-", linewidth=1.8, color="tab:orange", label="Hybrid forecast")
    ax.set_title(title)
    ax.set_ylabel("Units per day")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, filename)


def plot_model_comparison(metrics_table, filename):
    """WAPE by method, split by demand pattern (lower bars are better)."""
    t = metrics_table[metrics_table["segment_type"] == "pattern"]
    pivot = t.pivot(index="method", columns="segment", values="WAPE_pct")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("WAPE % (lower is better)")
    ax.set_xlabel("")
    ax.set_ylim(0, pivot.values.max() * 1.3)          # leave room for the legend
    ax.set_title("Backtest error by method and demand pattern")
    ax.legend(title="Demand pattern", loc="upper center", ncol=2)
    return _save(fig, filename)


def plot_reorder_chart(recs, filename):
    """Inventory position vs reorder point. REORDER NOW rows are drawn at the top."""
    urgency = {"REORDER NOW": 0, "OVERSTOCK RISK": 1, "OK": 2, "NO DEMAND FORECAST": 3}
    # barh draws the first row at the bottom, so sort the most urgent rows LAST
    r = recs.assign(_u=recs["status"].map(urgency))
    r = r.sort_values(["_u", "days_of_cover"], ascending=[False, False])
    labels = (r["store_city"] + " | " + r["item_name"]).tolist()
    all_colours = {"REORDER NOW": "tab:red", "OVERSTOCK RISK": "tab:purple",
                   "OK": "tab:green", "NO DEMAND FORECAST": "grey"}
    palette = {k: v for k, v in all_colours.items() if k in set(r["status"])}
    colors = [all_colours.get(status, "grey") for status in r["status"]]
    y = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, 11))
    ax.barh(y, r["inventory_position"].tolist(), color=colors)
    ax.scatter(r["reorder_point"].tolist(), y, marker="|", s=220, color="black", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-1, len(labels))

    handles = [Patch(color=c, label=s) for s, c in palette.items()]
    handles.append(Line2D([0], [0], marker="|", color="black", linestyle="None",
                          markersize=12, label="Reorder point"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.set_title("Inventory position vs reorder point (bar colour = status)")
    ax.set_xlabel("Units (on hand + on order)")
    return _save(fig, filename)
