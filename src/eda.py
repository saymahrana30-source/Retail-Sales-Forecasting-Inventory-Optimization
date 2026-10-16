"""
eda.py
------
Exploratory Data Analysis: understand the data BEFORE modelling.
Writes a summary table to outputs/ and five charts to images/.
"""

import matplotlib
matplotlib.use("Agg")          # draw charts straight to files, no window needed
import matplotlib.pyplot as plt

import config

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def build_item_summary(observed):
    """One row per item: average sales, zero-day share and promotion lift."""
    base = observed.groupby(["item_id", "item_name", "category"]).agg(
        avg_units_per_day=("qty_sold", "mean"),
        zero_day_share=("qty_sold", lambda s: (s == 0).mean()),
    ).reset_index()

    promo = observed.groupby(["item_id", "on_promo"])["qty_sold"].mean().unstack()
    promo_lift = (promo[1] / promo[0]).rename("promo_lift").reset_index()

    return base.merge(promo_lift, on="item_id", how="left").round(3)


def _save(fig, filename):
    path = config.IMAGES_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_eda_charts(clean, observed, summary):
    """Create the five EDA charts."""
    # 1) Total sales over time, with a 7-day average to show the trend
    daily = clean.groupby("date")["qty_sold"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values, alpha=0.4, label="Daily units")
    ax.plot(daily.index, daily.rolling(7).mean().values, linewidth=2, label="7-day average")
    ax.set_title("Total units sold per day, all stores and items")
    ax.set_ylabel("Units")
    ax.legend()
    _save(fig, "01_sales_trend.png")

    # 2) Category totals
    by_cat = observed.groupby("category")["qty_sold"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(by_cat.index, by_cat.values, color="tab:blue")
    ax.set_title("Units sold by category (uncensored days)")
    ax.set_xlabel("Units")
    _save(fig, "02_category_sales.png")

    # 3) Weekday pattern, scaled per series so big items do not dominate
    scaled = observed.assign(
        dow=observed["date"].dt.day_name(),
        relative=observed["qty_sold"] / observed.groupby(["store_id", "item_id"])["qty_sold"].transform("mean"),
    )
    weekday = scaled.groupby("dow")["relative"].mean().reindex(WEEKDAYS)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(weekday.index, weekday.values, color="tab:green")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Weekday pattern: demand relative to each series' average")
    ax.set_ylabel("Relative demand (1.0 = average)")
    plt.setp(ax.get_xticklabels(), rotation=20)
    _save(fig, "03_weekday_pattern.png")

    # 4) Promotion lift per item
    lift = summary.sort_values("promo_lift")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(lift["item_name"], lift["promo_lift"], color="tab:orange")
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Promotion lift: average sales on promo days / normal days")
    ax.set_xlabel("Lift (1.0 = no effect)")
    _save(fig, "04_promo_lift.png")

    # 5) Intermittency: share of zero-sales days per item
    zeros = summary.sort_values("zero_day_share")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(zeros["item_name"], zeros["zero_day_share"], color="tab:purple")
    ax.axvline(config.INTERMITTENT_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"Intermittent threshold ({config.INTERMITTENT_THRESHOLD:.0%})")
    ax.set_title("Share of zero-sales days per item (store average)")
    ax.set_xlabel("Share of days with zero sales")
    ax.legend(loc="lower right")
    _save(fig, "05_intermittency.png")


def run_eda(clean):
    """Run all EDA steps and return the item summary table."""
    observed = clean[clean["stockout_flag"] == 0]   # clean demand only
    summary = build_item_summary(observed)
    summary.to_csv(config.OUTPUTS_DIR / "eda_item_summary.csv", index=False)
    save_eda_charts(clean, observed, summary)
    return summary
