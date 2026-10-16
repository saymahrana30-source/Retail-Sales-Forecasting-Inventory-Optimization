"""
config.py
---------
Every setting for the project lives here. To change the forecast horizon,
the service level or the holding cost, edit this file and re-run main.py.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Folder paths (relative to this file, so the project works on any computer)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = PROJECT_ROOT / "images"
REPORTS_DIR = PROJECT_ROOT / "reports"

SALES_FILE = RAW_DIR / "retail_sales_daily.csv"
ITEMS_FILE = RAW_DIR / "items_master.csv"
STORES_FILE = RAW_DIR / "stores_master.csv"

# ----------------------------------------------------------------------------
# Synthetic data (simulation)
# ----------------------------------------------------------------------------
RANDOM_SEED = 42
SIM_START_DATE = "2024-01-01"
SIM_END_DATE = "2025-12-31"

# Festival calendar: known in advance, so it is safe to use as a forecast input
FESTIVAL_DATES = [(1, 1), (1, 14), (3, 25), (8, 15), (10, 2), (10, 31), (11, 1), (12, 25)]

# ----------------------------------------------------------------------------
# Forecasting
# ----------------------------------------------------------------------------
HORIZON_DAYS = 28                   # forecast 4 weeks ahead
N_BACKTEST_FOLDS = 3                # number of past 4-week windows used for testing
CLASSIFY_WINDOW_DAYS = 90           # recent window used to decide regular vs intermittent
INTERMITTENT_THRESHOLD = 0.30       # more than 30% zero-sales days -> intermittent
CROSTON_ALPHA = 0.1                 # smoothing factor for Croston / SBA
# Weight of the ML model for regular series (the rest goes to seasonal naive).
# Backtest result: blending in seasonal naive made accuracy WORSE (see docs/SIMULATION.md),
# so the final setting is 1.0 (ML only). Try 0.7 to see the blend for yourself.
ML_WEIGHT = 1.0

# ----------------------------------------------------------------------------
# Inventory policy
# ----------------------------------------------------------------------------
HOLDING_RATE = 0.20                 # yearly cost of holding stock = 20% of unit cost
ORDERING_COST = 500.0               # fixed cost of raising one purchase order (INR)
DAYS_PER_YEAR = 365
OVERSTOCK_DAYS_OF_COVER = 60        # raise an alert if stock lasts longer than this
DEFAULT_SERVICE_LEVEL = 0.95        # probability of not running out during a replenishment cycle
SERVICE_LEVEL_BY_CATEGORY = {
    "Dairy": 0.98,                  # perishable and high-traffic: protect availability
    "Bakery": 0.98,
    "Snacks": 0.95,
    "Beverages": 0.95,
    "Household": 0.92,
    "Personal Care": 0.90,          # slower movers: accept a little more risk
}
