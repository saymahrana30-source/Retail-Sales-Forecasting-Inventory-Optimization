"""
make_screenshots.py
-------------------
Creates the proof images in images/screenshots/.

What it makes
  01_dataset_preview.png        spreadsheet-style preview of the raw sales data
  02_data_quality_report.png    the data quality report, shown as a terminal printout
  03_terminal_run.png           the terminal output of `python main.py`
  08_tests_passing.png          the terminal output of the test suite
  09_purchase_orders.png        the purchase orders, as a spreadsheet-style table
  code_*.png                    key functions, copied from the real source files
  05_recommendations_table.png  the REORDER NOW rows of the recommendations file

Every image is rendered from the real run output and the real files, so it always
matches the code. These are rendered images, not operating-system screenshots.
The dashboard, the GitHub page and the README preview need a real screen capture.

Usage, from the project root (takes about 40 seconds because it re-runs the pipeline):
    python scripts/make_screenshots.py
"""

import ast
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images" / "screenshots"

# Colours loosely follow a dark code editor theme
BG, BAR, FG, GREY = "#1e1e1e", "#323232", "#d4d4d4", "#858585"
GREEN, RED, BLUE, ORANGE, LIGHT_GREEN, PROMPT = "#6a9955", "#f14c4c", "#569cd6", "#ce9178", "#b5cea8", "#4ec9b0"

FONT_SIZE = 10
ADVANCE = 0.602                       # width of one character in DejaVu Sans Mono, in em
CHAR_W = ADVANCE * FONT_SIZE / 72     # inches per character
LINE_H = 1.5 * FONT_SIZE / 72         # inches per line

KEYWORDS = ("def|return|if|elif|else|for|in|import|from|class|with|as|not|and|or|None|True|"
            "False|try|except|lambda|while|break|continue|pass|raise|yield")
TOKEN = re.compile(
    r"(#.*$)"                        # comment
    r"|(\"[^\"]*\"|'[^']*')"         # string
    r"|\b(" + KEYWORDS + r")\b"      # keyword
    r"|\b(\d+(?:\.\d+)?)\b"          # number
)


def run(command):
    """Run a command in the project folder and return its printed lines and exit code."""
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    text = (result.stdout + result.stderr).rstrip()
    return text.splitlines(), result.returncode


def _canvas(width_chars, n_lines, title):
    """A dark window with a title bar, sized to fit the text."""
    width = width_chars * CHAR_W + 0.8
    height = 0.35 + 0.2 + n_lines * LINE_H + 0.25
    fig = plt.figure(figsize=(width, height), facecolor=BG, dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)            # y grows downwards, like a page
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, 0.35, color=BAR))
    for i, colour in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(Circle((0.25 + i * 0.22, 0.175), 0.065, color=colour))
    ax.text(width / 2, 0.175, title, color=GREY, fontsize=9, ha="center", va="center")
    return fig, ax


def _save(fig, filename):
    path = OUT / filename
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def terminal_colour(line):
    if line.startswith(("$", "(.venv)")):
        return PROMPT
    if re.search(r"Traceback|Error|\bFAIL\b", line):
        return RED
    if "PASS" in line or "0 failed" in line:
        return GREEN
    return FG


def terminal_image(lines, filename, title):
    """Plain terminal printout, colouring prompts, passes and errors."""
    width_chars = max(len(line.replace("\t", "    ")) for line in lines) + 2
    fig, ax = _canvas(width_chars, len(lines), title)
    for i, line in enumerate(lines):
        ax.text(0.4, 0.55 + i * LINE_H, line.replace("\t", "    "), color=terminal_colour(line),
                fontsize=FONT_SIZE, family="DejaVu Sans Mono", va="top", ha="left")
    return _save(fig, filename)


def colour_for(match):
    if match.group(1):
        return GREEN
    if match.group(2):
        return ORANGE
    if match.group(3):
        return BLUE
    return LIGHT_GREEN


def segments(code):
    """Split one line of Python into (text, colour) pieces for simple syntax colouring."""
    pieces, pos = [], 0
    for m in TOKEN.finditer(code):
        if m.start() > pos:
            pieces.append((code[pos:m.start()], FG))
        pieces.append((m.group(0), colour_for(m)))
        pos = m.end()
    if pos < len(code):
        pieces.append((code[pos:], FG))
    return pieces


def code_image(lines, first_line_number, filename, title):
    """Source code with line numbers and simple syntax colouring."""
    lines = [line.replace("\t", "    ") for line in lines]
    width_chars = max(len(line) for line in lines) + 8        # room for the line numbers
    fig, ax = _canvas(width_chars, len(lines), title)
    for i, line in enumerate(lines):
        y = 0.55 + i * LINE_H
        ax.text(0.4, y, f"{first_line_number + i:>4}", color=GREY,
                fontsize=FONT_SIZE, family="DejaVu Sans Mono", va="top", ha="left")
        x0, col = 0.4 + 6 * CHAR_W, 0
        for text, colour in segments(line):
            ax.text(x0 + col * CHAR_W, y, text, color=colour, fontsize=FONT_SIZE,
                    family="DejaVu Sans Mono", va="top", ha="left")
            col += len(text)
    return _save(fig, filename)


def function_source(relative_path, name):
    """Return the lines of a function or test, and the line it starts on."""
    path = ROOT / relative_path
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            return lines[start - 1:node.end_lineno], start
    raise KeyError(f"{name} not found in {relative_path}")


def table_image(df, filename, title, font_size=9):
    """Spreadsheet-style table, as a PNG."""
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(max(6, 1.35 * n_cols), max(1.5, 0.33 * (n_rows + 1) + 0.9)), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=df.astype(str).values, colLabels=list(df.columns),
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.auto_set_column_width(list(range(n_cols)))
    table.scale(1, 1.3)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9e1f2")
            cell.set_text_props(weight="bold")
    ax.set_title(title, fontsize=10, loc="left")
    fig.savefig(OUT / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return OUT / filename


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    made = []

    # 1) Run the real pipeline and keep its printout
    print("Running main.py (about 40 seconds) ...")
    lines, code = run([sys.executable, "main.py"])
    if code != 0:
        raise SystemExit("main.py failed, so no screenshots were made:\n" + "\n".join(lines[-20:]))
    made.append(terminal_image(["(.venv) $ python main.py", *lines],
                               "03_terminal_run.png", "Terminal: python main.py"))

    # 2) Run the real tests and keep their printout
    print("Running the tests ...")
    lines, _ = run([sys.executable, "tests/run_tests.py"])
    made.append(terminal_image(["(.venv) $ python tests/run_tests.py", *lines],
                               "08_tests_passing.png", "Terminal: running the test suite"))

    # 3) The data quality report, as a printout
    report = (ROOT / "reports" / "data_quality_report.md").read_text(encoding="utf-8").splitlines()
    made.append(terminal_image(["$ cat reports/data_quality_report.md", *report],
                               "02_data_quality_report.png", "Terminal: data quality report"))

    # 4) Spreadsheet-style previews of the real data and results
    sales = pd.read_csv(ROOT / "data/raw/retail_sales_daily.csv")
    preview = sales[(sales["store_id"] == "S01") & (sales["item_id"] == "ITM001")].head(10)
    preview = preview[["date", "store_city", "item_name", "qty_sold", "price", "on_promo",
                       "stock_on_hand", "stockout_flag"]]
    made.append(table_image(preview, "01_dataset_preview.png",
                            "retail_sales_daily.csv: Mumbai, Full Cream Milk, first 10 days"))

    recs = pd.read_csv(ROOT / "outputs/replenishment_recommendations.csv")
    urgent = recs[recs["status"] == "REORDER NOW"][[
        "store_city", "item_name", "on_hand", "on_order", "reorder_point",
        "safety_stock", "eoq", "recommended_order_qty", "days_of_cover"]]
    made.append(table_image(urgent, "05_recommendations_table.png",
                            "replenishment_recommendations.csv: REORDER NOW rows"))

    po = pd.read_csv(ROOT / "outputs/po_export.csv").head(9)
    made.append(table_image(po[["store_city", "item_name", "recommended_order_qty", "unit_cost",
                                "order_value_inr", "expected_arrival_date"]],
                            "09_purchase_orders.png", "po_export.csv: purchase orders to raise"))

    # 5) Key code, copied from the real files
    snippets = [
        ("src/inventory.py", "compute_policy", "code_01_inventory_policy.png",
         "src/inventory.py: compute_policy"),
        ("src/models.py", "croston_sba", "code_02_croston_sba.png",
         "src/models.py: croston_sba"),
        ("src/models.py", "recursive_forecast", "code_03_recursive_forecast.png",
         "src/models.py: recursive_forecast"),
        ("src/features.py", "_series_features", "code_04_features_no_leakage.png",
         "src/features.py: _series_features"),
        ("src/data_quality.py", "clean_sales", "code_05_clean_sales.png",
         "src/data_quality.py: clean_sales"),
        ("src/simulate_data.py", "make_promotions", "code_06_simulate_promotions.png",
         "src/simulate_data.py: make_promotions"),
        ("tests/test_features.py", "test_features_do_not_leak_future_information",
         "code_07_leakage_test.png", "tests/test_features.py: leakage test"),
    ]
    for path, name, filename, title in snippets:
        body, start = function_source(path, name)
        made.append(code_image(body, start, filename, title))

    print("\nCreated:")
    for path in made:
        print("  ", Path(path).relative_to(ROOT))
    print(f"\n{len(made)} images in {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
