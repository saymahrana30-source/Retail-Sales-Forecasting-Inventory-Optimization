# Troubleshooting Guide

Each entry gives the **reason**, the **solution** and a **prevention tip**.

---

## Installation errors

**Problem: `python` is not recognised, or opens the Microsoft Store.**
- Reason: Python is not on the PATH, or Windows is using a placeholder.
- Solution: try `py --version` (Windows launcher). If that works, use `py` in place of `python`, for example `py -m venv .venv`. Otherwise reinstall Python from python.org and tick "Add Python to PATH".
- Prevention: check `python --version` before starting.

**Problem: `running scripts is disabled on this system` when activating the venv in PowerShell.**
- Reason: PowerShell blocks scripts by default.
- Solution: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, answer `Y`, then activate again with `.venv\Scripts\activate`.
- Prevention: do this once on a new machine.

**Problem: `pip install` fails with a permission error.**
- Reason: Python is installed system-wide and you are not in a virtual environment.
- Solution: create and activate `.venv` first (see README). As a last resort, `pip install --user -r requirements.txt`.
- Prevention: always create a virtual environment for each project.

**Problem: `Microsoft Visual C++ 14.0 is required` while installing a package.**
- Reason: a package has no pre-built wheel for your Python version and must be compiled.
- Solution: upgrade pip (`python -m pip install --upgrade pip`), then reinstall. This project's libraries all have pre-built wheels for recent Python versions. If you use a very new or old Python, install Python 3.11 or 3.12 instead.
- Prevention: use Python 3.11 or 3.12.

**Problem: `streamlit: command not found` after installing.**
- Reason: the virtual environment is not active, so the installed copy is not on the PATH.
- Solution: activate the environment, then run `streamlit run app/app.py`. Or use `python -m streamlit run app/app.py`.
- Prevention: the prompt should show `(.venv)` before you run anything.

---

## Library conflicts

**Problem: `ImportError` or `AttributeError` mentioning pandas or numpy after an upgrade.**
- Reason: a library was built against an older version of another library (a binary mismatch).
- Solution: reinstall cleanly inside the environment: `pip install --upgrade --force-reinstall numpy pandas scikit-learn`, then `pip install -r requirements.txt`.
- Prevention: keep one virtual environment per project and do not install into the global Python.

**Problem: a warning or error about a removed or renamed argument.**
- Reason: pandas or scikit-learn changed a function's arguments between versions.
- Solution: check the version (`python -c "import pandas, sklearn; print(pandas.__version__, sklearn.__version__)"`). This project was tested with pandas 3.0 and scikit-learn 1.8. Upgrading to those versions usually resolves it.
- Prevention: pin versions with `pip freeze > requirements.txt` once everything works.

---

## Dataset loading issues

**Problem: `FileNotFoundError: data/raw/retail_sales_daily.csv`.**
- Reason: `main.py` was run before the dataset existed, or from a different folder.
- Solution: run `python main.py` from the project root (the folder that contains `main.py`). The first run creates the dataset automatically. You can also run `python main.py --regenerate`.
- Prevention: always start commands from the project root. Check with `pwd` (Mac/Linux) or `cd` (Windows).

**Problem: `ModuleNotFoundError: No module named 'src'` or `'config'`.**
- Reason: Python is started from a different folder, so the project root is not on the import path.
- Solution: run from the project root, using `python main.py` or `python -m pytest`. For the dashboard, `streamlit run app/app.py` is fine because the app adds the project root itself.
- Prevention: do not move `main.py` out of the project root.

**Problem: `ValueError: Sales file is missing required columns`.**
- Reason: you replaced the CSV with a file that lacks `date`, `store_id`, `item_id` or `qty_sold`.
- Solution: restore the file with `python main.py --regenerate`, or rename your columns to match.
- Prevention: keep your own data in a separate file and convert it to this layout first.

---

## Date parsing issues

**Problem: dates are read as text, so the time-based features are wrong or the code fails.**
- Reason: the date column was not parsed as a date.
- Solution: the project uses `parse_dates=["date"]` in `load_sales()`. If you load the file another way, pass the same argument. Check with `df["date"].dtype` (it should show `datetime64`).
- Prevention: always parse dates when you read the file, and check the dtype.

**Problem: a date range is missing days, or features look shifted.**
- Reason: the data has gaps (missing days), so "7 rows ago" is not "7 days ago".
- Solution: the daily panel fills every calendar day with `build_daily_panel()`. Do not compute lags on raw rows without filling gaps first.
- Prevention: check `panel.groupby(["store_id","item_id"]).size()` has one value (a test already does this).

---

## Model training issues

**Problem: the model trains but accuracy is far worse than expected.**
- Reason: often the target includes stockout days (hidden demand), or features leak future values.
- Solution: confirm `train[train["qty"].notna()]` is used (the project does this), and run `python tests/run_tests.py`. The leakage test must pass.
- Prevention: never train on raw `qty_sold` when `stockout_flag` is 1.

**Problem: `ValueError: Input contains NaN` from a non-tree model.**
- Reason: lag and rolling features are NaN at the start of each series. The gradient-boosting model handles NaN, but linear models and some others do not.
- Solution: either use `HistGradientBoostingRegressor` (as this project does), or drop NaN rows before fitting the other model.
- Prevention: check `df.isna().sum()` before fitting any model.

**Problem: `categorical_features` error or unexpected category values.**
- Reason: category codes must be non-negative integers, and the model expects the same codes at prediction time.
- Solution: the project builds `store_code` and `item_code` once, on the full panel. Do not rebuild them on a subset.
- Prevention: build codes once and reuse them.

**Problem: training is slow.**
- Reason: a large number of iterations or a long history on a slow laptop.
- Solution: lower `max_iter` in `src/models.py` (for example to 100) for quicker experiments, then restore it for the final run.
- Prevention: time one experiment before running the full backtest.

---

## Forecasting errors

**Problem: forecasts are all zero or negative.**
- Reason: a series has no observed sales in the recent window, or a negative prediction was not clipped.
- Solution: the project clips predictions at zero. For an all-zero series, the forecast is correctly zero; check the item's `zero_day_share` in `outputs/eda_item_summary.csv`.
- Prevention: review zero-share before trusting a forecast for a new product.

**Problem: `IndexError` or empty forecasts for the last few days.**
- Reason: the forecast horizon runs past the end of the data, or the future calendar was not built.
- Solution: call `build_future_calendar(panel)` before `generate_forecasts`, as `main.py` does.
- Prevention: keep the pipeline steps in the order used in `main.py`.

**Problem: forecasts look flat when they should follow the weekly pattern.**
- Reason: the model had too little history, or the day-of-week feature was not included.
- Solution: confirm `dow` is in `FEATURE_COLUMNS` and that the series has at least 28 days of history.
- Prevention: plot one forecast against history (chart 06 does this).

---

## Incorrect inventory calculations

**Problem: reorder point seems too high or too low.**
- Reason: units mismatch (lead time in days but forecast in weeks), or the wrong lead time was used.
- Solution: lead time and forecast must both be in days. Recalculate one row by hand using the worked example in `docs/SIMULATION.md`. Check `lead_time_demand` sums the right number of days.
- Prevention: keep a worked example and check it after every change.

**Problem: recommended order is a very large number.**
- Reason: EOQ is large when holding cost is small compared with ordering cost. The Glucose Biscuits order (2,135 units) is an example.
- Solution: this is the formula working as designed. In practice, planners cap orders by storage space and shelf life. Lower `ORDERING_COST` in `config.py` to see smaller orders, or add a cap as an improvement.
- Prevention: review order quantities against days of cover before raising a purchase order.

**Problem: a store shows REORDER NOW but has plenty of stock.**
- Reason: stock on order is not counted, or the on-order figure is wrong.
- Solution: `inventory_position = on_hand + on_order`. Check the `on_order` column in the recommendations file.
- Prevention: always include pending orders in the position.

**Problem: safety stock is zero.**
- Reason: sigma is zero, usually because a series has too few backtest days, or the service level is 50% (z = 0).
- Solution: check `outputs/residual_stats.csv`. Every series should have around 80 days. A zero sigma means the error was zero, which is unlikely; investigate.
- Prevention: check the residual file after each run.

---

## Plotting errors

**Problem: a chart window does not appear, or the script hangs on `plt.show()`.**
- Reason: the project saves charts to files and does not open windows. This error comes from code you added yourself.
- Solution: save with `fig.savefig(...)` instead of `plt.show()`. The project already sets `matplotlib.use("Agg")`.
- Prevention: keep chart output as files in `images/`.

**Problem: `No module named 'tkinter'` or a backend error.**
- Reason: matplotlib tried to use a GUI backend.
- Solution: the project forces the `Agg` (file) backend. If the error persists, set `MPLBACKEND=Agg` before running.
- Prevention: do not change the backend line at the top of `visualize.py` or `eda.py`.

**Problem: labels overlap or are cut off.**
- Reason: long item names on a small figure.
- Solution: increase `figsize`, or reduce `fontsize`, as in `plot_reorder_chart`.
- Prevention: check each new chart at full size before committing it.

---

## Git errors

**Problem: `fatal: not a git repository`.**
- Reason: Git was not initialised in this folder.
- Solution: run `git init` in the project folder, then `git add .` and `git commit`.
- Prevention: check with `git status`, which fails with this error if the folder is not a repository.

**Problem: `error: remote origin already exists`.**
- Reason: a remote was added before.
- Solution: `git remote remove origin`, then add it again with the correct URL. Or update it: `git remote set-url origin <URL>`.
- Prevention: `git remote -v` shows the current remote.

**Problem: `fatal: 'main' does not appear to be a git repository` on push.**
- Reason: the remote URL is wrong, or it was typed with a mistake.
- Solution: copy the exact URL from your GitHub repository page and run `git remote set-url origin <URL>`.
- Prevention: paste the URL rather than typing it.

**Problem: `committing venv` or `thousands of files staged`.**
- Reason: the `.venv` folder was added before `.gitignore` existed.
- Solution: `git rm -r --cached .venv`, make sure `.gitignore` contains `.venv/`, commit again.
- Prevention: create `.gitignore` before the first `git add`.

**Problem: `LF will be replaced by CRLF` warnings on Windows.**
- Reason: Windows line-ending settings differ from GitHub's.
- Solution: these are warnings, not errors. They are harmless. To silence them: `git config --global core.autocrlf true`.
- Prevention: none needed.

---

## GitHub upload issues

**Problem: `failed to push some refs`, with a message about fetch first.**
- Reason: the GitHub repository has content (for example, a README created on GitHub) that your local copy does not have.
- Solution: `git pull origin main --allow-unrelated-histories`, resolve any conflicts, then `git push -u origin main`. Or create the repository empty next time (see GitHub guide, step 3).
- Prevention: do not tick "Add a README" when creating the repository for an existing project.

**Problem: authentication fails, or the password is rejected.**
- Reason: GitHub no longer accepts account passwords for Git operations.
- Solution: create a Personal Access Token (GitHub → Settings → Developer settings → Personal access tokens). Use it as the password when Git asks. On Windows, Git Credential Manager can store it.
- Prevention: store the token in the credential manager once.

**Problem: a file is rejected because it is larger than 100 MB.**
- Reason: GitHub's limit is 100 MB per file. This project's files are well under that (the dataset is about 3 MB).
- Solution: if you added a large file, remove it from history or use Git LFS. Keep models out of the repository (they are already ignored).
- Prevention: check file sizes before committing (`ls -lh` or `dir`).

**Problem: images do not appear in the README on GitHub.**
- Reason: the path is wrong, the capital letters differ, or the file was never committed.
- Solution: check the path is relative to the README (for example `images/01_sales_trend.png`), that the case matches exactly, and that `git status` shows the image is tracked.
- Prevention: view the README on GitHub after each push, not just on your computer.

**Problem: the repository looks empty or some folders are missing.**
- Reason: empty folders are not tracked by Git.
- Solution: add a placeholder file such as `.gitkeep` inside the folder. The project already does this for `models/`, `data/processed/` and `images/screenshots/`.
- Prevention: keep a `.gitkeep` in any folder that must exist but has no files.

---

## When all else fails

1. Copy the exact error message.
2. Check which step failed in the terminal output.
3. Run `python tests/run_tests.py`. If a test fails, the message usually points to the broken function.
4. Search the error text with the library name in front of it.
5. Restart with a clean environment: delete `.venv`, create it again, and `pip install -r requirements.txt`.
