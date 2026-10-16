"""
A tiny test runner, used when pytest is not installed.
Run from the project root:  python tests/run_tests.py
(With pytest installed, use:  python -m pytest)
"""
import importlib
import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

passed, failed = 0, []
for module_name in ["test_inventory", "test_forecasting", "test_data", "test_features"]:
    module = importlib.import_module(module_name)
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_"):
            try:
                func()
                passed += 1
                print(f"PASS  {module_name}.{name}")
            except Exception:
                failed.append(f"{module_name}.{name}")
                print(f"FAIL  {module_name}.{name}")
                traceback.print_exc()

print(f"\n{passed} passed, {len(failed)} failed")
sys.exit(1 if failed else 0)
