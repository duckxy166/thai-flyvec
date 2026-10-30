import importlib


def test_modules_import():
    for name in ("preprocess", "inputs", "flyvec", "baselines", "evaluate"):
        importlib.import_module(f"thai_flyvec.{name}")
