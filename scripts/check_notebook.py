import json, sys

try:
    with open("AI_Forex_Brain_2.ipynb", "r") as f:
        json.load(f)
    print("[OK] Notebook JSON is valid")
except Exception as e:
    print("[ERROR] Notebook JSON is corrupted!", e)
    sys.exit(1)
