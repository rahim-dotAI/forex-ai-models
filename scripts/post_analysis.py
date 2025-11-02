import os, glob

print("[SCAN] Checking for model files...")
pkl_files = glob.glob("output/*.pkl")
if pkl_files:
    print("[OK] Found", len(pkl_files), "trained model(s).")
else:
    print("[WARN] No .pkl models found in output/")

csv_files = glob.glob("output/*.csv")
if csv_files:
    print("[INFO] CSV result files detected:")
    for f in csv_files:
        try:
            lines = len(open(f).readlines())
            print("-", f, f"({lines} lines)")
        except Exception as e:
            print("Error reading", f, ":", e)
else:
    print("[WARN] No CSV outputs detected.")
