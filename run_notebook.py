import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
import os
from datetime import datetime

# Load notebook
print("Loading AI_Forex_Brain_2.ipynb...")
with open('AI_Forex_Brain_2.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

# Check environment
skip_av = os.environ.get('SKIP_ALPHA_VANTAGE', 'false').lower() == 'true'

print("=" * 70)
print("🚀 STARTING FOREX AI BRAIN EXECUTION")
print("=" * 70)
print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"🔧 Alpha Vantage: {'SKIPPED ⏭️' if skip_av else 'ACTIVE ✅'}")
print(f"📊 Total cells: {len([c for c in nb.cells if c.cell_type == 'code'])}")
print("=" * 70)
print()

# Execute notebook
ep = ExecutePreprocessor(timeout=2400, kernel_name='python3', allow_errors=True)

try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("\n" + "=" * 70)
    print("✅ EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    sys.exit(0)
except Exception as e:
    print("\n" + "=" * 70)
    print("❌ EXECUTION FAILED")
    print("=" * 70)
    print(f"Error: {str(e)}")
    print("=" * 70)
    sys.exit(1)
