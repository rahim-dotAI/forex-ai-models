import os
import sys
import nbformat
from nbformat import NotebookNode
from nbconvert.preprocessors import ExecutePreprocessor

# Timeout per cell in seconds
TIMEOUT = 600

# Find all notebooks in the repo
notebooks = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".ipynb"):
            notebooks.append(os.path.join(root, file))

if not notebooks:
    print("ℹ️ No notebooks found to execute.")
    sys.exit(0)

for nb_file in notebooks:
    print(f"➡️ Running notebook: {nb_file}")
    try:
        with open(nb_file, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3")

        # Execute notebook with cell-by-cell logging
        for cell_index, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                print(f"\n💡 Executing cell {cell_index}...")
                try:
                    ep.preprocess_cell(cell, {}, cell_index)
                    print("✅ Cell executed successfully.")
                except Exception as e:
                    print(f"⚠️ Cell {cell_index} failed: {e}")
                    raise

        print(f"✅ Notebook {nb_file} executed successfully.")

    except Exception as e:
        print(f"❌ Error executing notebook {nb_file}: {e}")
        sys.exit(1)
