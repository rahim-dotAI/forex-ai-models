#!/usr/bin/env python3
"""
run_notebooks.py

Execute all Jupyter notebooks in a repo and log outputs cell by cell.
Compatible with nbclient 0.7+
"""

import os
import sys
import nbformat
from nbclient import NotebookClient  # Removed CellExecutionError

NOTEBOOK_DIR = "."  # Change if your notebooks are in a subfolder
TIMEOUT = 600       # Maximum seconds per notebook
ALLOW_ERRORS = False

def execute_notebook(path):
    print(f"\n➡️ Executing notebook: {path}")
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3", allow_errors=ALLOW_ERRORS)
    
    try:
        client.execute()
        print(f"✅ Successfully executed: {path}")
    except Exception as e:
        print(f"⚠️ Error in notebook '{path}': {e}")
        if not ALLOW_ERRORS:
            sys.exit(1)

def main():
    notebooks = [os.path.join(dp, f) for dp, dn, filenames in os.walk(NOTEBOOK_DIR) for f in filenames if f.endswith(".ipynb")]
    if not notebooks:
        print("ℹ️ No notebooks found to execute.")
        sys.exit(0)

    for nb_path in notebooks:
        execute_notebook(nb_path)

if __name__ == "__main__":
    main()
