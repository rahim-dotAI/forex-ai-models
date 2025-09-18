#!/usr/bin/env python3
import nbformat
import os
import sys
from nbclient import NotebookClient

def find_notebooks(base_path="."):
    """Recursively find all .ipynb files in the repo"""
    for root, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".ipynb"):
                yield os.path.join(root, f)

def execute_notebook(path):
    """Execute a notebook with cell-by-cell logging"""
    print(f"➡️ Starting notebook: {path}")
    with open(path) as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=False)
    try:
        client.execute()
        print(f"✅ Notebook executed successfully: {path}")
    except Exception as e:
        print(f"⚠️ Error executing notebook {path}: {e}")
        sys.exit(1)

def main():
    notebooks = list(find_notebooks("."))
    if not notebooks:
        print("ℹ️ No notebooks found to execute.")
        sys.exit(0)

    for nb_file in notebooks:
        execute_notebook(nb_file)

if __name__ == "__main__":
    main()
