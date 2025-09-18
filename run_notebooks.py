#!/usr/bin/env python3
import os
import sys
import nbformat
from nbclient import NotebookClient

def execute_notebook(nb_path):
    """Execute a notebook and log output cell by cell."""
    print(f"\n➡️ Starting execution: {nb_path}\n{'='*60}")
    try:
        with open(nb_path) as f:
            nb = nbformat.read(f, as_version=4)

        client = NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=True)

        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                print(f"\n--- Executing cell {i} ---")
                try:
                    client.execute_cell(cell, i)
                    outputs = cell.get("outputs", [])
                    for output in outputs:
                        if "text" in output:
                            print(output["text"])
                        elif "data" in output and "text/plain" in output["data"]:
                            print(output["data"]["text/plain"])
                        elif "ename" in output and "evalue" in output:
                            print(f"Error: {output['ename']}: {output['evalue']}")
                except Exception as e:
                    print(f"⚠️ Exception in cell {i}: {e}")
        print(f"\n✅ Finished notebook: {nb_path}\n{'='*60}")
    except Exception as e:
        print(f"❌ Failed to execute notebook {nb_path}: {e}")
        sys.exit(1)

def main():
    notebook_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_files.append(os.path.join(root, file))

    if not notebook_files:
        print("ℹ️ No notebooks found to execute.")
        return

    for nb_file in notebook_files:
        execute_notebook(nb_file)

if __name__ == "__main__":
    main()
