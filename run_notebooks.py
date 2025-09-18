import os
import sys
import nbformat
from nbclient import NotebookClient

NOTEBOOK_DIR = "."  # Change if your notebooks are in a subfolder
TIMEOUT = 600       # Max seconds per notebook

def run_notebook(path):
    print(f"\n➡️ Running notebook: {path}")
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    client = NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3", allow_errors=True)
    
    try:
        client.execute()
        # Print cell outputs
        for cell in nb.cells:
            if "outputs" in cell:
                for output in cell.get("outputs", []):
                    if "text" in output:
                        print(output["text"])
                    elif "data" in output and "text/plain" in output["data"]:
                        print(output["data"]["text/plain"])
        print(f"✅ Successfully executed: {path}")
    except Exception as e:
        print(f"⚠️ Error executing {path}: {e}")
        sys.exit(1)

def main():
    notebooks = [os.path.join(dp, f) for dp, dn, filenames in os.walk(NOTEBOOK_DIR)
                 for f in filenames if f.endswith(".ipynb")]
    if not notebooks:
        print("ℹ️ No notebooks found to execute.")
        return

    for nb_file in notebooks:
        run_notebook(nb_file)

if __name__ == "__main__":
    main()
