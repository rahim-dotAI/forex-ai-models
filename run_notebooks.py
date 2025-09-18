# run_notebooks.py
import os
import sys
import nbformat
from nbclient import NotebookClient

repo_path = os.getcwd()  # run from repo root

for nb_file in os.listdir(repo_path):
    if nb_file.endswith(".ipynb"):
        print(f"➡️ Running notebook: {nb_file}")
        try:
            with open(os.path.join(repo_path, nb_file)) as f:
                nb = nbformat.read(f, as_version=4)
            client = NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=False)
            client.execute()
            print(f"✅ Successfully executed {nb_file}")
        except Exception as e:
            print(f"⚠️ Error executing {nb_file}: {e}")
            sys.exit(1)
