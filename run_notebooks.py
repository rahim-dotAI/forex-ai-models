import os
import time
import requests
from nbformat import read
from nbclient import NotebookClient

# ---------------- CONFIG ----------------
TRIGGER_FILE = "colab_trigger.txt"
BRANCH = os.environ.get("BRANCH", "main")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
HUB_PAT = os.environ.get("HUB_PAT")
LOCAL_REPO_PATH = os.getcwd()
RETRY_LIMIT = 3

# ---------------- FUNCTIONS ----------------

def trigger_exists():
    """Check if trigger file exists on GitHub"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{TRIGGER_FILE}?ref={BRANCH}"
    headers = {"Authorization": f"Bearer {HUB_PAT}"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return True, r.json()
    return False, None

def delete_trigger(sha):
    """Delete trigger file from GitHub using HUB_PAT"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{TRIGGER_FILE}"
    headers = {"Authorization": f"Bearer {HUB_PAT}"}
    payload = {
        "message": "Remove trigger after notebook run",
        "sha": sha,
        "branch": BRANCH
    }
    r = requests.delete(url, headers=headers, json=payload)
    if r.status_code in [200, 204]:
        print("✅ Trigger deleted successfully")
    else:
        print(f"⚠️ Failed to delete trigger: {r.status_code} - {r.text}")

def execute_notebook(path):
    """Execute notebook cell by cell, retrying failed cells"""
    with open(path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3", allow_errors=False)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            print(f"➡️ Executing notebook: {path} (Attempt {attempt}/{RETRY_LIMIT})")
            client.execute()
            # Log outputs cell by cell
            for i, cell in enumerate(nb.cells):
                if "outputs" in cell:
                    for output in cell.outputs:
                        if "text" in output:
                            print(f"[Cell {i}] Output:\n{output.text}")
            print(f"✅ Notebook executed successfully: {path}\n")
            break
        except Exception as e:
            print(f"⚠️ Notebook execution failed (attempt {attempt}): {e}")
            if attempt == RETRY_LIMIT:
                print(f"❌ Notebook failed after {RETRY_LIMIT} attempts: {path}")
            else:
                time.sleep(2)
                print("🔄 Retrying...")

# ---------------- MAIN ----------------

exists, trigger_info = trigger_exists()
if exists:
    print(f"✅ Trigger found. Running notebooks in {LOCAL_REPO_PATH}...\n")
    for file in os.listdir(LOCAL_REPO_PATH):
        if file.endswith(".ipynb"):
            execute_notebook(os.path.join(LOCAL_REPO_PATH, file))
    # Delete trigger after all notebooks executed
    delete_trigger(trigger_info['sha'])
else:
    print("ℹ️ No trigger found. Nothing to run.")
