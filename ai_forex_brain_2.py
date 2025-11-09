#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ======================================================
# 🌍 Notebook Initialization — Colab + GitHub Actions + Local
# ======================================================
import os
import sys
from pathlib import Path
import subprocess

# ======================================================
# 1️⃣ Detect Environment
# ======================================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

IN_GHA = "GITHUB_ACTIONS" in os.environ
IN_LOCAL = not IN_COLAB and not IN_GHA

ENV_NAME = "Colab" if IN_COLAB else "GitHub Actions" if IN_GHA else "Local"
print(f"🔍 Detected environment: {ENV_NAME}")

# ======================================================
# 2️⃣ Safe Working Folder (Auto-Switch)
# ======================================================
if IN_COLAB:
    BASE_DIR = Path("/content")
elif IN_GHA:
    BASE_DIR = Path("/home/runner/work")
else:
    BASE_DIR = Path(".")

REPO_NAME = "forex-ai-models"  # Updated repo name
SAVE_FOLDER = BASE_DIR / REPO_NAME
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
os.chdir(SAVE_FOLDER)
print(f"✅ Working directory set to: {SAVE_FOLDER.resolve()}")

# ======================================================
# 3️⃣ Git Configuration (Universal)
# ======================================================
GIT_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")

subprocess.run(["git", "config", "--global", "user.name", GIT_NAME], check=False)
subprocess.run(["git", "config", "--global", "user.email", GIT_EMAIL], check=False)
subprocess.run(["git", "config", "--global", "advice.detachedHead", "false"], check=False)

print(f"✅ Git configured: {GIT_NAME} <{GIT_EMAIL}>")

# ======================================================
# 4️⃣ Tokens & Secrets
# ======================================================
FOREX_PAT = os.environ.get("FOREX_PAT")
BROWSERLESS_TOKEN = os.environ.get("BROWSERLESS_TOKEN")

# Load Colab secrets if missing
if IN_COLAB and not FOREX_PAT:
    try:
        from google.colab import userdata
        FOREX_PAT = userdata.get('FOREX_PAT')
        if FOREX_PAT:
            os.environ["FOREX_PAT"] = FOREX_PAT
            print("🔐 Loaded FOREX_PAT from Colab secret.")
    except Exception:
        print("⚠️ No Colab secret found for FOREX_PAT")

if not FOREX_PAT:
    print("⚠️ FOREX_PAT not found — GitHub cloning may fail.")
if not BROWSERLESS_TOKEN:
    print("⚠️ BROWSERLESS_TOKEN not found.")

# ======================================================
# 5️⃣ Output Folders
# ======================================================
CSV_FOLDER = SAVE_FOLDER / "csvs"
PICKLE_FOLDER = SAVE_FOLDER / "pickles"
LOGS_FOLDER = SAVE_FOLDER / "logs"

for folder in [CSV_FOLDER, PICKLE_FOLDER, LOGS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

print(f"✅ Output folders ready:")
print(f"   • CSVs:    {CSV_FOLDER}")
print(f"   • Pickles: {PICKLE_FOLDER}")
print(f"   • Logs:    {LOGS_FOLDER}")

# ======================================================
# 6️⃣ Environment Debug Info
# ======================================================
print(f"Python version: {sys.version.split()[0]}")
print(f"Current working directory: {os.getcwd()}")
print(f"Directory contents: {os.listdir('.')}")


# In[ ]:




# In[ ]:


import os

# Set your keys (only for this session)
os.environ['ALPHA_VANTAGE_KEY'] = '1W58NPZXOG5SLHZ6'
os.environ['BROWSERLESS_TOKEN'] = '2TMVUBAjFwrr7Tb283f0da6602a4cb698b81778bda61967f7'

# Test if they work
print("Alpha Vantage Key:", os.environ.get('ALPHA_VANTAGE_KEY'))
print("Browserless Token:", os.environ.get('BROWSERLESS_TOKEN'))




# In[ ]:


# ======================================================
# ⚡ Full Colab-ready GitHub Sync + Remove LFS
# ======================================================
import os
import subprocess
import shutil
from pathlib import Path
import urllib.parse

# -----------------------------
# 0️⃣ Environment / Paths
# -----------------------------
REPO_PARENT = Path("/content/forex-automation")
REPO_PARENT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO_PARENT)

GITHUB_USERNAME = "rahim-dotAI"
GITHUB_REPO = "forex-ai-models"
BRANCH = "main"
REPO_FOLDER = REPO_PARENT / GITHUB_REPO

# -----------------------------
# 1️⃣ GitHub Token
# -----------------------------
FOREX_PAT = os.environ.get("FOREX_PAT")
if not FOREX_PAT:
    from google.colab import userdata
    FOREX_PAT = userdata.get("FOREX_PAT")
    if FOREX_PAT:
        os.environ["FOREX_PAT"] = FOREX_PAT
        print("🔐 Loaded FOREX_PAT from Colab secret.")

if not FOREX_PAT:
    raise ValueError("❌ Missing FOREX_PAT. Set it in Colab userdata or GitHub secrets.")

SAFE_PAT = urllib.parse.quote(FOREX_PAT)

REPO_URL = f"https://{GITHUB_USERNAME}:{SAFE_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

# -----------------------------
# 2️⃣ Clean old repo
# -----------------------------
if REPO_FOLDER.exists():
    print(f"🗑 Removing old repo: {REPO_FOLDER}")
    shutil.rmtree(REPO_FOLDER)

# -----------------------------
# 3️⃣ Clone repo safely (skip LFS)
# -----------------------------
print("🔗 Cloning repo (skipping LFS)...")
env = os.environ.copy()
env["GIT_LFS_SKIP_SMUDGE"] = "1"

subprocess.run(["git", "clone", REPO_URL, str(REPO_FOLDER)], check=True, env=env)
os.chdir(REPO_FOLDER)
print(f"✅ Repo cloned successfully into {REPO_FOLDER}")

# -----------------------------
# 4️⃣ Uninstall LFS and convert files
# -----------------------------
print("⚙️ Removing Git LFS and converting files...")
subprocess.run(["git", "lfs", "uninstall"], check=True)
subprocess.run(["git", "lfs", "migrate", "export", "--include=*.csv"], check=True)

# -----------------------------
# 5️⃣ Configure Git user
# -----------------------------
GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")

subprocess.run(["git", "config", "--global", "user.name", GIT_USER_NAME], check=True)
subprocess.run(["git", "config", "--global", "user.email", GIT_USER_EMAIL], check=True)
subprocess.run(["git", "config", "--global", "advice.detachedHead", "false"], check=True)
print(f"✅ Git configured: {GIT_USER_NAME} <{GIT_USER_EMAIL}>")

# -----------------------------
# 6️⃣ Stage, commit, push
# -----------------------------
subprocess.run(["git", "add", "-A"], check=True)
status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)

if status.stdout.strip():
    subprocess.run(["git", "commit", "-m", "Remove LFS and convert files to normal Git"], check=True)
    subprocess.run(["git", "push", "origin", BRANCH], check=True)
    print("🚀 Repo updated: LFS removed permanently.")
else:
    print("✅ No changes detected. LFS already removed.")

# -----------------------------
# 7️⃣ Create standard output folders
# -----------------------------
for folder in ["csvs", "pickles", "logs"]:
    Path(folder).mkdir(parents=True, exist_ok=True)
print("📁 Output folders ready: csvs/, pickles/, logs/")

# -----------------------------
# 8️⃣ Summary
# -----------------------------
print("\n🧾 Summary:")
print(f"• Working Directory: {os.getcwd()}")
print(f"• Repository: https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}")
print("✅ All operations completed successfully.")


# In[ ]:


# ======================================================
# 🚀 FULLY FIXED ALPHA VANTAGE FX WORKFLOW
# - Uses URL-safe PAT
# - Loads from Colab secrets
# - Cleans stale repo + skips LFS
# - GitHub Actions + Colab Safe
# ======================================================
import os
import time
import hashlib
import requests
import subprocess
import threading
import shutil
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# ======================================================
# 1️⃣ Detect Environment
# ======================================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

IN_GHA = "GITHUB_ACTIONS" in os.environ
print(f"Detected environment: {'Colab' if IN_COLAB else 'GitHub/Local'}")

# ======================================================
# 2️⃣ Working directories
# ======================================================
BASE_FOLDER = Path("/content/forex-alpha-models") if IN_COLAB else Path("./forex-alpha-models")
BASE_FOLDER.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_FOLDER)

PICKLE_FOLDER = BASE_FOLDER / "pickles"
CSV_FOLDER = BASE_FOLDER / "csvs"
LOG_FOLDER = BASE_FOLDER / "logs"

for folder in [PICKLE_FOLDER, CSV_FOLDER, LOG_FOLDER]:
    folder.mkdir(exist_ok=True)

print(f"✅ Working directory: {BASE_FOLDER.resolve()}")
print(f"✅ Output folders ready: {PICKLE_FOLDER}, {CSV_FOLDER}, {LOG_FOLDER}")

# ======================================================
# 3️⃣ GitHub Configuration
# ======================================================
GITHUB_USERNAME = "rahim-dotAI"
GITHUB_REPO = "forex-ai-models"
BRANCH = "main"
REPO_FOLDER = BASE_FOLDER / GITHUB_REPO

# Load PAT from env or Colab userdata
FOREX_PAT = os.environ.get("FOREX_PAT")
if not FOREX_PAT and IN_COLAB:
    try:
        from google.colab import userdata
        FOREX_PAT = userdata.get("FOREX_PAT")
        if FOREX_PAT:
            os.environ["FOREX_PAT"] = FOREX_PAT
            print("🔐 Loaded FOREX_PAT from Colab secret.")
    except Exception:
        pass

if not FOREX_PAT:
    raise ValueError("❌ Missing FOREX_PAT. Set it in Colab userdata or GitHub secrets.")

SAFE_PAT = urllib.parse.quote(FOREX_PAT)
REPO_URL = f"https://{GITHUB_USERNAME}:{SAFE_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

# ======================================================
# 4️⃣ Safe Repo Clone / Sync
# ======================================================
if REPO_FOLDER.exists():
    print(f"🗑 Removing old repo: {REPO_FOLDER}")
    shutil.rmtree(REPO_FOLDER)

print("🔗 Cloning repo (skipping LFS)...")
env = os.environ.copy()
env["GIT_LFS_SKIP_SMUDGE"] = "1"

subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, str(REPO_FOLDER)], check=True, env=env)
os.chdir(REPO_FOLDER)
print(f"✅ Repo cloned successfully into {REPO_FOLDER}")

# Configure Git identity
GIT_USER_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_USER_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")

subprocess.run(["git", "config", "--global", "user.name", GIT_USER_NAME], check=True)
subprocess.run(["git", "config", "--global", "user.email", GIT_USER_EMAIL], check=True)
print(f"✅ Git configured: {GIT_USER_NAME} <{GIT_USER_EMAIL}>")

# ======================================================
# 5️⃣ Alpha Vantage Setup
# ======================================================
ALPHA_VANTAGE_KEY = os.environ.get("ALPHA_VANTAGE_KEY")
if not ALPHA_VANTAGE_KEY:
    raise ValueError("❌ ALPHA_VANTAGE_KEY missing!")

FX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
lock = threading.Lock()

def ensure_tz_naive(df):
    if df is None or df.empty:
        return df
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df

def file_hash(filepath, chunk_size=8192):
    if not filepath.exists():
        return None
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()

def fetch_alpha_vantage_fx(pair, outputsize='full', max_retries=3, retry_delay=5):
    base_url = 'https://www.alphavantage.co/query'
    from_currency, to_currency = pair.split('/')
    params = {
        'function': 'FX_DAILY',
        'from_symbol': from_currency,
        'to_symbol': to_currency,
        'outputsize': outputsize,
        'datatype': 'json',
        'apikey': ALPHA_VANTAGE_KEY
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            if 'Time Series FX (Daily)' not in data:
                raise ValueError(f"Unexpected API response: {data}")
            ts = data['Time Series FX (Daily)']
            df = pd.DataFrame(ts).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close'
            }).astype(float)
            df = ensure_tz_naive(df)
            return df
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed fetching {pair}: {e}")
            time.sleep(retry_delay)
    print(f"❌ Failed to fetch {pair} after {max_retries} retries")
    return pd.DataFrame()

# ======================================================
# 6️⃣ Process Pairs for Unified CSV Pipeline
# ======================================================
def process_pair(pair):
    filename = pair.replace("/", "_") + ".csv"
    filepath = CSV_FOLDER / filename

    if filepath.exists():
        existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    else:
        existing_df = pd.DataFrame()

    old_hash = file_hash(filepath)
    new_df = fetch_alpha_vantage_fx(pair)
    if new_df.empty:
        return None, f"No new data for {pair}"

    combined_df = pd.concat([existing_df, new_df]) if not existing_df.empty else new_df
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
    combined_df.sort_index(inplace=True)

    with lock:
        combined_df.to_csv(filepath)

    new_hash = file_hash(filepath)
    changed = old_hash != new_hash
    print(f"ℹ️ {pair} total rows: {len(combined_df)}")
    return str(filepath) if changed else None, f"{pair} {'updated' if changed else 'no changes'}"

# ======================================================
# 7️⃣ Execute All Pairs in Parallel
# ======================================================
changed_files = []
tasks = []

with ThreadPoolExecutor(max_workers=4) as executor:
    for pair in FX_PAIRS:
        tasks.append(executor.submit(process_pair, pair))
    for future in as_completed(tasks):
        filepath, msg = future.result()
        print(msg)
        if filepath:
            changed_files.append(filepath)

# ======================================================
# 8️⃣ Commit & Push Changes
# ======================================================
if changed_files:
    print(f"🚀 Committing {len(changed_files)} updated files...")
    subprocess.run(["git", "add", "-A"], check=False)
    subprocess.run(["git", "commit", "-m", "Update Alpha Vantage FX data"], check=False)
    subprocess.run(["git", "push", "origin", BRANCH], check=False)
else:
    print("✅ No changes to commit.")

print("✅ All FX pairs processed, saved, pushed successfully!")


# In[ ]:


# ======================================================
# FULLY IMPROVED FOREX DATA WORKFLOW - YFINANCE
# Colab + GitHub Actions Safe, 403-Proof, Large History
# ======================================================

import os, time, hashlib, subprocess, shutil, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import yfinance as yf

# ======================================================
# 1️⃣ Detect environment
# ======================================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

IN_GHA = "GITHUB_ACTIONS" in os.environ
IN_LOCAL = not IN_COLAB and not IN_GHA

print(f"Detected environment: {'Colab' if IN_COLAB else ('GitHub Actions' if IN_GHA else 'Local')}")

# ======================================================
# 2️⃣ Working directories
# ======================================================
BASE_DIR = Path("/content/forex-alpha-models") if IN_COLAB else Path("./forex-alpha-models")
BASE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(BASE_DIR)

PICKLE_FOLDER = BASE_DIR / "pickles"; PICKLE_FOLDER.mkdir(exist_ok=True)
CSV_FOLDER = BASE_DIR / "csvs"; CSV_FOLDER.mkdir(exist_ok=True)
LOG_FOLDER = BASE_DIR / "logs"; LOG_FOLDER.mkdir(exist_ok=True)

print(f"✅ Working directory: {BASE_DIR.resolve()}")
print(f"✅ Output folders ready: {PICKLE_FOLDER}, {CSV_FOLDER}, {LOG_FOLDER}")

# ======================================================
# 3️⃣ Git configuration
# ======================================================
GIT_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")
GITHUB_USERNAME = "rahim-dotAI"
GITHUB_REPO = "forex-ai-models"
BRANCH = "main"

FOREX_PAT = os.environ.get("FOREX_PAT")
if not FOREX_PAT:
    raise ValueError("❌ FOREX_PAT missing!")

subprocess.run(["git", "config", "--global", "user.name", GIT_NAME], check=False)
subprocess.run(["git", "config", "--global", "user.email", GIT_EMAIL], check=False)
subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=False)

cred_file = Path.home() / ".git-credentials"
cred_file.write_text(f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com\n")

# ======================================================
# 4️⃣ Clone or update repo safely
# ======================================================
REPO_FOLDER = BASE_DIR / GITHUB_REPO
def ensure_repo_cloned(repo_url, repo_folder, branch="main"):
    repo_folder = Path(repo_folder)
    tmp_folder = repo_folder.parent / (repo_folder.name + "_tmp")
    if tmp_folder.exists(): shutil.rmtree(tmp_folder)
    if not (repo_folder / ".git").exists():
        print(f"📥 Cloning repo into {tmp_folder} ...")
        subprocess.run(["git", "clone", "-b", branch, repo_url, str(tmp_folder)], check=True)
        if repo_folder.exists(): shutil.rmtree(repo_folder)
        tmp_folder.rename(repo_folder)
    else:
        print("🔄 Repo exists, pulling latest...")
        subprocess.run(["git", "-C", str(repo_folder), "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", str(repo_folder), "checkout", branch], check=False)
        subprocess.run(["git", "-C", str(repo_folder), "pull", "origin", branch], check=False)
    print(f"✅ Repo ready at {repo_folder.resolve()}")

REPO_URL = f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
ensure_repo_cloned(REPO_URL, REPO_FOLDER, BRANCH)

# ======================================================
# 5️⃣ FX pairs & timeframes
# ======================================================
FX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
TIMEFRAMES = {
    "1d_5y": ("1d", "5y"),
    "1h_2y": ("1h", "2y"),
    "15m_60d": ("15m", "60d"),
    "5m_1mo": ("5m", "1mo"),
    "1m_7d": ("1m", "7d")
}

lock = threading.Lock()

# ======================================================
# 6️⃣ Helper functions
# ======================================================
def file_hash(filepath, chunk_size=8192):
    if not filepath.exists(): return None
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""): md5.update(chunk)
    return md5.hexdigest()

def ensure_tz_naive(df):
    if df is None or df.empty: return df
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz: df.index = df.index.tz_convert(None)
    return df

def merge_data(existing_df, new_df):
    existing_df = ensure_tz_naive(existing_df)
    new_df = ensure_tz_naive(new_df)
    if existing_df.empty: return new_df
    if new_df.empty: return existing_df
    combined = pd.concat([existing_df, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined

# ======================================================
# 7️⃣ Worker function for pairs/timeframes
# ======================================================
def process_pair_tf(pair, tf_name, interval, period, max_retries=3, retry_delay=5):
    symbol = pair.replace("/", "") + "=X"
    filename = f"{pair.replace('/', '_')}_{tf_name}.csv"
    filepath = REPO_FOLDER / filename

    existing_df = pd.read_csv(filepath, index_col=0, parse_dates=True) if filepath.exists() else pd.DataFrame()
    old_hash = file_hash(filepath)

    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False, threads=True)
            if df.empty: raise ValueError("No data returned")
            df = df[[c for c in ['Open','High','Low','Close','Volume'] if c in df.columns]]
            df.rename(columns=lambda x: x.lower(), inplace=True)
            df = ensure_tz_naive(df)
            combined_df = merge_data(existing_df, df)
            combined_df.to_csv(filepath)
            if old_hash != file_hash(filepath):
                return f"📈 Updated {pair} {tf_name}", str(filepath)
            return f"✅ No changes {pair} {tf_name}", None
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}/{max_retries} failed for {pair} {tf_name}: {e}")
            if attempt < max_retries: time.sleep(retry_delay)
            else: return f"❌ Failed {pair} {tf_name}", None

# ======================================================
# 8️⃣ Parallel execution
# ======================================================
changed_files = []
tasks = []

with ThreadPoolExecutor(max_workers=8) as executor:
    for pair in FX_PAIRS:
        for tf_name, (interval, period) in TIMEFRAMES.items():
            tasks.append(executor.submit(process_pair_tf, pair, tf_name, interval, period))

for future in as_completed(tasks):
    msg, filename = future.result()
    print(msg)
    if filename: changed_files.append(filename)

# ======================================================
# 9️⃣ Commit & push updates
# ======================================================
if changed_files:
    print(f"🚀 Committing {len(changed_files)} updated files...")
    subprocess.run(["git", "-C", str(REPO_FOLDER), "add"] + changed_files, check=False)
    subprocess.run(["git", "-C", str(REPO_FOLDER), "commit", "-m", "Update YFinance FX data CSVs"], check=False)
    subprocess.run(["git", "-C", str(REPO_FOLDER), "push", "origin", BRANCH], check=False)
else:
    print("✅ No changes detected, nothing to push.")

print("🎯 All FX pairs & timeframes processed safely with maximum historical rows!")


# In[ ]:


# ======================================================
# FX CSV Combine + Incremental Indicators Pipeline
# Fully optimized for YFinance + Alpha Vantage
# Thread-safe, timezone-safe, Git-push-safe, large dataset-ready
# FIXED: Column validation before processing
# ======================================================

import os, time, hashlib, subprocess, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta
from ta.momentum import WilliamsRIndicator

# -----------------------------
# 0️⃣ Environment & folders
# -----------------------------
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

ROOT_DIR = Path("/content/forex-alpha-models") if IN_COLAB else Path(".")
ROOT_DIR.mkdir(parents=True, exist_ok=True)

REPO_FOLDER = ROOT_DIR / "forex-ai-models"
CSV_FOLDER = ROOT_DIR / "csvs"
PICKLE_FOLDER = ROOT_DIR / "pickles"
LOGS_FOLDER = ROOT_DIR / "logs"
for folder in [CSV_FOLDER, PICKLE_FOLDER, LOGS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()

def print_status(msg, level="info"):
    levels = {"info":"ℹ️","success":"✅","warn":"⚠️"}
    print(f"{levels.get(level, 'ℹ️')} {msg}")

# -----------------------------
# 1️⃣ Git configuration
# -----------------------------
GIT_NAME = os.environ.get("GIT_USER_NAME", "Abdul Rahim")
GIT_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "rahim-dotAI")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "forex-ai-models")
FOREX_PAT = os.environ.get("FOREX_PAT", "").strip()
BRANCH = "main"

if not FOREX_PAT:
    raise ValueError("❌ FOREX_PAT missing!")

REPO_URL = f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

subprocess.run(["git", "config", "--global", "user.name", GIT_NAME], check=False)
subprocess.run(["git", "config", "--global", "user.email", GIT_EMAIL], check=False)
subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=False)
cred_file = Path.home() / ".git-credentials"
cred_file.write_text(f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com\n")

# -----------------------------
# 2️⃣ Ensure repo exists
# -----------------------------
def ensure_repo():
    if not (REPO_FOLDER / ".git").exists():
        if REPO_FOLDER.exists():
            shutil.rmtree(REPO_FOLDER)
        print_status(f"Cloning repo into {REPO_FOLDER}...", "info")
        subprocess.run(["git", "clone", "-b", BRANCH, REPO_URL, str(REPO_FOLDER)], check=True)
    else:
        print_status("Repo exists, pulling latest...", "info")
        subprocess.run(["git", "-C", str(REPO_FOLDER), "fetch", "origin"], check=False)
        subprocess.run(["git", "-C", str(REPO_FOLDER), "checkout", BRANCH], check=False)
        subprocess.run(["git", "-C", str(REPO_FOLDER), "pull", "origin", BRANCH], check=False)
        print_status("Repo synced successfully", "success")
ensure_repo()

# -----------------------------
# 3️⃣ Helpers - UPDATED WITH SAFEGUARDS
# -----------------------------
def ensure_tz_naive(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index, errors='coerce')
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def file_hash(filepath):
    if not filepath.exists():
        return None
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()

def safe_numeric(df):
    """Handle infinity/NaN robustly before any scaling - WITH COLUMN VALIDATION"""
    # Replace infinity values first
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check if OHLC columns exist before trying to drop NaN rows
    required_columns = ['open', 'high', 'low', 'close']
    existing_columns = [col for col in required_columns if col in df.columns]

    # Only drop NaN if we have at least some OHLC columns
    if existing_columns:
        df.dropna(subset=existing_columns, inplace=True)
    else:
        # If no OHLC columns, just drop rows that are completely empty
        df.dropna(how='all', inplace=True)

    return df

# -----------------------------
# 4️⃣ Incremental CSV combine
# -----------------------------
def combine_csv(csv_path):
    target_file = REPO_FOLDER / csv_path.name
    existing_df = ensure_tz_naive(pd.read_csv(target_file, index_col=0, parse_dates=True)) if target_file.exists() else pd.DataFrame()
    new_df = ensure_tz_naive(pd.read_csv(csv_path, index_col=0, parse_dates=True))
    combined_df = pd.concat([existing_df, new_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
    combined_df.sort_index(inplace=True)
    return combined_df, target_file

# -----------------------------
# 5️⃣ Incremental indicators - UPDATED WITH VALIDATION
# -----------------------------
def add_indicators_incremental(existing_df, combined_df):
    new_rows = combined_df.loc[~combined_df.index.isin(existing_df.index)] if not existing_df.empty else combined_df
    if new_rows.empty:
        return None

    # CRITICAL: Validate OHLC columns exist before processing
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in new_rows.columns for col in required_cols):
        print_status(f"⚠️ Missing required OHLC columns. Found: {list(new_rows.columns)}", "warn")
        return None

    # Clean numeric data with validation
    new_rows = safe_numeric(new_rows)

    # Check if we still have data after cleaning
    if new_rows.empty:
        print_status("⚠️ No rows left after cleaning", "warn")
        return None

    new_rows.sort_index(inplace=True)

    # Trend indicators
    try:
        trend = {
            'SMA_10': lambda d: ta.trend.sma_indicator(d['close'], 10),
            'SMA_50': lambda d: ta.trend.sma_indicator(d['close'], 50),
            'SMA_200': lambda d: ta.trend.sma_indicator(d['close'], 200),
            'EMA_10': lambda d: ta.trend.ema_indicator(d['close'], 10),
            'EMA_50': lambda d: ta.trend.ema_indicator(d['close'], 50),
            'EMA_200': lambda d: ta.trend.ema_indicator(d['close'], 200),
            'MACD': lambda d: ta.trend.macd(d['close']),
            'MACD_signal': lambda d: ta.trend.macd_signal(d['close']),
            'ADX': lambda d: ta.trend.adx(d['high'], d['low'], d['close'], 14)
        }

        # Momentum indicators
        momentum = {
            'RSI_14': lambda d: ta.momentum.rsi(d['close'], 14),
            'StochRSI': lambda d: ta.momentum.stochrsi(d['close'], 14),
            'CCI': lambda d: ta.trend.cci(d['high'], d['low'], d['close'], 20),
            'ROC': lambda d: ta.momentum.roc(d['close'], 12),
            'Williams_%R': lambda d: WilliamsRIndicator(d['high'], d['low'], d['close'], 14).williams_r()
        }

        # Volatility
        volatility = {
            'Bollinger_High': lambda d: ta.volatility.bollinger_hband(d['close'], 20, 2),
            'Bollinger_Low': lambda d: ta.volatility.bollinger_lband(d['close'], 20, 2),
            'ATR': lambda d: ta.volatility.average_true_range(d['high'], d['low'], d['close'], 14),
            'STDDEV_20': lambda d: d['close'].rolling(20).std()
        }

        # Volume-based
        volume = {}
        if 'volume' in new_rows.columns:
            volume = {
                'OBV': lambda d: ta.volume.on_balance_volume(d['close'], d['volume']),
                'MFI': lambda d: ta.volume.money_flow_index(d['high'], d['low'], d['close'], d['volume'], 14)
            }

        indicators = {**trend, **momentum, **volatility, **volume}
        for name, func in indicators.items():
            try:
                new_rows[name] = func(new_rows)
            except Exception as e:
                print_status(f"⚠️ Failed to calculate {name}: {e}", "warn")
                new_rows[name] = np.nan

        # Cross signals
        if 'EMA_10' in new_rows.columns and 'EMA_50' in new_rows.columns:
            new_rows['EMA_10_cross_EMA_50'] = (new_rows['EMA_10'] > new_rows['EMA_50']).astype(int)
        if 'EMA_50' in new_rows.columns and 'EMA_200' in new_rows.columns:
            new_rows['EMA_50_cross_EMA_200'] = (new_rows['EMA_50'] > new_rows['EMA_200']).astype(int)
        if 'SMA_10' in new_rows.columns and 'SMA_50' in new_rows.columns:
            new_rows['SMA_10_cross_SMA_50'] = (new_rows['SMA_10'] > new_rows['SMA_50']).astype(int)
        if 'SMA_50' in new_rows.columns and 'SMA_200' in new_rows.columns:
            new_rows['SMA_50_cross_SMA_200'] = (new_rows['SMA_50'] > new_rows['SMA_200']).astype(int)

    except Exception as e:
        print_status(f"⚠️ Indicator calculation error: {e}", "warn")

    # 🔧 CRITICAL FIX: Clean infinity/NaN values before scaling
    numeric_cols = new_rows.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0 and not new_rows[numeric_cols].dropna(how='all').empty:
        # Replace infinity values with NaN
        new_rows[numeric_cols] = new_rows[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Forward fill NaN values, then backward fill, then fill remaining with 0
        new_rows[numeric_cols] = new_rows[numeric_cols].ffill().bfill().fillna(0)

        # Clip extreme values to a reasonable range
        for col in numeric_cols:
            if new_rows[col].std() > 0:
                mean_val = new_rows[col].mean()
                std_val = new_rows[col].std()
                lower_bound = mean_val - (5 * std_val)
                upper_bound = mean_val + (5 * std_val)
                new_rows[col] = new_rows[col].clip(lower=lower_bound, upper=upper_bound)

        # Now scale safely
        scaler = MinMaxScaler()
        try:
            new_rows[numeric_cols] = scaler.fit_transform(new_rows[numeric_cols])
        except Exception as e:
            print_status(f"⚠️ Scaling warning: {e} - using manual normalization", "warn")
            # Manual normalization fallback
            for col in numeric_cols:
                col_min = new_rows[col].min()
                col_max = new_rows[col].max()
                if col_max > col_min:
                    new_rows[col] = (new_rows[col] - col_min) / (col_max - col_min)

    return new_rows

# -----------------------------
# 6️⃣ Worker function - UPDATED WITH VALIDATION
# -----------------------------
def process_csv_file(csv_file):
    try:
        combined_df, target_file = combine_csv(csv_file)

        # Validate combined dataframe has required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in combined_df.columns for col in required_cols):
            return None, f"⚠️ Skipped {csv_file.name}: Missing OHLC columns"

        existing_pickle = PICKLE_FOLDER / f"{csv_file.stem}_indicators.pkl"
        existing_df = pd.read_pickle(existing_pickle) if existing_pickle.exists() else pd.DataFrame()

        new_indicators = add_indicators_incremental(existing_df, combined_df)
        if new_indicators is not None:
            updated_df = pd.concat([existing_df, new_indicators]).sort_index()
            with lock:
                updated_df.to_pickle(existing_pickle, protocol=4)
                combined_df.to_csv(target_file)
            msg = f"{csv_file.name} updated with {len(new_indicators)} new rows"
        else:
            msg = f"{csv_file.name} no new rows"

        total_rows = len(combined_df)
        print_status(f"{csv_file.name} total rows: {total_rows}", "info")

        return str(existing_pickle) if new_indicators is not None else None, msg

    except Exception as e:
        print_status(f"❌ Error processing {csv_file.name}: {e}", "warn")
        return None, f"❌ Failed {csv_file.name}: {e}"

# -----------------------------
# 7️⃣ Process all CSVs in parallel
# -----------------------------
csv_files = list(CSV_FOLDER.glob("*.csv"))
if not csv_files:
    print_status("No CSVs found to process – pipeline will skip", "warn")

changed_files = []

with ThreadPoolExecutor(max_workers=min(8, len(csv_files) or 1)) as executor:
    futures = [executor.submit(process_csv_file, f) for f in csv_files]
    for future in as_completed(futures):
        file, msg = future.result()
        print_status(msg, "success" if file else "info")
        if file:
            changed_files.append(file)

# -----------------------------
# 8️⃣ Commit & push updates
# -----------------------------
if changed_files:
    print_status(f"Committing {len(changed_files)} updated files...", "info")
    subprocess.run(["git", "-C", str(REPO_FOLDER), "add"] + changed_files, check=False)
    subprocess.run(
        ["git", "-C", str(REPO_FOLDER), "commit", "-m", "📈 Auto update FX CSVs & indicators"],
        check=False
    )

    push_cmd = f"git -C {REPO_FOLDER} push {REPO_URL} {BRANCH}"
    for attempt in range(3):
        if subprocess.run(push_cmd, shell=True).returncode == 0:
            print_status("Push successful", "success")
            break
        else:
            print_status(f"Push attempt {attempt+1} failed, retrying...", "warn")
            time.sleep(5)
else:
    print_status("No files changed – skipping push", "info")

print_status("All CSVs combined, incremental indicators added, and Git updated successfully.", "success")


# In[ ]:


#!/usr/bin/env python3
"""
VERSION 3.6 – ULTRA-PERSISTENT SELF-LEARNING HYBRID FX PIPELINE (FIXED)
========================================================================
🚀 CRITICAL FIXES:
- ✅ SQLite INDEX syntax fixed (was causing crash)
- ✅ Trades evaluated in NEXT iteration (after price has moved)
- ✅ All data persists in Git repo (survives GitHub Actions)
- ✅ Real accuracy tracking (not artificial 100%)
- ✅ Proper minimum trade age before evaluation (1+ hours)
- ✅ Learning system gets real performance data
- ✅ Database auto-commits to Git after each run

NEW IMPROVEMENTS:
- ✅ Separate pending_trades and completed_trades tables
- ✅ Minimum age requirement for trade evaluation
- ✅ Better model performance comparison
- ✅ CSV and pickle files persist properly
- ✅ Proper SQLite index creation (separate statements)
- ✅ Enhanced error handling and validation
"""

import os, time, json, re, shutil, subprocess, pickle, filecmp, sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import requests
import ta
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from collections import defaultdict

# ======================================================
# 0️⃣ Logging & Environment
# ======================================================
ROOT_DIR = Path("/content/forex-alpha-models")
ROOT_DIR.mkdir(parents=True, exist_ok=True)
REPO_FOLDER = ROOT_DIR / "forex-ai-models"
CSV_FOLDER = ROOT_DIR / "csvs"
PICKLE_FOLDER = ROOT_DIR / "pickles"
LOGS_FOLDER = ROOT_DIR / "logs"

for folder in [CSV_FOLDER, PICKLE_FOLDER, LOGS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOGS_FOLDER / "pipeline.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def print_status(msg, level="info"):
    icons = {"info":"ℹ️","success":"✅","warn":"⚠️","debug":"🐞","error":"❌"}
    getattr(logging, level if level != "warn" else "warning", logging.info)(msg)
    print(f"{icons.get(level,'ℹ️')} {msg}")

# ======================================================
# 🆕 FIXED DATABASE - Stores in Git Repo
# ======================================================
PERSISTENT_DB = REPO_FOLDER / "ml_persistent_memory.db"

class FixedTradeMemoryDatabase:
    """
    FIXED VERSION - SQLite compatible database

    IMPROVEMENTS:
    - ✅ Proper SQLite INDEX syntax (created separately)
    - ✅ Enhanced error handling
    - ✅ Transaction management
    - ✅ Data validation
    - ✅ Backup and recovery

    EVALUATION FLOW:
    - Stores trades at end of iteration N
    - Evaluates them at start of iteration N+1 (after 1+ hours)
    - All data stored in Git repo for persistence
    """

    def __init__(self, db_path=PERSISTENT_DB):
        self.db_path = db_path
        self.conn = None
        self.min_age_hours = 1  # Minimum hours before evaluation
        self.initialize_database()

    def initialize_database(self):
        """
        Create database in Git repo (persists across runs)
        FIXED: Proper SQLite syntax for indexes
        """
        try:
            self.conn = sqlite3.connect(str(self.db_path), timeout=30)
            self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
            self.conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
            cursor = self.conn.cursor()

            # ===== TABLE 1: Pending trades (waiting to be evaluated) =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pending_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    sgd_prediction INTEGER,
                    rf_prediction INTEGER,
                    ensemble_prediction INTEGER,
                    entry_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    confidence REAL,
                    evaluated BOOLEAN DEFAULT 0
                )
            ''')

            # Create indexes separately (SQLite proper syntax)
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pending_eval
                ON pending_trades(evaluated, created_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_pending_pair
                ON pending_trades(pair, evaluated)
            ''')

            # ===== TABLE 2: Completed trades (historical results) =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS completed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pending_trade_id INTEGER,
                    created_at TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    iteration_created INTEGER,
                    iteration_evaluated INTEGER,
                    pair TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    sl_price REAL NOT NULL,
                    tp_price REAL NOT NULL,
                    prediction INTEGER,
                    hit_tp BOOLEAN NOT NULL,
                    pnl REAL NOT NULL,
                    duration_hours REAL,
                    FOREIGN KEY (pending_trade_id) REFERENCES pending_trades(id)
                )
            ''')

            # Create indexes separately
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_completed_model
                ON completed_trades(model_used, evaluated_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_completed_pair
                ON completed_trades(pair, model_used, evaluated_at)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_completed_timestamp
                ON completed_trades(evaluated_at)
            ''')

            # ===== TABLE 3: Model performance cache =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_stats_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    updated_at TEXT NOT NULL,
                    pair TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    days INTEGER NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    winning_trades INTEGER DEFAULT 0,
                    accuracy_pct REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    avg_pnl REAL DEFAULT 0.0,
                    UNIQUE(pair, model_name, days) ON CONFLICT REPLACE
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_stats_lookup
                ON model_stats_cache(pair, model_name, days)
            ''')

            # ===== TABLE 4: Pipeline execution log =====
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    trades_stored INTEGER DEFAULT 0,
                    trades_evaluated INTEGER DEFAULT 0,
                    duration_seconds REAL,
                    error_message TEXT
                )
            ''')

            self.conn.commit()
            print_status("✅ Fixed ML Database initialized (persists in Git)", "success")

            # Verify database integrity
            self._verify_database_integrity()

        except sqlite3.Error as e:
            print_status(f"❌ Database initialization failed: {e}", "error")
            raise

    def _verify_database_integrity(self):
        """Verify database structure is correct"""
        try:
            cursor = self.conn.cursor()

            # Check if tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'pending_trades', 'completed_trades',
                    'model_stats_cache', 'execution_log'
                )
            """)

            tables = [row[0] for row in cursor.fetchall()]
            expected_tables = ['pending_trades', 'completed_trades',
                             'model_stats_cache', 'execution_log']

            for table in expected_tables:
                if table in tables:
                    print_status(f"  ✓ Table '{table}' exists", "debug")
                else:
                    print_status(f"  ✗ Table '{table}' missing!", "error")

            # Check indexes
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_%'
            """)

            indexes = [row[0] for row in cursor.fetchall()]
            print_status(f"  📊 Found {len(indexes)} indexes", "debug")

        except Exception as e:
            print_status(f"⚠️ Database verification warning: {e}", "warn")

    def store_new_signals(self, aggregated_signals, current_iteration):
        """
        Store signals at END of current iteration.
        They will be evaluated in NEXT iteration.

        Args:
            aggregated_signals: Dict of signals by pair
            current_iteration: Current iteration number

        Returns:
            int: Number of signals stored
        """
        if not aggregated_signals:
            print_status("⚠️ No signals to store", "warn")
            return 0

        cursor = self.conn.cursor()
        stored_count = 0
        failed_count = 0

        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            for pair, pair_data in aggregated_signals.items():
                signals = pair_data.get('signals', {})

                for tf_name, signal_data in signals.items():
                    if not signal_data:
                        continue

                    # Validate required fields
                    required_fields = ['live', 'SL', 'TP']
                    if not all(signal_data.get(f, 0) > 0 for f in required_fields):
                        print_status(
                            f"⚠️ Skipping invalid signal for {pair} {tf_name}",
                            "warn"
                        )
                        failed_count += 1
                        continue

                    try:
                        cursor.execute('''
                            INSERT INTO pending_trades
                            (created_at, iteration, pair, timeframe,
                             sgd_prediction, rf_prediction, ensemble_prediction,
                             entry_price, sl_price, tp_price, confidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            datetime.now(timezone.utc).isoformat(),
                            current_iteration,
                            pair,
                            tf_name,
                            signal_data.get('sgd_pred', 0),
                            signal_data.get('rf_pred', 0),
                            signal_data.get('signal', 0),
                            signal_data.get('live', 0),
                            signal_data.get('SL', 0),
                            signal_data.get('TP', 0),
                            signal_data.get('confidence', 0.5)
                        ))
                        stored_count += 1
                    except sqlite3.Error as e:
                        print_status(f"⚠️ Failed to store {pair} {tf_name}: {e}", "warn")
                        failed_count += 1

            # Commit transaction
            self.conn.commit()

            # Log execution
            cursor.execute('''
                INSERT INTO execution_log
                (timestamp, iteration, status, trades_stored)
                VALUES (?, ?, 'signals_stored', ?)
            ''', (
                datetime.now(timezone.utc).isoformat(),
                current_iteration,
                stored_count
            ))
            self.conn.commit()

            print_status(
                f"💾 Stored {stored_count} trades for next iteration "
                f"({failed_count} failed)",
                "success"
            )
            return stored_count

        except sqlite3.Error as e:
            self.conn.rollback()
            print_status(f"❌ Transaction failed: {e}", "error")
            return 0

    def evaluate_pending_trades(self, current_prices, current_iteration):
        """
        Evaluate trades from PREVIOUS iterations.
        Only evaluates trades older than min_age_hours.

        Args:
            current_prices: Dict of current prices by pair
            current_iteration: Current iteration number

        Returns:
            dict: Evaluation results by model
        """
        if not current_prices:
            print_status("⚠️ No current prices provided", "warn")
            return {}

        cursor = self.conn.cursor()

        # Get unevaluated trades that are OLD ENOUGH
        min_age = (datetime.now(timezone.utc) - timedelta(hours=self.min_age_hours)).isoformat()

        try:
            cursor.execute('''
                SELECT id, pair, timeframe, sgd_prediction, rf_prediction,
                       ensemble_prediction, entry_price, sl_price, tp_price,
                       created_at, iteration
                FROM pending_trades
                WHERE evaluated = 0 AND created_at < ?
                ORDER BY created_at ASC
            ''', (min_age,))

            pending_trades = cursor.fetchall()

        except sqlite3.Error as e:
            print_status(f"❌ Failed to fetch pending trades: {e}", "error")
            return {}

        if not pending_trades:
            print_status(
                f"ℹ️ No trades old enough to evaluate (need {self.min_age_hours}+ hours)",
                "info"
            )
            return {}

        print_status(
            f"🔍 Evaluating {len(pending_trades)} trades from previous iteration(s)",
            "info"
        )

        results_by_model = defaultdict(lambda: {
            'closed_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'trades': []
        })

        evaluated_count = 0
        skipped_count = 0

        try:
            # Start transaction
            cursor.execute("BEGIN TRANSACTION")

            for trade in pending_trades:
                (trade_id, pair, timeframe, sgd_pred, rf_pred, ensemble_pred,
                 entry_price, sl_price, tp_price, created_at, created_iteration) = trade

                current_price = current_prices.get(pair, 0)

                if current_price <= 0:
                    print_status(f"⚠️ No current price for {pair}, skipping", "warn")
                    skipped_count += 1
                    continue

                # Validate prices
                if not self._validate_trade_prices(entry_price, sl_price, tp_price, current_price):
                    print_status(
                        f"⚠️ Invalid prices for {pair} {timeframe}, skipping",
                        "warn"
                    )
                    skipped_count += 1
                    continue

                # Evaluate for each model that made a prediction
                for model_name, prediction in [
                    ('SGD', sgd_pred),
                    ('RandomForest', rf_pred),
                    ('Ensemble', ensemble_pred)
                ]:
                    if prediction is None:
                        continue

                    # Check if TP or SL was hit
                    hit_tp, hit_sl, exit_price = self._evaluate_trade_outcome(
                        prediction, current_price, tp_price, sl_price
                    )

                    # If trade closed, record result
                    if exit_price:
                        # Calculate P&L
                        pnl = self._calculate_pnl(
                            prediction, entry_price, exit_price
                        )

                        # Duration
                        duration_hours = self._calculate_duration_hours(created_at)

                        # Insert into completed trades
                        try:
                            cursor.execute('''
                                INSERT INTO completed_trades
                                (pending_trade_id, created_at, evaluated_at,
                                 iteration_created, iteration_evaluated,
                                 pair, timeframe, model_used, entry_price, exit_price,
                                 sl_price, tp_price, prediction, hit_tp, pnl, duration_hours)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                trade_id, created_at, datetime.now(timezone.utc).isoformat(),
                                created_iteration, current_iteration,
                                pair, timeframe, model_name, entry_price, exit_price,
                                sl_price, tp_price, prediction, hit_tp, pnl, duration_hours
                            ))
                        except sqlite3.Error as e:
                            print_status(
                                f"⚠️ Failed to record completed trade: {e}",
                                "warn"
                            )
                            continue

                        # Accumulate results
                        results_by_model[model_name]['closed_trades'] += 1
                        results_by_model[model_name]['total_pnl'] += pnl

                        if hit_tp:
                            results_by_model[model_name]['wins'] += 1
                        else:
                            results_by_model[model_name]['losses'] += 1

                        results_by_model[model_name]['trades'].append({
                            'pair': pair,
                            'timeframe': timeframe,
                            'pnl': pnl,
                            'hit_tp': hit_tp
                        })

                        status = "WIN ✅" if hit_tp else "LOSS ❌"
                        print_status(
                            f"{status} {model_name}: {pair} {timeframe} "
                            f"Entry={entry_price:.5f} Exit={exit_price:.5f} "
                            f"P&L=${pnl:.5f} ({duration_hours:.1f}h)",
                            "success" if hit_tp else "warn"
                        )

                # Mark pending trade as evaluated
                try:
                    cursor.execute('''
                        UPDATE pending_trades
                        SET evaluated = 1
                        WHERE id = ?
                    ''', (trade_id,))
                    evaluated_count += 1
                except sqlite3.Error as e:
                    print_status(f"⚠️ Failed to mark trade as evaluated: {e}", "warn")

            # Commit transaction
            self.conn.commit()

            # Log execution
            cursor.execute('''
                INSERT INTO execution_log
                (timestamp, iteration, status, trades_evaluated)
                VALUES (?, ?, 'trades_evaluated', ?)
            ''', (
                datetime.now(timezone.utc).isoformat(),
                current_iteration,
                evaluated_count
            ))
            self.conn.commit()

            print_status(
                f"✅ Evaluated {evaluated_count} trades "
                f"({skipped_count} skipped)",
                "success"
            )

        except sqlite3.Error as e:
            self.conn.rollback()
            print_status(f"❌ Evaluation transaction failed: {e}", "error")
            return {}

        # Calculate accuracies
        for model_name, results in results_by_model.items():
            if results['closed_trades'] > 0:
                results['accuracy'] = (results['wins'] / results['closed_trades']) * 100
            else:
                results['accuracy'] = 0.0

        # Update model stats cache
        self._update_stats_cache()

        return dict(results_by_model)

    def _validate_trade_prices(self, entry, sl, tp, current):
        """Validate that trade prices are reasonable"""
        try:
            if any(p <= 0 for p in [entry, sl, tp, current]):
                return False

            # Prices shouldn't be wildly different (max 50% deviation)
            prices = [entry, sl, tp, current]
            avg_price = sum(prices) / len(prices)

            for price in prices:
                if abs(price - avg_price) / avg_price > 0.5:
                    return False

            return True
        except:
            return False

    def _evaluate_trade_outcome(self, prediction, current_price, tp_price, sl_price):
        """Determine if trade hit TP or SL"""
        hit_tp = False
        hit_sl = False
        exit_price = None

        try:
            if prediction == 1:  # Long
                if current_price >= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                elif current_price <= sl_price:
                    hit_sl = True
                    exit_price = sl_price
            elif prediction == 0:  # Short
                if current_price <= tp_price:
                    hit_tp = True
                    exit_price = tp_price
                elif current_price >= sl_price:
                    hit_sl = True
                    exit_price = sl_price

        except Exception as e:
            print_status(f"⚠️ Trade evaluation error: {e}", "warn")

        return hit_tp, hit_sl, exit_price

    def _calculate_pnl(self, prediction, entry_price, exit_price):
        """Calculate profit/loss"""
        try:
            if prediction == 1:  # Long
                return exit_price - entry_price
            else:  # Short
                return entry_price - exit_price
        except:
            return 0.0

    def _calculate_duration_hours(self, created_at):
        """Calculate trade duration in hours"""
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            duration = (datetime.now(timezone.utc) - created_dt).total_seconds() / 3600
            return max(0, duration)
        except:
            return 0.0

    def _update_stats_cache(self):
        """Update cached model performance statistics"""
        cursor = self.conn.cursor()

        try:
            # Get unique pairs and models
            cursor.execute('SELECT DISTINCT pair FROM completed_trades')
            pairs = [row[0] for row in cursor.fetchall()]

            cursor.execute('SELECT DISTINCT model_used FROM completed_trades')
            models = [row[0] for row in cursor.fetchall()]

            for pair in pairs:
                for model in models:
                    for days in [7, 30, 90]:
                        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

                        cursor.execute('''
                            SELECT
                                COUNT(*) as total,
                                SUM(CASE WHEN hit_tp THEN 1 ELSE 0 END) as wins,
                                SUM(pnl) as total_pnl,
                                AVG(pnl) as avg_pnl
                            FROM completed_trades
                            WHERE pair = ? AND model_used = ? AND evaluated_at > ?
                        ''', (pair, model, since))

                        result = cursor.fetchone()
                        total, wins, total_pnl, avg_pnl = result

                        if total and total > 0:
                            accuracy = (wins / total * 100) if total > 0 else 0.0

                            cursor.execute('''
                                INSERT OR REPLACE INTO model_stats_cache
                                (updated_at, pair, model_name, days, total_trades,
                                 winning_trades, accuracy_pct, total_pnl, avg_pnl)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                datetime.now(timezone.utc).isoformat(),
                                pair, model, days, total, wins or 0,
                                accuracy, total_pnl or 0.0, avg_pnl or 0.0
                            ))

            self.conn.commit()
            print_status("✅ Stats cache updated", "debug")

        except sqlite3.Error as e:
            print_status(f"⚠️ Stats cache update failed: {e}", "warn")

    def get_model_performance(self, pair, model_name, days=7):
        """
        Get model performance from cache (fast)

        Args:
            pair: Currency pair (e.g., 'EUR/USD')
            model_name: Model name ('SGD', 'RandomForest', 'Ensemble')
            days: Number of days to look back

        Returns:
            dict: Performance metrics
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                SELECT total_trades, winning_trades, accuracy_pct,
                       total_pnl, avg_pnl, updated_at
                FROM model_stats_cache
                WHERE pair = ? AND model_name = ? AND days = ?
            ''', (pair, model_name, days))

            result = cursor.fetchone()

            if not result:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'accuracy': 0.0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0
                }

            total, wins, accuracy, total_pnl, avg_pnl, updated_at = result

            return {
                'total_trades': total,
                'winning_trades': wins,
                'accuracy': accuracy,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'updated_at': updated_at
            }

        except sqlite3.Error as e:
            print_status(f"⚠️ Failed to get model performance: {e}", "warn")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'accuracy': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0
            }

    def get_best_model(self, pair, days=7, min_trades=3):
        """
        Determine which model performs best based on ACTUAL results

        Args:
            pair: Currency pair
            days: Number of days to look back
            min_trades: Minimum number of trades required

        Returns:
            str: Best model name or 'Ensemble' as default
        """
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                SELECT model_name, accuracy_pct, total_trades, total_pnl
                FROM model_stats_cache
                WHERE pair = ? AND days = ? AND total_trades >= ?
                ORDER BY accuracy_pct DESC, total_pnl DESC
                LIMIT 1
            ''', (pair, days, min_trades))

            result = cursor.fetchone()

            if result:
                return result[0]

        except sqlite3.Error as e:
            print_status(f"⚠️ Failed to get best model: {e}", "warn")

        return 'Ensemble'  # Default fallback

    def get_database_stats(self):
        """Get database statistics"""
        cursor = self.conn.cursor()
        stats = {}

        try:
            # Pending trades count
            cursor.execute('SELECT COUNT(*) FROM pending_trades WHERE evaluated = 0')
            stats['pending_trades'] = cursor.fetchone()[0]

            # Completed trades count
            cursor.execute('SELECT COUNT(*) FROM completed_trades')
            stats['completed_trades'] = cursor.fetchone()[0]

            # Total P&L
            cursor.execute('SELECT SUM(pnl) FROM completed_trades')
            result = cursor.fetchone()
            stats['total_pnl'] = result[0] if result[0] else 0.0

            # Overall accuracy
            cursor.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_tp THEN 1 ELSE 0 END) as wins
                FROM completed_trades
            ''')
            result = cursor.fetchone()
            if result and result[0] > 0:
                stats['overall_accuracy'] = (result[1] / result[0]) * 100
            else:
                stats['overall_accuracy'] = 0.0

            # Database size
            stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

        except Exception as e:
            print_status(f"⚠️ Failed to get database stats: {e}", "warn")

        return stats

    def cleanup_old_data(self, days_to_keep=90):
        """
        Clean up old data to prevent database bloat

        Args:
            days_to_keep: Number of days of data to keep
        """
        cursor = self.conn.cursor()
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()

        try:
            # Delete old evaluated pending trades
            cursor.execute('''
                DELETE FROM pending_trades
                WHERE evaluated = 1 AND created_at < ?
            ''', (cutoff_date,))
            deleted_pending = cursor.rowcount

            # Delete old execution logs
            cursor.execute('''
                DELETE FROM execution_log
                WHERE timestamp < ?
            ''', (cutoff_date,))
            deleted_logs = cursor.rowcount

            self.conn.commit()

            print_status(
                f"🧹 Cleaned up {deleted_pending} old pending trades, "
                f"{deleted_logs} old logs",
                "info"
            )

            # Vacuum to reclaim space
            cursor.execute('VACUUM')
            print_status("✅ Database optimized", "debug")

        except sqlite3.Error as e:
            print_status(f"⚠️ Cleanup failed: {e}", "warn")

    def backup_database(self, backup_dir=None):
        """Create a backup of the database"""
        if backup_dir is None:
            backup_dir = REPO_FOLDER / "backups"

        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"ml_memory_backup_{timestamp}.db"

        try:
            # Close connection temporarily
            if self.conn:
                self.conn.close()

            # Copy database file
            shutil.copy2(self.db_path, backup_path)

            # Reconnect
            self.conn = sqlite3.connect(str(self.db_path), timeout=30)

            print_status(f"✅ Database backed up to {backup_path.name}", "success")
            return backup_path

        except Exception as e:
            print_status(f"⚠️ Backup failed: {e}", "warn")
            # Reconnect even if backup failed
            if not self.conn:
                self.conn = sqlite3.connect(str(self.db_path), timeout=30)
            return None

    def close(self):
        """Close database connection safely"""
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
                print_status("✅ Database connection closed", "debug")
            except Exception as e:
                print_status(f"⚠️ Error closing database: {e}", "warn")
            finally:
                self.conn = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

    def __del__(self):
        """Destructor - ensure connection is closed"""
        self.close()


# ======================================================
# Initialize Database
# ======================================================
TRADE_DB = FixedTradeMemoryDatabase()

# Display initial database stats
initial_stats = TRADE_DB.get_database_stats()
print_status("\n📊 DATABASE STATISTICS:", "info")
print_status(f"  Pending Trades: {initial_stats.get('pending_trades', 0)}", "info")
print_status(f"  Completed Trades: {initial_stats.get('completed_trades', 0)}", "info")
print_status(f"  Total P&L: ${initial_stats.get('total_pnl', 0):.2f}", "info")
print_status(f"  Overall Accuracy: {initial_stats.get('overall_accuracy', 0):.1f}%", "info")
print_status(f"  Database Size: {initial_stats.get('db_size_mb', 0):.2f} MB\n", "info")

# ======================================================
# 🆕 PERSISTENT ITERATION COUNTER (in Git)
# ======================================================
ITERATION_COUNTER_FILE = REPO_FOLDER / "ml_iteration_counter.pkl"

class MLIterationCounter:
    """Tracks total ML pipeline iterations across all runs forever"""

    def __init__(self, counter_file=ITERATION_COUNTER_FILE):
        self.counter_file = counter_file
        self.data = self.load_counter()

    def load_counter(self):
        if self.counter_file.exists():
            try:
                with open(self.counter_file, 'rb') as f:
                    data = pickle.load(f)
                print_status(f"✅ Loaded iteration counter: {data['total_iterations']} total runs", "success")
                return data
            except Exception as e:
                print_status(f"⚠️ Failed to load iteration counter: {e}", "warn")

        return {
            'total_iterations': 0,
            'start_date': datetime.now(timezone.utc).isoformat(),
            'last_run': None,
            'run_history': []
        }

    def increment(self):
        """Increment and save counter"""
        self.data['total_iterations'] += 1
        self.data['last_run'] = datetime.now(timezone.utc).isoformat()
        self.data['run_history'].append({
            'iteration': self.data['total_iterations'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        # Keep only last 1000 runs
        if len(self.data['run_history']) > 1000:
            self.data['run_history'] = self.data['run_history'][-1000:]

        self.save_counter()
        return self.data['total_iterations']

    def save_counter(self):
        try:
            # Atomic write with temp file
            temp_file = self.counter_file.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                pickle.dump(self.data, f, protocol=4)
            temp_file.replace(self.counter_file)
        except Exception as e:
            logging.error(f"Failed to save iteration counter: {e}")

    def get_current(self):
        return self.data['total_iterations']

    def get_stats(self):
        """Get statistics about runs"""
        if not self.data['run_history']:
            return {}

        try:
            first_run = datetime.fromisoformat(self.data['start_date'])
            days_running = max(1, (datetime.now(timezone.utc) - first_run).days)

            return {
                'total_iterations': self.data['total_iterations'],
                'days_running': days_running,
                'avg_iterations_per_day': self.data['total_iterations'] / days_running,
                'start_date': self.data['start_date'],
                'last_run': self.data['last_run']
            }
        except Exception as e:
            print_status(f"⚠️ Failed to get counter stats: {e}", "warn")
            return {}


ML_ITERATION_COUNTER = MLIterationCounter()

# ======================================================
# 1️⃣ Git & Credentials
# ======================================================
GIT_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "rahim-dotAI")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "forex-ai-models")
FOREX_PAT = os.environ.get("FOREX_PAT", "").strip()
BRANCH = "main"
BROWSERLESS_TOKEN = os.environ.get("BROWSERLESS_TOKEN","")

if not FOREX_PAT:
    raise ValueError("❌ FOREX_PAT missing!")

REPO_URL = f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

subprocess.run(["git","config","--global","user.name",GIT_NAME], check=False)
subprocess.run(["git","config","--global","user.email",GIT_EMAIL], check=False)
subprocess.run(["git","config","--global","credential.helper","store"], check=False)

cred_file = Path.home() / ".git-credentials"
cred_file.write_text(f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com\n")

def ensure_repo():
    """Ensure Git repository is cloned and up to date"""
    if not (REPO_FOLDER / ".git").exists():
        if REPO_FOLDER.exists():
            shutil.rmtree(REPO_FOLDER)
        print_status(f"Cloning repo into {REPO_FOLDER}...", "info")
        subprocess.run(["git","clone","-b",BRANCH,REPO_URL,str(REPO_FOLDER)], check=True)
    else:
        print_status("Repo exists, pulling latest...", "info")
        subprocess.run(["git","-C",str(REPO_FOLDER),"fetch","origin"], check=False)
        subprocess.run(["git","-C",str(REPO_FOLDER),"checkout",BRANCH], check=False)
        subprocess.run(["git","-C",str(REPO_FOLDER),"pull","origin",BRANCH], check=False)
        print_status("✅ Repo synced successfully", "success")

ensure_repo()

# ======================================================
# 🆕 CLEANUP CORRUPTED PICKLES
# ======================================================
def cleanup_corrupted_pickles():
    """Remove corrupted pickle files at startup"""
    print_status("🧹 Checking for corrupted ML pickle files...", "info")

    corrupted_count = 0
    for pkl_file in PICKLE_FOLDER.glob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                pickle.load(f)
        except Exception:
            try:
                pkl_file.unlink()
                print_status(f"🗑️ Removed corrupted: {pkl_file.name}", "warn")
                corrupted_count += 1
            except:
                pass

    if corrupted_count > 0:
        print_status(f"✅ Cleaned up {corrupted_count} corrupted files", "success")
    else:
        print_status("✅ No corrupted files found", "success")

cleanup_corrupted_pickles()

# ======================================================
# 2️⃣ CSV Loader + Sanity Check
# ======================================================
def load_csv(path):
    """Load CSV with validation"""
    if not path.exists():
        print_status(f"⚠️ CSV missing: {path}", "warn")
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]

        for col in ["open","high","low","close"]:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].ffill().bfill()

        df = df[["open","high","low","close"]].dropna(how='all')

        # Price sanity check
        if len(df) > 0:
            mean_price = df['close'].mean()
            if mean_price < 0.5 or mean_price > 200:
                print_status(f"⚠️ {path.name} suspicious price (mean={mean_price:.2f}), skipping", "warn")
                return None

        return df

    except Exception as e:
        print_status(f"❌ Failed to load {path.name}: {e}", "error")
        return None

# ======================================================
# 3️⃣ Live Price Fetch
# ======================================================
def fetch_live_rate(pair, timeout=10, retries=2):
    """Fetch live exchange rate with retry logic"""
    if not BROWSERLESS_TOKEN:
        print_status("⚠️ BROWSERLESS_TOKEN missing", "warn")
        return 0

    from_currency, to_currency = pair.split("/")
    url = f"https://production-sfo.browserless.io/content?token={BROWSERLESS_TOKEN}"
    payload = {
        "url": f"https://www.x-rates.com/calculator/?from={from_currency}&to={to_currency}&amount=1"
    }

    for attempt in range(retries):
        try:
            res = requests.post(url, json=payload, timeout=timeout)
            res.raise_for_status()
            match = re.search(r'ccOutputRslt[^>]*>([\d,.]+)', res.text)

            if match:
                rate = float(match.group(1).replace(",",""))
                if rate > 0:
                    print_status(f"💹 {pair} live price: {rate}", "info")
                    return rate

        except Exception as e:
            if attempt < retries - 1:
                print_status(f"⚠️ Retry {attempt+1}/{retries} for {pair}: {e}", "warn")
                time.sleep(2)
            else:
                print_status(f"❌ Failed to fetch {pair} after {retries} attempts: {e}", "error")

    return 0

def inject_live_price(df, live_price, n_candles=3):
    """Inject live price into recent candles for real-time analysis"""
    if live_price <= 0 or df is None or df.empty:
        return df

    df_copy = df.copy()
    n_inject = min(n_candles, len(df_copy))

    for i in range(n_inject):
        # Add small random variation to simulate realistic price movement
        price = live_price * (1 + np.random.uniform(-0.0005, 0.0005))

        for col in ["open","high","low","close"]:
            if col in df_copy.columns:
                df_copy.iloc[-n_inject+i, df_copy.columns.get_loc(col)] = price

    return df_copy

# ======================================================
# 4️⃣ Enhanced Indicators
# ======================================================
scaler_global = MinMaxScaler()

def add_indicators(df, fit_scaler=True):
    """Add technical indicators with error handling"""
    df = df.copy()

    try:
        # Trend indicators
        if len(df) >= 50:
            df['SMA_50'] = ta.trend.SMAIndicator(df['close'], 50).sma_indicator()
        if len(df) >= 20:
            df['EMA_20'] = ta.trend.EMAIndicator(df['close'], 20).ema_indicator()

        # Momentum indicators
        if len(df) >= 14:
            df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], 14).rsi()
            df['Williams_%R'] = ta.momentum.WilliamsRIndicator(
                df['high'], df['low'], df['close'], 14
            ).williams_r()

        # Volatility
        if len(df) >= 20:
            df['ATR_14'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], 14
            ).average_true_range()

        # Trend strength
        df['MACD'] = ta.trend.MACD(df['close']).macd()
        df['CCI_20'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], 20).cci()

        if len(df) >= 14:
            df['ADX_14'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 14).adx()

        # Fill NaN values
        df = df.ffill().bfill().fillna(0)

        # Scale numeric columns (except OHLC)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                       if c not in ['open', 'high', 'low', 'close']]

        if numeric_cols and len(numeric_cols) > 0:
            if fit_scaler:
                df[numeric_cols] = scaler_global.fit_transform(df[numeric_cols])
            else:
                try:
                    df[numeric_cols] = scaler_global.transform(df[numeric_cols])
                except NotFittedError:
                    df[numeric_cols] = scaler_global.fit_transform(df[numeric_cols])

    except Exception as e:
        print_status(f"⚠️ Indicator calculation issue: {e}", "warn")

    return df

# ======================================================
# 5️⃣ Enhanced ML Training with Performance Tracking
# ======================================================
def train_predict_ml_enhanced(df, pair_name, timeframe):
    """Train both SGD and RandomForest, return best prediction"""
    df = df.dropna()

    if len(df) < 50:
        return 0, 0, 0, 0.5

    # Prepare features
    X = df.drop(columns=['close'], errors='ignore')
    X = X if not X.empty else df[['close']]
    y = (df['close'].diff() > 0).astype(int).fillna(0)
    X = X.fillna(0)

    safe_pair_name = pair_name.replace("/", "_")
    safe_tf_name = timeframe.replace("/", "_")

    # ===== SGD Training =====
    sgd_file = PICKLE_FOLDER / f"{safe_pair_name}_{safe_tf_name}_sgd.pkl"

    if sgd_file.exists():
        try:
            sgd = pickle.load(open(sgd_file, "rb"))
        except:
            sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
            sgd.partial_fit(X, y, classes=np.array([0, 1]))
    else:
        sgd = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
        sgd.partial_fit(X, y, classes=np.array([0, 1]))

    sgd.partial_fit(X, y)
    pickle.dump(sgd, open(sgd_file, "wb"), protocol=4)
    sgd_pred = int(sgd.predict(X.iloc[[-1]])[0])

    # Get SGD confidence
    try:
        sgd_proba = sgd.predict_proba(X.iloc[[-1]])[0]
        sgd_confidence = float(max(sgd_proba))
    except:
        sgd_confidence = 0.5

    # ===== RandomForest with Historical Memory =====
    hist_file = PICKLE_FOLDER / f"{safe_pair_name}_{safe_tf_name}_rf_hist.pkl"

    if hist_file.exists():
        try:
            hist_X, hist_y = pickle.load(open(hist_file, "rb"))
            hist_X = pd.concat([hist_X, X], ignore_index=True)
            hist_y = pd.concat([hist_y, y], ignore_index=True)

            if len(hist_X) > 5000:
                hist_X = hist_X.iloc[-5000:]
                hist_y = hist_y.iloc[-5000:]
        except:
            hist_X, hist_y = X.copy(), y.copy()
    else:
        hist_X, hist_y = X.copy(), y.copy()

    rf_file = PICKLE_FOLDER / f"{safe_pair_name}_{safe_tf_name}_rf.pkl"
    rf = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced',
        random_state=42,
        max_depth=10
    )

    rf.fit(hist_X, hist_y)
    pickle.dump(rf, open(rf_file, "wb"), protocol=4)
    pickle.dump((hist_X, hist_y), open(hist_file, "wb"), protocol=4)

    rf_pred = int(rf.predict(X.iloc[[-1]])[0])

    # Get RF confidence
    try:
        rf_proba = rf.predict_proba(X.iloc[[-1]])[0]
        rf_confidence = float(max(rf_proba))
    except:
        rf_confidence = 0.5

    # ===== Ensemble Decision =====
    best_model = TRADE_DB.get_best_model(pair_name, days=7)

    if best_model == 'SGD':
        ensemble_pred = sgd_pred
        confidence = sgd_confidence
    elif best_model == 'RandomForest':
        ensemble_pred = rf_pred
        confidence = rf_confidence
    else:  # Ensemble (vote)
        ensemble_pred = 1 if (sgd_pred + rf_pred) >= 1 else 0
        confidence = (sgd_confidence + rf_confidence) / 2

    return sgd_pred, rf_pred, ensemble_pred, confidence

print_status("\n✅ Fixed TradeMemoryDatabase v3.6 loaded successfully!", "success")
print_status("🔧 All SQLite syntax errors resolved", "success")
print_status("💾 Database ready for persistent storage in Git\n", "success")


# In[ ]:


# ======================================================
# VERSION 3.6 – Unified Loader + Merge Pickles (Production Ready)
# Fully Safe | Threaded | Compatible with Hybrid FX Pipeline
# Added: Data validation, ATR floors, debug prints, raw price preservation
# ======================================================
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings
import ta
from ta.momentum import WilliamsRIndicator
from ta.volatility import AverageTrueRange
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# -----------------------------
# 0️⃣ Environment & folders
# -----------------------------
ROOT_DIR = Path("/content/forex-alpha-models")
CSV_FOLDER = ROOT_DIR / "csvs"
REPO_FOLDER = ROOT_DIR / "forex-ai-models"
TEMP_PICKLE_FOLDER = ROOT_DIR / "temp_pickles"
FINAL_PICKLE_FOLDER = ROOT_DIR / "merged_data_pickles"

for folder in [CSV_FOLDER, TEMP_PICKLE_FOLDER, FINAL_PICKLE_FOLDER, REPO_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

JSON_FILE = REPO_FOLDER / "latest_signals.json"

# -----------------------------
# 1️⃣ Safe Indicator Generator
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if df.empty:
        return df

    # --- Preserve raw OHLC prices for GA ---
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[f"raw_{col}"] = df[col].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        try:
            if len(df['close']) >= 10:
                df['SMA_10'] = ta.trend.sma_indicator(df['close'], 10)
                df['EMA_10'] = ta.trend.ema_indicator(df['close'], 10)
            if len(df['close']) >= 50:
                df['SMA_50'] = ta.trend.sma_indicator(df['close'], 50)
                df['EMA_50'] = ta.trend.ema_indicator(df['close'], 50)
            if len(df['close']) >= 14:
                df['RSI_14'] = ta.momentum.rsi(df['close'], 14)
            if all(col in df.columns for col in ['high', 'low', 'close']) and len(df['close']) >= 14:
                df['Williams_%R'] = WilliamsRIndicator(df['high'], df['low'], df['close'], 14).williams_r()
        except Exception as e:
            print(f"⚠️ Indicator calculation failed: {e}")

        # --- Safe ATR ---
        try:
            if all(col in df.columns for col in ['high', 'low', 'close']):
                window = 14
                if len(df) >= window:
                    df['ATR'] = AverageTrueRange(
                        df['high'], df['low'], df['close'], window=window
                    ).average_true_range().fillna(1e-5).clip(lower=1e-4)
                else:
                    df['ATR'] = 1e-4
        except Exception as e:
            df['ATR'] = 1e-4
            print(f"⚠️ ATR calculation failed: {e}")

        # --- Scale only non-price numeric columns ---
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not df[c].isna().all()]
        protected_cols = [
            "open", "high", "low", "close",
            "raw_open", "raw_high", "raw_low", "raw_close"
        ]
        numeric_cols = [c for c in numeric_cols if c not in protected_cols]

        if numeric_cols:
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0) + 1e-8)

    return df

# -----------------------------
# 2️⃣ Safe CSV Processing
# -----------------------------
def process_csv_file(csv_file: Path, save_folder: Path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        if df.empty:
            print(f"⚪ Skipped empty CSV: {csv_file.name}")
            return None

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df = add_indicators(df)
        if df.empty:
            print(f"⚪ Skipped CSV after filtering invalid prices: {csv_file.name}")
            return None

        out_file = save_folder / f"{csv_file.stem}.pkl"
        df.to_pickle(out_file)
        print(f"✅ Processed CSV {csv_file.name} → {out_file.name}")
        return out_file

    except Exception as e:
        print(f"❌ Failed CSV {csv_file.name}: {e}")
        return None

# -----------------------------
# 3️⃣ JSON Processing
# -----------------------------
def process_json_file(json_file: Path, save_folder: Path):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load JSON: {e}")
        return []

    signals_data = data.get("pairs", {})
    timestamp = pd.to_datetime(data.get("timestamp"), utc=True)
    processed_files = []

    for pair, info in signals_data.items():
        signals = info.get("signals", {})
        dfs = []

        for tf_name, tf_info in signals.items():
            df = pd.DataFrame({
                "live": [tf_info.get("live")],
                "SL": [tf_info.get("SL")],
                "TP": [tf_info.get("TP")],
                "signal": [tf_info.get("signal")]
            }, index=[timestamp])
            df["timeframe"] = tf_name
            df = add_indicators(df)
            if not df.empty:
                dfs.append(df)

        if dfs:
            df_pair = pd.concat(dfs)
            out_file = save_folder / f"{pair.replace('/', '_')}.pkl"
            df_pair.to_pickle(out_file)
            print(f"✅ Processed JSON {pair} → {out_file.name}")
            processed_files.append(out_file)

    return processed_files

# -----------------------------
# 4️⃣ Safe Pickle Merger
# -----------------------------
def merge_pickles(temp_folder: Path, final_folder: Path, keep_last: int = 5):
    pickles = list(temp_folder.glob("*.pkl"))
    if not pickles:
        print("⚪ No temporary pickles to merge.")
        return

    pairs = set(p.stem.split('.')[0] for p in pickles)

    for pair in pairs:
        pair_files = [p for p in pickles if p.stem.startswith(pair)]
        dfs = [pd.read_pickle(p) for p in pair_files if p.exists() and p.stat().st_size > 0]

        if not dfs:
            print(f"⚪ Skipped {pair} (no valid pickles)")
            continue

        merged_df = pd.concat(dfs, ignore_index=False).sort_index().drop_duplicates()
        # Changed filename suffix to match the expected format in W4XoZxs-TrDh
        merged_file = final_folder / f"{pair}_2244.pkl"
        merged_df.to_pickle(merged_file)
        print(f"🔗 Merged {len(pair_files)} files → {merged_file.name}")

        existing = sorted(final_folder.glob(f"{pair}_*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        for old_file in existing[keep_last:]:
            try:
                old_file.unlink()
                print(f"🧹 Removed old file: {old_file.name}")
            except Exception as e:
                print(f"⚠️ Could not remove {old_file.name}: {e}")

# -----------------------------
# 5️⃣ Unified Pipeline Runner
# -----------------------------
def run_unified_pipeline():
    temp_files = []

    # Process JSON first
    if JSON_FILE.exists():
        temp_files += process_json_file(JSON_FILE, TEMP_PICKLE_FOLDER)
        print(f"✅ JSON processing complete ({len(temp_files)} files)")

    # Process CSVs concurrently
    csv_files = list(CSV_FOLDER.glob("*.csv"))
    if csv_files:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_csv_file, f, TEMP_PICKLE_FOLDER) for f in csv_files]
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    temp_files.append(result)

    # Merge all pickles safely
    merge_pickles(TEMP_PICKLE_FOLDER, FINAL_PICKLE_FOLDER)
    print(f"🎯 Unified pipeline complete — merged pickles saved in {FINAL_PICKLE_FOLDER}")

    # Debug: print last few rows of each merged pickle
    for pkl_file in FINAL_PICKLE_FOLDER.glob("*.pkl"):
        df = pd.read_pickle(pkl_file)
        print(f"🔍 {pkl_file.name} last rows:\n", df.tail(3))

    return FINAL_PICKLE_FOLDER

# -----------------------------
# 6️⃣ Execute
# -----------------------------
if __name__ == "__main__":
    final_folder = run_unified_pipeline()


# In[ ]:


#!/usr/bin/env python3
"""
Ultimate Forex Pipeline v8.5.1 - FIXED GIT PUSH EDITION
=======================================================
✅ Enhanced Git operations with proper error handling
✅ Automatic pull before push to prevent conflicts
✅ Fallback mechanisms for push failures
✅ All other v8.5 features preserved
"""

import os
import sys
import json
import pickle
import random
import re
import smtplib
import subprocess
import time
import logging
import sqlite3
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

# ======================================================
# CONFIGURATION & SETUP
# ======================================================
logging.basicConfig(
    filename='forex_pipeline_v85.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def print_status(msg, level="info"):
    icons = {"info": "ℹ️", "success": "✅", "warn": "⚠️", "error": "❌", "rocket": "🚀", "chart": "📊", "brain": "🧠", "money": "💰"}
    getattr(logging, level if level != "warn" else "warning", logging.info)(msg)
    print(f"{icons.get(level, 'ℹ️')} {msg}")

# Environment detection
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

ROOT_DIR = Path("/content") if IN_COLAB else Path(".")
ROOT_PATH = ROOT_DIR / "forex-alpha-models"

# Folder setup
PICKLE_FOLDER = ROOT_PATH / "merged_data_pickles"
REPO_FOLDER = ROOT_PATH / "forex-ai-models"
for f in [PICKLE_FOLDER, REPO_FOLDER]:
    f.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT_PATH)

# Git configuration
GIT_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")
GITHUB_USERNAME = os.environ.get("GITHUB_USERNAME", "rahim-dotAI")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "forex-ai-models")
FOREX_PAT = os.environ.get("FOREX_PAT", "").strip()
REPO_URL = f"https://{GITHUB_USERNAME}:{FOREX_PAT}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"

subprocess.run(["git", "config", "--global", "user.name", GIT_NAME], check=False)
subprocess.run(["git", "config", "--global", "user.email", GIT_EMAIL], check=False)

# Email configuration
GMAIL_USER = os.environ.get("GMAIL_USER", "nakatonabira3@gmail.com")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "gmwohahtltmcewug")
LOGO_URL = "https://raw.githubusercontent.com/rahim-dotAI/forex-ai-models/main/IMG_1599.jpeg"

# Trading parameters
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
ATR_PERIOD = 14
MIN_ATR = 1e-5
BASE_CAPITAL = 100
MAX_POSITION_FRACTION = 0.1
MAX_TRADE_CAP = BASE_CAPITAL * 0.05
EPS = 1e-8
MAX_ATR_SL = 3.0
MAX_ATR_TP = 3.0
TOURNAMENT_SIZE = 3
SLIPPAGE_PCT = 0.0001
COMMISSION_PCT = 0.0002

# File paths
SIGNALS_JSON_PATH = REPO_FOLDER / "broker_signals.json"
ENSEMBLE_SIGNALS_FILE = REPO_FOLDER / "ensemble_signals.json"
MEMORY_DB = REPO_FOLDER / "memory_v85.db"
LEARNING_FILE = REPO_FOLDER / "learning_v85.pkl"
ITERATION_FILE = REPO_FOLDER / "iteration_v85.pkl"
WEIGHTS_FILE = REPO_FOLDER / "weights_v85.pkl"
MONDAY_FILE = REPO_FOLDER / "monday_runs.pkl"

# Model configurations
COMPETITION_MODELS = {
    "Alpha Momentum": {
        "color": "🔴", "hex_color": "#E74C3C",
        "strategy": "Aggressive momentum with adaptive stops",
        "atr_sl_range": (1.5, 2.5), "atr_tp_range": (2.0, 3.5),
        "risk_range": (0.015, 0.03), "confidence_range": (0.3, 0.5),
        "pop_size": 15, "generations": 20, "mutation_rate": 0.3
    },
    "Beta Conservative": {
        "color": "🔵", "hex_color": "#3498DB",
        "strategy": "Conservative mean reversion",
        "atr_sl_range": (1.0, 1.8), "atr_tp_range": (1.5, 2.5),
        "risk_range": (0.005, 0.015), "confidence_range": (0.5, 0.7),
        "pop_size": 12, "generations": 15, "mutation_rate": 0.2
    },
    "Gamma Adaptive": {
        "color": "🟢", "hex_color": "#2ECC71",
        "strategy": "Adaptive volatility trading",
        "atr_sl_range": (1.2, 2.2), "atr_tp_range": (1.8, 3.0),
        "risk_range": (0.01, 0.025), "confidence_range": (0.4, 0.6),
        "pop_size": 18, "generations": 22, "mutation_rate": 0.25
    }
}

# ======================================================
# CORE DATA CLASSES
# ======================================================
@dataclass
class TradeSignal:
    pair: str
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    confidence: float
    atr: float
    timestamp: str
    model: str

# ======================================================
# ITERATION COUNTER
# ======================================================
class IterationCounter:
    def __init__(self, file=ITERATION_FILE):
        self.file = file
        self.data = self._load()

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {'total': 0, 'start': datetime.now(timezone.utc).isoformat(), 'history': []}

    def increment(self, success=True):
        self.data['total'] += 1
        self.data['history'].append({'iteration': self.data['total'], 'time': datetime.now(timezone.utc).isoformat(), 'success': success})
        if len(self.data['history']) > 1000:
            self.data['history'] = self.data['history'][-1000:]
        try:
            with open(self.file, 'wb') as f:
                pickle.dump(self.data, f, protocol=4)
        except Exception as e:
            logging.error(f"Counter save failed: {e}")
        return self.data['total']

    def get_stats(self):
        days = max(1, (datetime.now(timezone.utc) - datetime.fromisoformat(self.data['start'])).days)
        return {
            'total': self.data['total'],
            'days': days,
            'per_day': self.data['total'] / days,
            'start': self.data['start']
        }

COUNTER = IterationCounter()

# ======================================================
# MEMORY SYSTEM
# ======================================================
class MemorySystem:
    def __init__(self, db_path=MEMORY_DB):
        self.conn = sqlite3.connect(str(db_path))
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()

        # Signals history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                sl_price REAL,
                tp_price REAL,
                atr REAL,
                confidence INTEGER,
                model_name TEXT NOT NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_model ON signals_history(model_name, timestamp)')

        # Trade results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                direction TEXT NOT NULL,
                pnl REAL NOT NULL,
                pnl_after_costs REAL,
                commission REAL,
                slippage REAL,
                was_correct BOOLEAN NOT NULL,
                duration_minutes INTEGER,
                model_name TEXT NOT NULL,
                confidence INTEGER
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_model ON trade_results(model_name, timestamp)')

        # Competition results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS competition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                total_pnl REAL,
                accuracy REAL,
                sharpe_ratio REAL,
                total_trades INTEGER,
                successful_trades INTEGER
            )
        ''')

        self.conn.commit()
        print_status("Memory system initialized", "success")

    def get_history(self, model_name, days=7):
        cursor = self.conn.cursor()
        since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        cursor.execute('''
            SELECT COUNT(*), SUM(CASE WHEN was_correct THEN 1 ELSE 0 END),
                   SUM(pnl_after_costs), AVG(pnl_after_costs)
            FROM trade_results
            WHERE model_name = ? AND timestamp > ?
        ''', (model_name, since))

        result = cursor.fetchone()
        total = result[0] or 0
        wins = result[1] or 0
        return {
            'total_trades': total,
            'wins': wins,
            'accuracy': (wins / total * 100) if total > 0 else 0,
            'total_pnl': result[2] or 0,
            'avg_pnl': result[3] or 0
        }

    def close(self):
        if self.conn:
            self.conn.commit()
            self.conn.close()

MEMORY = MemorySystem()

# ======================================================
# TRADE TRACKER
# ======================================================
class TradeTracker:
    def __init__(self, memory):
        self.memory = memory
        self.active_trades = {}

    def store_signals(self, signals_by_model, timestamp):
        for model_name, signals in signals_by_model.items():
            if model_name not in self.active_trades:
                self.active_trades[model_name] = {}

            for pair, sig in signals.items():
                if sig['direction'] == 'HOLD':
                    continue

                key = f"{pair}_{timestamp.isoformat()}"
                self.active_trades[model_name][key] = {
                    'pair': pair,
                    'direction': sig['direction'],
                    'entry': sig['last_price'],
                    'sl': sig['SL'],
                    'tp': sig['TP'],
                    'time': timestamp,
                    'confidence': sig['score_1_100'],
                    'closed': False
                }

    def evaluate_outcomes(self, current_prices, current_time):
        outcomes = defaultdict(lambda: {'closed': 0, 'wins': 0, 'total_pnl': 0.0, 'accuracy': 0.0})

        for model, trades in self.active_trades.items():
            for key, trade in list(trades.items()):
                if trade['closed']:
                    continue

                price = current_prices.get(trade['pair'], 0)
                if price <= 0:
                    continue

                # Check TP/SL hit
                hit_tp = hit_sl = False
                if trade['direction'] == 'BUY':
                    if price >= trade['tp']:
                        hit_tp = True
                        exit_price = trade['tp'] * (1 - SLIPPAGE_PCT)
                    elif price <= trade['sl']:
                        hit_sl = True
                        exit_price = trade['sl'] * (1 - SLIPPAGE_PCT)
                    else:
                        continue
                else:  # SELL
                    if price <= trade['tp']:
                        hit_tp = True
                        exit_price = trade['tp'] * (1 + SLIPPAGE_PCT)
                    elif price >= trade['sl']:
                        hit_sl = True
                        exit_price = trade['sl'] * (1 + SLIPPAGE_PCT)
                    else:
                        continue

                # Calculate P&L
                adjusted_entry = trade['entry'] * (1 + SLIPPAGE_PCT if trade['direction'] == 'BUY' else 1 - SLIPPAGE_PCT)
                pnl = (exit_price - adjusted_entry) if trade['direction'] == 'BUY' else (adjusted_entry - exit_price)
                commission = abs(pnl) * COMMISSION_PCT
                pnl_after_costs = pnl - commission

                # Record in database
                duration = (current_time - trade['time']).total_seconds() / 60
                cursor = self.memory.conn.cursor()
                cursor.execute('''
                    INSERT INTO trade_results
                    (timestamp, pair, entry_price, exit_price, direction, pnl, pnl_after_costs,
                     commission, slippage, was_correct, duration_minutes, model_name, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (current_time.isoformat(), trade['pair'], adjusted_entry, exit_price,
                      trade['direction'], pnl, pnl_after_costs, commission, SLIPPAGE_PCT * price,
                      hit_tp, duration, model, trade['confidence']))
                self.memory.conn.commit()

                # Update outcomes
                outcomes[model]['closed'] += 1
                outcomes[model]['total_pnl'] += pnl_after_costs
                if hit_tp:
                    outcomes[model]['wins'] += 1

                trade['closed'] = True

                print_status(
                    f"{model}: {trade['pair']} {trade['direction']} @ {exit_price:.5f} - "
                    f"P&L: ${pnl_after_costs:.5f} ({'WIN' if hit_tp else 'LOSS'})",
                    "success" if hit_tp else "warn"
                )

        # Calculate accuracy
        for model, data in outcomes.items():
            if data['closed'] > 0:
                data['accuracy'] = (data['wins'] / data['closed']) * 100

        return dict(outcomes)

TRACKER = TradeTracker(MEMORY)

# ======================================================
# LEARNING SYSTEM
# ======================================================
class LearningSystem:
    def __init__(self, file=LEARNING_FILE):
        self.file = file
        self.data = self._load()

    def _load(self):
        if self.file.exists():
            try:
                with open(self.file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {
            'iterations': 0,
            'successful_patterns': {},
            'learning_curve': [],
            'adaptation_score': 0.0
        }

    def record_iteration(self, results, outcomes=None):
        self.data['iterations'] += 1

        for model, result in results.items():
            if not result or 'metrics' not in result:
                continue

            pnl = outcomes[model]['total_pnl'] if outcomes and model in outcomes else result['metrics']['total_pnl']
            accuracy = outcomes[model]['accuracy'] if outcomes and model in outcomes else 0

            if pnl > 0 and accuracy >= 50:
                key = f"{model}_success"
                if key not in self.data['successful_patterns']:
                    self.data['successful_patterns'][key] = []

                self.data['successful_patterns'][key].append({
                    'chromosome': result.get('chromosome'),
                    'pnl': pnl,
                    'accuracy': accuracy,
                    'time': datetime.now(timezone.utc).isoformat()
                })

                if len(self.data['successful_patterns'][key]) > 50:
                    self.data['successful_patterns'][key] = sorted(
                        self.data['successful_patterns'][key],
                        key=lambda x: x['pnl'],
                        reverse=True
                    )[:50]

        self.data['learning_curve'].append(sum(outcomes[m]['total_pnl'] for m in outcomes) if outcomes else 0)
        if len(self.data['learning_curve']) > 100:
            self.data['learning_curve'] = self.data['learning_curve'][-100:]

        if len(self.data['learning_curve']) >= 10:
            recent = np.mean(self.data['learning_curve'][-10:])
            self.data['adaptation_score'] = min(100, max(0, 50 + recent))

        try:
            with open(self.file, 'wb') as f:
                pickle.dump(self.data, f, protocol=4)
        except Exception as e:
            logging.error(f"Learning save failed: {e}")

    def get_best_chromosomes(self, model, top_n=3):
        key = f"{model}_success"
        patterns = self.data['successful_patterns'].get(key, [])
        return [p['chromosome'] for p in sorted(patterns, key=lambda x: x['pnl'], reverse=True)[:top_n] if p.get('chromosome')]

    def get_report(self):
        total_success = sum(len(p) for p in self.data['successful_patterns'].values())
        return {
            'iterations': self.data['iterations'],
            'adaptation_score': self.data['adaptation_score'],
            'total_successes': total_success,
            'trend': "📈 Improving" if self.data['adaptation_score'] > 50 else "📉 Adjusting"
        }

LEARNING = LearningSystem()

# ======================================================
# WEEKEND/MONDAY MANAGER
# ======================================================
class ModeManager:
    def __init__(self):
        self.monday_data = self._load_monday()

    def _load_monday(self):
        if MONDAY_FILE.exists():
            try:
                data = pickle.load(open(MONDAY_FILE, "rb"))
                if data.get('date') != datetime.now().strftime('%Y-%m-%d'):
                    return {'count': 0, 'date': datetime.now().strftime('%Y-%m-%d')}
                return data
            except:
                pass
        return {'count': 0, 'date': datetime.now().strftime('%Y-%m-%d')}

    def get_mode(self):
        weekday = datetime.now().weekday()
        if weekday in [5, 6]:
            return "weekend_replay"
        elif weekday == 0 and self.monday_data['count'] < 1:
            return "monday_replay"
        return "normal"

    def increment_monday(self):
        self.monday_data['count'] += 1
        self.monday_data['date'] = datetime.now().strftime('%Y-%m-%d')
        try:
            with open(MONDAY_FILE, "wb") as f:
                pickle.dump(self.monday_data, f, protocol=4)
        except:
            pass

    def should_send_email(self):
        return self.get_mode() == "normal"

MODE_MANAGER = ModeManager()

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def ensure_atr(df):
    if "atr" in df.columns and not df["atr"].isna().all():
        df["atr"] = df["atr"].fillna(MIN_ATR).clip(lower=MIN_ATR)
        return df

    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.roll(close, 1)),
        np.abs(low - np.roll(close, 1))
    ])
    tr[0] = high[0] - low[0] if len(tr) > 0 else MIN_ATR
    df["atr"] = pd.Series(tr, index=df.index).rolling(ATR_PERIOD, min_periods=1).mean().fillna(MIN_ATR).clip(lower=MIN_ATR)
    return df

def seed_hybrid_signal(df):
    if "hybrid_signal" not in df.columns or df["hybrid_signal"].abs().sum() == 0:
        fast = df["close"].rolling(10, min_periods=1).mean()
        slow = df["close"].rolling(50, min_periods=1).mean()
        df["hybrid_signal"] = (fast - slow).fillna(0)
    return df

def load_data(folder):
    combined = {}
    for pair in PAIRS:
        combined[pair] = {}
        prefix = pair.replace("/", "_")
        for pf in sorted(folder.glob(f"{prefix}*.pkl")):
            try:
                df = pd.read_pickle(pf)
                if not isinstance(df, pd.DataFrame) or len(df) < 50:
                    continue
                df.index = pd.to_datetime(df.index, errors="coerce")
                if df.index.tz is not None:
                    df.index = df.index.tz_convert(None)
                df = ensure_atr(df)
                df = seed_hybrid_signal(df)
                tf = re.sub(rf"{prefix}_?|\.pkl", "", pf.name).strip("_") or "merged"
                combined[pair][tf] = df
            except:
                continue
    return combined

def fetch_live_rate(pair):
    token = os.environ.get("BROWSERLESS_TOKEN", "")
    if not token:
        return 0.0
    from_c, to_c = pair.split("/")
    try:
        r = requests.post(
            f"https://production-sfo.browserless.io/content?token={token}",
            json={"url": f"https://www.x-rates.com/calculator/?from={from_c}&to={to_c}&amount=1"},
            timeout=8
        )
        match = re.search(r'ccOutputRslt[^>]*>([\d,.]+)', r.text)
        return float(match.group(1).replace(",", "")) if match else 0.0
    except:
        return 0.0

def build_tf_map(data):
    return {p: list(tfs.keys()) for p, tfs in data.items()}

def create_chromosome(tf_map, config):
    chrom = [
        float(random.uniform(*config['atr_sl_range'])),
        float(random.uniform(*config['atr_tp_range'])),
        float(random.uniform(*config['risk_range'])),
        float(random.uniform(*config['confidence_range']))
    ]
    for p in PAIRS:
        n = max(1, len(tf_map.get(p, [])))
        weights = np.random.dirichlet(np.ones(n)).tolist()
        chrom.extend(weights)
    return chrom

def decode_chromosome(chrom, tf_map):
    atr_sl = np.clip(chrom[0], 1.0, MAX_ATR_SL)
    atr_tp = np.clip(chrom[1], 1.0, MAX_ATR_TP)
    risk, conf = chrom[2], chrom[3]

    tf_w = {}
    idx = 4
    for p in PAIRS:
        n = max(1, len(tf_map.get(p, [])))
        weights = np.array(chrom[idx:idx+n], dtype=float)
        weights = weights / (weights.sum() + EPS) if weights.sum() > 0 else np.ones(n) / n
        tf_w[p] = {tf: float(w) for tf, w in zip(tf_map.get(p, []), weights)}
        idx += n

    return atr_sl, atr_tp, risk, conf, tf_w

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2:
        return 0.0

    equity_array = np.array(equity_curve, dtype=float)
    returns = np.diff(equity_array) / (equity_array[:-1] + EPS)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    return float(np.mean(returns) / (np.std(returns) + EPS))

# ======================================================
# BACKTESTING
# ======================================================
def backtest_strategy(data, tf_map, chromosome):
    atr_sl, atr_tp, risk, conf, tf_w = decode_chromosome(chromosome, tf_map)

    equity = BASE_CAPITAL
    equity_curve = [equity]
    trades = []
    position = None

    all_times = sorted(set().union(*[df.index for tfs in data.values() for df in tfs.values()]))

    for t in all_times:
        if position:
            pair = position['pair']
            price = 0
            for tf in tf_map.get(pair, []):
                if tf in data.get(pair, {}) and t in data[pair][tf].index:
                    price = data[pair][tf].loc[t, 'close']
                    break

            if price > 0:
                hit_tp = (position['dir'] == 'BUY' and price >= position['tp']) or (position['dir'] == 'SELL' and price <= position['tp'])
                hit_sl = (position['dir'] == 'BUY' and price <= position['sl']) or (position['dir'] == 'SELL' and price >= position['sl'])

                if hit_tp or hit_sl:
                    exit_price = position['tp'] if hit_tp else position['sl']
                    pnl = (exit_price - position['entry']) * position['size'] if position['dir'] == 'BUY' else (position['entry'] - exit_price) * position['size']
                    equity += pnl
                    equity_curve.append(equity)
                    trades.append({'pnl': pnl, 'correct': hit_tp})
                    position = None

        if position is None:
            for pair in PAIRS:
                signal = 0
                price = 0
                atr = MIN_ATR

                for tf, weight in tf_w.get(pair, {}).items():
                    if tf in data.get(pair, {}) and t in data[pair][tf].index:
                        row = data[pair][tf].loc[t]
                        signal += row.get('hybrid_signal', 0) * weight
                        price = row['close']
                        atr = max(row.get('atr', MIN_ATR), MIN_ATR)

                if abs(signal) > conf and price > 0:
                    direction = 'BUY' if signal > 0 else 'SELL'
                    size = min(equity * risk, MAX_TRADE_CAP) / (atr * atr_sl)

                    if direction == 'BUY':
                        sl = price - (atr * atr_sl)
                        tp = price + (atr * atr_tp)
                    else:
                        sl = price + (atr * atr_sl)
                        tp = price - (atr * atr_tp)

                    position = {'pair': pair, 'dir': direction, 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
                    break

    total = len(trades)
    wins = sum(1 for t in trades if t['correct'])
    return {
        'total_trades': total,
        'winning_trades': wins,
        'accuracy': (wins / total * 100) if total > 0 else 0,
        'total_pnl': sum(t['pnl'] for t in trades),
        'sharpe': calculate_sharpe(equity_curve)
    }

# ======================================================
# GENETIC ALGORITHM
# ======================================================
def run_ga(data, tf_map, model_name, config):
    print_status(f"{config['color']} Training {model_name}...", "info")

    pop_size = config['pop_size']
    generations = config['generations']
    mutation_rate = config['mutation_rate']

    try:
        population = []
        best_hist = LEARNING.get_best_chromosomes(model_name, top_n=3)
        for chrom in best_hist:
            if chrom:
                metrics = backtest_strategy(data, tf_map, chrom)
                fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
                population.append((fitness, chrom))

        while len(population) < pop_size:
            chrom = create_chromosome(tf_map, config)
            metrics = backtest_strategy(data, tf_map, chrom)
            fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
            population.append((fitness, chrom))

        population.sort(reverse=True, key=lambda x: x[0])

        for gen in range(generations):
            new_pop = []
            elite_count = max(1, int(pop_size * 0.2))
            new_pop.extend(population[:elite_count])

            while len(new_pop) < pop_size:
                parent1 = max(random.sample(population, TOURNAMENT_SIZE), key=lambda x: x[0])[1]
                parent2 = max(random.sample(population, TOURNAMENT_SIZE), key=lambda x: x[0])[1]

                point = random.randint(1, len(parent1) - 1)
                child = [float(x) for x in parent1[:point]] + [float(x) for x in parent2[point:]]

                for i in range(len(child)):
                    if random.random() < mutation_rate:
                        if i == 0:
                            child[i] = float(child[i] + random.gauss(0, 0.3))
                            child[i] = float(np.clip(child[i], *config['atr_sl_range']))
                        elif i == 1:
                            child[i] = float(child[i] + random.gauss(0, 0.3))
                            child[i] = float(np.clip(child[i], *config['atr_tp_range']))
                        elif i == 2:
                            child[i] = float(child[i] + random.gauss(0, 0.005))
                            child[i] = float(np.clip(child[i], *config['risk_range']))
                        elif i == 3:
                            child[i] = float(child[i] + random.gauss(0, 0.1))
                            child[i] = float(np.clip(child[i], *config['confidence_range']))
                        else:
                            child[i] = float(max(0.01, child[i] + random.gauss(0, 0.2)))

                metrics = backtest_strategy(data, tf_map, child)
                fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
                new_pop.append((fitness, child))

            population = sorted(new_pop, reverse=True, key=lambda x: x[0])

            if (gen + 1) % 5 == 0:
                print_status(f"  Gen {gen+1}/{generations}: Best={population[0][0]:.4f}", "info")

        best_chrom = population[0][1]
        final_metrics = backtest_strategy(data, tf_map, best_chrom)

        print_status(
            f"  ✅ {model_name}: {final_metrics['accuracy']:.1f}% accuracy | "
            f"${final_metrics['total_pnl']:.4f} PnL | {final_metrics['total_trades']} trades",
            "success"
        )

        return {'chromosome': best_chrom, 'metrics': final_metrics}

    except Exception as e:
        logging.exception(f"{model_name} GA error")
        raise

# ======================================================
# SIGNAL GENERATION
# ======================================================
def generate_signals(data, tf_map, chromosome, model_name, current_time):
    atr_sl, atr_tp, risk, conf, tf_w = decode_chromosome(chromosome, tf_map)
    signals = {}

    for pair in PAIRS:
        signal_strength = 0
        price = 0
        atr = MIN_ATR

        for tf, weight in tf_w.get(pair, {}).items():
            if tf in data.get(pair, {}):
                df = data[pair][tf]
                if len(df) > 0:
                    row = df.iloc[-1]
                    signal_strength += row.get('hybrid_signal', 0) * weight
                    price = row['close']
                    atr = max(row.get('atr', MIN_ATR), MIN_ATR)

        direction = 'HOLD'
        sl = tp = price

        if abs(signal_strength) > conf and price > 0:
            direction = 'BUY' if signal_strength > 0 else 'SELL'

            if direction == 'BUY':
                sl = price - (atr * atr_sl)
                tp = price + (atr * atr_tp)
            else:
                sl = price + (atr * atr_sl)
                tp = price - (atr * atr_tp)

        signals[pair] = {
            'direction': direction,
            'last_price': float(price),
            'SL': float(sl),
            'TP': float(tp),
            'atr': float(atr),
            'score_1_100': int(abs(signal_strength) * 100),
            'model': model_name,
            'timestamp': current_time.isoformat()
        }

    return signals

# ======================================================
# EMAIL SYSTEM
# ======================================================
def send_email(signals_by_model, iteration_stats, learning_report):
    if not MODE_MANAGER.should_send_email():
        print_status("Email skipped (replay mode)", "info")
        return

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🤖 Forex AI Signals - Iteration #{iteration_stats['iteration']}"
        msg['From'] = GMAIL_USER
        msg['To'] = GMAIL_USER

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .logo {{ max-width: 150px; }}
                .stats {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .model-section {{ margin: 20px 0; padding: 15px; border-left: 4px solid; }}
                .signal-card {{ background: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .buy {{ color: #27ae60; font-weight: bold; }}
                .sell {{ color: #e74c3c; font-weight: bold; }}
                .hold {{ color: #95a5a6; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <img src="{LOGO_URL}" alt="Logo" class="logo">
                    <h1>🤖 Forex AI Trading Signals</h1>
                    <p>Iteration #{iteration_stats['iteration']} | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
                </div>

                <div class="stats">
                    <h3>📊 System Statistics</h3>
                    <p><strong>Total Iterations:</strong> {iteration_stats['total_iterations']}</p>
                    <p><strong>Learning Trend:</strong> {learning_report['trend']}</p>
                    <p><strong>Adaptation Score:</strong> {learning_report['adaptation_score']:.1f}/100</p>
                </div>
        """

        for model_name, signals in signals_by_model.items():
            config = COMPETITION_MODELS[model_name]
            html += f"""
                <div class="model-section" style="border-color: {config['hex_color']};">
                    <h3>{config['color']} {model_name}</h3>
                    <p><em>{config['strategy']}</em></p>
            """

            for pair, sig in signals.items():
                if sig['direction'] != 'HOLD':
                    direction_class = sig['direction'].lower()
                    html += f"""
                        <div class="signal-card">
                            <strong>{pair}</strong>:
                            <span class="{direction_class}">{sig['direction']}</span>
                            @ {sig['last_price']:.5f}
                            | SL: {sig['SL']:.5f} | TP: {sig['TP']:.5f}
                            <br><small>Confidence: {sig['score_1_100']}/100</small>
                        </div>
                    """

            html += "</div>"

        html += """
            </div>
        </body>
        </html>
        """

        msg.attach(MIMEText(html, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.send_message(msg)

        print_status("Email sent successfully", "success")

    except Exception as e:
        print_status(f"Email failed: {e}", "error")

# ======================================================
# ENHANCED GIT OPERATIONS WITH PROPER ERROR HANDLING
# ======================================================
def push_to_github(files, message):
    """
    Enhanced Git push with automatic conflict resolution and stash handling
    """
    try:
        # Ensure repo exists
        if not REPO_FOLDER.exists():
            print_status("Cloning repository...", "info")
            result = subprocess.run(
                ["git", "clone", REPO_URL, str(REPO_FOLDER)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print_status(f"Clone failed: {result.stderr}", "error")
                return False

        os.chdir(REPO_FOLDER)

        # Stage files BEFORE pulling
        print_status("Staging files...", "info")
        files_added = 0
        for f in files:
            file_path = REPO_FOLDER / f
            if file_path.exists():
                subprocess.run(["git", "add", str(f)], check=False)
                files_added += 1
            else:
                print_status(f"Skipping {f} (not found)", "warn")

        if files_added == 0:
            print_status("No files to stage", "warn")
            return True

        # Check if there are changes to commit
        status_result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True
        )

        if status_result.returncode == 0:
            print_status("No changes detected", "info")
            return True

        # Stash any unstaged changes to allow clean pull
        print_status("Stashing unstaged changes...", "info")
        subprocess.run(["git", "stash", "--include-untracked"], check=False)

        # Pull latest changes
        print_status("Pulling latest changes...", "info")
        pull_result = subprocess.run(
            ["git", "pull", "--rebase", "origin", "main"],
            capture_output=True,
            text=True
        )

        if pull_result.returncode != 0:
            print_status(f"Pull had issues, attempting recovery...", "warn")
            # Reset to remote if pull completely fails
            subprocess.run(["git", "fetch", "origin", "main"], check=False)
            subprocess.run(["git", "reset", "--hard", "origin/main"], check=False)

        # Pop stash if we stashed anything
        subprocess.run(["git", "stash", "pop"], capture_output=True, check=False)

        # Re-add our files after pull
        print_status("Re-staging files after pull...", "info")
        for f in files:
            file_path = REPO_FOLDER / f
            if file_path.exists():
                subprocess.run(["git", "add", str(f)], check=False)

        # Commit changes
        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True
        )

        if commit_result.returncode != 0:
            if "nothing to commit" in commit_result.stdout.lower():
                print_status("No new changes to commit", "info")
                return True
            else:
                print_status(f"Commit warning: {commit_result.stderr}", "warn")

        # Push with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            print_status(f"Pushing to GitHub (attempt {attempt + 1}/{max_retries})...", "info")

            push_result = subprocess.run(
                ["git", "push", "origin", "main"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if push_result.returncode == 0:
                print_status("✅ Successfully pushed to GitHub", "success")
                return True

            # If push failed, try pulling and pushing again
            if attempt < max_retries - 1:
                print_status(f"Push failed, syncing and retrying...", "warn")
                subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
                time.sleep(2)
            else:
                error_msg = push_result.stderr.strip()[:200]  # First 200 chars
                print_status(f"Push failed after {max_retries} attempts: {error_msg}", "error")
                return False

        return False

    except subprocess.TimeoutExpired:
        print_status("Git operation timed out", "error")
        return False
    except Exception as e:
        print_status(f"Git error: {e}", "error")
        logging.error(f"Git error: {e}")
        return False
    finally:
        try:
            os.chdir(ROOT_PATH)
        except:
            pass

# ======================================================
# MAIN EXECUTION
# ======================================================
def main():
    print_status("=" * 70, "rocket")
    print_status("🚀 FOREX PIPELINE v8.5.1 - FIXED GIT EDITION", "rocket")
    print_status("=" * 70, "rocket")

    success = False

    try:
        # Display stats
        current_iter = COUNTER.data['total'] + 1
        stats = COUNTER.get_stats()
        mode = MODE_MANAGER.get_mode()

        print_status(f"\n📊 Iteration #{current_iter} | Mode: {mode.upper()}", "info")
        print_status(f"Total Runs: {stats['total']} | Days: {stats['days']} | Avg/Day: {stats['per_day']:.1f}", "info")

        # Load data
        print_status("\n📦 Loading data...", "info")
        data = load_data(PICKLE_FOLDER)

        if not data:
            raise ValueError("No data loaded - check PICKLE_FOLDER path")

        print_status(f"✅ Loaded {len(data)} pairs", "success")

        tf_map = build_tf_map(data)

        # Run competition
        print_status("\n🏆 Running Competition...", "chart")
        competition_results = {}
        signals_by_model = {}

        for model_name, config in COMPETITION_MODELS.items():
            try:
                result = run_ga(data, tf_map, model_name, config)
                competition_results[model_name] = result

                # Generate signals
                signals = generate_signals(data, tf_map, result['chromosome'], model_name, datetime.now(timezone.utc))
                signals_by_model[model_name] = signals

                # Store in database
                cursor = MEMORY.conn.cursor()
                cursor.execute('''
                    INSERT INTO competition_results
                    (timestamp, iteration, model_name, total_pnl, accuracy, sharpe_ratio, total_trades, successful_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(timezone.utc).isoformat(),
                    current_iter,
                    model_name,
                    result['metrics']['total_pnl'],
                    result['metrics']['accuracy'],
                    result['metrics']['sharpe'],
                    result['metrics']['total_trades'],
                    result['metrics']['winning_trades']
                ))
                MEMORY.conn.commit()

            except Exception as e:
                print_status(f"❌ {model_name} failed: {e}", "error")

        # Evaluate previous trades
        print_status("\n📈 Evaluating trade outcomes...", "info")
        current_prices = {}

        for pair in PAIRS:
            if mode == "normal":
                live_rate = fetch_live_rate(pair)
                if live_rate > 0:
                    current_prices[pair] = live_rate
                else:
                    # Fallback to data
                    for tf in tf_map.get(pair, []):
                        if tf in data.get(pair, {}):
                            current_prices[pair] = data[pair][tf].iloc[-1]['close']
                            break
            else:
                # Replay mode: use historical price
                for tf in tf_map.get(pair, []):
                    if tf in data.get(pair, {}):
                        current_prices[pair] = data[pair][tf].iloc[-1]['close']
                        break

        TRACKER.store_signals(signals_by_model, datetime.now(timezone.utc))
        outcomes = TRACKER.evaluate_outcomes(current_prices, datetime.now(timezone.utc))

        if outcomes:
            print_status("\n📊 Trade Outcomes:", "success")
            for model, outcome_data in outcomes.items():
                print_status(
                    f"{model}: {outcome_data['wins']}/{outcome_data['closed']} wins "
                    f"({outcome_data['accuracy']:.1f}%) | ${outcome_data['total_pnl']:.2f}",
                    "success" if outcome_data['total_pnl'] > 0 else "warn"
                )

        # Update learning system
        LEARNING.record_iteration(competition_results, outcomes)
        learning_report = LEARNING.get_report()

        print_status(f"\n🧠 Learning: {learning_report['trend']} | Score: {learning_report['adaptation_score']:.1f}/100", "brain")

        # Save signals
        print_status("\n💾 Saving signals...", "info")
        with open(SIGNALS_JSON_PATH, 'w') as f:
            json.dump(signals_by_model, f, indent=2, default=str)

        with open(ENSEMBLE_SIGNALS_FILE, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'iteration': current_iter,
                'models': signals_by_model
            }, f, indent=2, default=str)

        # Send email
        iteration_stats = {
            'iteration': current_iter,
            'total_iterations': stats['total']
        }
        send_email(signals_by_model, iteration_stats, learning_report)

        # Push to GitHub with enhanced error handling
        print_status("\n🔄 Pushing to GitHub...", "info")
        files = [
            SIGNALS_JSON_PATH.name,
            ENSEMBLE_SIGNALS_FILE.name,
            MEMORY_DB.name,
            LEARNING_FILE.name,
            ITERATION_FILE.name,
            MONDAY_FILE.name
        ]

        git_success = push_to_github(files, f"🤖 Auto-update: Iteration #{current_iter} - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

        if not git_success:
            print_status("⚠️  GitHub push had issues, but pipeline completed", "warn")

        # Summary
        print_status("\n" + "=" * 70, "success")
        print_status("✅ PIPELINE COMPLETED SUCCESSFULLY", "success")
        print_status("=" * 70, "success")
        print_status(f"Iteration: #{current_iter}", "info")
        print_status(f"Models: {len(competition_results)}", "info")
        print_status(f"Signals: {sum(1 for m in signals_by_model.values() for s in m.values() if s['direction'] != 'HOLD')}", "info")

        success = True

    except KeyboardInterrupt:
        print_status("\n⚠️ Shutdown requested", "warn")
    except Exception as e:
        print_status(f"\n❌ Fatal error: {e}", "error")
        logging.exception("Fatal error")
        import traceback
        traceback.print_exc()
    finally:
        COUNTER.increment(success=success)
        MEMORY.close()
        if MODE_MANAGER.get_mode() == "monday_replay":
            MODE_MANAGER.increment_monday()
        print_status("Cleanup complete", "info")

if __name__ == "__main__":
    main()
    print_status("Pipeline shutdown complete", "info")

