#!/usr/bin/env python3
"""
Ultimate Forex Pipeline v8.5.1 - FIXED GIT EDITION
Simplified for GitHub Actions execution
"""

import os
import sys
import json
import pickle
import random
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ======================================================
# CONFIGURATION
# ======================================================
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def print_status(msg, level="info"):
    icons = {"info": "ℹ️", "success": "✅", "warn": "⚠️", "error": "❌"}
    print(f"{icons.get(level, 'ℹ️')} {msg}")

# Paths - all in current directory for GitHub Actions
ROOT_PATH = Path(".")
PICKLE_FOLDER = ROOT_PATH
REPO_FOLDER = ROOT_PATH

# Git config
GIT_NAME = os.environ.get("GIT_USER_NAME", "Forex AI Bot")
GIT_EMAIL = os.environ.get("GIT_USER_EMAIL", "nakatonabira3@gmail.com")

# Trading parameters
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD"]
ATR_PERIOD = 14
MIN_ATR = 1e-5
BASE_CAPITAL = 100
EPS = 1e-8

# File paths
SIGNALS_JSON = ROOT_PATH / "broker_signals.json"
ENSEMBLE_JSON = ROOT_PATH / "ensemble_signals.json"
MEMORY_DB = ROOT_PATH / "memory_v85.db"
LEARNING_FILE = ROOT_PATH / "learning_v85.pkl"
ITERATION_FILE = ROOT_PATH / "iteration_v85.pkl"

# Model configs
COMPETITION_MODELS = {
    "Alpha Momentum": {
        "color": "🔴",
        "atr_sl_range": (1.5, 2.5),
        "atr_tp_range": (2.0, 3.5),
        "risk_range": (0.015, 0.03),
        "confidence_range": (0.3, 0.5),
        "pop_size": 15,
        "generations": 20,
        "mutation_rate": 0.3
    },
    "Beta Conservative": {
        "color": "🔵",
        "atr_sl_range": (1.0, 1.8),
        "atr_tp_range": (1.5, 2.5),
        "risk_range": (0.005, 0.015),
        "confidence_range": (0.5, 0.7),
        "pop_size": 12,
        "generations": 15,
        "mutation_rate": 0.2
    },
    "Gamma Adaptive": {
        "color": "🟢",
        "atr_sl_range": (1.2, 2.2),
        "atr_tp_range": (1.8, 3.0),
        "risk_range": (0.01, 0.025),
        "confidence_range": (0.4, 0.6),
        "pop_size": 18,
        "generations": 22,
        "mutation_rate": 0.25
    }
}

# ======================================================
# CORE CLASSES
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
        return {'total': 0, 'start': datetime.now(timezone.utc).isoformat()}
    
    def increment(self):
        self.data['total'] += 1
        try:
            with open(self.file, 'wb') as f:
                pickle.dump(self.data, f, protocol=4)
        except Exception as e:
            logging.error(f"Counter save failed: {e}")
        return self.data['total']
    
    def get_stats(self):
        days = max(1, (datetime.now(timezone.utc) - datetime.fromisoformat(self.data['start'])).days)
        return {'total': self.data['total'], 'days': days, 'per_day': self.data['total'] / days}

COUNTER = IterationCounter()

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
    if "hybrid_signal" not in df.columns:
        fast = df["close"].rolling(10, min_periods=1).mean()
        slow = df["close"].rolling(50, min_periods=1).mean()
        df["hybrid_signal"] = (fast - slow).fillna(0)
    return df

def load_data(folder):
    combined = {}
    for pair in PAIRS:
        prefix = pair.replace("/", "_")
        pkl_file = folder / f"{prefix}_2244.pkl"
        
        if not pkl_file.exists():
            print_status(f"⚠️  Missing {pkl_file.name}", "warn")
            continue
        
        try:
            df = pd.read_pickle(pkl_file)
            if not isinstance(df, pd.DataFrame) or len(df) < 50:
                continue
            
            df.index = pd.to_datetime(df.index, errors="coerce")
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            
            df = ensure_atr(df)
            df = seed_hybrid_signal(df)
            
            combined[pair] = {"merged": df}
            print_status(f"✅ Loaded {pair}: {len(df)} rows", "success")
            
        except Exception as e:
            print_status(f"❌ Error loading {pkl_file.name}: {e}", "error")
    
    return combined

def create_chromosome(config):
    return [
        float(random.uniform(*config['atr_sl_range'])),
        float(random.uniform(*config['atr_tp_range'])),
        float(random.uniform(*config['risk_range'])),
        float(random.uniform(*config['confidence_range']))
    ]

def decode_chromosome(chrom):
    return np.clip(chrom[0], 1.0, 3.0), np.clip(chrom[1], 1.0, 3.0), chrom[2], chrom[3]

# ======================================================
# BACKTEST
# ======================================================
def backtest_strategy(data, chromosome):
    atr_sl, atr_tp, risk, conf = decode_chromosome(chromosome)
    
    equity = BASE_CAPITAL
    trades = []
    position = None
    
    all_times = sorted(set().union(*[df.index for tfs in data.values() for df in tfs.values()]))
    
    for t in all_times:
        if position:
            pair = position['pair']
            if pair in data and "merged" in data[pair] and t in data[pair]["merged"].index:
                price = data[pair]["merged"].loc[t, 'close']
                
                hit_tp = (position['dir'] == 'BUY' and price >= position['tp']) or (position['dir'] == 'SELL' and price <= position['tp'])
                hit_sl = (position['dir'] == 'BUY' and price <= position['sl']) or (position['dir'] == 'SELL' and price >= position['sl'])
                
                if hit_tp or hit_sl:
                    exit_price = position['tp'] if hit_tp else position['sl']
                    pnl = (exit_price - position['entry']) * position['size'] if position['dir'] == 'BUY' else (position['entry'] - exit_price) * position['size']
                    equity += pnl
                    trades.append({'pnl': pnl, 'correct': hit_tp})
                    position = None
        
        if position is None:
            for pair in PAIRS:
                if pair not in data or "merged" not in data[pair]:
                    continue
                
                if t in data[pair]["merged"].index:
                    row = data[pair]["merged"].loc[t]
                    signal = row.get('hybrid_signal', 0)
                    
                    if abs(signal) > conf:
                        price = row['close']
                        atr = max(row.get('atr', MIN_ATR), MIN_ATR)
                        direction = 'BUY' if signal > 0 else 'SELL'
                        size = min(equity * risk, BASE_CAPITAL * 0.05) / (atr * atr_sl)
                        
                        if direction == 'BUY':
                            sl, tp = price - (atr * atr_sl), price + (atr * atr_tp)
                        else:
                            sl, tp = price + (atr * atr_sl), price - (atr * atr_tp)
                        
                        position = {'pair': pair, 'dir': direction, 'entry': price, 'sl': sl, 'tp': tp, 'size': size}
                        break
    
    total = len(trades)
    wins = sum(1 for t in trades if t['correct'])
    return {
        'total_trades': total,
        'winning_trades': wins,
        'accuracy': (wins / total * 100) if total > 0 else 0,
        'total_pnl': sum(t['pnl'] for t in trades)
    }

# ======================================================
# GENETIC ALGORITHM
# ======================================================
def run_ga(data, model_name, config):
    print_status(f"{config['color']} Training {model_name}...", "info")
    
    pop_size = config['pop_size']
    generations = config['generations']
    mutation_rate = config['mutation_rate']
    
    population = []
    for _ in range(pop_size):
        chrom = create_chromosome(config)
        metrics = backtest_strategy(data, chrom)
        fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
        population.append((fitness, chrom))
    
    population.sort(reverse=True, key=lambda x: x[0])
    
    for gen in range(generations):
        new_pop = population[:max(1, int(pop_size * 0.2))]
        
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            point = random.randint(1, len(p1[1]) - 1)
            child = p1[1][:point] + p2[1][point:]
            
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = float(child[i] + random.gauss(0, 0.1))
            
            metrics = backtest_strategy(data, child)
            fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
            new_pop.append((fitness, child))
        
        population = sorted(new_pop, reverse=True, key=lambda x: x[0])
        
        if (gen + 1) % 5 == 0:
            print_status(f"  Gen {gen+1}/{generations}: Best={population[0][0]:.4f}", "info")
    
    best_chrom = population[0][1]
    final_metrics = backtest_strategy(data, best_chrom)
    
    print_status(
        f"  ✅ {model_name}: {final_metrics['accuracy']:.1f}% accuracy | "
        f"${final_metrics['total_pnl']:.4f} PnL | {final_metrics['total_trades']} trades",
        "success"
    )
    
    return {'chromosome': best_chrom, 'metrics': final_metrics}

# ======================================================
# SIGNAL GENERATION
# ======================================================
def generate_signals(data, chromosome, model_name):
    atr_sl, atr_tp, risk, conf = decode_chromosome(chromosome)
    signals = {}
    
    for pair in PAIRS:
        if pair not in data or "merged" not in data[pair]:
            continue
        
        df = data[pair]["merged"]
        if len(df) == 0:
            continue
        
        row = df.iloc[-1]
        signal_strength = row.get('hybrid_signal', 0)
        price = row['close']
        atr = max(row.get('atr', MIN_ATR), MIN_ATR)
        
        direction = 'HOLD'
        sl = tp = price
        
        if abs(signal_strength) > conf:
            direction = 'BUY' if signal_strength > 0 else 'SELL'
            
            if direction == 'BUY':
                sl, tp = price - (atr * atr_sl), price + (atr * atr_tp)
            else:
                sl, tp = price + (atr * atr_sl), price - (atr * atr_tp)
        
        signals[pair] = {
            'direction': direction,
            'last_price': float(price),
            'SL': float(sl),
            'TP': float(tp),
            'atr': float(atr),
            'score_1_100': int(abs(signal_strength) * 100),
            'model': model_name,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    return signals

# ======================================================
# MAIN
# ======================================================
def main():
    print_status("=" * 70, "info")
    print_status("🚀 FOREX PIPELINE v8.5.1", "success")
    print_status("=" * 70, "info")
    
    try:
        current_iter = COUNTER.increment()
        stats = COUNTER.get_stats()
        
        print_status(f"\n📊 Iteration #{current_iter}", "info")
        print_status(f"Total: {stats['total']} | Days: {stats['days']} | Avg/Day: {stats['per_day']:.1f}", "info")
        
        print_status("\n📦 Loading data...", "info")
        data = load_data(PICKLE_FOLDER)
        
        if not data:
            raise ValueError("No data loaded!")
        
        print_status(f"✅ Loaded {len(data)} pairs", "success")
        
        print_status("\n🏆 Running Competition...", "info")
        results = {}
        signals_by_model = {}
        
        for model_name, config in COMPETITION_MODELS.items():
            result = run_ga(data, model_name, config)
            results[model_name] = result
            signals = generate_signals(data, result['chromosome'], model_name)
            signals_by_model[model_name] = signals
        
        print_status("\n💾 Saving signals...", "info")
        with open(SIGNALS_JSON, 'w') as f:
            json.dump(signals_by_model, f, indent=2, default=str)
        
        with open(ENSEMBLE_JSON, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'iteration': current_iter,
                'models': signals_by_model
            }, f, indent=2, default=str)
        
        print_status("\n" + "=" * 70, "success")
        print_status("✅ PIPELINE COMPLETED", "success")
        print_status("=" * 70, "success")
        
    except Exception as e:
        print_status(f"\n❌ Error: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
