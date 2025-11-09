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
    print_status(f"\n{'='*70}", "info")
    print_status(f"{config['color']} TRAINING: {model_name}", "info")
    print_status(f"{'='*70}", "info")
    print_status(f"Population Size: {config['pop_size']}", "info")
    print_status(f"Generations: {config['generations']}", "info")
    print_status(f"Mutation Rate: {config['mutation_rate']}", "info")
    print_status(f"ATR SL Range: {config['atr_sl_range']}", "info")
    print_status(f"ATR TP Range: {config['atr_tp_range']}", "info")
    print_status(f"Risk Range: {config['risk_range']}", "info")
    print_status(f"Confidence Range: {config['confidence_range']}", "info")
    print_status(f"{'='*70}\n", "info")
    
    pop_size = config['pop_size']
    generations = config['generations']
    mutation_rate = config['mutation_rate']
    
    # Initialize population
    print_status(f"🧬 Initializing population with {pop_size} chromosomes...", "info")
    population = []
    for i in range(pop_size):
        chrom = create_chromosome(config)
        metrics = backtest_strategy(data, chrom)
        fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
        population.append((fitness, chrom))
        if (i + 1) % 5 == 0 or (i + 1) == pop_size:
            print_status(f"  Created {i+1}/{pop_size} chromosomes...", "info")
    
    population.sort(reverse=True, key=lambda x: x[0])
    print_status(f"✅ Initial population created. Best fitness: {population[0][0]:.4f}\n", "success")
    
    # Evolution loop
    print_status("🔄 Starting evolution...", "info")
    best_fitness_history = []
    
    for gen in range(generations):
        # Selection: Keep top 20%
        elite_size = max(1, int(pop_size * 0.2))
        new_pop = population[:elite_size]
        
        # Crossover and mutation
        offspring_created = 0
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population, 2)
            point = random.randint(1, len(p1[1]) - 1)
            child = p1[1][:point] + p2[1][point:]
            
            # Mutation
            mutations = 0
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = float(child[i] + random.gauss(0, 0.1))
                    mutations += 1
            
            metrics = backtest_strategy(data, child)
            fitness = metrics['total_pnl'] + (metrics['accuracy'] / 100) * 10
            new_pop.append((fitness, child))
            offspring_created += 1
        
        population = sorted(new_pop, reverse=True, key=lambda x: x[0])
        best_fitness = population[0][0]
        best_fitness_history.append(best_fitness)
        
        # Calculate improvement
        improvement = ""
        if gen > 0:
            diff = best_fitness - best_fitness_history[gen-1]
            if diff > 0:
                improvement = f" ⬆️ +{diff:.4f}"
            elif diff < 0:
                improvement = f" ⬇️ {diff:.4f}"
            else:
                improvement = " ➡️ No change"
        
        # Progress bar
        progress = (gen + 1) / generations
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        percentage = progress * 100
        
        print_status(
            f"Gen [{gen+1:2d}/{generations:2d}] [{bar}] {percentage:5.1f}% | "
            f"Best Fitness: {best_fitness:8.4f}{improvement} | "
            f"Elite: {elite_size} | Offspring: {offspring_created}",
            "info"
        )
    
    # Final evaluation
    print_status(f"\n{'='*70}", "success")
    best_chrom = population[0][1]
    final_metrics = backtest_strategy(data, best_chrom)
    
    atr_sl, atr_tp, risk, conf = decode_chromosome(best_chrom)
    
    print_status(f"🏆 FINAL RESULTS - {model_name}", "success")
    print_status(f"{'='*70}", "success")
    print_status(f"Accuracy:        {final_metrics['accuracy']:.2f}%", "success")
    print_status(f"Total PnL:       ${final_metrics['total_pnl']:.4f}", "success")
    print_status(f"Total Trades:    {final_metrics['total_trades']}", "success")
    print_status(f"Winning Trades:  {final_metrics['winning_trades']}", "success")
    print_status(f"Losing Trades:   {final_metrics['total_trades'] - final_metrics['winning_trades']}", "success")
    print_status(f"Win Rate:        {final_metrics['accuracy']:.2f}%", "success")
    print_status(f"\nBest Parameters:", "success")
    print_status(f"  ATR Stop Loss:  {atr_sl:.4f}", "success")
    print_status(f"  ATR Take Profit: {atr_tp:.4f}", "success")
    print_status(f"  Risk Per Trade: {risk:.4f} ({risk*100:.2f}%)", "success")
    print_status(f"  Confidence:     {conf:.4f}", "success")
    print_status(f"{'='*70}\n", "success")
    
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
    print_status("🚀 FOREX PIPELINE v8.5.1 - DETAILED MODE", "success")
    print_status("=" * 70, "info")
    
    try:
        # Iteration tracking
        print_status("\n📊 ITERATION TRACKING", "info")
        print_status("-" * 70, "info")
        current_iter = COUNTER.increment()
        stats = COUNTER.get_stats()
        
        print_status(f"Current Iteration: #{current_iter}", "info")
        print_status(f"Total Iterations:  {stats['total']}", "info")
        print_status(f"Days Running:      {stats['days']}", "info")
        print_status(f"Avg Per Day:       {stats['per_day']:.2f}", "info")
        print_status(f"Timestamp:         {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", "info")
        print_status("-" * 70 + "\n", "info")
        
        # Data loading
        print_status("📦 LOADING DATA", "info")
        print_status("-" * 70, "info")
        print_status(f"Looking for pickle files in: {PICKLE_FOLDER}", "info")
        print_status(f"Expected pairs: {', '.join(PAIRS)}\n", "info")
        
        data = load_data(PICKLE_FOLDER)
        
        if not data:
            raise ValueError("No data loaded!")
        
        print_status(f"\n✅ Successfully loaded {len(data)} currency pairs", "success")
        
        # Data summary
        print_status("\n📈 DATA SUMMARY", "info")
        print_status("-" * 70, "info")
        for pair, pair_data in data.items():
            df = pair_data['merged']
            print_status(f"{pair}:", "info")
            print_status(f"  Rows:           {len(df)}", "info")
            print_status(f"  Date Range:     {df.index[0]} to {df.index[-1]}", "info")
            print_status(f"  Last Close:     {df['close'].iloc[-1]:.5f}", "info")
            print_status(f"  Last ATR:       {df['atr'].iloc[-1]:.5f}", "info")
            print_status(f"  Hybrid Signal:  {df['hybrid_signal'].iloc[-1]:.5f}", "info")
        print_status("-" * 70 + "\n", "info")
        
        # Model competition
        print_status("🏆 STARTING MODEL COMPETITION", "info")
        print_status("=" * 70, "info")
        print_status(f"Number of Models: {len(COMPETITION_MODELS)}", "info")
        print_status(f"Models: {', '.join(COMPETITION_MODELS.keys())}\n", "info")
        
        results = {}
        signals_by_model = {}
        
        for idx, (model_name, config) in enumerate(COMPETITION_MODELS.items(), 1):
            print_status(f"\n{'='*70}", "info")
            print_status(f"MODEL {idx}/{len(COMPETITION_MODELS)}: {model_name} {config['color']}", "info")
            print_status(f"{'='*70}", "info")
            
            result = run_ga(data, model_name, config)
            results[model_name] = result
            signals = generate_signals(data, result['chromosome'], model_name)
            signals_by_model[model_name] = signals
            
            # Show generated signals
            print_status(f"📡 Generated Signals for {model_name}:", "info")
            print_status("-" * 70, "info")
            for pair, signal in signals.items():
                direction_icon = "🟢" if signal['direction'] == 'BUY' else "🔴" if signal['direction'] == 'SELL' else "⚪"
                print_status(
                    f"{direction_icon} {pair}: {signal['direction']} | "
                    f"Price: {signal['last_price']:.5f} | "
                    f"SL: {signal['SL']:.5f} | "
                    f"TP: {signal['TP']:.5f} | "
                    f"Score: {signal['score_1_100']}/100",
                    "info"
                )
            print_status("-" * 70 + "\n", "info")
        
        # Competition summary
        print_status("\n" + "=" * 70, "success")
        print_status("🏆 COMPETITION RESULTS SUMMARY", "success")
        print_status("=" * 70, "success")
        
        ranked_models = sorted(
            results.items(),
            key=lambda x: x[1]['metrics']['total_pnl'],
            reverse=True
        )
        
        for rank, (model_name, result) in enumerate(ranked_models, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            metrics = result['metrics']
            print_status(
                f"{medal} {model_name}: "
                f"${metrics['total_pnl']:.2f} PnL | "
                f"{metrics['accuracy']:.1f}% accuracy | "
                f"{metrics['total_trades']} trades | "
                f"Win rate: {metrics['winning_trades']}/{metrics['total_trades']}",
                "success"
            )
        print_status("=" * 70 + "\n", "success")
        
        # Save signals
        print_status("💾 SAVING SIGNALS", "info")
        print_status("-" * 70, "info")
        
        print_status(f"Saving to {SIGNALS_JSON}...", "info")
        with open(SIGNALS_JSON, 'w') as f:
            json.dump(signals_by_model, f, indent=2, default=str)
        print_status(f"✅ Saved broker signals", "success")
        
        print_status(f"Saving to {ENSEMBLE_JSON}...", "info")
        with open(ENSEMBLE_JSON, 'w') as f:
            json.dump({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'iteration': current_iter,
                'models': signals_by_model,
                'performance': {
                    model: {
                        'pnl': results[model]['metrics']['total_pnl'],
                        'accuracy': results[model]['metrics']['accuracy'],
                        'trades': results[model]['metrics']['total_trades']
                    }
                    for model in signals_by_model.keys()
                }
            }, f, indent=2, default=str)
        print_status(f"✅ Saved ensemble signals", "success")
        print_status("-" * 70 + "\n", "success")
        
        # Final summary
        print_status("=" * 70, "success")
        print_status("✅ PIPELINE COMPLETED SUCCESSFULLY", "success")
        print_status("=" * 70, "success")
        print_status(f"Execution Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", "success")
        print_status(f"Files Generated: {SIGNALS_JSON.name}, {ENSEMBLE_JSON.name}", "success")
        print_status(f"Next Action: Signals ready for broker execution", "success")
        print_status("=" * 70 + "\n", "success")
        
    except Exception as e:
        print_status("\n" + "=" * 70, "error")
        print_status("❌ PIPELINE FAILED", "error")
        print_status("=" * 70, "error")
        print_status(f"Error: {e}", "error")
        print_status("\nFull Traceback:", "error")
        import traceback
        traceback.print_exc()
        print_status("=" * 70, "error")
        sys.exit(1)

if __name__ == "__main__":
    main()
