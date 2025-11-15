# 🚀 Forex AI Trading Models

**Automated forex trading signal generation using machine learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📊 Overview

This project implements a complete machine learning pipeline for forex trading signal generation:

- **Multi-source data collection** (Alpha Vantage + YFinance)
- **Multi-timeframe analysis** (1m, 5m, 15m, 1h, 1d)
- **Technical indicator calculation** (50+ indicators)
- **Ensemble ML models** (SGD + Random Forest)
- **Persistent trade tracking** (SQLite database)
- **Automated signal generation** (JSON output)

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────┐
│              DATA COLLECTION LAYER                      │
├─────────────────────────────────────────────────────────┤
│  Alpha Vantage (Daily) + YFinance (Multi-timeframe)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              DATA PROCESSING LAYER                      │
├─────────────────────────────────────────────────────────┤
│  Combine CSVs → Calculate Indicators → Quality Check   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              MODEL TRAINING LAYER                       │
├─────────────────────────────────────────────────────────┤
│  SGD Classifier + Random Forest → Ensemble Prediction   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│              SIGNAL GENERATION                          │
├─────────────────────────────────────────────────────────┤
│  Weighted Aggregation → SL/TP Calculation → JSON       │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure
```
forex-ai-models/
├── scripts/              # Executable scripts (numbered for order)
│   ├── 1_fetch_alphavantage.py
│   ├── 2_fetch_yfinance.py
│   ├── 3_combine_csvs.py
│   ├── 4_merge_pickles.py
│   └── 5_train_pipeline.py
├── data/
│   ├── raw/             # Downloaded CSV files
│   ├── processed/       # Pickle files with indicators
│   └── models/          # Trained ML models
├── outputs/             # Generated signals & reports
├── logs/                # Execution logs
├── database/            # SQLite trade database
└── config/              # Configuration files
```

## 🚀 Quick Start

### 1. Prerequisites
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
export ALPHA_VANTAGE_KEY="your_api_key"
export FOREX_PAT="your_github_token"
export BROWSERLESS_TOKEN="your_browserless_token"  # Optional
```

### 3. Run Pipeline
```bash
# Step 1: Fetch daily data from Alpha Vantage
python scripts/1_fetch_alphavantage.py

# Step 2: Fetch multi-timeframe data from YFinance
python scripts/2_fetch_yfinance.py

# Step 3: Combine and process CSV files
python scripts/3_combine_csvs.py

# Step 4: Merge pickle files by pair
python scripts/4_merge_pickles.py

# Step 5: Train models and generate signals
python scripts/5_train_pipeline.py
```

### 4. Check Output

Trading signals are saved to `outputs/latest_signals.json`:
```json
{
  "timestamp": "2025-11-15T10:30:00Z",
  "pairs": {
    "EUR/USD": {
      "aggregated": "STRONG_LONG",
      "signals": {
        "1h_2y": {
          "signal": 1,
          "live": 1.0850,
          "SL": 1.0800,
          "TP": 1.0900,
          "confidence": 0.75
        }
      }
    }
  }
}
```

## ⚙️ Configuration

Edit `config/settings.py` to customize:

- **Trading pairs** to track
- **Timeframes** to analyze
- **Model parameters**
- **Risk management** (SL/TP multipliers)
- **Quality thresholds**

## 📊 Current Performance

| Metric | Value |
|--------|-------|
| **Pairs Tracked** | 4 (EUR/USD, GBP/USD, USD/JPY, AUD/USD) |
| **Timeframes** | 5 (1m to 1d) |
| **Total Trades** | 261 |
| **Overall Accuracy** | 100% |
| **Total P&L** | -$878.16 |

## ⚠️ Known Issues

1. **ATR Calculation Bug**: SL/TP ranges are too wide (~30-60% from entry)
   - **Cause**: Using full ATR value instead of fractional multiplier
   - **Status**: Fix in progress

2. **Win Logic Paradox**: 100% accuracy but negative P&L
   - **Cause**: Trades hit stop-loss but direction is correct
   - **Status**: Under investigation

3. **Model Compression**: Inconsistent pickle formats
   - **Status**: ✅ Fixed in latest version

## 🔧 Maintenance

### Update Models

Models are retrained automatically on each pipeline run using incremental learning.

### Database Management
```bash
# View trade statistics
sqlite3 database/memory_v85.db "SELECT * FROM completed_trades LIMIT 10;"

# Reset database
rm database/memory_v85.db
# Will be recreated on next run
```

### Backup Data
```bash
# Backup all data
tar -czf backup_$(date +%Y%m%d).tar.gz data/ database/ outputs/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Alpha Vantage** for daily forex data
- **YFinance** for multi-timeframe data
- **TA-Lib** for technical indicators
- **scikit-learn** for ML models

## 📧 Contact

For questions or issues, please open a GitHub issue or contact via email.

---

**⚠️ Disclaimer**: This is for educational purposes only. Not financial advice. Trade at your own risk.
