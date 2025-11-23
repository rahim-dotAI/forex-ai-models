# 🧠 Forex AI Brain - Autonomous Trading System

[![GitHub Actions](https://img.shields.io/badge/Automated-GitHub%20Actions-blue)](https://github.com/features/actions)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-success)](https://github.com/rahim-dotAI/forex-ai-models)

> **A fully autonomous AI-powered Forex trading system that learns, adapts, and trades 24/7 using Deep Q-Learning and Machine Learning**

---

## 🌟 **Key Features**

### 🤖 **Autonomous Operation**
- **100% automated** - Runs on GitHub Actions (no server costs!)
- **Weekend learning mode** - Backtests and trains on historical data
- **Live trading mode** - Executes real trades during weekdays
- **Self-learning** - Improves performance over time through experience

### 🧠 **Advanced AI Architecture**
- **Deep Q-Learning** - Neural network-based decision making
- **Experience Replay** - Learns from 3,800+ past experiences
- **Dual Networks** - Q-network + Target network for stability
- **Dynamic Exploration** - Epsilon-greedy strategy with decay

### 📊 **Multi-Source Data Integration**
- **Alpha Vantage** - Daily OHLC data (optimized to 4 calls/day)
- **YFinance** - Multiple timeframes (1m, 5m, 15m, 1h, 1d)
- **24 data streams** - Comprehensive market coverage
- **Quality validation** - Automatic data integrity checks

### 💡 **Smart Features**
- **ATR-based risk management** - Dynamic stop loss & take profit
- **Confidence scoring** - Only trades high-probability setups
- **Multi-timeframe analysis** - Combines 1m to 1d data
- **Technical indicators** - 30+ features per trade decision

---

## 📈 **Current Performance**

| Metric | Value | Status |
|--------|-------|--------|
| **Win Rate** | 42.2% | 🟡 Improving |
| **Total Trades** | 1,000+ | ✅ Strong experience |
| **Total P&L** | $4.23 | ✅ Profitable |
| **Experience Pool** | 3,891 samples | ✅ Deep learning |
| **API Efficiency** | 16% of limit | ✅ Optimized |

---

## 🎯 **Trading Pairs**

Currently trading 4 major Forex pairs:
- 🇪🇺 **EUR/USD** - Euro / US Dollar
- 🇬🇧 **GBP/USD** - British Pound / US Dollar
- 🇯🇵 **USD/JPY** - US Dollar / Japanese Yen
- 🇦🇺 **AUD/USD** - Australian Dollar / US Dollar

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────┐
│           GitHub Actions (Automated Pipeline)           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    Data Collection                       │
│  ┌──────────────┐              ┌──────────────┐        │
│  │ Alpha Vantage│              │   YFinance   │        │
│  │ (Daily OHLC) │              │ (5 timeframes)│        │
│  └──────────────┘              └──────────────┘        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Data Processing & Indicators                │
│  • ATR, RSI, MACD, Bollinger Bands                      │
│  • Quality validation & cleaning                        │
│  • Multi-timeframe consolidation                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                ML Pipeline (Pipeline v5.0)               │
│  • SGD Classifier (fast incremental learning)           │
│  • Random Forest (ensemble predictions)                 │
│  • Fresh model training each run                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           Trade Beacon RL Agent (v18.1)                  │
│  ┌─────────────────────────────────────────┐            │
│  │    Deep Q-Network (128→64→32 nodes)    │            │
│  │    • State: 30 features                 │            │
│  │    • Actions: BUY, SELL, HOLD           │            │
│  │    • Reward: P&L + Risk-adjusted return │            │
│  └─────────────────────────────────────────┘            │
│                                                          │
│  Weekend: Backtest & Learn (631 trades/run)             │
│  Weekday: Live Trading (Real money)                     │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Trade Execution                        │
│  • Browserless API for live prices                      │
│  • Dynamic position sizing                              │
│  • ATR-based SL/TP (2x/3x multipliers)                  │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 **Repository Structure**

```
forex-ai-models/
├── 📓 AI_Forex_Brain_2.ipynb        # Main notebook (8 cells)
├── ⚙️  .github/workflows/
│   └── main.yml                      # GitHub Actions automation
├── 📊 data/
│   ├── raw/
│   │   ├── yfinance/                 # YFinance CSVs (20 files)
│   │   └── alpha_vantage/            # Alpha Vantage CSVs (4 files)
│   ├── processed/                    # Processed pickles (24 files)
│   └── quarantine/                   # Failed quality checks
├── 💾 database/
│   └── memory_v85.db                 # SQLite trade history
├── 🧠 rl_memory/
│   ├── experience_replay.json.gz     # 3,891 experiences
│   ├── network_weights.json          # Q-network weights
│   ├── learning_stats.json           # Training metrics
│   └── trade_history.json            # Historical trades
├── 📤 outputs/
│   └── omega_signals.json            # Latest trading signals
├── 🌀 omega_state/
│   └── omega_iteration.json          # Run counter & history
├── 💼 backups/                       # Automatic backups
└── 📝 logs/                          # Execution logs
```

---

## ⚡ **Quick Start**

### **Prerequisites**
- GitHub account (for Actions)
- Alpha Vantage API key (free)
- Browserless API token (optional, for live prices)
- Gmail account (for reports)

### **Setup**

1. **Fork this repository**

2. **Add GitHub Secrets** (Settings → Secrets → Actions):
   ```
   FOREX_PAT              # GitHub Personal Access Token
   ALPHA_VANTAGE_KEY      # Alpha Vantage API key
   BROWSERLESS_TOKEN      # Browserless API token
   GMAIL_USER             # Your Gmail address
   GMAIL_APP_PASSWORD     # Gmail app password
   ```

3. **Enable GitHub Actions**
   - Go to Actions tab
   - Click "Enable workflows"

4. **That's it!** 🎉
   - System runs automatically every 2 hours (weekdays)
   - Runs every 30 minutes (weekends)
   - Alpha Vantage fetch once daily at midnight

---

## 📅 **Automated Schedule**

| Time | Action | Mode |
|------|--------|------|
| **Weekdays** | Every 2 hours | 🔴 **LIVE TRADING** |
| **Weekends** | Every 30 minutes | 🏖️ **LEARNING MODE** |
| **Midnight UTC** | Daily | 🌙 **Alpha Vantage Fetch** |

---

## 🧪 **How It Works**

### **Weekend Learning Mode** 🏖️
1. Loads historical data (24 pickle files)
2. Runs 500-step backtest per currency pair
3. Generates ~630 trade simulations
4. Trains Q-network on experiences
5. Updates exploration strategy (epsilon decay)
6. Saves learned weights for Monday

### **Live Trading Mode** 💰
1. Fetches real-time prices via Browserless API
2. Calculates 30-feature state vector
3. Q-network predicts best action (BUY/SELL/HOLD)
4. Confidence system filters trades (>25% threshold)
5. Executes trades with ATR-based risk management
6. Monitors open positions for SL/TP exits
7. Records outcomes for continuous learning

---

## 🎓 **Technical Details**

### **State Vector (30 Features)**
- Price momentum & trend (5 periods)
- RSI (14-period, 1h & 1d)
- MACD & signal line
- Bollinger Bands position & width
- ATR & volatility metrics
- EMA crossovers (12/26 periods)
- Volume ratios
- Time-of-day features (3 sessions)
- Market regime indicators

### **Reward Function**
```python
Reward = (
    P&L × 500                    # Profit/Loss scaled
    + Win Bonus (50)             # For TP hits
    - Loss Penalty (10)          # For SL hits
    + Risk-Adjusted Return × 30  # Sharpe-like metric
    + Duration Bonus/Penalty     # Favor quick wins
)
```

### **Risk Management**
- **Stop Loss**: Entry ± (2 × ATR)
- **Take Profit**: Entry ± (3 × ATR)
- **Position Sizing**: 2% of capital per trade
- **Max Positions**: 2 concurrent trades
- **Max Trade Size**: $10 equivalent

---

## 📊 **API Optimization**

### **Alpha Vantage Efficiency**
- **Old approach**: 48 calls/day (hourly fetching)
- **New approach**: 4 calls/day (midnight only)
- **Savings**: 44 calls/day = **92% reduction**
- **Why**: Daily OHLC doesn't change intraday

### **Rate Limits**
- Alpha Vantage: 25 calls/day (using 16%)
- YFinance: 2,000 calls/hour (using <1%)
- Browserless: 1,000 requests/month

---

## 📈 **Performance Tracking**

### **Live Dashboard**
Check latest run: `.github/run_history/latest_run.json`

```json
{
  "timestamp": "2025-11-23T11:30:00Z",
  "iteration": 29,
  "mode": "WEEKEND_LEARNING",
  "rl_stats": {
    "total_trades": 1000,
    "win_rate": 0.422,
    "total_pnl": 4.23,
    "epsilon": 0.10
  }
}
```

### **Email Reports**
- Sent every 10 runs (weekdays only)
- Contains: Win rate, P&L, active signals, epsilon

---

## 🔧 **Configuration**

### **Key Parameters** (in Trade Beacon v18.1):
```python
# Q-Learning
STATE_SIZE = 30              # Input features
ACTION_SPACE = 3             # BUY, SELL, HOLD
LEARNING_RATE = 0.0005       # Neural network learning rate
GAMMA = 0.95                 # Discount factor
EPSILON_START = 1.0          # Initial exploration
EPSILON_MIN = 0.10           # Minimum exploration
EPSILON_DECAY = 0.995        # Decay per update

# Training
BATCH_SIZE = 64              # Training batch size
MEMORY_SIZE = 15000          # Max experiences to store
MIN_REPLAY_SIZE = 200        # Min before training
TARGET_UPDATE_FREQ = 25      # Target network sync

# Risk Management
ATR_SL_MULTIPLIER = 2.0      # Stop loss distance
ATR_TP_MULTIPLIER = 3.0      # Take profit distance
MAX_RISK_PER_TRADE = 0.02    # 2% per trade
MAX_POSITIONS = 2            # Concurrent trades
```

---

## 🐛 **Troubleshooting**

### **Pipeline fails**
- Check GitHub Actions logs
- Verify secrets are set correctly
- Ensure API keys are valid

### **Low win rate (<30%)**
- System is still learning (needs more data)
- Continue weekend training
- Check that all 24 data files are present

### **No trades executed**
- Confidence threshold too high (normal)
- Agent in exploration mode (epsilon > 0.5)
- No high-probability setups found

### **Data quality warnings**
- Some files quarantined automatically
- Check `data/quarantine/` for reports
- System continues with good data

---

## 🚀 **Roadmap**

### **Current Version: v18.1**
- ✅ Corruption-free architecture
- ✅ Weekend backtest learning
- ✅ Fixed model persistence
- ✅ Multi-pair support

### **Planned Features**
- [ ] Support for 8+ currency pairs
- [ ] Advanced sentiment analysis
- [ ] Multi-model ensemble voting
- [ ] Real-time dashboard UI
- [ ] Telegram bot notifications
- [ ] Cloud deployment option
- [ ] Backtesting web interface

---

## 📚 **Documentation**

### **Key Files**
- `AI_Forex_Brain_2.ipynb` - Main pipeline (8 cells)
- Cell 1: API Keys Configuration
- Cell 2: Environment Setup
- Cell 3: GitHub Sync
- Cell 4: Dependencies
- Cell 5: Alpha Vantage Fetcher
- Cell 6: YFinance Fetcher
- Cell 7: CSV Combiner
- Cell 8: ML Pipeline v5.0
- Cell 9: Trade Beacon RL Agent v18.1

### **References**
- [Alpha Vantage API Docs](https://www.alphavantage.co/documentation/)
- [YFinance Documentation](https://pypi.org/project/yfinance/)
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [GitHub Actions Guide](https://docs.github.com/en/actions)

---

## ⚠️ **Disclaimer**

**THIS SOFTWARE IS FOR EDUCATIONAL PURPOSES ONLY.**

- Trading forex involves substantial risk of loss
- Past performance does not guarantee future results
- Only trade with money you can afford to lose
- This is NOT financial advice
- The authors are NOT responsible for any trading losses

**USE AT YOUR OWN RISK**

---

## 🤝 **Contributing**

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 **License**

MIT License - See [LICENSE](LICENSE) file for details

---

## 👨‍💻 **Author**

**Rahim Dotai**
- GitHub: [@rahim-dotAI](https://github.com/rahim-dotAI)
- Email: nakatonabira3@gmail.com

---

## 🙏 **Acknowledgments**

- Alpha Vantage for free financial data API
- YFinance for multi-timeframe market data
- GitHub Actions for free automation
- The open-source community

---

## 📊 **Stats**

![GitHub last commit](https://img.shields.io/github/last-commit/rahim-dotAI/forex-ai-models)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/rahim-dotAI/forex-ai-models)
![Lines of code](https://img.shields.io/tokei/lines/github/rahim-dotAI/forex-ai-models)

---

<div align="center">

### **⭐ Star this repo if you find it useful!**

**Made with ❤️ and 🧠 by Rahim Dotai**

</div>
