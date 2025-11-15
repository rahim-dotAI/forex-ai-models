# 🚀 Forex AI Trading Models

Automated forex trading signal generation using machine learning.

## 📁 Repository Structure

```
forex-ai-models/
├── data/
│   ├── raw/          - CSV data files (gitignored)
│   ├── processed/    - Processed pickle files (gitignored)
│   └── models/       - Trained ML models (gitignored)
├── outputs/          - Trading signals (JSON)
├── scripts/          - Executable Python scripts
├── notebooks/        - Jupyter notebooks
├── logs/             - Execution logs (gitignored)
├── database/         - SQLite databases (gitignored)
└── config/           - Configuration files
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run main pipeline
python run_pipeline.py
```

## ⚙️ Configuration

Edit `config/settings.py` to customize trading pairs, timeframes, and model parameters.

## ⚠️ Disclaimer

For educational purposes only. Not financial advice. Trade at your own risk.
