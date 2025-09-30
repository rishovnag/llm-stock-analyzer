# LLM Stock Prediction & Backtesting System

A sophisticated system for analyzing Indian stock indices (NIFTY50, NIFTYBANK, NIFTYIT) using Large Language Models (LLMs) and technical analysis, with comprehensive backtesting capabilities.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [API Keys Setup](#api-keys-setup)
- [Output Files](#output-files)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

## Overview

This project consists of two main components:

1. **llm_stock_analyzer.py**: Multi-model LLM prediction system that analyzes stock movements using various thresholds (direction, 1%, 2%, 3% movements)
2. **advanced_backtest_analyzer.py**: Comprehensive backtesting framework with variable train-test splits and rolling window analysis

## Features

### LLM Stock Analyzer
- **Multi-Provider Support**: OpenAI (GPT-4), Anthropic (Claude), Google (Gemini)
- **Movement Types**: 
  - Direction (UP/DOWN)
  - Significant movements (>1%, >2%, >3%)
- **Technical Indicators**: 50+ indicators including RSI, MACD, Bollinger Bands, Moving Averages
- **Async Processing**: Parallel prediction requests for efficiency
- **Unified Interface**: Consistent API across different LLM providers

### Advanced Backtest Analyzer
- **Multiple Split Strategies**:
  - Fixed ratio splits (70/30, 75/25, 80/20, 85/15, 90/10, 95/5)
  - Rolling window analysis (12, 18, 24 months)
- **Comprehensive Metrics**:
  - Classification: Accuracy, Precision, Recall, F1-Score, AUC
  - Trading: Sharpe Ratio, Win Rate, Max Drawdown, Profit Factor
- **Excel Reports**: Multi-sheet analysis with detailed breakdowns
- **Rule-Based Predictions**: Technical analysis-based prediction system

## Project Structure

```
project_root/
│
├── llm_stock_analyzer.py          # Main LLM prediction system
├── advanced_backtest_analyzer.py  # Backtesting framework
├── README.md                       # This file
├── .env                           # API keys (create this)
├── requirements.txt               # Python dependencies
│
├── data/                          # Input data directory
│   ├── NIFTY50_*_Direction.csv
│   ├── NIFTY50_*_SigMove_*.csv
│   ├── NIFTYBANK_*_Direction.csv
│   ├── NIFTYBANK_*_SigMove_*.csv
│   ├── NIFTYIT_*_Direction.csv
│   └── NIFTYIT_*_SigMove_*.csv
│
├── results/                       # LLM prediction outputs
│   └── {SYMBOL}_all_movements_{timestamp}.json
│
└── advanced_backtest_results/    # Backtest reports
    ├── comprehensive_backtest_results_{timestamp}.xlsx
    └── comprehensive_backtest_report_{timestamp}.txt
```

## Installation

### Requirements
- Python 3.8+
- 8GB+ RAM recommended
- API keys for LLM providers (at least one)

### Step 1: Clone/Setup Project
```bash
# Create project directory
mkdir llm-stock-prediction
cd llm-stock-prediction

# Copy the Python files
# Copy llm_stock_analyzer.py
# Copy advanced_backtest_analyzer.py
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Create requirements.txt
```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
python-dotenv>=1.0.0
aiohttp>=3.8.0
tqdm>=4.65.0
openpyxl>=3.0.0
openai>=1.0.0          # For GPT models
anthropic>=0.18.0      # For Claude models
google-generativeai>=0.3.0  # For Gemini models
```

## Configuration

### API Keys Setup

Create a `.env` file in the project root:

```env
# OpenAI API Key (for GPT-4 models)
OPENAI_API_KEY=sk-...

# Anthropic API Key (for Claude models)
ANTHROPIC_API_KEY=sk-ant-...

# Google API Key (for Gemini models)
GOOGLE_API_KEY=AIza...
```

**Note**: You need at least one API key. Gemini offers a free tier which is good for testing.

### Model Configuration

Models are pre-configured in `llm_stock_analyzer.py`:

```python
model_configs = {
    'gpt4o-mini': {'provider': 'openai', 'model': 'gpt-4o-mini'},
    'gpt4o': {'provider': 'openai', 'model': 'gpt-4o'},
    'claude-haiku': {'provider': 'anthropic', 'model': 'claude-3-haiku-20240307'},
    'claude-sonnet': {'provider': 'anthropic', 'model': 'claude-3-5-sonnet-20241022'},
    'gemini-flash': {'provider': 'google', 'model': 'gemini-1.5-flash'},
    'gemini-pro': {'provider': 'google', 'model': 'gemini-1.5-pro'}
}
```

## Usage

### 1. LLM Stock Predictions

```python
python llm_stock_analyzer.py
```

#### Programmatic Usage
```python
import asyncio
from llm_stock_analyzer import LLMStockPredictor

async def run_predictions():
    predictor = LLMStockPredictor("data")
    
    # Test with Gemini (free tier)
    models = ['gemini-flash']
    
    # Get predictions for all movement types
    results = await predictor.predict_all_movements(models, 'NIFTY50')
    predictor.display_movement_results('NIFTY50', results)
    predictor.save_movement_results('NIFTY50', results)

asyncio.run(run_predictions())
```

### 2. Backtesting Analysis

```python
python advanced_backtest_analyzer.py
```

#### Custom Configuration
```python
from advanced_backtest_analyzer import AdvancedBacktestEngine

engine = AdvancedBacktestEngine("data")

# Custom train-test ratios
results = engine.run_comprehensive_backtest(
    stock_symbols=['NIFTY50', 'NIFTYBANK'],
    train_ratios=[0.70, 0.80, 0.90],
    num_runs_per_config=10
)
```

## Data Requirements

### File Naming Convention
Files must follow this naming pattern:
- Direction: `{SYMBOL}_*_Direction.csv`
- Significant Moves: `{SYMBOL}_*_SigMove_{threshold}pct.csv`

Example:
- `NIFTY50_2007_2025_with_Direction.csv`
- `NIFTY50_2007_2025_Daily_with_SigMove_1pct.csv`

### Required Columns
```csv
Date,Open,High,Low,Close,Adj_Close,Volume,Direction/Significant_Move
2024-01-01,19000.50,19150.25,18950.75,19100.00,19100.00,250000000,1
```

- **Price columns**: Open, High, Low, Close, Adj_Close (numeric)
- **Volume**: Trading volume (can be string with commas)
- **Movement column**: 
  - Direction: 1 (UP) or 0 (DOWN)
  - Significant_Move: 1 (significant) or 0 (normal)

### Data Validation
The system automatically:
- Converts date strings to datetime
- Handles comma-separated numbers
- Removes invalid/NaN values
- Sorts data chronologically

## Output Files

### LLM Predictions (JSON)
Location: `results/{SYMBOL}_all_movements_{timestamp}.json`

Structure:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "stock_symbol": "NIFTY50",
  "movement_predictions": {
    "direction": [
      {
        "model_name": "gemini-flash",
        "prediction": 1,
        "confidence": 0.75,
        "reasoning": "Strong bullish momentum...",
        "technical_data": {...}
      }
    ],
    "sig_move_1pct": [...],
    "sig_move_2pct": [...],
    "sig_move_3pct": [...]
  }
}
```

### Backtest Reports (Excel)
Location: `advanced_backtest_results/comprehensive_backtest_results_{timestamp}.xlsx`

Sheets:
1. **All_Results**: Complete dataset of all runs
2. **Train_Ratio_Summary**: Performance by train-test ratio
3. **Stock_Movement_Summary**: Performance by stock and movement type
4. **Split_Type_Comparison**: Fixed vs rolling window comparison
5. **Best_Performers**: Top configurations for each metric
6. **Ratio_XX_Detail**: Detailed results for each train ratio

## Technical Details

### Technical Indicators Calculated
- **Price-based**: SMA (5,10,20,50), EMA (12,26), MACD
- **Momentum**: RSI, Rate of Change, Momentum (3,5,10 days)
- **Volatility**: Standard deviation, ATR, Bollinger Bands
- **Volume**: Volume ratio, Volume momentum
- **Support/Resistance**: Recent highs/lows, distance metrics

### Prediction Logic

#### LLM Predictions
1. Calculate technical indicators from historical data
2. Compute movement statistics
3. Generate detailed prompt with context
4. Send to LLM for analysis
5. Parse JSON response
6. Validate and store results

#### Rule-Based (Backtest)
1. Weight multiple technical signals
2. Combine indicators based on movement type
3. Apply thresholds for prediction
4. Calculate confidence scores

### Performance Metrics

**Classification Metrics**:
- Accuracy: Correct predictions / Total predictions
- Precision: True Positives / (True Positives + False Positives)
- Recall: True Positives / (True Positives + False Negatives)
- F1-Score: Harmonic mean of Precision and Recall
- AUC: Area Under the ROC Curve

**Trading Metrics**:
- Sharpe Ratio: Risk-adjusted returns (annualized)
- Win Rate: Profitable trades / Total trades
- Max Drawdown: Largest peak-to-trough decline
- Profit Factor: Gross profits / Gross losses

## Troubleshooting

### Common Issues

#### 1. API Key Errors
```
Error: API key not found for openai
```
**Solution**: Ensure .env file exists with correct API keys

#### 2. Data File Not Found
```
FileNotFoundError: No datasets found for NIFTY50
```
**Solution**: Check data directory and file naming convention

#### 3. Insufficient Data
```
Warning: Insufficient data for NIFTY50 - direction
```
**Solution**: Ensure dataset has at least 200 rows

#### 4. Memory Issues
```
MemoryError during backtesting
```
**Solution**: Reduce `num_runs_per_config` or process fewer stocks

#### 5. JSON Parsing Error
```
Failed to parse JSON response
```
**Solution**: LLM returned invalid JSON - retry or check API limits

### Performance Tips

1. **Start with Gemini Flash** (free tier) for testing
2. **Use async processing** for multiple predictions
3. **Limit backtest runs** for initial testing (5-10 runs)
4. **Process stocks sequentially** to manage memory
5. **Save results frequently** to avoid data loss

## License

This project is for educational and research purposes. Ensure compliance with:
- API provider terms of service
- Data source licensing
- Financial regulations in your jurisdiction

## Disclaimer

This system is for educational purposes only. Stock market predictions are inherently uncertain and past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API provider documentation
3. Ensure data format compliance
4. Verify environment setup

---

**Version**: 1.0.0  
**Last Updated**: January 2025  
**Python Version**: 3.8+
