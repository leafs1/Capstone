# Polymarket-SPY Granger Causality Analysis

Research project testing whether Polymarket prediction markets have predictive relationships with U.S. equity markets (SPY).

## Quick Start

### 1. Run Granger Causality Analysis
```bash
# Analyze all markets (350 tokens)
python Granger.py

# View results
python show_granger_results.py
python show_granger_results.py --direction poly_to_eq
python show_granger_results.py --direction both  # bidirectional only
```

### 2. Validate Results
```bash
# Validate Polymarket → SPY relationships
python validate_granger.py --direction poly_to_eq

# Validate specific lag ranges
python validate_granger.py --direction poly_to_eq --lag-range short   # 1-3 min
python validate_granger.py --direction poly_to_eq --lag-range medium  # 5-15 min
python validate_granger.py --direction poly_to_eq --lag-range long    # 16-30 min

# Validate SPY → Polymarket (reverse direction)
python validate_granger.py --direction eq_to_poly --limit 30

# Save validation results
python validate_granger.py --direction poly_to_eq --save-csv
```

### 3. Visualize Results
```bash
# Plot markets with significant Granger causality
python plot_granger_markets.py --direction poly_to_eq --limit 10

# Test insider trading hypothesis (jump prediction)
python test_jump_prediction.py
```

## Project Structure

### Core Analysis Scripts

- **`Granger.py`** - Main Granger causality analysis engine
  - Tests 350 Polymarket tokens against SPY
  - Implements stationarity tests, lag selection, Bonferroni correction
  - Stores results in `data/markets.duckdb`

- **`validate_granger.py`** - Validation module using lead-lag cross-correlation
  - Tests if Granger lags correspond to maximum correlation
  - Classifies results: CONFIRMED, SYNCHRONOUS, REVERSED, etc.
  - Generates correlation plots in `plots/validation/`

- **`test_jump_prediction.py`** - Tests insider trading hypothesis
  - Identifies large price jumps (>3%, >2σ)
  - Tests if Polymarket jumps predict SPY movements
  - Tests asymmetry between directions

- **`show_granger_results.py`** - Display and export Granger results
  - Filter by direction (poly_to_eq, eq_to_poly, both)
  - Show timing statistics and data quality
  - Export to CSV

- **`plot_granger_markets.py`** - Visualization tool
  - Plots Polymarket vs SPY prices for significant markets
  - Shows minute-resolution data

### Data Loading Scripts

- **`PolyMarket.py`** - Polymarket data ingestion
  - Fetches market and price data from Polymarket API
  - Stores in `data/research.duckdb`

- **`FinData.py`** - Databento historical data loading
  - Loads SPY minute bars from Databento
  - Stores in `data/markets.duckdb`

- **`load_market_prices.py`** - Market price loading utilities

### Utilities

- **`DbController.py`** - Database management
- **`utils.py`** - Shared utilities
- **`main.py`** - Main orchestration script

## Key Findings

### Initial Results (Granger Tests)
- **49 markets**: Polymarket → SPY (p<0.05, Bonferroni corrected)
- **44 markets**: SPY → Polymarket  
- **16 markets**: Bidirectional
- Lags: 1-30 minutes (median ~20 min)
- P-values: Some as low as 10⁻⁶⁹

### Validation Results (Lead-Lag Correlation)
- **83% of markets**: Maximum correlation at lag=0 (SYNCHRONOUS)
- **0% of markets**: Confirmed Granger lag
- Correlation 8-16x stronger at lag=0 vs Granger lag
- **Conclusion**: Results are spurious, markets move synchronously

### Jump Prediction Results
- **0/7 markets**: Polymarket jumps predict SPY
- **1/7 markets**: SPY jumps predict Polymarket (opposite direction!)
- **Conclusion**: No evidence of insider trading via price jumps

### Final Conclusion
Both markets process information **synchronously** (within seconds), not predictively. Granger causality results were driven by **common information shocks**, not true causal relationships. Both markets are **equally efficient**.

## Data

### Databases (data/)
- **markets.duckdb**: SPY prices (100,096 minute bars) + Granger results
- **research.duckdb**: Polymarket data (440 tokens, 44.6M prices)

### Coverage
- **Period**: October 25, 2024 - November 3, 2025
- **Trading days**: 256 days
- **Polymarket markets**: Federal Reserve policy, inflation, GDP, economic indicators

## Outputs

### Validation Results (CSV)
- `validation_short_lags.csv` - Ultra-short lag markets (1-3 min)
- `validation_medium_long_lags.csv` - Medium/long lag markets (5-30 min)
- `validation_eq_to_poly.csv` - Reverse direction (SPY → Polymarket)

### Plots
- `plots/validation/` - Lead-lag correlation plots for validated markets

### Documentation
- **`RESEARCH_REPORT.md`** - Comprehensive research report
- **`CLEANUP_PLAN.md`** - Code organization documentation

## Environment Setup

```bash
# Install dependencies
pip install duckdb pandas numpy matplotlib scipy statsmodels

# Set environment variables (optional)
export MKT_DB="./data/markets.duckdb"
export POLY_DB="./data/research.duckdb"
export GRANGER_MAXLAG="30"
```

## Archive

The `archive/` directory contains legacy validation scripts and diagnostic tools that have been superseded by `validate_granger.py`. See `archive/README.md` for details.

## Research Team

- Michael - Lead Researcher
- Capstone Project 2025

## References

- Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models"
- Polymarket API: https://docs.polymarket.com/
- Databento: Historical market data provider
