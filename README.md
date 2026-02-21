# Polymarket-SPY Granger Causality Analysis

Research project testing whether Polymarket prediction markets have predictive relationships with U.S. equity markets (SPY).

## 📊 Key Finding

**Both markets move synchronously** - no predictive relationship in either direction. Markets process information equally fast (within seconds), contradicting initial Granger causality results.

## 🚀 Quick Start

```bash
# Run Granger analysis
python Granger.py

# Validate results
python validate_granger.py --direction poly_to_eq

# View results  
python show_granger_results.py

# Plot markets
python plot_granger_markets.py --limit 10
```

## 📁 Project Structure

```
capstone/
├── Core Analysis Scripts
│   ├── Granger.py                    # Main Granger causality analysis
│   ├── validate_granger.py           # Lead-lag validation (consolidated)
│   ├── test_jump_prediction.py       # Insider trading hypothesis test
│   ├── plot_granger_markets.py       # Visualization
│   └── show_granger_results.py       # Results viewer
│
├── Data Loading
│   ├── PolyMarket.py                 # Polymarket API integration
│   ├── FinData.py                    # Databento SPY data
│   └── load_market_prices.py         # Loading utilities
│
├── Utilities
│   ├── DbController.py               # Database management
│   ├── utils.py                      # Shared utilities
│   └── main.py                       # Orchestration
│
├── data/
│   ├── markets.duckdb                # SPY data + Granger results
│   └── research.duckdb               # Polymarket data
│
├── docs/                             # 📚 Documentation
│   ├── README.md                     # Detailed usage guide
│   ├── RESEARCH_REPORT.md            # Full research report
│   ├── CLEANUP_PLAN.md               # Code organization
│   ├── VALIDATION_*.md               # Validation findings
│   └── next_steps.md                 # Future work
│
├── results/                          # 📈 Analysis Results
│   ├── validation_short_lags.csv
│   ├── validation_medium_long_lags.csv
│   └── validation_eq_to_poly.csv
│
├── plots/validation/                 # 📉 Plots
│
└── archive/                          # 🗄️ Legacy scripts
    └── (obsolete validation scripts)
```

## 📖 Documentation

- **[Full Documentation](docs/README.md)** - Detailed usage guide
- **[Research Report](docs/RESEARCH_REPORT.md)** - Complete findings and methodology
- **[Cleanup Plan](docs/CLEANUP_PLAN.md)** - Code organization details

## 🔬 Research Summary

### Methodology
1. **Granger Causality Tests** on 350 Polymarket tokens vs SPY
2. **Lead-Lag Cross-Correlation** validation
3. **Jump Prediction Analysis** for insider trading detection

### Initial Results (Granger)
- 49 markets: Polymarket → SPY significant
- 44 markets: SPY → Polymarket significant  
- P-values as low as 10⁻⁶⁹

### Validation Results
- **83% synchronous** (max correlation at lag=0)
- **0% confirmed** (max correlation at Granger lag)
- Correlation 8-16x stronger at lag=0 vs Granger lag

### Conclusion
Markets are **equally efficient**. No exploitable lead-lag relationships. Granger results were **spurious**, driven by common information shocks.

## 📊 Data Coverage

- **Period**: Oct 2024 - Nov 2025 (256 trading days)
- **Polymarket**: 440 tokens, 44.6M prices
- **SPY**: 100,096 minute bars

## 🛠️ Environment

```bash
pip install duckdb pandas numpy matplotlib scipy statsmodels
```

## 👨‍🔬 Author

Michael - Capstone Project 2025

---

**See [docs/README.md](docs/README.md) for detailed documentation.**
