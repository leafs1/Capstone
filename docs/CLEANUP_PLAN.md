# Code Cleanup Plan

## Files to Remove (Replaced by validate_granger.py)

The following validation scripts have been consolidated into `validate_granger.py`:

- ❌ `validate_short_lags.py` - Replaced by: `validate_granger.py --lag-range short`
- ❌ `validate_medium_long_lags.py` - Replaced by: `validate_granger.py --lag-range medium/long`
- ❌ `validate_eq_to_poly.py` - Replaced by: `validate_granger.py --direction eq_to_poly`
- ❌ `validate_extended_lag.py` - Not needed (maxlag=60 was deemed suspicious)

## Files to Keep

### Core Analysis Scripts
✅ `Granger.py` - Main Granger causality analysis engine
✅ `validate_granger.py` - **NEW** Consolidated validation module
✅ `show_granger_results.py` - Display and export results
✅ `plot_granger_markets.py` - Visualization tool
✅ `test_jump_prediction.py` - Insider trading/jump analysis

### Data Loading Scripts  
✅ `PolyMarket.py` - Polymarket data ingestion
✅ `FinData.py` - Financial data loading (Databento)
✅ `load_market_prices.py` - Market price loading utility

### Database & Utilities
✅ `DbController.py` - Database management
✅ `utils.py` - Shared utilities
✅ `main.py` - Main orchestration script

### One-off/Diagnostic Scripts (Consider Archiving)
⚠️ `analyze_timestamps.py` - Timestamp diagnostic (used once)
⚠️ `test_merge.py` - Test script (used for debugging)
⚠️ `filter_short_lags.py` - Filtering utility (one-time use)
⚠️ `backfill_checkpoints.py` - Data backfill (one-time use)
⚠️ `fix_checkpoints_table.py` - Schema fix (one-time use)
⚠️ `check_progress.py` - Progress monitoring (diagnostic)
⚠️ `validate_market_variance.py` - Variance check (one-time validation)

## Recommended Actions

1. **Delete obsolete validation scripts:**
   ```bash
   rm validate_short_lags.py
   rm validate_medium_long_lags.py  
   rm validate_eq_to_poly.py
   rm validate_extended_lag.py
   ```

2. **Archive diagnostic scripts** (move to `archive/` folder):
   ```bash
   mkdir -p archive
   mv analyze_timestamps.py archive/
   mv test_merge.py archive/
   mv filter_short_lags.py archive/
   mv backfill_checkpoints.py archive/
   mv fix_checkpoints_table.py archive/
   mv check_progress.py archive/
   mv validate_market_variance.py archive/
   ```

3. **Keep CSV results** (useful for reference):
   - validation_short_lags.csv
   - validation_medium_long_lags.csv
   - validation_eq_to_poly.csv
   - token_timestamp_analysis.csv

## New Clean Structure

```
capstone/
├── core/                           # Main analysis modules
│   ├── Granger.py                  # Granger causality analysis
│   ├── validate_granger.py         # Validation module (NEW)
│   ├── test_jump_prediction.py     # Jump/insider trading analysis
│   └── plot_granger_markets.py     # Visualization
│
├── data_loading/                   # Data ingestion
│   ├── PolyMarket.py               # Polymarket API
│   ├── FinData.py                  # Databento integration
│   └── load_market_prices.py       # Price loading utilities
│
├── utilities/                      # Utilities & DB
│   ├── DbController.py             # Database management
│   ├── utils.py                    # Shared utilities
│   ├── show_granger_results.py     # Results viewer
│   └── main.py                     # Orchestration
│
├── data/                           # Databases
│   ├── markets.duckdb              # SPY + Granger results
│   └── research.duckdb             # Polymarket data
│
├── plots/                          # Generated plots
│   └── validation/                 # Validation plots
│
├── archive/                        # One-off scripts
│   └── (diagnostic scripts)
│
├── RESEARCH_REPORT.md              # Final research report
└── validation_*.csv                # Validation results
```

## Usage After Cleanup

### Run Granger Analysis:
```bash
python Granger.py
```

### Validate Results:
```bash
# Validate Polymarket → SPY (all lags)
python validate_granger.py --direction poly_to_eq

# Validate specific lag ranges
python validate_granger.py --direction poly_to_eq --lag-range short
python validate_granger.py --direction poly_to_eq --lag-range medium
python validate_granger.py --direction poly_to_eq --lag-range long

# Validate SPY → Polymarket
python validate_granger.py --direction eq_to_poly --limit 30

# Save results to CSV
python validate_granger.py --direction poly_to_eq --save-csv
```

### View Results:
```bash
# Show all significant results
python show_granger_results.py

# Filter by direction
python show_granger_results.py --direction both  # bidirectional only
python show_granger_results.py --direction poly_to_eq

# Export to CSV
python show_granger_results.py --export granger_results.csv
```

### Plot Markets:
```bash
python plot_granger_markets.py --direction poly_to_eq --limit 10
```

### Test Jump Prediction:
```bash
python test_jump_prediction.py
```

## Benefits of Cleanup

1. **Single source of truth** for validation (validate_granger.py)
2. **Reduced duplication** - shared functions for correlation, plotting, etc.
3. **Clearer organization** - obvious what each file does
4. **Easier maintenance** - fix bugs in one place
5. **Better discoverability** - new users can find relevant scripts quickly
