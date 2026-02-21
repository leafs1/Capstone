# Analysis Results

This directory contains CSV files with validation and analysis results.

## Validation Results

### Polymarket → SPY Direction

**validation_short_lags.csv** - Ultra-short lag markets (1-3 minutes)
- 13 markets tested
- **Finding**: 84.6% synchronous, 0% confirmed
- Mean correlation at lag=0: 0.095 vs Granger lag: 0.008 (12x stronger)

**validation_medium_long_lags.csv** - Medium and long lag markets (5-30 minutes)
- 28 markets tested (8 medium, 20 long)
- **Finding**: 82.1% synchronous overall, 0% confirmed
- Mean correlation at lag=0: 0.082 vs Granger lag: 0.008 (10x stronger)

### SPY → Polymarket Direction

**validation_eq_to_poly.csv** - Reverse direction validation
- 30 markets tested
- **Finding**: 83.3% synchronous, 3.3% plausible
- Mean correlation at lag=0: 0.057 vs Granger lag: 0.005 (11x stronger)

## Diagnostic Results

**token_timestamp_analysis.csv** - Timestamp alignment analysis
- Used to identify timestamp rounding issues
- Led to fix in Granger.py merge logic

## Key Takeaways

Across **all 71 validated markets** (both directions):
- **83.1% show maximum correlation at lag=0** (synchronous movement)
- **0% confirmed Granger lag** (no true predictive power)
- **Correlation ~11x stronger at lag=0** than at Granger lag

**Conclusion**: Markets react to same information simultaneously, not predictively.

## CSV Column Descriptions

### Validation CSVs

- **token_id**: Polymarket token identifier
- **question**: Market question/description
- **granger_lag**: Lag identified by Granger test (minutes)
- **max_corr_lag**: Lag where correlation is strongest (minutes)
- **corr_at_granger**: Correlation at Granger lag
- **corr_at_zero**: Correlation at lag=0 (synchronous)
- **corr_at_max**: Maximum correlation value
- **granger_p_value**: Permutation test p-value at Granger lag
- **zero_p_value**: Permutation test p-value at lag=0
- **verdict**: Classification (CONFIRMED, SYNCHRONOUS, REVERSED, etc.)
- **n_observations**: Number of aligned observations
- **plot_path**: Path to correlation plot

### Verdict Classifications

- **CONFIRMED**: Max correlation at Granger lag ✓
- **PLAUSIBLE**: Max correlation within ±3 min of Granger lag
- **NEAR**: Max correlation within ±5 min of Granger lag  
- **SYNCHRONOUS**: Max correlation at lag=0 (most common)
- **REVERSED**: Max correlation in opposite direction
- **MISMATCH**: Max correlation at inconsistent lag

## Usage

```python
import pandas as pd

# Load validation results
short_lags = pd.read_csv('results/validation_short_lags.csv')
medium_long = pd.read_csv('results/validation_medium_long_lags.csv')
eq_to_poly = pd.read_csv('results/validation_eq_to_poly.csv')

# Analyze synchronous markets
synchronous = short_lags[short_lags['verdict'] == 'SYNCHRONOUS']
print(f"Synchronous markets: {len(synchronous)} / {len(short_lags)}")

# Compare correlation strengths
ratio = short_lags['corr_at_zero'].abs() / short_lags['corr_at_granger'].abs()
print(f"Mean ratio (lag=0 / Granger): {ratio.mean():.2f}x")
```

## Related Documentation

- [Research Report](../docs/RESEARCH_REPORT.md) - Full analysis and interpretation
- [Validation Summaries](../docs/VALIDATION_*.md) - Detailed validation findings
- [Plots](../plots/validation/) - Visual evidence for each market
