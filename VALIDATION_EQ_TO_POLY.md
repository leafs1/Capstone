# Equity → Polymarket Validation Results

## Summary

Testing whether SPY movements predict Polymarket probabilities reveals the **same pattern** as the reverse direction:

### Results: 30 Markets Tested

| Verdict | Count | Percentage |
|---------|-------|------------|
| **Synchronous** (both move together) | 25 | **83.3%** ❌ |
| **Reversed** (Poly leads SPY, wrong direction) | 2 | 6.7% ❌ |
| **Plausible** (SPY might lead Poly) | 1 | **3.3%** ⚠️ |
| **Mismatch** (unclear pattern) | 2 | 6.7% ⚠️ |
| **TOTAL PREDICTIVE** | 1 | **3.3%** |

### Key Findings

1. **83.3% are synchronous** - Markets move together at the same time
2. **Only 1 market (3.3%) shows potential SPY → Poly prediction**
   - "September 2025 unemployment rate = 4.2%"
   - But this is just 1 out of 30 (likely random chance)
3. **Correlation at lag=0 is 11x stronger** than at Granger lag
   - At Granger lag: Mean |corr| = 0.0050
   - At lag=0: Mean |corr| = 0.0572 (11x stronger!)

### Interpretation

The **same conclusion** applies to both directions:

**❌ SPY does NOT predict Polymarket**
**❌ Polymarket does NOT predict SPY**

**✅ They move TOGETHER in response to the same news**

### What This Means

Both markets are:
1. **Equally efficient** at processing information
2. **Reacting simultaneously** to news (FOMC, CPI, tweets, etc.)
3. **Not leading/lagging** each other

When the Fed announces something or economic data drops:
- Stock traders see it immediately → react within seconds
- Polymarket traders see it immediately → react within seconds
- Both markets move at the same time
- No predictive relationship in either direction

### Statistical Evidence

**Example: NYSE Circuit Breaker Market**
- Granger: SPY predicts Poly at 30 min lag (p<0.001)
- Reality: Max correlation at lag=0 (synchronous)
- Correlation at 30 min: -0.003 (basically zero, wrong sign)
- **Verdict:** Synchronous, not predictive

**Example: Inflation >6% Market**
- Granger: SPY predicts Poly at 30 min lag (p<0.001)
- Reality: Max correlation at lag=0 (synchronous)
- Correlation at 30 min: -0.002 (basically zero, wrong sign)
- **Verdict:** Synchronous, not predictive

### Comparison with Poly → SPY Direction

| Metric | Poly → SPY | SPY → Poly |
|--------|------------|------------|
| Markets tested | 41 | 30 |
| Synchronous | 82.9% | 83.3% |
| Reversed | 9.8% | 6.7% |
| Predictive | 0% | 3.3% |
| Corr at Granger lag | 0.008 | 0.005 |
| Corr at lag=0 | 0.082 | 0.057 |

**Conclusion:** Both directions show the same pattern - **synchronous movement, not prediction**.

### Files Generated

- `validation_eq_to_poly.csv` - Detailed results (30 markets)
- `plots/validation_eq_to_poly/` - Lead-lag correlation plots (30 plots)

### Bottom Line

**Neither market predicts the other.** They both react to the same information simultaneously, demonstrating high market efficiency in both venues.

The Granger causality results in BOTH directions are spurious correlations caused by:
1. Simultaneous reactions to common news
2. Shared time trends
3. Multiple testing false positives

**There is no lead-lag relationship in either direction.**
