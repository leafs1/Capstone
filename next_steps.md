# Granger Analysis - Next Steps

## Current Status
✅ 49 significant Polymarket → Equity relationships (maxlag=30)
✅ 44 significant Equity → Polymarket relationships  
✅ 16 bidirectional relationships (feedback loops)
✅ All with Bonferroni correction (rigorous)

## Priority Actions

### 1. Fix Plot Quality Issue ⚠️ HIGH PRIORITY
**Problem:** Plots show different smoothness due to different time periods
- NYSE circuit breaker: 305 days (81K observations) → looks chunky
- Fed interest rates: 94 days (25K observations) → looks smooth

**Solution:** Add adaptive resampling to plot_granger_markets.py
```python
if days > 180:  resample_freq = '30min'
elif days > 90: resample_freq = '15min'  
elif days > 30: resample_freq = '5min'
else:           resample_freq = None  # minute data
```

### 2. Filter Out Non-Economic Markets 🔍 MEDIUM PRIORITY
**Problem:** Celebrity/entertainment markets showing causality is suspicious
- "A$AP Rocky release song" → SPY (p=0.022)
- "Lana Del Rey new song" → SPY (p=0.029)
- "Beyoncé new song" → SPY (p=0.047)

**Why suspicious?**
- No economic mechanism for music releases to affect S&P 500
- Likely spurious correlations or bot activity
- All have very short analysis windows (24 days)

**Solution:** Create filtered results focusing on:
- Fed policy markets (rates, Powell, inflation)
- Economic indicators (GDP, unemployment, stagflation)
- Market structure (circuit breakers)
- Major policy (tariffs, deportations, DOGE cuts)

### 3. Investigate Ultra-Short Lags 🔬 HIGH PRIORITY
**Markets with 1-2 minute lags:**
- Fed December meeting (2 min, p=0.030)
- Trump fire Powell (1 min, p=0.008)
- Fed 50+ bps cut (2 min, p=0.001)
- Trump deport (2 min, p=0.037)

**Questions:**
- Are these true predictive relationships?
- Or just high correlation (moving together)?
- Could test with **lead-lag cross-correlation** to see if Polymarket actually leads

**Test:** Run correlation analysis at different lags (-30 to +30 minutes)
```python
for lag in range(-30, 31):
    corr = poly_series.shift(lag).corr(spy_series)
# If max correlation at lag=0, it's synchronous (not predictive)
# If max at positive lag, Polymarket leads (good!)
```

### 4. Verify YES/NO Token Pairs 🔍 LOW PRIORITY
**Markets with duplicate entries:**
- NYSE circuit breaker (2 tokens)
- Inflation 6%, 8%, 10% (2 tokens each)
- Fed December meeting (2 tokens)

**Expected:** YES + NO probabilities should sum to ~1.0

**Check:** Verify these are complementary markets, not duplicates

### 5. Out-of-Sample Validation 📊 MEDIUM PRIORITY
**Current:** In-sample Granger tests (same data used to find relationships)

**Better:** Out-of-sample testing
1. Train on first 80% of data
2. Test if Polymarket predicts SPY in final 20%
3. Measure actual predictive power

**Metrics:**
- Direction accuracy (up/down)
- Magnitude correlation
- Sharpe ratio if trading on signals

### 6. Event Study Analysis 📅 LOW PRIORITY
**Hypothesis:** Relationships stronger around economic events

**Events to check:**
- FOMC announcements (8 per year)
- CPI releases (monthly)
- Jobs reports (monthly)
- Fed Chair speeches

**Analysis:** Compare Granger causality strength in:
- 3 days before/after events vs
- Baseline periods

### 7. Categorical Analysis 📋 MEDIUM PRIORITY
**Group markets by theme:**
- Fed Policy (rates, Powell, cuts)
- Inflation (CPI targets)
- Economic Growth (GDP, unemployment, stagflation)
- Market Structure (circuit breakers)
- Political Events (tariffs, deportations)
- Celebrity/Entertainment (filter out)

**Analysis:**
- Which categories have strongest causality?
- Which have shortest lags?
- Which are bidirectional vs unidirectional?

### 8. Liquidity Adjustment 🔍 LOW PRIORITY
**Concern:** Some markets may have low liquidity
- 479 observations = only 3 days of data (German tariff market)
- 152 observations = 3 days (Powell says "Trump")

**Filter:** Require minimum:
- 30 days of data (7,800 observations)
- OR document results separately as "short-term markets"

## Recommended Immediate Actions

**Today:**
1. Fix plot quality (adaptive resampling)
2. Create filtered results (economic markets only)
3. Investigate ultra-short lags (1-2 min markets)

**This Week:**
4. Test lead-lag correlations for validation
5. Categorical analysis (group by theme)
6. Document suspicious markets (entertainment)

**Optional (Research Extensions):**
7. Out-of-sample validation
8. Event study analysis
9. Trading strategy backtest

## Questions to Answer

1. **Are 1-2 minute lags too fast to be real?**
   - Could be HFT bots reading both markets
   - Or genuine information flow
   - Need lead-lag correlation test

2. **Why do entertainment markets show causality?**
   - Probably spurious
   - Or bot activity
   - Should filter out

3. **Are YES/NO pairs both included?**
   - May be double-counting same market
   - Check if they're complementary

4. **Can we actually trade on this?**
   - Need out-of-sample test
   - Need to account for:
     - Transaction costs
     - Polymarket fees
     - Execution delays

## Files to Create

- `validate_short_lags.py` - Test 1-2 min lag markets
- `filter_economic_markets.py` - Remove entertainment/celebrity
- `categorical_analysis.py` - Group by theme
- `lead_lag_correlation.py` - Validate predictive power
- `plot_granger_markets_v2.py` - Fixed adaptive resampling
