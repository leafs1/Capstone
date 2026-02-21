# Archive Directory

This directory contains one-off diagnostic and legacy scripts that were used during development but are no longer part of the main workflow.

## Obsolete Validation Scripts (Replaced by validate_granger.py)

- **validate_short_lags.py** - Validated ultra-short lag markets (1-3 minutes)
  - Replaced by: `validate_granger.py --lag-range short`
  
- **validate_medium_long_lags.py** - Validated medium and long lag markets (5-30 minutes)
  - Replaced by: `validate_granger.py --lag-range medium/long`
  
- **validate_eq_to_poly.py** - Validated reverse direction (SPY → Polymarket)
  - Replaced by: `validate_granger.py --direction eq_to_poly`
  
- **validate_extended_lag.py** - Tested extended lags up to 60 minutes
  - **Finding:** 80% of results hit the 54-60 minute boundary, deemed suspicious
  - **Decision:** Not pursued further, maxlag=30 is appropriate

## Diagnostic Scripts (One-time Use)

- **analyze_timestamps.py** - Analyzed timestamp alignment issues between Polymarket and SPY
  - **Purpose:** Discovered timestamp rounding needed for proper merging
  - **Outcome:** Fixed in Granger.py merge_poly_with_equity()
  
- **test_merge.py** - Test script for debugging data merge issues
  - **Purpose:** Debugging "insufficient data" errors
  - **Outcome:** Identified need for timestamp rounding to nearest minute

- **filter_short_lags.py** - Filtered Granger results for short lag analysis
  - **Purpose:** Extract markets with 1-3 minute lags for focused validation
  - **Outcome:** Used once to generate validation targets

- **validate_market_variance.py** - Checked if markets had sufficient price variation
  - **Purpose:** Ensure markets weren't flat/inactive during analysis window
  - **Outcome:** 87.5% had >10% price range (healthy variation)

## Database Maintenance Scripts (One-time Use)

- **backfill_checkpoints.py** - Backfilled missing checkpoint data
  - **Purpose:** Populate checkpoints table for existing data
  - **Outcome:** Run once, checkpoints now maintained by PolyMarket.py

- **fix_checkpoints_table.py** - Fixed checkpoint table schema
  - **Purpose:** Schema migration for checkpoint tracking
  - **Outcome:** Run once, no longer needed

- **check_progress.py** - Monitored data ingestion progress
  - **Purpose:** Track Polymarket data loading status
  - **Outcome:** Diagnostic tool, not part of main workflow

## These Files Are Safe to Delete

All functionality has been:
1. Integrated into main codebase
2. Superseded by better approaches
3. Or was one-time diagnostic work

Kept in archive for reference only. Can be permanently deleted if needed.
