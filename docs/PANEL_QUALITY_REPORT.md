# Candidate Panel Quality Report

`build_panel_quality_report.py` is the readiness-plan stage-3 diagnostic. It
does not bless a dataset for a paper; it makes the questions that must be
answered before baselines or models are rerun visible and hash-backed.

```bash
python -B scripts/build_panel_quality_report.py \
  --prices data/public/us_etf_2010_v1/ohlcv_yfinance.csv \
  --membership data/public/us_etf_2010_v1/membership_monthly.csv \
  --rebalance-frequency ME \
  --output data/public/us_etf_2010_v1/panel_quality_report.json
```

The report records:

1. per-asset first/last observation, gaps between those dates, early stops,
   and whether an asset stays eligible after its final price;
2. membership snapshot counts and additions/removals;
3. eligible-price coverage at every rebalance signal;
4. price-return outliers requiring an explanation—not automatic deletion;
5. crosswalk verification status and overlapping duplicate ISINs when a
   time-bounded identity crosswalk is provided; and
6. optional return-level reconciliation against an independent price source.

For an equity market, add `--identity-crosswalk` and a independently sourced
`--reference-prices` panel. A missing identity or reference source is reported
as a limitation, not silently treated as a pass. The public ETF reports are
development/reproducibility diagnostics only and remain non-paper evidence.
