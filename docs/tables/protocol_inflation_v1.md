| Evaluation shortcut | Panel / strategy | Reported | Honest | Inflation |
| --- | --- | ---: | ---: | ---: |
| Report the development window as if it were a held-out test | US 30-ETF: Causal dynamic graph-temporal model, 5 epochs | 0.775 (mean) | 0.685 (mean) | **+0.090** |
| Report the best random seed instead of the seed distribution | US 30-ETF: Causal dynamic graph-temporal model, outer window | 0.734 (max) | 0.729 (median) | **+0.005** |
| Report the best random seed instead of the seed distribution | US 50-ETF: Causal dynamic graph-temporal model, development window | 1.261 (max) | 0.759 (median) | **+0.502** |
| Report gross or near-zero transaction costs | US 30-ETF: Causal dynamic graph-temporal model (mean one-way turnover 56%/month) | 0.731 (mean) | 0.500 (mean) | **+0.232** |
| Report gross or near-zero transaction costs | US 30-ETF: Equal weight (low-turnover control: the shortcut should barely move it) | 0.720 (mean) | 0.710 (mean) | **+0.010** |
| Report gross or near-zero transaction costs | Confirmation panel (US 30-ETF, sealed): Dynamic graph-temporal, seed 7 (48.9%/month turnover) | 0.735 (mean) | 0.568 (mean) | **+0.167** |
| Report gross or near-zero transaction costs | Confirmation panel (US 30-ETF, sealed): Equal weight (low-turnover control: 2.0%/month) | 0.680 (mean) | 0.673 (mean) | **+0.007** |
| Report the best random seed instead of the seed distribution | Confirmation panel, 2020+ secondary window (sealed): Dynamic graph-temporal; best baseline on this window is 0.737 | 0.799 (max) | 0.714 (median) | **+0.085** |
| Report the development window as if it were a held-out test | US 30-ETF, declared 20-epoch training budget: Dynamic graph-temporal, 5 seeds (equal weight: 0.792 development, 0.718 outer) | 0.930 (median) | 0.604 (median) | **+0.326** |
| All three of the above at once | US 30-ETF, declared 20-epoch budget: Development window, best seed, zero costs, against held-out median at 10 bps | 1.095 (max) | 0.604 (median) | **+0.491** |
| Fix one favourite baseline instead of reporting a ladder | Confirmation panel (US 30-ETF, sealed): Dynamic graph-temporal, best seed vs the six pre-declared baselines | +0.045 edge | -0.025 edge | **+0.070** **sign flips** |
| Compare against a benchmark covering a different period | Confirmation panel, quarterly rebalance (sealed): Dynamic graph-temporal seed 7 (24-quarter warm-up starts 2020) vs equal weight | +0.538 edge | +0.045 edge | **+0.493** |
| Report the best grid cell without correcting for the search | US 30-ETF: Uncertainty-aware HMM conditional risk, 9-cell grid vs matched HRP-756 control | p=0.359 | p=0.680 | **+0.321** |
| Report the best candidate without correcting for the family | Confirmation panel (US 30-ETF, sealed): Seven-candidate confirmation family vs strongest baseline | p=0.131 | p=0.344 | **+0.213** |
| Report the best seed's p-value without correcting for the family | Confirmation panel, 2020+ secondary window (sealed): Seed 7 alone vs the seven-candidate family, against the strongest baseline | p=0.065 | p=0.185 | **+0.120** |

All allocation rows report net sharpe. Inflation is reported minus honest.
