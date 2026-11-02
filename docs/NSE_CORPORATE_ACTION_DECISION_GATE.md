# NSE Corporate-Action Decision Gate

## Status

The project has **not** approved any corporate-action price factor. The 107
rows in the existing split/bonus review queue are parsed proposals, not
adjustments. This is correct: a free-text event such as `Bonus 1:1` is not a
substitute for a documented treatment of the security and its price source.

Before an NSE raw-price panel can be adjusted, a reviewer must create a
decision CSV based on
[`nse_corporate_action_decisions_TEMPLATE.csv`](../configs/nse_corporate_action_decisions_TEMPLATE.csv).
It must have exactly one final decision per review-queue `source_record_id`.

Allowed decisions are:

| Decision | Meaning | Adjustment output |
| --- | --- | --- |
| `approve_price_scale_factor` | Primary evidence confirms a conventional equity split/bonus and the reviewed factor. | Emits one guarded split/bonus factor. |
| `exclude_from_raw_panel` | The event cannot safely be represented in raw-price history. | No factor; the affected security is not eligible for that raw panel. |
| `require_total_return_source` | Cash or other value treatment requires a documented total-return source. | No factor; raw prices remain insufficient. |

Every row needs a reviewer, UTC review timestamp, direct evidence URL, and an
explanatory note. A complex scheme/reorganisation or unparsed candidate can
never receive a price-scale approval. A reviewer may override the parser's
proposed factor only with a substantive note and primary evidence.

## Reproducible materialisation

```bash
python -B scripts/build_nse_approved_price_scale_actions.py \
  --review data/exploratory/kaggle_nifty50_weights_v4/nse_price_scale_action_review_community_symbols_v2.csv \
  --decisions path/to/nse_corporate_action_decisions_v1.csv \
  --output path/to/nse_price_scale_actions_v1.csv
```

The command fails if any candidate lacks a decision, if a decision references
an unknown event, if a factor is non-finite/zero/one, or if a complex action is
approved. It writes a hash-backed sidecar covering both inputs.

The generated output may be passed to the existing guarded price-adjustment
routine. It is still **price-scale only**: dividends, rights, mergers,
demergers, spin-offs, delistings, index membership, and security identities
remain independent requirements. Therefore the decision gate makes the India
blocker explicit and reviewable; it does not claim to resolve it.
