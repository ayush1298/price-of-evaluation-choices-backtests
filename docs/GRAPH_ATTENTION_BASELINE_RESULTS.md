# Causal graph-attention baseline diagnostic

This is the graph-attention comparator required before claiming an advantage
for the repository's dynamic graph Transformer. It is deliberately separate
from the proposed model: only the graph message-passing operator changes.

## Implementation

`DynamicGraphTransformer` now supports `graph_encoder="attention"`. At every
causal graph snapshot it performs query/key/value attention only across
observed graph edges plus a self-loop. The non-negative graph similarity is a
log-attention prior, and absent edges are masked rather than converted into
learned links. The subsequent causal temporal Transformer, node features,
expanding training labels, allocator, execution lag, cap, and transaction
costs are identical to the similarity-message-passing model.

The frozen configurations are:

- `configs/public_us_etf_2010_graph_attention_dynamic_graph.json`
- `configs/public_india_etf_2014_graph_attention_dynamic_graph.json`

## US development result

The US seed-7, five-epoch development run covers 2014-02-03 to 2019-12-31
(1,489 realised daily observations) with 10 bps one-way costs. It is compared
to the existing causal similarity-message-passing model over precisely the
same return dates.

| Graph encoder | Annualised return | Annualised volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Similarity message passing | 7.96% | 9.92% | 0.821 | -13.88% |
| Edge-masked graph attention | 8.05% | 10.24% | 0.807 | -17.15% |

The attention model's 21-session, 5,000-replication paired moving-block
annualised mean-return difference is +0.11%, but the confidence interval spans
zero and its one-sided p-value is 0.474. Its primary development Sharpe and
drawdown are both worse. It therefore fails the pre-2020 development gate and
is not advanced to the 2020+ outer diagnostic or India study.

This is still a useful paper-control result: the final method cannot attribute
any result to merely replacing similarity propagation with graph attention.
