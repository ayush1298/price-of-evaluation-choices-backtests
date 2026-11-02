"""A dependency-light temporal graph-attention model for fixed-universe panels.

The legacy DeepCNL learns one static graph through a CNN/LSTM objective. This
candidate consumes a sequence of *causally constructed* graph snapshots and
node features. Similarity-weighted message passing captures cross-sectional
structure at each date; a causal Transformer then captures temporal patterns
for each asset. The model itself does not construct graphs or decide portfolio
weights, keeping those choices testable and independently ablatable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, pi

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class GraphForecast:
    """Per-asset predicted return score and positive scale estimate."""

    score: Tensor
    scale: Tensor


class _SimilarityMessagePassing(nn.Module):
    """Residual message passing for non-negative similarity adjacencies."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(dimension, dimension, bias=False)
        self.update = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        # Add a self-loop for isolated assets, then make every row a convex
        # combination. Canonical graph builders retain those isolated nodes.
        asset_count = adjacency.shape[-1]
        identity = torch.eye(asset_count, dtype=adjacency.dtype, device=adjacency.device)
        normalized = adjacency + identity
        normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(nodes.dtype).eps)
        aggregated = torch.einsum("btij,btjd->btid", normalized, self.message(nodes))
        update = self.update(torch.cat([nodes, aggregated], dim=-1))
        return self.norm(nodes + update)


class _GraphAttentionMessagePassing(nn.Module):
    """Edge-masked graph attention with observed similarity as an attention prior."""

    def __init__(self, dimension: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(dimension, dimension, bias=False)
        self.key = nn.Linear(dimension, dimension, bias=False)
        self.value = nn.Linear(dimension, dimension, bias=False)
        self.update = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, dimension),
        )
        self.norm = nn.LayerNorm(dimension)

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        asset_count = adjacency.shape[-1]
        identity = torch.eye(asset_count, dtype=adjacency.dtype, device=adjacency.device)
        prior = adjacency + identity
        mask = prior > 0
        scale = nodes.shape[-1] ** -0.5
        scores = torch.einsum("btid,btjd->btij", self.query(nodes), self.key(nodes)) * scale
        # Preserve graph sparsity while allowing attention to learn how much
        # each observed edge matters. log(prior) is a finite similarity prior;
        # absent edges remain impossible rather than receiving a learned link.
        scores = scores + prior.clamp_min(torch.finfo(nodes.dtype).eps).log()
        scores = scores.masked_fill(~mask, torch.finfo(nodes.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        aggregated = torch.einsum("btij,btjd->btid", attention, self.value(nodes))
        update = self.update(torch.cat([nodes, aggregated], dim=-1))
        return self.norm(nodes + update)


class DynamicGraphTransformer(nn.Module):
    """Forecast per-asset signals from dynamic graphs and historical features.

    Parameters
    ----------
    input_features:
        Number of causal node features per asset/date (for example returns,
        realised volatility, volume features, and centrality measures).
    model_dimension:
        Shared graph/temporal representation width.
    temporal_layers:
        Number of causal Transformer encoder layers after graph propagation.
    graph_layers:
        Number of similarity-weighted graph message-passing layers per date.

    Notes
    -----
    ``forward`` expects ``features`` with shape ``[batch, time, assets,
    input_features]`` and ``adjacency`` with shape ``[batch, time, assets,
    assets]``. All dates must be on or before the signal date. The predicted
    score is not a portfolio weight; constrained allocation remains a separate
    stage of the canonical pipeline.
    """

    def __init__(
        self,
        input_features: int,
        *,
        model_dimension: int = 64,
        attention_heads: int = 4,
        temporal_layers: int = 2,
        graph_layers: int = 2,
        dropout: float = 0.1,
        maximum_sequence_length: int = 512,
        regime_expert_heads: int = 0,
        graph_encoder: str = "similarity",
    ) -> None:
        super().__init__()
        if input_features < 1 or model_dimension < 1:
            raise ValueError("input_features and model_dimension must be positive")
        if attention_heads < 1 or model_dimension % attention_heads:
            raise ValueError("model_dimension must be divisible by attention_heads")
        if temporal_layers < 0 or graph_layers < 0 or (temporal_layers == 0 and graph_layers == 0):
            raise ValueError("Use at least one non-negative graph or temporal layer")
        if maximum_sequence_length < 1:
            raise ValueError("maximum_sequence_length must be positive")
        if regime_expert_heads < 0:
            raise ValueError("regime_expert_heads cannot be negative")
        if regime_expert_heads and input_features < regime_expert_heads:
            raise ValueError("input_features must include every regime probability when expert heads are enabled")
        if graph_encoder not in {"similarity", "attention"}:
            raise ValueError("graph_encoder must be 'similarity' or 'attention'")
        self.input_features = input_features
        self.maximum_sequence_length = maximum_sequence_length
        self.temporal_layers = temporal_layers
        self.regime_expert_heads = regime_expert_heads
        self.graph_encoder = graph_encoder
        self.feature_projection = nn.Linear(input_features, model_dimension)
        self.position_embedding = nn.Parameter(torch.empty(maximum_sequence_length, model_dimension))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        graph_block = _SimilarityMessagePassing if graph_encoder == "similarity" else _GraphAttentionMessagePassing
        self.graph_blocks = nn.ModuleList([graph_block(model_dimension, dropout) for _ in range(graph_layers)])
        if temporal_layers:
            layer = nn.TransformerEncoderLayer(
                d_model=model_dimension,
                nhead=attention_heads,
                dim_feedforward=model_dimension * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            # Nested-tensor acceleration is not applicable with pre-norm layers
            # and dense fixed-universe panels; disable it explicitly to avoid a
            # misleading runtime warning during every experiment.
            self.temporal_encoder: nn.Module | None = nn.TransformerEncoder(
                layer, num_layers=temporal_layers, enable_nested_tensor=False
            )
        else:
            self.temporal_encoder = None
        self.score_head = nn.Linear(model_dimension, 1)
        self.scale_head = nn.Sequential(nn.Linear(model_dimension, 1), nn.Softplus())
        self.regime_score_heads = nn.ModuleList(
            [nn.Linear(model_dimension, 1) for _ in range(regime_expert_heads)]
        )
        self.regime_scale_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(model_dimension, 1), nn.Softplus()) for _ in range(regime_expert_heads)]
        )

    def forward(self, features: Tensor, adjacency: Tensor) -> GraphForecast:
        self._validate_inputs(features, adjacency)
        batch_size, sequence_length, asset_count, _ = features.shape
        nodes = self.feature_projection(features)
        for block in self.graph_blocks:
            nodes = block(nodes, adjacency)
        temporal = nodes.permute(0, 2, 1, 3).reshape(batch_size * asset_count, sequence_length, -1)
        temporal = temporal + self.position_embedding[:sequence_length].unsqueeze(0)
        causal_mask = torch.ones(
            (sequence_length, sequence_length), dtype=torch.bool, device=features.device
        ).triu(diagonal=1)
        encoded = self.temporal_encoder(temporal, mask=causal_mask) if self.temporal_encoder else temporal
        final_state = encoded[:, -1, :].reshape(batch_size, asset_count, -1)
        if not self.regime_expert_heads:
            return GraphForecast(
                score=self.score_head(final_state).squeeze(-1),
                scale=self.scale_head(final_state).squeeze(-1).clamp_min(torch.finfo(features.dtype).eps),
            )
        probabilities = self._regime_probabilities(features)
        expert_scores = torch.stack(
            [head(final_state).squeeze(-1) for head in self.regime_score_heads], dim=-1
        )
        expert_scales = torch.stack(
            [head(final_state).squeeze(-1) for head in self.regime_scale_heads], dim=-1
        ).clamp_min(torch.finfo(features.dtype).eps)
        mixture = probabilities.unsqueeze(1)
        score = (expert_scores * mixture).sum(dim=-1)
        # A mixture's variance combines each expert's aleatoric variance and
        # the uncertainty between state-specific conditional means.
        second_moment = (mixture * (expert_scales.square() + expert_scores.square())).sum(dim=-1)
        variance = (second_moment - score.square()).clamp_min(torch.finfo(features.dtype).eps)
        return GraphForecast(score=score, scale=variance.sqrt())

    def gaussian_negative_log_likelihood(
        self,
        forecast: GraphForecast,
        realised_returns: Tensor,
        *,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Return mean Gaussian NLL for a future-return training label.

        ``realised_returns`` is a label strictly after the input sequence and
        therefore belongs in an outer walk-forward training split, not in the
        feature construction path.
        """
        if realised_returns.shape != forecast.score.shape:
            raise ValueError("realised_returns must match the [batch, assets] forecast shape")
        if not torch.isfinite(realised_returns).all():
            raise ValueError("realised_returns must be finite")
        variance = forecast.scale.square()
        losses = 0.5 * (
            torch.log(variance)
            + (realised_returns - forecast.score).square() / variance
            + log(2.0 * pi)
        )
        if target_mask is None:
            return losses.mean()
        if target_mask.shape != losses.shape:
            raise ValueError("target_mask must match the [batch, assets] forecast shape")
        if target_mask.device != losses.device:
            raise ValueError("target_mask must be on the forecast device")
        mask = target_mask.to(dtype=losses.dtype)
        if (mask < 0).any() or (mask > 1).any():
            raise ValueError("target_mask must be boolean or in [0, 1]")
        if mask.sum() <= 0:
            raise ValueError("target_mask must include at least one target")
        return (losses * mask).sum() / mask.sum()

    def cross_sectional_information_coefficient_loss(
        self,
        forecast: GraphForecast,
        realised_returns: Tensor,
        *,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``1 - IC`` over each valid cross-section in a batch.

        This auxiliary objective aligns forecast training with the ordering
        consumed by the long-only allocator. It never observes a later return
        at prediction time: callers supply the same strictly completed labels
        already used by Gaussian likelihood training. Samples with fewer than
        two non-constant eligible returns are excluded rather than assigned an
        arbitrary correlation.
        """
        if realised_returns.shape != forecast.score.shape:
            raise ValueError("realised_returns must match the [batch, assets] forecast shape")
        if not torch.isfinite(realised_returns).all():
            raise ValueError("realised_returns must be finite")
        if target_mask is not None and target_mask.shape != forecast.score.shape:
            raise ValueError("target_mask must match the [batch, assets] forecast shape")
        mask = torch.ones_like(forecast.score, dtype=torch.bool) if target_mask is None else target_mask.bool()
        losses: list[Tensor] = []
        epsilon = torch.finfo(forecast.score.dtype).eps
        for predicted, realised, valid in zip(forecast.score, realised_returns, mask):
            if int(valid.sum().item()) < 2:
                continue
            predicted = predicted[valid]
            realised = realised[valid]
            predicted_centered = predicted - predicted.mean()
            realised_centered = realised - realised.mean()
            denominator = predicted_centered.square().sum().sqrt() * realised_centered.square().sum().sqrt()
            if float(denominator.detach()) <= epsilon:
                continue
            correlation = (predicted_centered * realised_centered).sum() / denominator.clamp_min(epsilon)
            losses.append(1.0 - correlation)
        if not losses:
            return forecast.score.sum() * 0.0
        return torch.stack(losses).mean()

    def _validate_inputs(self, features: Tensor, adjacency: Tensor) -> None:
        if features.ndim != 4:
            raise ValueError("features must have shape [batch, time, assets, input_features]")
        if adjacency.ndim != 4:
            raise ValueError("adjacency must have shape [batch, time, assets, assets]")
        if features.shape[:3] != adjacency.shape[:3] or adjacency.shape[-1] != features.shape[2]:
            raise ValueError("features and adjacency must agree on batch, time, and asset dimensions")
        if features.shape[-1] != self.input_features:
            raise ValueError(f"Expected {self.input_features} node features, got {features.shape[-1]}")
        if features.shape[1] > self.maximum_sequence_length:
            raise ValueError(
                f"Sequence length {features.shape[1]} exceeds maximum_sequence_length={self.maximum_sequence_length}"
            )
        if features.device != adjacency.device or features.dtype != adjacency.dtype:
            raise ValueError("features and adjacency must share device and dtype")
        if not torch.isfinite(features).all() or not torch.isfinite(adjacency).all():
            raise ValueError("features and adjacency must be finite")
        if (adjacency < 0).any():
            raise ValueError("adjacency must contain non-negative similarity weights")

    def _regime_probabilities(self, features: Tensor) -> Tensor:
        """Read exogenous final-date state probabilities for gated expert heads.

        The canonical regime broadcaster appends globally shared probability
        columns after the regular node features. Averaging over assets makes
        the gate invariant to the ordering of the fixed asset universe; the
        inputs are validated rather than silently normalised from arbitrary
        feature values.
        """
        probabilities = features[:, -1, :, -self.regime_expert_heads :]
        tolerance = 1e-5
        if (probabilities < -tolerance).any():
            raise ValueError("regime expert probabilities must be non-negative")
        per_asset_total = probabilities.sum(dim=-1)
        if not torch.allclose(per_asset_total, torch.ones_like(per_asset_total), atol=tolerance, rtol=tolerance):
            raise ValueError("regime expert probabilities must sum to one for every asset")
        gate = probabilities.mean(dim=1).clamp_min(0.0)
        return gate / gate.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(features.dtype).eps)
