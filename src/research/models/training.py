"""Expanding-window training utilities for the dynamic graph candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .dataset import DynamicGraphWindowDataset
from .dynamic_graph_transformer import DynamicGraphTransformer, GraphForecast


@dataclass(frozen=True)
class DynamicGraphTrainingConfig:
    """Training settings recorded in every proposed-model experiment."""

    model_dimension: int = 64
    attention_heads: int = 4
    temporal_layers: int = 2
    graph_layers: int = 2
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    regime_expert_heads: int = 0
    graph_encoder: str = "similarity"
    information_coefficient_loss_weight: float = 0.0
    random_seed: int = 42
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.model_dimension < 1 or self.attention_heads < 1:
            raise ValueError("model_dimension and attention_heads must be positive")
        if self.model_dimension % self.attention_heads:
            raise ValueError("model_dimension must be divisible by attention_heads")
        if self.temporal_layers < 0 or self.graph_layers < 0 or (self.temporal_layers == 0 and self.graph_layers == 0):
            raise ValueError("Use at least one non-negative graph or temporal layer")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Invalid optimizer or gradient-clipping setting")
        if self.regime_expert_heads < 0:
            raise ValueError("regime_expert_heads cannot be negative")
        if self.graph_encoder not in {"similarity", "attention"}:
            raise ValueError("graph_encoder must be 'similarity' or 'attention'")
        if self.information_coefficient_loss_weight < 0:
            raise ValueError("information_coefficient_loss_weight cannot be negative")


@dataclass(frozen=True)
class DynamicGraphTrainingResult:
    """A fitted causal model plus the exact training-label cutoff evidence."""

    model: DynamicGraphTransformer
    decision_date: pd.Timestamp
    training_indices: tuple[int, ...]
    training_input_end_dates: tuple[pd.Timestamp, ...]
    training_target_end_dates: tuple[pd.Timestamp, ...]
    epoch_losses: tuple[float, ...]
    device: str


def train_expanding_dynamic_graph_transformer(
    dataset: DynamicGraphWindowDataset,
    decision_date: pd.Timestamp | str,
    *,
    config: DynamicGraphTrainingConfig | None = None,
) -> DynamicGraphTrainingResult:
    """Train using only labels fully observed before a decision date.

    The strict ``target_end_date < decision_date`` filter is the critical
    leakage guard. A feature window ending at an earlier signal date is not
    considered trainable until its subsequent return label has completed.
    """
    config = config or DynamicGraphTrainingConfig()
    decision = pd.Timestamp(decision_date)
    indices = tuple(
        index
        for index, target_end in enumerate(dataset.target_end_dates)
        if pd.Timestamp(target_end) < decision
    )
    if not indices:
        raise ValueError(
            f"No completed forward-return labels are available before decision date {decision.date()}"
        )
    device = torch.device(config.device)
    torch.manual_seed(config.random_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.random_seed)
    model = DynamicGraphTransformer(
        input_features=dataset.features.shape[-1],
        model_dimension=config.model_dimension,
        attention_heads=config.attention_heads,
        temporal_layers=config.temporal_layers,
        graph_layers=config.graph_layers,
        dropout=config.dropout,
        maximum_sequence_length=dataset.features.shape[1],
        regime_expert_heads=config.regime_expert_heads,
        graph_encoder=config.graph_encoder,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    ordered_indices = torch.tensor(indices, dtype=torch.long)
    generator = torch.Generator(device="cpu").manual_seed(config.random_seed)
    epoch_losses: list[float] = []
    model.train()
    for _ in range(config.epochs):
        permutation = ordered_indices[torch.randperm(len(ordered_indices), generator=generator)]
        weighted_loss = 0.0
        valid_targets = 0
        for start in range(0, len(permutation), config.batch_size):
            batch_indices = permutation[start : start + config.batch_size]
            features = dataset.features[batch_indices].to(device)
            adjacency = dataset.adjacency[batch_indices].to(device)
            targets = dataset.target_returns[batch_indices].to(device)
            mask = dataset.target_mask[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            forecast = model(features, adjacency)
            loss = model.gaussian_negative_log_likelihood(forecast, targets, target_mask=mask)
            if config.information_coefficient_loss_weight:
                loss = loss + config.information_coefficient_loss_weight * model.cross_sectional_information_coefficient_loss(
                    forecast, targets, target_mask=mask
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()
            batch_count = int(mask.sum().item())
            weighted_loss += float(loss.detach().cpu()) * batch_count
            valid_targets += batch_count
        epoch_losses.append(weighted_loss / valid_targets)
    model.eval()
    return DynamicGraphTrainingResult(
        model=model,
        decision_date=decision,
        training_indices=indices,
        training_input_end_dates=tuple(dataset.input_end_dates[index] for index in indices),
        training_target_end_dates=tuple(dataset.target_end_dates[index] for index in indices),
        epoch_losses=tuple(epoch_losses),
        device=str(device),
    )


def predict_dynamic_graph_forecast(
    result: DynamicGraphTrainingResult,
    dataset: DynamicGraphWindowDataset,
    input_end_date: pd.Timestamp | str,
) -> GraphForecast:
    """Forecast one declared signal date from a model trained before it."""
    date = pd.Timestamp(input_end_date)
    try:
        index = dataset.input_end_dates.index(date)
    except ValueError as error:
        raise ValueError(f"No dynamic-graph model window ends on {date.date()}") from error
    if date > result.decision_date:
        raise ValueError("A forecast input cannot end after the model decision date")
    device = torch.device(result.device)
    with torch.no_grad():
        forecast = result.model(
            dataset.features[index : index + 1].to(device), dataset.adjacency[index : index + 1].to(device)
        )
    return GraphForecast(score=forecast.score.cpu(), scale=forecast.scale.cpu())


def completed_training_mask(dataset: DynamicGraphWindowDataset, decision_date: pd.Timestamp | str) -> np.ndarray:
    """Expose the strict label-completion filter for audit tables and tests."""
    decision = pd.Timestamp(decision_date)
    return np.array([pd.Timestamp(date) < decision for date in dataset.target_end_dates], dtype=bool)
