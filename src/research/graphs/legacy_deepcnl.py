"""Reproducible adapter for the preserved MTP1 DeepCNL implementation.

This module intentionally delegates model training to ``src.mtp1`` rather than
rewriting its CNN-LSTM architecture.  It isolates the model checkpoint in a
temporary directory, validates genuine OHLCV inputs, and records the seed and
training interval required to reproduce the graph.
"""

from __future__ import annotations

import contextlib
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import networkx as nx
import numpy as np
import pandas as pd

from src.research.data.schema import validate_price_panel
from src.research.graphs.centrality import align_graph_to_universe


@dataclass(frozen=True)
class LegacyDeepCNLConfig:
    """Configuration for an exact, seeded DeepCNL baseline run."""

    ticker_num: int = 470
    window: int = 32
    feature_num: int = 4
    rare_ratio: float = 0.002
    epochs: int = 200
    seed: int = 42
    implementation: str = "igo"

    def __post_init__(self) -> None:
        if self.ticker_num < 3:
            raise ValueError("DeepCNL requires at least three tickers")
        if self.window < 2 or self.feature_num < 1:
            raise ValueError("DeepCNL window and feature_num must be positive")
        if not 0 < self.rare_ratio <= 1:
            raise ValueError("rare_ratio must be in (0, 1]")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")


@dataclass(frozen=True)
class LegacyDeepCNLRun:
    """A graph plus enough metadata to reproduce its construction."""

    graph: nx.Graph
    metadata: dict[str, object]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextlib.contextmanager
def _temporary_legacy_settings(config: LegacyDeepCNLConfig) -> Iterator[object]:
    """Apply temporary MTP1 settings and isolate its model checkpoint."""
    from src.mtp1 import stock_network_analysis as legacy

    original_epochs = legacy.EPOCH_NUM
    original_window = legacy.WINDOW
    original_features = legacy.FEATURE_NUM
    cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="deepcnl-") as run_directory:
        try:
            legacy.EPOCH_NUM = config.epochs
            legacy.WINDOW = config.window
            legacy.FEATURE_NUM = config.feature_num
            os.chdir(run_directory)
            yield legacy
        finally:
            os.chdir(cwd)
            legacy.EPOCH_NUM = original_epochs
            legacy.WINDOW = original_window
            legacy.FEATURE_NUM = original_features


def run_legacy_deepcnl(
    price_path: str | Path,
    target_path: str | Path,
    *,
    training_start: str,
    training_end: str,
    config: LegacyDeepCNLConfig | None = None,
) -> LegacyDeepCNLRun:
    """Train the preserved DeepCNL model and return one annual/rolling graph.

    ``price_path`` must contain real OHLCV observations; no fields are
    synthesised. The target path is the market-index series used by the
    original model. Portfolio evaluation must be performed in a later period.
    """
    config = config or LegacyDeepCNLConfig()
    price_path = Path(price_path).resolve()
    target_path = Path(target_path).resolve()
    prices = pd.read_csv(price_path)
    validate_price_panel(prices, require_ohlcv=True)
    _seed_everything(config.seed)

    with _temporary_legacy_settings(config) as legacy:
        datatool = legacy.Data_util(
            ticker_num=config.ticker_num,
            window=config.window,
            feature_num=config.feature_num,
            input_path=str(price_path),
            target_path=str(target_path),
        )
        experiment = legacy.Experimental_platform(datatool)
        train_period = [training_start, training_end]
        train_x = datatool.load_x(train_period)
        train_y = datatool.load_y(train_period)
        if len(train_y) == 0:
            raise ValueError("DeepCNL target series has no usable labels in the training interval")
        learned_graph = experiment.deep_CNL(config.implementation, train_x, train_y, config.rare_ratio)
        universe = [str(symbol) for symbol in datatool.dict_ticker.values()]
        graph = align_graph_to_universe(learned_graph, universe)

    metadata: dict[str, object] = {
        "model": "legacy_deepcnl_cnn_lstm",
        "training_start": training_start,
        "training_end": training_end,
        "config": asdict(config),
        "actual_ticker_num": datatool.actual_ticker_num,
        "eligible_universe": universe,
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "input_price_path": str(price_path),
        "target_path": str(target_path),
    }
    return LegacyDeepCNLRun(graph=graph, metadata=metadata)
