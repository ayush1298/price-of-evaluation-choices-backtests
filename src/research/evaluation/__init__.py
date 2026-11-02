"""Performance metrics and robust evaluation helpers."""

from .metrics import performance_metrics
from .factor_models import FactorRegressionResult, fit_factor_model
from .canonical_runs import canonical_paper_table, collect_canonical_runs
from .ensembles import equally_average_target_weights
from .inference import (
    BlockBootstrapComparison,
    DeflatedSharpeResult,
    WhiteRealityCheckResult,
    deflated_sharpe_ratio,
    paired_moving_block_bootstrap,
    white_reality_check,
)
from .candidate_ledger import (
    CandidateLedgerAudit,
    audit_validation_candidate_ledger,
    freeze_validation_selection,
    load_candidate_ledger,
)

__all__ = [
    "BlockBootstrapComparison",
    "CandidateLedgerAudit",
    "DeflatedSharpeResult",
    "FactorRegressionResult",
    "canonical_paper_table",
    "audit_validation_candidate_ledger",
    "collect_canonical_runs",
    "deflated_sharpe_ratio",
    "equally_average_target_weights",
    "fit_factor_model",
    "freeze_validation_selection",
    "load_candidate_ledger",
    "paired_moving_block_bootstrap",
    "performance_metrics",
    "WhiteRealityCheckResult",
    "white_reality_check",
]
