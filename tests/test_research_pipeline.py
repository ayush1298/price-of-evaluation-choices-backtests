from __future__ import annotations

import io
import json
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.research.backtest import BacktestConfig, rebalance_signal_dates, run_backtest
from src.research.backtest.walk_forward import WalkForwardConfig, run_walk_forward
from src.research.data import (
    eligible_universe,
    build_fixed_universe_membership,
    parse_fama_french_daily_factors,
    synchronise_fixed_universe_panel,
    validate_membership_history,
    validate_point_in_time_market_data,
    validate_price_panel,
    validate_data_contract,
    verify_saved_contract_report,
    normalise_yfinance_download,
    normalise_yfinance_download_audited,
    normalise_kaggle_current_sp500_stocks,
    drop_incomplete_ohlcv_dates,
    drop_symbols_with_materially_invalid_ohlcv,
    select_sector_balanced_current_survivor_cohort,
    membership_from_kaggle_nifty50_weights,
    business_days_on_or_before,
    apply_price_scale_actions,
    extract_price_scale_review_candidates,
    materialise_approved_price_scale_actions,
    build_panel_quality_report,
    map_membership_to_asset_ids,
    map_price_panel_to_asset_ids,
    validate_identity_crosswalk,
    audit_identity_evidence,
    nse_equity_bhavcopy_url,
    parse_nse_equity_bhavcopy,
    nse_corporate_actions_url,
    parse_nse_corporate_actions,
    reconcile_nse_corporate_action_records,
)
from src.research.evaluation import (
    canonical_paper_table,
    collect_canonical_runs,
    audit_validation_candidate_ledger,
    freeze_validation_selection,
    deflated_sharpe_ratio,
    paired_moving_block_bootstrap,
    white_reality_check,
    performance_metrics,
    fit_factor_model,
    equally_average_target_weights,
)
from src.research.experiments import (
    BaselineSignalBuilder,
    BaselineStrategyConfig,
    DynamicGraphExperimentConfig,
    ProbabilisticRegimeRouter,
    RegimeConditionalRiskConfig,
    SoftRegimeConditionalHRPBuilder,
    run_dynamic_graph_walk_forward,
)
from src.research.experiments.dynamic_graph import _graph_signal_coverage
from src.research.graphs import (
    GraphSnapshotConfig,
    adjacency_matrix,
    build_graph_snapshots,
    build_partial_correlation_graph,
    build_pcc_graph,
    compute_centrality_scores,
    edge_jaccard,
    edge_stability,
)
from src.research.regimes import (
    ExpandingGaussianHMMRegime,
    ExpandingKMeansRegime,
    ExpandingSoftKMeansRegime,
    broadcast_regime_probabilities_to_nodes,
    build_market_regime_features,
    build_price_regime_covariates,
    merge_external_regime_covariates,
)
from src.research.models import (
    DynamicGraphTrainingConfig,
    DynamicGraphTransformer,
    GraphForecast,
    NodeFeatureConfig,
    build_dynamic_graph_window_dataset,
    build_node_feature_snapshots,
    build_rebalance_forward_returns,
    completed_training_mask,
    predict_dynamic_graph_forecast,
    train_expanding_dynamic_graph_transformer,
)
from src.research.allocation import (
    hierarchical_risk_parity_weights,
    hierarchical_risk_parity_weights_from_covariance,
    mean_cvar_weights,
)
from src.research.reproducibility import (
    input_paths_with_existing_source_sidecars,
    sha256_file,
    research_eligibility_from_config,
    write_run_manifest,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03",
                    "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05",
                ]
            ),
            "symbol": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "close": [100.0, 100.0, 110.0, 100.0, 121.0, 90.0, 121.0, 90.0],
        }
    )


def test_price_validation_rejects_constant_synthetic_volume() -> None:
    panel = _prices().assign(open=lambda x: x.close, high=lambda x: x.close, low=lambda x: x.close, volume=1_000_000)
    with pytest.raises(ValueError, match="Volume is constant"):
        validate_price_panel(panel, require_ohlcv=True)


def test_data_contract_binds_fixed_universe_and_source_metadata(tmp_path: Path) -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "open": [10.0, 20.0, 11.0, 19.0],
            "high": [11.0, 21.0, 12.0, 20.0],
            "low": [9.0, 19.0, 10.0, 18.0],
            "close": [10.5, 20.5, 11.5, 19.5],
            "volume": [100.0, 120.0, 101.0, 122.0],
        }
    )
    membership = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["AAA", "BBB", "AAA", "BBB"],
            "eligible": [True, True, True, True],
        }
    )
    prices.to_csv(tmp_path / "prices.csv", index=False)
    membership.to_csv(tmp_path / "membership.csv", index=False)
    pd.DataFrame({"symbol": ["AAA", "BBB"]}).to_csv(tmp_path / "universe.csv", index=False)
    (tmp_path / "prices.csv.source.json").write_text(
        json.dumps({"kind": "public", "provider": "example", "auto_adjust": True}), encoding="utf-8"
    )
    (tmp_path / "membership.csv.source.json").write_text(
        json.dumps({"kind": "fixed", "rebalance_frequency": "B"}), encoding="utf-8"
    )
    contract = {
        "kind": "research-data-contract",
        "schema_version": 1,
        "contract_id": "unit-test-fixed-universe",
        "study_scope": "test",
        "paper_eligible": False,
        "market": {"code": "TEST", "currency": "USD"},
        "inputs": {"prices": "prices.csv", "membership": "membership.csv", "universe": "universe.csv"},
        "declared_universe": {"expected_assets": 2, "membership_must_match_declared": True},
        "validation": {
            "require_ohlcv": True,
            "required_price_columns": ["open", "high", "low", "close", "volume"],
            "expected_price_assets": 2,
            "minimum_start": "2024-01-01",
            "minimum_end": "2024-01-03",
            "rebalance_frequency": "B",
            "minimum_eligible_assets": 2,
            "maximum_membership_staleness_days": 0,
        },
        "source_sidecars": {
            "prices": {"required_values": {"provider": "example", "auto_adjust": True}},
            "membership": {"required_values": {"kind": "fixed"}},
        },
    }
    report = validate_data_contract(contract, project_root=tmp_path)
    assert report["validation_status"] == "passed"
    assert report["price_panel"]["n_assets"] == 2
    assert report["market_data_validation"]["minimum_eligible_assets_observed"] == 2

    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    report_path = tmp_path / "contract_validation.json"
    report_path.write_text(
        json.dumps({**report, "contract_file": {"sha256": sha256_file(contract_path)}}), encoding="utf-8"
    )
    # A saved report describing exactly these inputs is the only accepted provenance.
    verify_saved_contract_report(contract_path, report_path, report)

    stale = json.loads(report_path.read_text(encoding="utf-8"))
    stale["inputs"] = {**stale["inputs"], "prices": {"sha256": "0" * 64}}
    report_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="input hashes do not match"):
        verify_saved_contract_report(contract_path, report_path, report)

    unpassed = json.loads(report_path.read_text(encoding="utf-8"))
    unpassed["validation_status"] = "failed"
    report_path.write_text(json.dumps(unpassed), encoding="utf-8")
    with pytest.raises(ValueError, match="passed validation"):
        verify_saved_contract_report(contract_path, report_path, report)

    contract["source_sidecars"]["prices"]["required_values"]["auto_adjust"] = False
    with pytest.raises(ValueError, match="auto_adjust"):
        validate_data_contract(contract, project_root=tmp_path)


def test_candidate_ledger_open_end_date_must_match_an_open_run(tmp_path: Path) -> None:
    def _write_run(run_id: str, last_signal_date: str | None) -> None:
        directory = tmp_path / run_id
        directory.mkdir(parents=True)
        (directory / "performance_metrics.json").write_text(
            json.dumps({"sharpe": 1.0, "n_observations": 100}), encoding="utf-8"
        )
        (directory / "run_manifest.json").write_text(
            json.dumps(
                {
                    "kind": "canonical-run-manifest",
                    "run_id": run_id,
                    "extra": {
                        "study_scope": "test-scope",
                        "data_contract_id": "test-contract",
                        "data_contract": {"contract_id": "test-contract"},
                        "walk_forward": {
                            "first_signal_date": "2014-01-31",
                            "last_signal_date": last_signal_date,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    _write_run("run-open", None)
    _write_run("run-bounded", "2020-12-31")

    def _ledger(run_id: str, last: str | None) -> dict:
        return {
            "kind": "validation-candidate-ledger",
            "schema_version": 1,
            "ledger_id": "test",
            "study_scope": "test-scope",
            "data_contract_id": "test-contract",
            "selection_permitted": False,
            "selection_metric": "sharpe",
            "validation_window": {"first_signal_date": "2014-01-31", "last_signal_date": last},
            "candidates": [
                {"candidate_id": "c", "run_id": run_id, "expected_manifest_values": {}}
            ],
        }

    # A declared open end is satisfied only by a run that also ran open-ended.
    open_audit = audit_validation_candidate_ledger(_ledger("run-open", None), results_root=tmp_path)
    assert open_audit.incomplete_candidate_count == 0

    # A run that silently bounded itself is not the run that was declared.
    bounded_audit = audit_validation_candidate_ledger(_ledger("run-bounded", None), results_root=tmp_path)
    assert bounded_audit.incomplete_candidate_count == 1

    # And a declared bound is not satisfied by an open-ended run either.
    reverse_audit = audit_validation_candidate_ledger(
        _ledger("run-open", "2020-12-31"), results_root=tmp_path
    )
    assert reverse_audit.incomplete_candidate_count == 1


def _protocol_inflation_module():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_protocol_inflation_table.py"
    spec = importlib.util.spec_from_file_location("build_protocol_inflation_table", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_canonical_run(root: Path, run_id: str, sharpe: float, observations: int) -> None:
    directory = root / run_id
    directory.mkdir(parents=True)
    (directory / "performance_metrics.json").write_text(
        json.dumps({"sharpe": sharpe, "n_observations": observations}), encoding="utf-8"
    )
    (directory / "run_manifest.json").write_text(
        json.dumps({"kind": "canonical-run-manifest", "run_id": run_id}), encoding="utf-8"
    )


def test_protocol_inflation_reading_rules_require_a_matched_evaluation_period(tmp_path: Path) -> None:
    module = _protocol_inflation_module()
    _write_canonical_run(tmp_path, "run-a", 1.20, 1489)
    _write_canonical_run(tmp_path, "run-b", 0.80, 1489)

    side = module._load_side({"rule": "max", "runs": ["run-a", "run-b"]}, results_root=tmp_path, metric="sharpe")
    assert side["value"] == pytest.approx(1.20)
    assert side["n_observations"] == 1489
    median = module._load_side(
        {"rule": "median", "runs": ["run-a", "run-b"]}, results_root=tmp_path, metric="sharpe"
    )
    assert median["value"] == pytest.approx(1.00)

    # Reducing runs that cover different periods would compare unlike numbers.
    _write_canonical_run(tmp_path, "run-c", 0.90, 1627)
    with pytest.raises(ValueError, match="different evaluation lengths"):
        module._load_side({"rule": "max", "runs": ["run-a", "run-c"]}, results_root=tmp_path, metric="sharpe")

    # A named run that is absent must fail loudly rather than shrink the family.
    with pytest.raises(FileNotFoundError, match="silently drop"):
        module._load_side({"rule": "max", "runs": ["run-a", "absent"]}, results_root=tmp_path, metric="sharpe")


def test_benchmark_choice_row_detects_a_conclusion_that_flips_sign(tmp_path: Path) -> None:
    module = _protocol_inflation_module()
    _write_canonical_run(tmp_path, "candidate", 0.72, 3120)
    _write_canonical_run(tmp_path, "easy-benchmark", 0.68, 3137)
    _write_canonical_run(tmp_path, "strong-benchmark", 0.75, 3137)

    row = module._build_benchmark_choice_row(
        {
            "row_id": "r",
            "shortcut": "s",
            "panel": "p",
            "strategy": "t",
            "candidate_run": "candidate",
            "convenient_benchmark_run": "easy-benchmark",
            "baseline_pool": ["easy-benchmark", "strong-benchmark"],
        },
        results_root=tmp_path,
        metric="sharpe",
    )
    assert row["strongest_baseline"]["run_id"] == "strong-benchmark"
    assert row["reported_edge"] == pytest.approx(0.04)
    assert row["honest_edge"] == pytest.approx(-0.03)
    assert row["inflation"] == pytest.approx(0.07)
    assert row["conclusion_sign_flips"] is True

    # The convenient benchmark must be one the study actually declared, so a
    # weaker comparator cannot be introduced from outside the baseline ladder.
    with pytest.raises(ValueError, match="must itself be one of the declared baselines"):
        module._build_benchmark_choice_row(
            {
                "row_id": "r",
                "shortcut": "s",
                "panel": "p",
                "strategy": "t",
                "candidate_run": "candidate",
                "convenient_benchmark_run": "candidate",
                "baseline_pool": ["easy-benchmark", "strong-benchmark"],
            },
            results_root=tmp_path,
            metric="sharpe",
        )


def test_saved_contract_report_must_belong_to_the_supplied_contract_file(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps({"contract_id": "a"}), encoding="utf-8")
    other_path = tmp_path / "other_contract.json"
    other_path.write_text(json.dumps({"contract_id": "b"}), encoding="utf-8")
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "validation_status": "passed",
                "contract_file": {"sha256": sha256_file(other_path)},
                "inputs": {},
            }
        ),
        encoding="utf-8",
    )
    # A report generated for a different contract must never authorise this run.
    with pytest.raises(ValueError, match="does not match the supplied contract file"):
        verify_saved_contract_report(contract_path, report_path, {"inputs": {}})


def test_price_validation_allows_only_machine_precision_ohlc_rounding() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "symbol": ["A"],
            "open": [100.0],
            "high": [100.0 - 1e-13],
            "low": [99.0],
            "close": [100.0],
            "volume": [123.0],
        }
    )
    validate_price_panel(panel, require_ohlcv=True, reject_constant_volume=False)
    panel.loc[0, "high"] = 99.0
    with pytest.raises(ValueError, match="High must be"):
        validate_price_panel(panel, require_ohlcv=True, reject_constant_volume=False)


def test_nse_bhavcopy_parser_keeps_only_requested_equity_series() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            "cm31JAN2008bhav.csv",
            "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,TOTTRDQTY,TIMESTAMP,\n"
            "VSNL,EQ,500,580,481,519.85,1502292,31-JAN-2008,\n"
            "VSNL,BE,500,580,481,519.85,7,31-JAN-2008,\n"
            "OTHER,EQ,10,11,9,10.5,100,31-JAN-2008,\n",
        )
    panel = parse_nse_equity_bhavcopy(
        archive_buffer.getvalue(), observation_date="2008-01-31", symbols=["vsnl", "missing"]
    )
    assert panel.to_dict(orient="records") == [
        {
            "date": pd.Timestamp("2008-01-31"),
            "symbol": "VSNL",
            "open": 500,
            "high": 580,
            "low": 481,
            "close": 519.85,
            "volume": 1502292,
        }
    ]
    assert nse_equity_bhavcopy_url("2008-01-31") == (
        "https://archives.nseindia.com/content/historical/EQUITIES/2008/JAN/cm31JAN2008bhav.csv.zip"
    )
    with pytest.raises(ValueError, match="does not contain an ISIN"):
        parse_nse_equity_bhavcopy(
            archive_buffer.getvalue(), observation_date="2008-01-31", symbols=["VSNL"], include_isin=True
        )


def test_nse_bhavcopy_parser_supports_post_udiff_schema() -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr(
            "BhavCopy_NSE_CM_0_0_0_20240731_F_0000.csv",
            "TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,ISIN\n"
            "2024-07-31,HDFCBANK,EQ,1600,1625,1590,1610,3000000,INE040A01034\n"
            "2024-07-31,OTHER,BE,10,11,9,10.5,100,INE000A01001\n",
        )
    panel = parse_nse_equity_bhavcopy(
        archive_buffer.getvalue(), observation_date="2024-07-31", symbols=["HDFCBANK"]
    )
    assert panel.loc[0, "symbol"] == "HDFCBANK"
    assert panel.loc[0, "close"] == pytest.approx(1610.0)
    assert nse_equity_bhavcopy_url("2024-07-31") == (
        "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_20240731_F_0000.csv.zip"
    )
    assert parse_nse_equity_bhavcopy(
        archive_buffer.getvalue(), observation_date="2024-07-31", symbols=["HDFCBANK"], include_isin=True
    ).loc[0, "isin"] == "INE040A01034"


def test_nse_month_end_candidates_never_advance_a_weekend_signal() -> None:
    candidates = business_days_on_or_before("2024-06-30", maximum_prior_business_days=2)
    assert candidates == (
        pd.Timestamp("2024-06-28"),
        pd.Timestamp("2024-06-27"),
        pd.Timestamp("2024-06-26"),
    )


def test_nse_corporate_action_parser_keeps_raw_action_meaning_without_inferred_factor() -> None:
    raw = """[
        {
            "symbol": "ALLCARGO", "series": "EQ", "isin": "INE418H01029",
            "exDate": "02-Jan-2024", "recDate": "02-Jan-2024", "subject": "Bonus 3:1",
            "comp": "Allcargo Logistics Limited", "faceVal": "2", "bcStartDate": "-",
            "bcEndDate": "-", "ndStartDate": "-", "ndEndDate": "-", "caBroadcastDate": null
        },
        {
            "symbol": "83GS2040", "series": "GS", "isin": "IN0020100031",
            "exDate": "01-Jan-2024", "recDate": "01-Jan-2024", "subject": "Interest Payment",
            "comp": "GOVERNMENT OF INDIA"
        }
    ]"""
    events = parse_nse_corporate_actions(
        raw,
        source_url="https://www.nseindia.com/api/corporates-corporateActions?example=1",
        symbols=["allcargo"],
    )
    assert events["symbol"].tolist() == ["ALLCARGO"]
    assert events.loc[0, "subject"] == "Bonus 3:1"
    assert events.loc[0, "ex_date"] == pd.Timestamp("2024-01-02")
    assert pd.isna(events.loc[0, "book_closure_start"])
    assert events.loc[0, "source_record_id"].startswith("NSE-CA-ALLCARGO-20240102-")
    assert "backward_price_factor" not in events.columns
    assert nse_corporate_actions_url("2024-01-01", "2024-01-31") == (
        "https://www.nseindia.com/api/corporates-corporateActions?index=equities&from_date=01-01-2024&to_date=31-01-2024"
    )


def test_nse_corporate_action_parser_rejects_malformed_required_dates() -> None:
    with pytest.raises(ValueError, match="invalid exDate"):
        parse_nse_corporate_actions(
            '[{"symbol":"A", "series":"EQ", "isin":"INE000A01001", "exDate":"not-a-date", '
            '"subject":"Bonus 1:1", "comp":"Example"}]',
            source_url="https://example.test",
        )


def test_nse_corporate_action_parser_accepts_nse_empty_data_envelope() -> None:
    events = parse_nse_corporate_actions(
        '{"data": [], "msg": "no data found"}',
        source_url="https://example.test",
    )
    assert events.empty
    assert list(events.columns) == [
        "symbol",
        "series",
        "isin",
        "company",
        "ex_date",
        "record_date",
        "subject",
        "face_value",
        "book_closure_start",
        "book_closure_end",
        "no_delivery_start",
        "no_delivery_end",
        "corporate_action_broadcast_date",
        "evidence_url",
        "source_record_id",
    ]


def test_nse_corporate_action_source_id_includes_all_event_dates() -> None:
    base = {
        "symbol": "A", "series": "EQ", "isin": "INE000A01001", "exDate": "02-Jan-2024",
        "recDate": "02-Jan-2024", "subject": "Bonus 1:1", "comp": "Example", "faceVal": "10",
        "bcStartDate": "-", "bcEndDate": "-", "ndStartDate": "-", "ndEndDate": "-", "caBroadcastDate": None,
    }
    altered = {**base, "bcStartDate": "03-Jan-2024"}
    events = parse_nse_corporate_actions(
        json.dumps([base, altered]),
        source_url="https://example.test",
    )
    assert events["source_record_id"].nunique() == 2


def test_nse_corporate_action_reconciliation_collapses_only_identical_replays() -> None:
    raw = '[{"symbol":"A", "series":"EQ", "isin":"INE000A01001", "exDate":"02-Jan-2024", '
    raw += '"subject":"Bonus 1:1", "comp":"Example"}]'
    original = parse_nse_corporate_actions(raw, source_url="https://example.test/first")
    replay = original.assign(evidence_url="https://example.test/replay")
    reconciled, audit = reconcile_nse_corporate_action_records(pd.concat([original, replay], ignore_index=True))
    assert len(reconciled) == 1
    assert audit.input_records == 2
    assert audit.duplicate_records_collapsed == 1

    conflicting = replay.assign(subject="Different event")
    with pytest.raises(ValueError, match="conflicting records"):
        reconcile_nse_corporate_action_records(pd.concat([original, conflicting], ignore_index=True))


def test_price_scale_actions_adjust_only_bars_before_a_documented_split() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["A", "A", "A"],
            "open": [100.0, 50.0, 60.0],
            "high": [101.0, 51.0, 61.0],
            "low": [99.0, 49.0, 59.0],
            "close": [100.0, 50.0, 60.0],
            "volume": [100.0, 200.0, 210.0],
        }
    )
    actions = pd.DataFrame(
        {
            "symbol": ["A"],
            "ex_date": ["2024-01-03"],
            "action_type": ["split_bonus_price_factor"],
            "backward_price_factor": [0.5],
            "evidence_url": ["https://example.test/nse-action"],
            "source_record_id": ["NSE-CA-1"],
        }
    )
    adjusted, audit = apply_price_scale_actions(prices, actions)
    assert adjusted["close"].tolist() == [50.0, 50.0, 60.0]
    assert adjusted["volume"].tolist() == [200.0, 200.0, 210.0]
    assert audit.affected_price_rows == 1
    assert audit.adjustment_scope == "price_only_split_and_bonus"


def test_price_scale_actions_reject_unmodeled_events() -> None:
    actions = pd.DataFrame(
        {
            "symbol": ["A"],
            "ex_date": ["2024-01-03"],
            "action_type": ["cash_dividend"],
            "backward_price_factor": [0.99],
            "evidence_url": ["https://example.test/nse-action"],
            "source_record_id": ["NSE-CA-1"],
        }
    )
    with pytest.raises(ValueError, match="unsupported action types"):
        apply_price_scale_actions(
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "symbol": ["A", "A"],
                    "open": [100.0, 99.0],
                    "high": [100.0, 99.0],
                    "low": [100.0, 99.0],
                    "close": [100.0, 99.0],
                    "volume": [100.0, 101.0],
                }
            ),
            actions,
        )


def test_price_scale_action_review_candidates_require_human_approval() -> None:
    ledger = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "ex_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "subject": [
                "Bonus 1:1",
                "Face Value Split From Rs.10/- To Rs.2/-",
                "Bonus 2:3 And Face Value Split From Rs.10 To Re.1",
                "Sch Of Agmt- Bonus Deb1:1",
            ],
            "evidence_url": ["https://example.test/action"] * 4,
            "source_record_id": ["A1", "B1", "C1", "D1"],
        }
    )
    review, audit = extract_price_scale_review_candidates(ledger)
    assert review.set_index("symbol").loc["A", "proposed_backward_price_factor"] == pytest.approx(0.5)
    assert review.set_index("symbol").loc["B", "proposed_backward_price_factor"] == pytest.approx(0.2)
    assert review.set_index("symbol").loc["C", "proposed_backward_price_factor"] == pytest.approx(0.06)
    assert review.set_index("symbol").loc["D", "review_status"] == "blocked_complex_action"
    assert pd.isna(review.set_index("symbol").loc["D", "proposed_backward_price_factor"])
    assert audit.simple_candidates == 3
    assert audit.blocked_complex_events == 1
    assert "action_type" not in review.columns


def test_corporate_action_decisions_require_full_manual_coverage_before_materialising_actions() -> None:
    review, _ = extract_price_scale_review_candidates(
        pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "ex_date": ["2024-01-02", "2024-01-03"],
                "subject": ["Bonus 1:1", "Sch Of Agmt- Bonus Deb1:1"],
                "evidence_url": ["https://example.test/action-a", "https://example.test/action-b"],
                "source_record_id": ["A1", "B1"],
            }
        )
    )
    decisions = pd.DataFrame(
        {
            "source_record_id": ["A1", "B1"],
            "decision": ["approve_price_scale_factor", "exclude_from_raw_panel"],
            "approved_backward_price_factor": [0.5, None],
            "evidence_url": ["https://example.test/approved-a", "https://example.test/excluded-b"],
            "reviewed_by": ["reviewer@example.test", "reviewer@example.test"],
            "reviewed_at": ["2026-07-29T10:00:00Z", "2026-07-29T10:01:00Z"],
            "decision_note": [
                "Primary exchange notice confirms the 1:1 equity bonus factor.",
                "Complex scheme event is excluded from the raw-price panel.",
            ],
        }
    )
    actions, audit = materialise_approved_price_scale_actions(review, decisions)
    assert actions.to_dict(orient="records") == [
        {
            "symbol": "A",
            "ex_date": pd.Timestamp("2024-01-02"),
            "action_type": "split_bonus_price_factor",
            "backward_price_factor": 0.5,
            "evidence_url": "https://example.test/approved-a",
            "source_record_id": "A1",
        }
    ]
    assert audit.all_candidates_resolved
    assert audit.approved_price_scale_factors == 1
    assert audit.excluded_from_raw_panel == 1

    with pytest.raises(ValueError, match=r"lack final decisions: \['B1'\]"):
        materialise_approved_price_scale_actions(review, decisions.iloc[[0]])


def test_corporate_action_decisions_reject_automatic_approval_of_complex_event() -> None:
    review, _ = extract_price_scale_review_candidates(
        pd.DataFrame(
            {
                "symbol": ["A"],
                "ex_date": ["2024-01-02"],
                "subject": ["Sch Of Agmt- Bonus Deb1:1"],
                "evidence_url": ["https://example.test/action"],
                "source_record_id": ["A1"],
            }
        )
    )
    decisions = pd.DataFrame(
        {
            "source_record_id": ["A1"],
            "decision": ["approve_price_scale_factor"],
            "approved_backward_price_factor": [0.5],
            "evidence_url": ["https://example.test/incorrect"],
            "reviewed_by": ["reviewer@example.test"],
            "reviewed_at": ["2026-07-29T10:00:00Z"],
            "decision_note": ["Should never be accepted."],
        }
    )
    with pytest.raises(ValueError, match="Only conventional human-review candidates"):
        materialise_approved_price_scale_actions(review, decisions)


def test_panel_quality_report_exposes_coverage_membership_outliers_and_source_gaps() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["A", "B", "A", "A", "B"],
            "close": [100.0, 10.0, 120.0, 121.0, 11.0],
        }
    )
    membership = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-03", "2024-01-03"],
            "symbol": ["A", "B", "A", "B"],
            "eligible": [True, True, True, False],
        }
    )
    reference = prices.copy()
    reference.loc[(reference["date"] == "2024-01-02") & (reference["symbol"] == "A"), "close"] = 119.0
    report = build_panel_quality_report(
        prices,
        membership,
        rebalance_frequency="D",
        return_outlier_threshold=0.15,
        reference_prices=reference,
        reference_return_tolerance=0.005,
    )
    coverage = report.price_coverage
    b_row = next(row for row in coverage["per_asset"] if row["symbol"] == "B")
    assert b_row["missing_dates_between_first_last"] == 1
    assert coverage["rebalance_signal_coverage"]["minimum_eligible_assets_with_price"] == 1
    assert report.membership["total_removals_after_initial_snapshot"] == 1
    assert report.return_outliers["total_flagged_return_pairs"] == 1
    assert not report.identity["supplied"]
    assert report.reference_reconciliation["return_pairs_above_tolerance"] >= 1


def test_validation_candidate_ledger_selects_only_complete_predeclared_validation_runs(tmp_path: Path) -> None:
    def write_candidate(run_id: str, *, components: int, lookback: int, sharpe: float) -> None:
        destination = tmp_path / run_id
        destination.mkdir()
        manifest = {
            "run_id": run_id,
            "config": {"path": "configs/test.json", "sha256": "test-config-sha"},
            "extra": {
                "study_scope": "development",
                "data_contract": {"contract_id": "contract-v1"},
                "walk_forward": {"first_signal_date": "2020-01-31", "last_signal_date": "2020-12-31"},
                "hmm": {"n_components": components},
                "conditional_risk": {"lookback_sessions": lookback},
                "backtest": {"min_rebalance_turnover": 0.0},
            },
        }
        (destination / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (destination / "performance_metrics.json").write_text(json.dumps({"sharpe": sharpe}), encoding="utf-8")

    write_candidate("candidate-a", components=2, lookback=63, sharpe=0.4)
    write_candidate("candidate-b", components=3, lookback=126, sharpe=0.7)
    ledger = {
        "kind": "validation-candidate-ledger",
        "schema_version": 1,
        "ledger_id": "ledger-v1",
        "study_scope": "development",
        "data_contract_id": "contract-v1",
        "selection_permitted": True,
        "selection_metric": "sharpe",
        "validation_window": {"first_signal_date": "2020-01-31", "last_signal_date": "2020-12-31"},
        "candidates": [
            {
                "candidate_id": "a",
                "run_id": "candidate-a",
                "expected_manifest_values": {
                    "extra.hmm.n_components": 2,
                    "extra.conditional_risk.lookback_sessions": 63,
                    "extra.backtest.min_rebalance_turnover": 0.0,
                },
            },
            {
                "candidate_id": "b",
                "run_id": "candidate-b",
                "expected_manifest_values": {
                    "extra.hmm.n_components": 3,
                    "extra.conditional_risk.lookback_sessions": 126,
                    "extra.backtest.min_rebalance_turnover": 0.0,
                },
            },
        ],
    }
    audit = audit_validation_candidate_ledger(ledger, results_root=tmp_path)
    assert audit.selection_status == "selection_ready_to_freeze"
    assert audit.selected_candidate_id == "b"
    assert audit.selected_metric_value == pytest.approx(0.7)
    frozen = freeze_validation_selection(ledger, results_root=tmp_path)
    assert frozen["selected_candidate_id"] == "b"
    assert frozen["selected_validation_config"]["sha256"] == "test-config-sha"

    incomplete = {**ledger, "candidates": [*ledger["candidates"], {**ledger["candidates"][0], "candidate_id": "missing", "run_id": "missing-run"}]}
    blocked = audit_validation_candidate_ledger(incomplete, results_root=tmp_path)
    assert blocked.selection_status == "selection_blocked_incomplete_candidates"
    assert blocked.incomplete_candidate_count == 1


def test_white_reality_check_adjusts_for_the_complete_candidate_family() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    benchmark = pd.Series(0.0, index=dates)
    candidates = pd.DataFrame(
        {
            "positive": np.full(len(dates), 0.001),
            "matched": np.zeros(len(dates)),
        },
        index=dates,
    )
    result = white_reality_check(
        candidates,
        benchmark,
        block_length=5,
        replications=500,
        random_seed=7,
    )
    assert result.candidates == 2
    assert result.best_candidate == "positive"
    assert result.best_annualized_excess_return == pytest.approx(0.252)
    assert result.reality_check_p_value < 0.05


def _verified_same_security_crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_symbol": ["A", "A"],
            "asset_id": ["INE000A01001", "INE000A01001"],
            "provider_symbol": ["OLD_A", "A"],
            "start_date": ["2024-01-01", "2024-01-03"],
            "end_date": ["2024-01-02", "2024-12-31"],
            "isin": ["INE000A01001", "INE000A01001"],
            "relationship": ["same_security_rename", "same_security_rename"],
            "continuity": ["continuous", "continuous"],
            "verification_status": ["verified", "verified"],
            "evidence_url": ["https://example.test/evidence", "https://example.test/evidence"],
            "source_record_id": ["NSE-1", "NSE-2"],
        }
    )


def test_identity_crosswalk_preserves_continuity_across_verified_rename() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "symbol": ["OLD_A", "A", "A"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [100.0, 110.0, 120.0],
        }
    )
    membership = pd.DataFrame(
        {"date": ["2024-01-02", "2024-01-03"], "symbol": ["A", "A"], "eligible": [True, True]}
    )
    mapped_prices, price_audit = map_price_panel_to_asset_ids(prices, _verified_same_security_crosswalk())
    mapped_membership, membership_audit = map_membership_to_asset_ids(
        membership, _verified_same_security_crosswalk()
    )
    assert mapped_prices["symbol"].tolist() == ["INE000A01001"] * 3
    assert mapped_prices["provider_symbol"].tolist() == ["OLD_A", "A", "A"]
    assert mapped_membership["symbol"].tolist() == ["INE000A01001", "INE000A01001"]
    assert mapped_membership["source_symbol"].tolist() == ["A", "A"]
    assert price_audit.mapped_rows == 3
    assert membership_audit.mapped_rows == 2


def test_identity_crosswalk_rejects_unverified_or_false_continuity() -> None:
    candidate = _verified_same_security_crosswalk()
    candidate.loc[0, "verification_status"] = "candidate"
    with pytest.raises(ValueError, match="non-verified identity mappings"):
        map_price_panel_to_asset_ids(
            pd.DataFrame(
                {
                    "date": ["2024-01-02", "2024-01-03"],
                    "symbol": ["OLD_A", "A"],
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.0, 101.0],
                    "volume": [100.0, 110.0],
                }
            ),
            candidate,
        )
    broken = _verified_same_security_crosswalk().iloc[[0]].copy()
    broken.loc[:, "relationship"] = "reorganisation_successor"
    with pytest.raises(ValueError, match="continuity='break'"):
        validate_identity_crosswalk(broken)


def test_identity_evidence_prevents_unsupported_crosswalk_promotion() -> None:
    crosswalk = _verified_same_security_crosswalk()
    evidence = pd.DataFrame(
        {
            "evidence_id": ["E-1", "E-2"],
            "source_record_id": ["NSE-1", "NSE-2"],
            "source_tier": ["official_bhavcopy", "official_exchange"],
            "source_url": ["https://example.test/old", "https://example.test/new"],
            "provider_symbol_supported": [True, True],
            "effective_interval_supported": [True, True],
            "isin_supported": [True, False],
            "continuity_supported": [True, True],
            "review_status": ["verified", "candidate"],
            "review_note": ["Complete historic-security evidence.", "ISIN evidence remains missing."],
        }
    )
    with pytest.raises(ValueError, match="NSE-2"):
        audit_identity_evidence(crosswalk, evidence)

    crosswalk.loc[1, "verification_status"] = "candidate"
    audit = audit_identity_evidence(crosswalk, evidence)
    assert audit.verified_crosswalk_rows == 1
    assert audit.fully_supported_source_records == 1
    assert audit.source_records_missing_fact == 1


def test_run_manifest_inputs_include_existing_source_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.research.reproducibility as reproducibility

    monkeypatch.setattr(reproducibility, "PROJECT_ROOT", tmp_path)
    price_path = tmp_path / "prices.csv"
    price_path.write_text("date,symbol,close\n2024-01-02,A,1\n", encoding="utf-8")
    sidecar_path = Path(f"{price_path}.source.json")
    sidecar_path.write_text("{}\n", encoding="utf-8")
    assert input_paths_with_existing_source_sidecars([price_path]) == [price_path, sidecar_path]


def test_paper_release_eligibility_requires_explicit_completed_evidence() -> None:
    exploratory = research_eligibility_from_config({"study_scope": "exploratory"})
    assert not exploratory["paper_eligible"]
    assert exploratory["release_status"] == "nonpaper"
    assert set(exploratory["missing_paper_evidence"])
    paper_ready = research_eligibility_from_config(
        {
            "study_scope": "paper_confirmation",
            "paper_eligible": True,
            "paper_evidence": {
                "universe_contract_verified": True,
                "price_adjustment_verified": True,
                "evaluation_precommitted": True,
                "independent_confirmation": True,
                "robustness_matrix_completed": True,
            },
        }
    )
    assert paper_ready["paper_eligible"]
    assert paper_ready["release_status"] == "paper_ready"


def test_manifest_centrally_enforces_nonpaper_release_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    import src.research.reproducibility as reproducibility

    monkeypatch.setattr(reproducibility, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text('{"study_scope":"public_diagnostic","paper_eligible":false}\n', encoding="utf-8")
    input_path = tmp_path / "prices.csv"
    input_path.write_text("date,symbol,close\n2024-01-02,A,1\n", encoding="utf-8")
    manifest_path = write_run_manifest(
        "scope-test",
        config_path=config_path,
        input_paths=[input_path],
        extra={"paper_eligible": True, "study_scope": "incorrect_runner_override"},
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert not manifest["extra"]["paper_eligible"]
    assert manifest["extra"]["study_scope"] == "public_diagnostic"
    assert manifest["extra"]["paper_eligibility"]["release_status"] == "nonpaper"


def test_fixed_universe_membership_has_current_rebalance_snapshots_and_rejects_gaps() -> None:
    dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29"])
    prices = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "symbol": ["ETF_A"] * len(dates) + ["ETF_B"] * len(dates),
            "close": [100.0, 101.0, 102.0, 103.0, 50.0, 51.0, 52.0, 53.0],
        }
    )
    membership = build_fixed_universe_membership(prices, ["ETF_A", "ETF_B"])
    assert membership["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-31",
        "2024-01-31",
        "2024-02-29",
        "2024-02-29",
    ]
    assert membership["eligible"].all()
    with pytest.raises(ValueError, match="missing a rebalance-date close"):
        build_fixed_universe_membership(
            prices.loc[
                ~((prices["date"] == pd.Timestamp("2024-02-29")) & (prices["symbol"] == "ETF_B"))
            ],
            ["ETF_A", "ETF_B"],
        )


def test_synchronised_fixed_universe_panel_removes_incomplete_dates_without_filling() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-04"],
            "symbol": ["A", "B", "A", "A", "B"],
            "open": [100.0, 50.0, 101.0, 102.0, 51.0],
            "high": [100.0, 50.0, 101.0, 102.0, 51.0],
            "low": [100.0, 50.0, 101.0, 102.0, 51.0],
            "close": [100.0, 50.0, 101.0, 102.0, 51.0],
            "volume": [100.0, 200.0, 101.0, 102.0, 202.0],
        }
    )
    synchronized = synchronise_fixed_universe_panel(prices, ["A", "B"])
    assert synchronized["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-01-02", "2024-01-04"]
    assert len(synchronized) == 4


def test_centrality_adds_isolated_universe_assets_and_uses_similarity() -> None:
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=10.0)
    scores = compute_centrality_scores(graph, universe=["A", "B", "C"])
    assert set(scores["ticker"]) == {"A", "B", "C"}
    assert scores.loc[scores["ticker"] == "C", "degree_centrality"].item() == 0.0


def test_graph_stability_reports_edge_overlap_across_seeds() -> None:
    first = nx.Graph([("A", "B"), ("B", "C")])
    second = nx.Graph([("A", "B"), ("A", "C")])
    assert edge_jaccard(first, second) == pytest.approx(1.0 / 3.0)
    stability = edge_stability([first, second], seeds=[7, 17])
    assert stability.loc[0, "seed_left"] == 7
    assert stability.loc[0, "edge_jaccard"] == pytest.approx(1.0 / 3.0)


def test_correlation_baselines_preserve_isolated_assets() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
            "B": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
            "C": [0.02, -0.01, 0.01, -0.02, 0.02, -0.01],
        }
    )
    pcc = build_pcc_graph(returns, edge_fraction=0.33, min_observations=4)
    partial = build_partial_correlation_graph(returns, edge_fraction=0.33)
    assert set(pcc.nodes) == {"A", "B", "C"}
    assert set(partial.nodes) == {"A", "B", "C"}
    assert pcc.number_of_edges() == 1


def test_graph_snapshots_are_point_in_time_and_keep_eligible_isolates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "A", "close": [100, 102, 104, 103, 106]}),
            pd.DataFrame({"date": dates, "symbol": "B", "close": [100, 101, 99, 101, 102]}),
            pd.DataFrame({"date": [dates[0]], "symbol": ["C"], "close": [100]}),
        ],
        ignore_index=True,
    )
    universe = pd.DataFrame(
        {"date": [dates[0]] * 3, "symbol": ["A", "B", "C"], "eligible": [True, True, True]}
    )
    result = build_graph_snapshots(
        prices,
        universe,
        [dates[-1]],
        config=GraphSnapshotConfig(lookback_sessions=4, edge_fraction=1.0, minimum_fit_assets=2),
    )
    graph = result.graphs[dates[-1]]
    assert set(graph.nodes) == {"A", "B", "C"}
    assert graph.degree("C") == 0
    assert adjacency_matrix(graph, ["A", "B", "C"]).shape == (3, 3)


def test_backtest_executes_after_signal_and_charges_costs() -> None:
    targets = pd.DataFrame(
        {"signal_date": [pd.Timestamp("2024-01-02")], "symbol": ["A"], "weight": [1.0]}
    )
    result = run_backtest(
        _prices(), targets, config=BacktestConfig(transaction_cost_bps=100.0, execution_lag_sessions=1)
    )
    # Signal on the 2nd executes at the 3rd close: it pays cost then, and the
    # new A position first earns the following session's close-to-close return.
    assert result.executions.loc[0, "execution_date"] == pd.Timestamp("2024-01-03")
    assert result.daily.index.min() == pd.Timestamp("2024-01-03")
    assert result.daily.loc[pd.Timestamp("2024-01-03"), "net_return"] == pytest.approx(-0.01)
    assert result.daily.loc[pd.Timestamp("2024-01-04"), "net_return"] == pytest.approx(0.10)
    assert result.daily["drawdown"].min() == pytest.approx(-0.01)


def test_backtest_no_trade_band_skips_small_realised_rebalance() -> None:
    targets = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "symbol": ["A", "A"],
            "weight": [1.0, 1.0],
        }
    )
    result = run_backtest(
        _prices(),
        targets,
        config=BacktestConfig(transaction_cost_bps=100.0, min_rebalance_turnover=0.1),
    )
    assert len(result.executions) == 2
    assert result.executions.loc[0, "turnover"] == pytest.approx(1.0)
    assert not result.executions.loc[0, "skipped_by_no_trade_band"]
    assert result.executions.loc[1, "requested_turnover"] == pytest.approx(0.0)
    assert result.executions.loc[1, "turnover"] == pytest.approx(0.0)
    assert result.executions.loc[1, "skipped_by_no_trade_band"]


def test_metrics_use_daily_drawdown() -> None:
    metrics = performance_metrics(pd.Series([0.10, -0.20, 0.10]))
    assert metrics["max_drawdown"] == pytest.approx(-0.20)
    assert performance_metrics(pd.Series([-0.20]))["max_drawdown"] == pytest.approx(-0.20)


def test_expanding_kmeans_never_trains_on_prediction_date() -> None:
    features = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=6, freq=pd.offsets.MonthEnd()),
            "density": [0.1, 0.2, 0.15, 0.8, 0.75, 0.7],
            "path": [4.0, 3.8, 3.9, 2.0, 2.1, 2.2],
        }
    )
    predicted = ExpandingKMeansRegime(n_clusters=2, min_train_size=3).predict_expanding(
        features, ["density", "path"]
    )
    assert len(predicted) == 3
    assert (predicted["training_end"] < predicted["date"]).all()


def test_expanding_hmm_emits_causal_probabilistic_states() -> None:
    generator = np.random.default_rng(7)
    features = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=14, freq="ME"),
            "realized_volatility": np.r_[
                generator.normal(0.08, 0.01, 7), generator.normal(0.30, 0.02, 7)
            ],
            "market_return": np.r_[generator.normal(0.01, 0.01, 7), generator.normal(-0.02, 0.02, 7)],
        }
    )
    predicted = ExpandingGaussianHMMRegime(
        n_components=2, min_train_size=6, random_state=7, n_iter=25
    ).predict_expanding(features, ["realized_volatility", "market_return"], ordering_feature="realized_volatility")
    assert (predicted["training_end"] < predicted["date"]).all()
    assert predicted[["probability_state_0", "probability_state_1"]].sum(axis=1).tolist() == pytest.approx(
        [1.0] * len(predicted)
    )


def test_expanding_hmm_stabilises_unvisited_transition_rows() -> None:
    generator = np.random.default_rng(17)
    features = pd.DataFrame(
        {
            "date": pd.date_range("2018-01-31", periods=42, freq="ME"),
            "realized_volatility": generator.normal(0.15, 0.002, 42),
            "market_return": generator.normal(0.005, 0.001, 42),
        }
    )
    predicted = ExpandingGaussianHMMRegime(
        n_components=3, min_train_size=36, random_state=7, n_iter=20
    ).predict_expanding(features, ["realized_volatility", "market_return"])
    probabilities = predicted.filter(like="probability_state_")
    assert len(predicted) == 6
    assert probabilities.sum(axis=1).tolist() == pytest.approx([1.0] * len(predicted))


def test_expanding_soft_kmeans_emits_causal_ordered_probabilities() -> None:
    dates = pd.date_range("2020-01-31", periods=9, freq="ME")
    features = pd.DataFrame(
        {
            "date": dates,
            "volatility": [0.1, 0.2, 0.15, 0.5, 0.45, 0.6, 0.12, 0.18, 0.4],
            "drawdown": [-0.01, -0.02, -0.01, -0.2, -0.15, -0.25, -0.02, -0.03, -0.1],
        }
    )
    predicted = ExpandingSoftKMeansRegime(
        n_components=2, min_train_size=4, random_state=7
    ).predict_expanding(features, ["volatility", "drawdown"], ordering_feature="volatility")
    assert len(predicted) == 5
    assert (predicted["training_end"] < predicted["date"]).all()
    assert predicted[["probability_state_0", "probability_state_1"]].sum(axis=1).tolist() == pytest.approx(
        [1.0] * len(predicted)
    )


def test_broadcast_regime_probabilities_requires_exact_dates() -> None:
    nodes = pd.DataFrame(
        {
            "date": ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"],
            "symbol": ["A", "B", "A", "B"],
            "one_session_return": [0.0, 0.0, 0.0, 0.0],
        }
    )
    probabilities = pd.DataFrame(
        {
            "date": ["2024-02-29"],
            "probability_state_0": [0.3],
            "probability_state_1": [0.7],
        }
    )
    merged, columns = broadcast_regime_probabilities_to_nodes(nodes, probabilities)
    assert columns == ("probability_state_0", "probability_state_1")
    assert merged["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2024-02-29"]
    assert merged["probability_state_1"].tolist() == pytest.approx([0.7, 0.7])


def test_regime_features_at_a_date_do_not_depend_on_later_prices() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
            "close": [
                value
                for row in zip(
                    [100, 102, 101, 104, 106, 105, 107, 108],
                    [100, 99, 101, 100, 103, 102, 104, 106],
                )
                for value in row
            ],
        }
    )
    full = build_market_regime_features(
        prices, volatility_window=3, drawdown_window=3, minimum_cross_sectional_assets=2
    )
    as_of = full.iloc[0]["date"]
    truncated = build_market_regime_features(
        prices.loc[prices["date"] <= as_of],
        volatility_window=3,
        drawdown_window=3,
        minimum_cross_sectional_assets=2,
    )
    assert truncated.iloc[-1]["realized_volatility"] == pytest.approx(
        full.loc[full["date"] == as_of, "realized_volatility"].item()
    )


def test_external_price_covariates_are_causal_and_merge_on_exact_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    covariate_prices = pd.DataFrame(
        {"date": dates, "symbol": ["VIX"] * len(dates), "close": [15, 16, 14, 17, 18, 16, 15, 14]}
    )
    full = build_price_regime_covariates(covariate_prices, symbol="VIX", prefix="vix", lookback_sessions=3)
    as_of = full.iloc[0]["date"]
    truncated = build_price_regime_covariates(
        covariate_prices.loc[covariate_prices["date"] <= as_of], symbol="VIX", prefix="vix", lookback_sessions=3
    )
    assert truncated.iloc[-1]["vix_level"] == pytest.approx(full.iloc[0]["vix_level"])
    market = pd.DataFrame({"date": [as_of, as_of + pd.offsets.BDay(1)], "market_return": [0.01, 0.02]})
    merged = merge_external_regime_covariates(market, full.loc[[0], ["date", "vix_level"]])
    assert merged["date"].tolist() == [as_of]
    assert merged["vix_level"].tolist() == pytest.approx([full.iloc[0]["vix_level"]])


def test_point_in_time_universe_uses_latest_available_snapshot() -> None:
    history = validate_membership_history(
        pd.DataFrame(
            {
                "date": ["2020-01-31", "2020-01-31", "2020-02-28", "2020-02-28"],
                "symbol": ["A", "B", "A", "B"],
                "eligible": [True, True, True, False],
            }
        )
    )
    assert eligible_universe(history, "2020-02-15") == ["A", "B"]
    assert eligible_universe(history, "2020-02-28") == ["A"]


def test_market_data_gate_rejects_stale_constituent_history() -> None:
    dates = pd.bdate_range("2024-01-01", periods=65)
    prices = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
            "close": [100.0 + index for index in range(len(dates)) for _ in (0, 1)],
        }
    )
    fresh_membership = pd.DataFrame(
        {
            "date": [dates[0], dates[0], dates[22], dates[22], dates[44], dates[44]],
            "symbol": ["A", "B"] * 3,
            "eligible": [True] * 6,
        }
    )
    report = validate_point_in_time_market_data(
        prices,
        fresh_membership,
        market="US",
        currency="usd",
        minimum_eligible_assets=2,
        maximum_membership_staleness_days=35,
    )
    assert report.currency == "USD"
    with pytest.raises(ValueError, match="stale"):
        validate_point_in_time_market_data(
            prices,
            fresh_membership.iloc[:2],
            market="US",
            currency="USD",
            minimum_eligible_assets=2,
            maximum_membership_staleness_days=10,
        )


def test_public_ohlcv_adapter_requires_and_preserves_actual_ohlcv_fields() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["A", "B"]])
    raw = pd.DataFrame(
        [
            [100, 200, 102, 202, 99, 199, 101, 201, 1000, 2000],
            [101, 201, 103, 203, 100, 200, 102, 202, 1100, 2100],
            [102, 202, 104, 204, 101, 201, 103, 203, 1200, 2200],
        ],
        index=dates,
        columns=columns,
    )
    panel = normalise_yfinance_download(raw, ["A", "B"])
    assert set(panel.columns) == {"date", "symbol", "open", "high", "low", "close", "volume"}
    assert len(panel) == 6
    assert panel.loc[panel["symbol"] == "A", "volume"].tolist() == [1000, 1100, 1200]


def test_public_ohlcv_audit_separates_missing_and_invalid_symbols() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([fields, ["A", "B"]])
    raw = pd.DataFrame(
        [
            [100, 200, 102, 199, 99, 198, 101, 201, 1000, 2000],
            [101, 201, 103, 200, 100, 199, 102, 202, 1100, 2100],
            [102, 202, 104, 201, 101, 200, 103, 203, 1200, 2200],
        ],
        index=dates,
        columns=columns,
    )
    audit = normalise_yfinance_download_audited(raw, ["A", "B", "MISSING"])
    assert audit.downloaded_symbols == ("A",)
    assert audit.missing_symbols == ("MISSING",)
    assert set(audit.invalid_symbols) == {"B"}
    assert audit.panel["symbol"].unique().tolist() == ["A"]


def test_kaggle_current_sp500_adapter_keeps_provider_close_and_standardises_fields() -> None:
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02"],
            "Symbol": ["B", "A"],
            "Open": [20.0, 10.0],
            "High": [21.0, 11.0],
            "Low": [19.0, 9.0],
            "Close": [20.5, 10.5],
            "Volume": [200, 100],
            "Adj Close": [20.1, 10.2],
        }
    )
    panel = normalise_kaggle_current_sp500_stocks(raw)
    assert panel.columns.tolist() == ["date", "symbol", "open", "high", "low", "close", "volume"]
    assert panel["symbol"].tolist() == ["A", "B"]
    assert panel["close"].tolist() == [10.5, 20.5]


def test_kaggle_nifty_weights_become_50_member_point_in_time_snapshots() -> None:
    tickers = [f"S{index:02d}" for index in range(50)]
    raw = pd.DataFrame(
        [
            ["2024-01-31", *([2.0] * 50)],
            ["2024-02-29", *([2.0] * 50)],
        ],
        columns=["DATE", *tickers],
    )
    membership = membership_from_kaggle_nifty50_weights(raw)
    assert len(membership) == 100
    assert membership.groupby("date")["index_weight_pct"].sum().tolist() == pytest.approx([100.0, 100.0])
    assert membership.groupby("date")["eligible"].sum().tolist() == [50, 50]
    with pytest.raises(ValueError, match="sum to 100"):
        membership_from_kaggle_nifty50_weights(raw.assign(S00=1.0))


def test_kaggle_adapter_drops_an_entire_partially_published_date_without_imputation() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["A", "B", "A", "B"],
            "open": [1.0, 2.0, None, 2.1],
            "high": [1.1, 2.1, None, 2.2],
            "low": [0.9, 1.9, None, 2.0],
            "close": [1.0, 2.0, None, 2.1],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )
    cleaned, dropped = drop_incomplete_ohlcv_dates(panel)
    assert dropped.tolist() == [pd.Timestamp("2024-01-03")]
    assert cleaned["date"].tolist() == ["2024-01-02", "2024-01-02"]


def test_kaggle_adapter_drops_a_symbol_with_a_materially_invalid_ohlc_bar() -> None:
    panel = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "symbol": ["A", "B", "A", "B"],
            "open": [1.0, 2.0, 1.1, 2.1],
            "high": [1.2, 2.2, 1.3, 2.3],
            "low": [0.9, 1.9, 1.0, 2.2],
            "close": [1.1, 2.1, 1.2, 2.15],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )
    cleaned, dropped = drop_symbols_with_materially_invalid_ohlcv(panel)
    assert dropped == ("B",)
    assert cleaned["symbol"].tolist() == ["A", "A"]


def test_kaggle_sector_balanced_survivor_cohort_requires_complete_coverage() -> None:
    prices = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"] * 4 + ["2024-01-02"],
            "symbol": ["A", "A", "B", "B", "C", "C", "D", "D", "E"],
        }
    )
    companies = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "company": ["A Co", "B Co", "C Co", "D Co", "E Co"],
            "sector": ["One", "One", "Two", "Two", "One"],
            "date_added": ["2000-01-01"] * 5,
        }
    )
    cohort = select_sector_balanced_current_survivor_cohort(
        prices, companies, start="2024-01-02", end="2024-01-03", per_sector=2
    )
    assert cohort["symbol"].tolist() == ["A", "B", "C", "D"]


def test_mean_cvar_optimizer_respects_constraints() -> None:
    history = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.01, -0.02],
            "B": [0.01, 0.00, 0.01, -0.01],
            "C": [-0.02, 0.01, 0.00, 0.00],
        }
    )
    weights = mean_cvar_weights(history, alpha=0.25, cvar_aversion=3.0, max_weight=0.6)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.6 + 1e-12).all()


def test_hierarchical_risk_parity_weights_are_long_only_and_deterministic() -> None:
    history = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.01, -0.02, 0.01, 0.00],
            "B": [0.01, -0.01, 0.00, -0.01, 0.01, 0.00],
            "C": [-0.02, 0.01, -0.01, 0.02, -0.01, 0.01],
        }
    )
    weights = hierarchical_risk_parity_weights(history, max_weight=0.6)
    repeated = hierarchical_risk_parity_weights(history, max_weight=0.6)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.6 + 1e-12).all()
    assert weights.to_dict() == pytest.approx(repeated.to_dict())


def test_hierarchical_risk_parity_from_covariance_respects_constraints() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.02, 0.005], [0.00, 0.005, 0.03]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = hierarchical_risk_parity_weights_from_covariance(covariance, max_weight=0.6)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert (weights <= 0.6 + 1e-12).all()


def test_soft_regime_conditional_hrp_uses_only_current_or_past_regime_probabilities() -> None:
    dates = pd.bdate_range("2024-01-01", periods=14)
    prices = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
            "close": [
                value
                for row in zip(
                    [100, 101, 99, 102, 103, 104, 102, 105, 107, 106, 109, 110, 111, 113],
                    [100, 99, 101, 100, 102, 101, 103, 104, 102, 105, 104, 106, 107, 108],
                )
                for value in row
            ],
        }
    )
    universe = pd.DataFrame(
        {"date": [dates[0], dates[0]], "symbol": ["A", "B"], "eligible": [True, True]}
    )
    probabilities = pd.DataFrame(
        {
            "date": [dates[5], dates[9], dates[-1], dates[-1] + pd.Timedelta(days=5)],
            "probability_state_0": [0.7, 0.4, 0.2, 0.99],
            "probability_state_1": [0.3, 0.6, 0.8, 0.01],
        }
    )
    builder = SoftRegimeConditionalHRPBuilder(
        RegimeConditionalRiskConfig(
            lookback_sessions=6,
            minimum_assets=2,
            minimum_effective_observations=2,
            covariance_shrinkage=0.25,
            max_weight=0.6,
        ),
        universe,
        probabilities,
    )
    weights = builder(prices, dates[-1])
    assert weights.sum() == pytest.approx(1.0)
    diagnostics = builder.diagnostics_frame().loc[0]
    assert diagnostics["current_probability_state_0"] == pytest.approx(0.2)
    assert diagnostics["n_return_observations"] == 6


def test_regime_conditional_hrp_falls_back_when_state_posterior_is_uniform() -> None:
    dates = pd.bdate_range("2024-01-01", periods=14)
    prices = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "symbol": ["A", "B"] * len(dates),
            "close": [
                value
                for row in zip(
                    [100, 101, 99, 102, 103, 104, 102, 105, 107, 106, 109, 110, 111, 113],
                    [100, 99, 101, 100, 102, 101, 103, 104, 102, 105, 104, 106, 107, 108],
                )
                for value in row
            ],
        }
    )
    universe = pd.DataFrame({"date": [dates[0], dates[0]], "symbol": ["A", "B"], "eligible": [True, True]})
    probabilities = pd.DataFrame(
        {
            "date": [dates[5], dates[9], dates[-1]],
            "training_end": [dates[4], dates[8], dates[-2]],
            "probability_state_0": [0.9, 0.2, 0.5],
            "probability_state_1": [0.1, 0.8, 0.5],
        }
    )
    builder = SoftRegimeConditionalHRPBuilder(
        RegimeConditionalRiskConfig(
            lookback_sessions=6,
            minimum_assets=2,
            minimum_effective_observations=2,
            covariance_shrinkage=0.25,
            max_weight=0.6,
            uncertainty_to_unconditional=True,
            minimum_regime_confidence=0.0,
            require_causal_training_end=True,
        ),
        universe,
        probabilities,
    )
    weights = builder(prices, dates[-1])
    diagnostics = builder.diagnostics_frame().loc[0]
    assert weights.sum() == pytest.approx(1.0)
    assert diagnostics["regime_entropy"] == pytest.approx(np.log(2.0))
    assert diagnostics["regime_confidence"] == pytest.approx(0.0)
    assert diagnostics["regime_training_end"] == dates[-2]


def test_group_aware_regime_conditional_hrp_requires_complete_groups_and_records_diagnostics() -> None:
    dates = pd.bdate_range("2024-01-01", periods=14)
    values = {
        "A": [100, 102, 101, 104, 103, 106, 105, 107, 110, 108, 111, 112, 114, 113],
        "B": [100, 99, 101, 100, 102, 101, 103, 104, 102, 105, 104, 106, 107, 108],
        "C": [100, 101, 103, 102, 104, 106, 105, 107, 108, 110, 109, 111, 112, 114],
    }
    prices = pd.DataFrame(
        [
            {"date": date, "symbol": symbol, "close": close}
            for symbol, series in values.items()
            for date, close in zip(dates, series)
        ]
    )
    universe = pd.DataFrame(
        {"date": [dates[0]] * 3, "symbol": ["A", "B", "C"], "eligible": [True] * 3}
    )
    probabilities = pd.DataFrame(
        {
            "date": [dates[5], dates[9], dates[-1]],
            "probability_state_0": [0.8, 0.3, 0.9],
            "probability_state_1": [0.2, 0.7, 0.1],
        }
    )
    config = RegimeConditionalRiskConfig(
        lookback_sessions=6,
        minimum_assets=3,
        minimum_effective_observations=2,
        covariance_shrinkage=0.25,
        max_weight=0.5,
        group_aware_within_group_only=True,
    )
    builder = SoftRegimeConditionalHRPBuilder(
        config,
        universe,
        probabilities,
        asset_groups={"A": "equity", "B": "equity", "C": "bonds"},
    )
    weights = builder(prices, dates[-1])
    diagnostics = builder.diagnostics_frame().iloc[0]
    assert weights.sum() == pytest.approx(1.0)
    assert diagnostics["group_aware_within_group_only"]
    assert diagnostics["conditional_group_count"] == 2
    assert diagnostics["group_covariance_psd_adjustment_frobenius"] >= 0.0

    missing_group_builder = SoftRegimeConditionalHRPBuilder(
        config,
        universe,
        probabilities,
        asset_groups={"A": "equity", "B": "equity"},
    )
    with pytest.raises(ValueError, match="no group mapping for assets"):
        missing_group_builder(prices, dates[-1])


def test_walk_forward_only_passes_history_through_signal_date() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=45).repeat(2),
            "symbol": ["A", "B"] * 45,
            "close": [100.0 + index for index in range(45) for _ in (0, 1)],
        }
    )
    observed_signal_dates: list[pd.Timestamp] = []

    def builder(history: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Series:
        assert history["date"].max() <= signal_date
        observed_signal_dates.append(signal_date)
        return pd.Series({"A": 0.5, "B": 0.5})

    result = run_walk_forward(
        prices,
        builder,
        walk_forward_config=WalkForwardConfig(minimum_history_sessions=20),
    )
    assert observed_signal_dates
    assert not result.targets.empty


def test_walk_forward_honors_requested_rebalance_frequency() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=140).repeat(2),
            "symbol": ["A", "B"] * 140,
            "close": [100.0 + index for index in range(140) for _ in (0, 1)],
        }
    )
    monthly = rebalance_signal_dates(prices, "ME")
    quarterly = rebalance_signal_dates(prices, "QE")
    assert len(quarterly) < len(monthly)


def test_fama_french_parser_extracts_decimal_daily_returns() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as bundle:
        bundle.writestr(
            "factors.csv",
            "Created for a parser test\n"
            ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
            "20240102,1.00,-0.20,0.30,0.10,-0.10,0.02\n"
            "20240103,-0.50,0.10,0.20,-0.10,0.00,0.02\n"
            "Copyright footer\n",
        )
    factors = parse_fama_french_daily_factors(payload.getvalue())
    assert factors["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert factors.loc[0, "Mkt-RF"] == pytest.approx(0.01)
    assert factors.loc[1, "RF"] == pytest.approx(0.0002)


def test_factor_model_recovers_known_alpha_and_betas() -> None:
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2024-01-02", periods=80)
    factors = pd.DataFrame(
        {
            "date": dates,
            "Mkt-RF": rng.normal(0.0004, 0.01, len(dates)),
            "SMB": rng.normal(0.0, 0.006, len(dates)),
            "HML": rng.normal(0.0, 0.005, len(dates)),
            "RMW": rng.normal(0.0, 0.004, len(dates)),
            "CMA": rng.normal(0.0, 0.004, len(dates)),
            "RF": np.full(len(dates), 0.0001),
        }
    )
    returns = pd.Series(
        0.0003
        + factors["RF"].to_numpy()
        + 0.8 * factors["Mkt-RF"].to_numpy()
        - 0.25 * factors["SMB"].to_numpy()
        + 0.1 * factors["HML"].to_numpy(),
        index=dates,
    )
    result = fit_factor_model(
        returns,
        factors,
        factor_columns=["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
        hac_lags=5,
    )
    assert result.observations == len(dates)
    assert result.daily_alpha == pytest.approx(0.0003, abs=1e-12)
    assert result.factor_betas["Mkt-RF"] == pytest.approx(0.8, abs=1e-10)
    assert result.factor_betas["SMB"] == pytest.approx(-0.25, abs=1e-10)
    assert result.r_squared == pytest.approx(1.0, abs=1e-12)


def test_walk_forward_can_lock_an_inclusive_outer_evaluation_boundary() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=90).repeat(2),
            "symbol": ["A", "B"] * 90,
            "close": [100.0 + index for index in range(90) for _ in (0, 1)],
        }
    )
    boundary = "2024-03-15"
    result = run_walk_forward(
        prices,
        lambda _history, _date: pd.Series({"A": 0.5, "B": 0.5}),
        walk_forward_config=WalkForwardConfig(
            minimum_history_sessions=20,
            first_signal_date=boundary,
            last_signal_date="2024-04-30",
        ),
    )
    assert result.targets["signal_date"].min() >= pd.Timestamp(boundary)
    assert result.targets["signal_date"].max() <= pd.Timestamp("2024-04-30")
    assert result.backtest.daily.index.max() <= pd.Timestamp("2024-04-30")


def test_baseline_builder_composes_pit_universe_graph_and_allocator() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "date": dates.repeat(3),
            "symbol": ["A", "B", "C"] * len(dates),
            "close": [
                value
                for row in zip(
                    [100, 101, 103, 102, 105, 107, 106, 110],
                    [100, 99, 101, 100, 102, 103, 102, 104],
                    [100, 102, 101, 103, 104, 103, 105, 106],
                )
                for value in row
            ],
        }
    )
    universe = pd.DataFrame(
        {"date": [dates[0]] * 3, "symbol": ["A", "B", "C"], "eligible": [True, True, True]}
    )
    builder = BaselineSignalBuilder(
        BaselineStrategyConfig(
            graph_method="pcc",
            selector="hub_avoid",
            allocator="inverse_volatility",
            lookback_sessions=5,
            minimum_assets=2,
            portfolio_size=2,
            edge_fraction=0.5,
            max_weight=0.6,
        ),
        universe,
    )
    weights = builder(prices, dates[-1])
    assert len(weights) == 2
    assert weights.sum() == pytest.approx(1.0)
    assert builder.diagnostics_frame().loc[0, "graph_nodes"] == 3


def test_probabilistic_regime_router_blends_allocations_without_hard_switching() -> None:
    date = pd.Timestamp("2024-03-29")
    router = ProbabilisticRegimeRouter(
        pd.DataFrame(
            {
                "date": [date],
                "training_end": [pd.Timestamp("2024-02-29")],
                "probability_state_0": [0.7],
                "probability_state_1": [0.3],
            }
        ),
        {
            0: lambda history, signal_date: pd.Series({"A": 1.0}),
            1: lambda history, signal_date: pd.Series({"B": 1.0}),
        },
    )
    weights = router(pd.DataFrame(), date)
    assert weights.to_dict() == pytest.approx({"A": 0.7, "B": 0.3})
    assert router.diagnostics_frame().loc[0, "selected_state"] == 0


def test_dynamic_graph_transformer_supports_isolated_assets_and_backpropagation() -> None:
    model = DynamicGraphTransformer(
        input_features=3,
        model_dimension=8,
        attention_heads=2,
        temporal_layers=1,
        graph_layers=1,
        dropout=0.0,
    )
    features = torch.randn(2, 4, 3, 3, requires_grad=True)
    adjacency = torch.zeros(2, 4, 3, 3)
    adjacency[:, :, 0, 1] = 0.8
    adjacency[:, :, 1, 0] = 0.8
    forecast = model(features, adjacency)
    assert forecast.score.shape == (2, 3)
    assert (forecast.scale > 0).all()
    loss = model.gaussian_negative_log_likelihood(
        forecast, torch.zeros(2, 3), target_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    )
    loss.backward()
    assert features.grad is not None


def test_dynamic_graph_transformer_supports_graph_and_temporal_ablations() -> None:
    features = torch.randn(1, 3, 2, 2)
    adjacency = torch.eye(2).repeat(1, 3, 1, 1)
    graph_only = DynamicGraphTransformer(
        input_features=2,
        model_dimension=4,
        attention_heads=2,
        temporal_layers=0,
        graph_layers=1,
        dropout=0.0,
    )
    temporal_only = DynamicGraphTransformer(
        input_features=2,
        model_dimension=4,
        attention_heads=2,
        temporal_layers=1,
        graph_layers=0,
        dropout=0.0,
    )
    assert graph_only(features, adjacency).score.shape == (1, 2)
    assert temporal_only(features, adjacency).score.shape == (1, 2)


def test_dynamic_graph_transformer_supports_causal_graph_attention() -> None:
    model = DynamicGraphTransformer(
        input_features=2,
        model_dimension=4,
        attention_heads=2,
        temporal_layers=1,
        graph_layers=1,
        graph_encoder="attention",
        dropout=0.0,
    )
    features = torch.randn(2, 3, 3, 2, requires_grad=True)
    adjacency = torch.zeros(2, 3, 3, 3)
    adjacency[:, :, 0, 1] = 0.7
    adjacency[:, :, 1, 0] = 0.7
    output = model(features, adjacency)
    assert output.score.shape == (2, 3)
    model.gaussian_negative_log_likelihood(output, torch.zeros(2, 3)).backward()
    assert features.grad is not None


def test_dynamic_graph_transformer_probability_gates_expert_forecasts() -> None:
    model = DynamicGraphTransformer(
        input_features=5,
        model_dimension=4,
        attention_heads=2,
        temporal_layers=0,
        graph_layers=1,
        dropout=0.0,
        regime_expert_heads=3,
    )
    model.eval()
    with torch.no_grad():
        for value, score_head, scale_head in zip(
            (1.0, 2.0, 3.0), model.regime_score_heads, model.regime_scale_heads
        ):
            score_head.weight.zero_()
            score_head.bias.fill_(value)
            scale_head[0].weight.zero_()
            scale_head[0].bias.fill_(0.0)
    adjacency = torch.eye(2).repeat(2, 3, 1, 1)
    features = torch.zeros(2, 3, 2, 5)
    features[0, -1, :, -3:] = torch.tensor([1.0, 0.0, 0.0])
    features[1, -1, :, -3:] = torch.tensor([0.0, 0.0, 1.0])
    forecast = model(features, adjacency)
    assert forecast.score[0].tolist() == pytest.approx([1.0, 1.0])
    assert forecast.score[1].tolist() == pytest.approx([3.0, 3.0])
    assert (forecast.scale > 0).all()


def test_dynamic_graph_transformer_information_coefficient_loss_respects_cross_sectional_order() -> None:
    model = DynamicGraphTransformer(
        input_features=2,
        model_dimension=4,
        attention_heads=2,
        temporal_layers=0,
        graph_layers=1,
    )
    targets = torch.tensor([[0.01, 0.02, 0.03]])
    aligned = GraphForecast(score=torch.tensor([[1.0, 2.0, 3.0]]), scale=torch.ones(1, 3))
    reversed_order = GraphForecast(score=torch.tensor([[3.0, 2.0, 1.0]]), scale=torch.ones(1, 3))
    aligned_loss = model.cross_sectional_information_coefficient_loss(aligned, targets)
    reversed_loss = model.cross_sectional_information_coefficient_loss(reversed_order, targets)
    assert float(aligned_loss) == pytest.approx(0.0)
    assert float(reversed_loss) == pytest.approx(2.0)


def test_equal_target_ensemble_requires_aligned_members_and_averages_weights() -> None:
    first = pd.DataFrame(
        {
            "signal_date": ["2024-01-31", "2024-01-31", "2024-02-29", "2024-02-29"],
            "symbol": ["A", "B", "A", "B"],
            "weight": [0.6, 0.4, 0.5, 0.5],
        }
    )
    second = first.assign(weight=[0.4, 0.6, 0.2, 0.8])
    averaged = equally_average_target_weights([first, second])
    assert averaged["weight"].tolist() == pytest.approx([0.5, 0.5, 0.35, 0.65])
    misaligned = second.loc[second["symbol"] == "A"].copy()
    with pytest.raises(ValueError, match="identical"):
        equally_average_target_weights([first, misaligned])


def test_dynamic_graph_dataset_preserves_fixed_assets_and_masks_ineligible_nodes() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.DataFrame(
        {
            "date": dates.repeat(3),
            "symbol": ["A", "B", "C"] * len(dates),
            "close": [
                value
                for row in zip(
                    [100, 101, 103, 102, 104, 105, 106, 108],
                    [100, 99, 101, 102, 101, 103, 104, 106],
                    [100, 100, 101, 100, 102, 103, 102, 104],
                )
                for value in row
            ],
        }
    )
    membership = pd.DataFrame(
        {"date": [dates[0]] * 3, "symbol": ["A", "B", "C"], "eligible": [True, True, False]}
    )
    signal_dates = list(dates[2:])
    graphs = build_graph_snapshots(
        prices,
        membership,
        signal_dates,
        config=GraphSnapshotConfig(lookback_sessions=2, edge_fraction=1.0, minimum_fit_assets=2),
    ).graphs
    node_features = build_node_feature_snapshots(
        prices, membership, graphs, ["A", "B", "C"], config=NodeFeatureConfig(lookback_sessions=2)
    )
    labels = build_rebalance_forward_returns(prices, signal_dates)
    dataset = build_dynamic_graph_window_dataset(
        node_features, graphs, labels, ["A", "B", "C"], sequence_length=2
    )
    assert dataset.features.shape[1:] == (2, 3, 5)
    assert dataset.target_mask[:, 2].sum().item() == 0
    assert dataset.universe == ("A", "B", "C")
    decision_date = dataset.input_end_dates[-1]
    training = train_expanding_dynamic_graph_transformer(
        dataset,
        decision_date,
        config=DynamicGraphTrainingConfig(
            model_dimension=8,
            attention_heads=2,
            temporal_layers=1,
            graph_layers=1,
            epochs=1,
            batch_size=2,
            dropout=0.0,
            random_seed=7,
        ),
    )
    assert all(date < decision_date for date in training.training_target_end_dates)
    assert completed_training_mask(dataset, decision_date).sum() == len(training.training_indices)
    forecast = predict_dynamic_graph_forecast(training, dataset, decision_date)
    assert forecast.score.shape == (1, 3)


def test_dashboard_marks_frozen_outputs_as_legacy_exploratory_artifacts() -> None:
    from web_app.app import app

    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"Legacy exploratory view" in response.data
    assert b"not canonical backtest evidence" in response.data


def test_dynamic_graph_experiment_runs_with_only_completed_labels() -> None:
    dates = pd.bdate_range("2020-01-01", periods=290)
    symbols = ["A", "B", "C"]
    prices = pd.DataFrame(
        {
            "date": dates.repeat(len(symbols)),
            "symbol": symbols * len(dates),
            "close": [
                value
                for index in range(len(dates))
                for value in (
                    100 + index * 0.08 + np.sin(index / 7),
                    95 + index * 0.05 + np.cos(index / 9),
                    105 + index * 0.06 + np.sin(index / 5),
                )
            ],
        }
    )
    membership = pd.DataFrame(
        {"date": [dates[0]] * 3, "symbol": symbols, "eligible": [True, True, True]}
    )
    config = DynamicGraphExperimentConfig(
        rebalance_frequency="ME",
        annualization_factor=252,
        minimum_eligible_assets=2,
        max_weight=0.6,
        graph=GraphSnapshotConfig(method="pcc", lookback_sessions=5, edge_fraction=1.0, minimum_fit_assets=2),
        node_features=NodeFeatureConfig(lookback_sessions=3),
        sequence_length=2,
        minimum_completed_training_windows=2,
        training=DynamicGraphTrainingConfig(
            model_dimension=8,
            attention_heads=2,
            temporal_layers=1,
            graph_layers=1,
            dropout=0.0,
            epochs=1,
            batch_size=4,
            random_seed=7,
        ),
        allocation_lookback_sessions=5,
        cvar_alpha=0.25,
        cvar_aversion=1.0,
        turnover_penalty=0.0,
        backtest=BacktestConfig(transaction_cost_bps=0.0, max_weight=0.6),
        first_signal_date="2020-09-01",
        last_signal_date="2020-12-31",
    )
    result = run_dynamic_graph_walk_forward(prices, membership, config=config)
    assert not result.targets.empty
    assert result.master_universe_date >= pd.Timestamp("2020-09-01")
    assert result.targets["signal_date"].min() >= pd.Timestamp("2020-09-01")
    assert result.targets["signal_date"].max() <= pd.Timestamp("2020-12-31")
    assert result.backtest.daily.index.max() <= pd.Timestamp("2020-12-31")
    assert result.backtest.daily.index.max() <= result.targets["signal_date"].max()
    assert (result.decision_diagnostics["latest_training_target_end"] < result.decision_diagnostics["signal_date"]).all()
    assert result.backtest.daily.index.min() > dates[0]


def test_dynamic_graph_causally_skips_unbuildable_warmup_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.concat(
        [
            pd.DataFrame({"date": dates, "symbol": "A", "close": range(100, 108)}),
            pd.DataFrame({"date": dates[3:], "symbol": "B", "close": range(50, 55)}),
        ],
        ignore_index=True,
    )
    membership = pd.DataFrame(
        {"date": [dates[0], dates[0]], "symbol": ["A", "B"], "eligible": [True, True]}
    )
    coverage = _graph_signal_coverage(
        prices,
        membership,
        pd.DatetimeIndex([dates[3], dates[6]]),
        GraphSnapshotConfig(lookback_sessions=3, edge_fraction=1.0, minimum_fit_assets=2),
    )
    assert coverage["n_with_complete_history"].tolist() == [1, 2]
    assert coverage["eligible_for_graph"].tolist() == [False, True]


def test_canonical_results_collector_excludes_legacy_and_incomplete_runs(tmp_path) -> None:
    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "run_manifest.json").write_text(
        '{"kind":"canonical-run-manifest","run_id":"completed","extra":{"paper_eligible":true,"strategy":{"allocator":"mean_cvar"}}}',
        encoding="utf-8",
    )
    (completed / "performance_metrics.json").write_text(
        '{"sharpe":1.2,"max_drawdown":-0.1}', encoding="utf-8"
    )
    incomplete = tmp_path / "graph_only"
    incomplete.mkdir()
    (incomplete / "run_manifest.json").write_text(
        '{"kind":"canonical-run-manifest","run_id":"graph_only"}', encoding="utf-8"
    )
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run_manifest.json").write_text(
        '{"kind":"legacy-manifest","run_id":"legacy"}', encoding="utf-8"
    )
    (legacy / "performance_metrics.json").write_text('{"sharpe":99}', encoding="utf-8")
    exploratory = tmp_path / "exploratory"
    exploratory.mkdir()
    (exploratory / "run_manifest.json").write_text(
        '{"kind":"canonical-run-manifest","run_id":"exploratory","extra":{"paper_eligible":false}}',
        encoding="utf-8",
    )
    (exploratory / "performance_metrics.json").write_text('{"sharpe":2.3}', encoding="utf-8")
    unclassified = tmp_path / "unclassified"
    unclassified.mkdir()
    (unclassified / "run_manifest.json").write_text(
        '{"kind":"canonical-run-manifest","run_id":"unclassified"}', encoding="utf-8"
    )
    (unclassified / "performance_metrics.json").write_text('{"sharpe":3.1}', encoding="utf-8")
    runs = collect_canonical_runs(tmp_path)
    assert runs["run_id"].tolist() == ["completed"]
    assert canonical_paper_table(runs).loc[0, "metric_sharpe"] == pytest.approx(1.2)
    assert collect_canonical_runs(tmp_path, include_exploratory=True)["run_id"].tolist() == [
        "completed",
        "exploratory",
        "unclassified",
    ]


def test_paired_block_bootstrap_and_deflated_sharpe_use_return_level_evidence() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    strategy = pd.Series(0.002, index=dates)
    benchmark = pd.Series(0.0, index=dates)
    comparison = paired_moving_block_bootstrap(
        strategy, benchmark, block_length=5, replications=200, random_seed=7
    )
    assert comparison.observations == 80
    assert comparison.bootstrap_ci_low > 0
    varied_returns = pd.Series(np.linspace(-0.01, 0.03, 80), index=dates)
    deflated = deflated_sharpe_ratio(varied_returns, [0.0, 0.01, 0.02, 0.03])
    assert 0.0 <= deflated.deflated_sharpe_probability <= 1.0
