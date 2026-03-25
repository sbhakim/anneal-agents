from pathlib import Path
from tempfile import TemporaryDirectory

from src.core.system import SelfEvolveSystem
from src.governance.canary import CanaryRunner
from src.governance.guard import Guard
from src.utils.config_loader import load_config


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_LEVEL_DIVERGENCE_INSTRUCTION = "Place an order for 2 monitor(s) for standard customer"


class _DummyRulePool:
    def snapshot(self):
        return {}

    def restore(self, snapshot):
        return None


def _always_stage(_patch):
    return True


def _always_pass(_example, _rule_pool):
    return {"ok": True, "reason": "synthetic pass"}


def _risky_schema_patch():
    return {
        "id": "probe-unsafe-001",
        "action": "UPDATE_TOOL_SCHEMA",
        "operator": "PlaceOrder",
        "details": "Change authentication method from legacy token flow to new token flow",
        "justification": "Required by upstream API update",
    }


def _low_risk_schema_patch():
    return {
        "id": "probe-safe-001",
        "action": "UPDATE_TOOL_SCHEMA",
        "operator": "PlaceOrder",
        "details": "Switch tool endpoint from /v1/book to /v2/book (deprecation/tool-drift).",
        "justification": "Observed endpoint deprecation",
    }


def _guard_context():
    return {
        "scores": {
            "risk": 0.5,
            "aggregate": 0.73,
            "plausibility": 0.7,
            "z3": {"verdict": "SAT", "parse_coverage": 1.0},
        },
        "trace": {"root_cause_candidates": []},
        "state": {
            "payment_method": "credit_card",
            "shipping_restrictions": [],
            "promo_policies": {},
            "return_window_days": 30,
            "expired_promos": [],
            "required_auth_mode": "legacy_auth_token",
            "allowed_auth_modes": ["legacy_auth_token", "signed_session_token"],
            "auth_schema_version": "legacy_auth_token",
        },
    }


def _governance_cfg(config_name):
    cfg = load_config(str(_REPO_ROOT / config_name))
    gov = dict(cfg.get("governance", {}) or {})
    gov.setdefault("domain", (cfg.get("scenario", {}) or {}).get("name", "ecommerce"))
    return gov


def _isolated_system_config(config_name, temp_dir):
    cfg = load_config(str(_REPO_ROOT / config_name))
    cfg = dict(cfg)
    cfg["fdka"] = dict(cfg.get("fdka", {}) or {})
    cfg["fdka"]["propose_edit"] = dict(cfg["fdka"].get("propose_edit", {}) or {})
    cfg["fdka"]["propose_edit"]["llm_provider"] = "mock"
    cfg["output"] = dict(cfg.get("output", {}) or {})
    cfg["output"]["results_dir"] = str(Path(temp_dir) / "results")
    cfg["logging"] = dict(cfg.get("logging", {}) or {})
    cfg["logging"]["log_file"] = str(Path(temp_dir) / "logs" / "test.log")
    cfg["governance"] = dict(cfg.get("governance", {}) or {})
    cfg["governance"]["provenance"] = dict(cfg["governance"].get("provenance", {}) or {})
    cfg["governance"]["provenance"]["log_path"] = str(Path(temp_dir) / "provenance.jsonl")
    return cfg


def _task_level_divergence_config(config_name, temp_dir):
    cfg = _isolated_system_config(config_name, temp_dir)
    cfg["system"] = dict(cfg.get("system", {}) or {})
    cfg["system"]["domain"] = "ecommerce"
    cfg["scenario"] = dict(cfg.get("scenario", {}) or {})
    cfg["scenario"]["name"] = "ecommerce"
    cfg["scenario"]["num_tasks"] = 1
    cfg["scenario"]["failure_rate"] = 1.0
    cfg["scenario"]["min_failures_in_prefix"] = 1
    cfg["scenario"]["prefix_len"] = 1
    cfg["scenario"]["task_overrides"] = [_TASK_LEVEL_DIVERGENCE_INSTRUCTION]
    cfg["scenario"]["placeorder_force_mode"] = "tool_schema_drift"
    return cfg



def test_full_governance_escalates_patch_that_nogovern_allows():
    patch = _risky_schema_patch()
    context = _guard_context()

    full_guard = Guard(_governance_cfg("config.yaml"))
    nogovern_guard = Guard(_governance_cfg("config_nogovern.yaml"))

    full_result = full_guard.check(patch, context=context)
    nogovern_result = nogovern_guard.check(patch, context=context)

    assert full_result["decision"] == "request_human"
    assert full_result["reason_code"] == "CAUSAL:HIGH_IMPACT"
    assert nogovern_result["decision"] == "allow"
    assert nogovern_result["value"]["reason_code"] == "VALUE:DISABLED"
    assert nogovern_result["causal"]["reason_code"] == "CAUSAL:DISABLED"



def test_nogovern_ablation_still_keeps_canary_gate_active():
    cfg = load_config(str(_REPO_ROOT / "config_nogovern.yaml"))
    canary = CanaryRunner((cfg.get("governance", {}) or {}).get("canary", {}) or {})

    result = canary.run(
        _risky_schema_patch(),
        {
            "rule_pool": _DummyRulePool(),
            "stage_fn": _always_stage,
            "simulator": _always_pass,
            "examples": [{"id": "ex-1"}],
            "scores": _guard_context()["scores"],
        },
    )

    assert result["passed"] is True
    assert result["mode"] == "low_power"



def test_nogovern_canary_simulation_allows_risky_patch_under_real_system_path():
    with TemporaryDirectory() as temp_dir:
        system = SelfEvolveSystem(_isolated_system_config("config_nogovern.yaml", temp_dir))
        instruction = system.scenario.get_task(1)
        example = {
            "trace_id": -1,
            "trace": [],
            "success": True,
            "operator": "PlaceOrder",
            "metadata": {"instruction": instruction, "task_id": "CANARY-ECOM-01"},
        }

        result = system.canary_runner.run(
            _risky_schema_patch(),
            {
                "rule_pool": system.rule_pool,
                "stage_fn": system.rule_pool.stage_patch,
                "simulator": system._run_canary_simulation,
                "examples": [example],
                "scores": _guard_context()["scores"],
            },
        )

    assert result["passed"] is True
    assert result["mode"] == "low_power"



def test_task_level_governance_activation_harness_diverges_between_full_and_nogovern():
    with TemporaryDirectory() as full_dir:
        full_system = SelfEvolveSystem(_task_level_divergence_config("config.yaml", full_dir))
        full_system.fdka.propose_edit = lambda trace, rule_pool: dict(_risky_schema_patch())
        full_result = full_system.run_task(0, full_system.scenario.get_task(0))
        full_stats = dict(full_system.metrics.governance_stats)
        full_committed = len(full_system._committed_patches)

    with TemporaryDirectory() as nogovern_dir:
        nogovern_system = SelfEvolveSystem(_task_level_divergence_config("config_nogovern.yaml", nogovern_dir))
        nogovern_system.fdka.propose_edit = lambda trace, rule_pool: dict(_risky_schema_patch())
        nogovern_result = nogovern_system.run_task(0, nogovern_system.scenario.get_task(0))
        nogovern_stats = dict(nogovern_system.metrics.governance_stats)
        nogovern_committed = len(nogovern_system._committed_patches)

    assert full_result["status"] == "failure"
    assert full_stats["causal_escalations"] == 1
    assert full_stats["canary_tests"] == 0
    assert full_committed == 0

    assert nogovern_result["status"] == "success"
    assert nogovern_stats["causal_escalations"] == 0
    assert nogovern_stats["canary_tests"] == 1
    assert nogovern_stats["canary_passes"] == 1
    assert nogovern_committed == 1



def test_ecommerce_valuekg_hydrates_from_runtime_state():
    guard = Guard(_governance_cfg("config.yaml"))
    context = _guard_context()
    context["state"] = {
        "payment_method": "credit_card",
        "shipping_restrictions": ["APO", "FPO"],
        "promo_policies": {"stackable_promos": False},
        "return_window_days": 14,
        "expired_promos": ["SAVE10"],
        "required_auth_mode": "signed_session_token",
        "allowed_auth_modes": ["legacy_auth_token", "signed_session_token"],
        "auth_schema_version": "signed_session_token",
    }

    result = guard.check(_low_risk_schema_patch(), context=context)

    assert result["decision"] == "allow"
    assert guard.value_kg is not None
    assert guard.value_kg.domain == "ecommerce"
    assert guard.value_kg.shipping_restrictions == ["APO", "FPO"]
    assert guard.value_kg.return_window_days == 14
    assert guard.value_kg.required_auth_mode == "signed_session_token"



def test_govern_stress_configs_diverge_with_real_proposal_path():
    with TemporaryDirectory() as full_dir:
        full_system = SelfEvolveSystem(_isolated_system_config("config_govern_stress.yaml", full_dir))
        full_result = full_system.run_task(0, full_system.scenario.get_task(0))
        full_stats = dict(full_system.metrics.governance_stats)
        full_committed = len(full_system._committed_patches)

    with TemporaryDirectory() as nogovern_dir:
        nogovern_system = SelfEvolveSystem(_isolated_system_config("config_govern_stress_nogovern.yaml", nogovern_dir))
        nogovern_result = nogovern_system.run_task(0, nogovern_system.scenario.get_task(0))
        nogovern_stats = dict(nogovern_system.metrics.governance_stats)
        nogovern_committed = len(nogovern_system._committed_patches)

    assert full_result["status"] == "failure"
    assert full_stats["causal_escalations"] == 1
    assert full_stats["canary_tests"] == 0
    assert full_committed == 0

    assert nogovern_result["status"] == "success"
    assert nogovern_stats["causal_escalations"] == 0
    assert nogovern_stats["canary_tests"] == 0
    assert nogovern_committed == 1


def test_govern_stress_suite_configs_diverge_over_multiple_tasks():
    with TemporaryDirectory() as full_dir:
        full_system = SelfEvolveSystem(_isolated_system_config("config_govern_stress_suite.yaml", full_dir))
        full_metrics = full_system.run_evaluation()
        full_summary = full_metrics.get_summary()
        full_stats = dict(full_system.metrics.governance_stats)
        full_committed = len(full_system._committed_patches)

    with TemporaryDirectory() as nogovern_dir:
        nogovern_system = SelfEvolveSystem(_isolated_system_config("config_govern_stress_suite_nogovern.yaml", nogovern_dir))
        nogovern_metrics = nogovern_system.run_evaluation()
        nogovern_summary = nogovern_metrics.get_summary()
        nogovern_stats = dict(nogovern_system.metrics.governance_stats)
        nogovern_committed = len(nogovern_system._committed_patches)

    assert full_summary["success_rate"] == 0.0
    assert full_stats["causal_escalations"] == 6
    assert full_stats["canary_tests"] == 0
    assert full_committed == 0

    assert nogovern_summary["success_rate"] == 1.0
    assert nogovern_stats["causal_escalations"] == 0
    assert nogovern_stats["canary_tests"] == 0
    assert nogovern_committed == 1
