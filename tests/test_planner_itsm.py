from src.core.planner import Planner
from src.knowledge.rule_pool import RulePool


def _planner():
    rule_pool = RulePool("data/knowledge/operators.json")
    return Planner({"type": "HTN", "max_depth": 5, "timeout_ms": 300}, rule_pool)


def test_extract_itsm_resource_handles_department_to_resource_form():
    planner = _planner()

    resource = planner._extract_itsm_resource(
        "Provision contributor access for charlie.wang in Operations to billing-system"
    )

    assert resource == "billing-system"


def test_compile_provision_access_keeps_resource_for_natural_itsm_instruction():
    planner = _planner()

    plan = planner.compile(
        "Provision contributor access for charlie.wang in Operations to billing-system",
        {},
    )

    assert len(plan) == 1
    assert plan[0]["operator"].name == "ProvisionAccess"
    assert plan[0]["params"]["user"] == "charlie.wang"
    assert plan[0]["params"]["resource"] == "billing-system"
    assert plan[0]["params"]["role"] == "contributor"


def test_compile_credential_rotation_routes_to_resetcredentials():
    planner = _planner()

    plan = planner.compile(
        "Perform credential rotation for grace.okonkwo on payroll-system",
        {},
    )

    assert len(plan) == 1
    assert plan[0]["operator"].name == "ResetCredentials"
    assert plan[0]["params"]["user"] == "grace.okonkwo"
    assert plan[0]["params"]["system"] == "payroll-system"


def test_extract_itsm_system_handles_on_the_system_form():
    planner = _planner()

    system = planner._extract_itsm_system(
        "Reset credentials for bob.smith on the code-repository system using security_key verification"
    )

    assert system == "code-repository"
