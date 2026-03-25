from src.baselines.reflexion import ReflexionAgent


def _config_for(domain: str) -> dict:
    return {
        "scenario": {
            "name": domain,
            "difficulty": "normal",
            "num_tasks": 1,
            "failure_rate": 0.0,
            "task_generation_seed": 7,
            "failure_injector_seed": 108,
        },
        "fdka": {
            "propose_edit": {
                "llm_provider": "mock",
                "model": "mock",
                "temperature": 0.3,
                "timeout_sec": 1.0,
            }
        },
        "logging": {"level": "WARNING"},
        "output": {"results_dir": "data/results/test_reflexion"},
    }


def test_reflexion_prompt_is_domain_scoped_for_ecommerce():
    agent = ReflexionAgent(_config_for("ecommerce"))
    prompt = agent._build_system_prompt()

    assert "PlaceOrder" in prompt
    assert "ApplyPromoCode" in prompt
    assert "CreateTicket" not in prompt
    assert "ProvisionAccess" not in prompt
    assert "BookHotel" not in prompt
    assert "Use only the actions listed for the current domain." in prompt
