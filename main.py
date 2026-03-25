# main.py
"""
ANNEAL: Adapting LLM Agents via Governed Symbolic Patch Learning
Main execution entry point.
"""

import argparse
import sys

from src.constants import SYSTEM_NAME, SYSTEM_KEY
# Import all agents: the main system and the baselines
from src.core.system import SelfEvolveSystem
from src.baselines.static_ns import StaticNSAgent
from src.baselines.llm_reflect import LLMReflectAgent
from src.baselines.verify_only import VerifyOnlyAgent
from src.baselines.react import ReActAgent
from src.baselines.reflexion import ReflexionAgent
from src.utils.config_loader import load_config


# Map agent names from the config file to their respective classes
AGENT_MAP = {
    SYSTEM_KEY: SelfEvolveSystem,
    "static_ns": StaticNSAgent,
    "llm_reflect": LLMReflectAgent,
    "verify_only": VerifyOnlyAgent,
    "react": ReActAgent,
    "reflexion": ReflexionAgent,
}


def main():
    """Main entry point: parses arguments and starts the selected system."""
    parser = argparse.ArgumentParser(
        description=f"{SYSTEM_NAME}: Adapting LLM Agents via Governed Symbolic Patch Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--mode', type=str, choices=['demo', 'eval'], default='demo',
        help='Execution mode (default: demo)'
    )
    args = parser.parse_args()

    # ASCII Banner
    print("\n" + "=" * 70)
    print(f"  {SYSTEM_NAME}: Adapting LLM Agents via Governed Symbolic Patch Learning")
    print("=" * 70)

    try:
        config = load_config(args.config)
        print(f"✅ Configuration loaded from: {args.config}\n")

        # Ensure expected top-level keys exist (keeps overrides fail-safe).
        config.setdefault('scenario', {})
        config.setdefault('metacognition', {})
        config.setdefault('experiment', {})

        if args.mode == 'demo':
            # Override settings for a shorter, more legible run.
            demo_overrides = config.get('demo', {})
            config['scenario']['num_tasks'] = int(demo_overrides.get('num_tasks', 20))
            config['scenario']['failure_rate'] = float(demo_overrides.get('failure_rate', 0.4))
            config['scenario']['min_failures_in_prefix'] = int(demo_overrides.get('min_failures_in_prefix', 5))

            # Faster adaptation detection for demos.
            config['metacognition']['adaptation_window'] = int(
                demo_overrides.get('metrics_window_size', 10)
            )

            print(
                "🎯 Running DEMO mode "
                f"({config['scenario']['num_tasks']} tasks, "
                f"failure_rate={config['scenario']['failure_rate']}, "
                f"adapt_window={config['metacognition']['adaptation_window']})...\n"
            )
            # Some scenarios may override failure_rate internally based on difficulty.
            # This print reflects the config override, not necessarily the final runtime value.
        else:
            print("📈 Running FULL EVALUATION...\n")

        # Determine which agent to run from the config
        agent_key = config.get('experiment', {}).get('agent_to_run', SYSTEM_KEY)
        AgentClass = AGENT_MAP.get(agent_key)

        if not AgentClass:
            print(f"❌ Unknown agent '{agent_key}' specified in config.yaml under 'experiment.agent_to_run'.")
            sys.exit(1)

        print(f"🔬 Running experiment with agent: {agent_key.upper()}")

        # Initialize and run the selected agent
        agent = AgentClass(config)
        agent.run_evaluation()

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✨ Done! Check data/results/ for outputs.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
