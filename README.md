# ANNEAL: Adapting LLM Agents via Governed Symbolic Patch Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> LLM-based agents recover from individual errors but repeatedly fail on the same fault when process knowledge remains unrepaired. **ANNEAL** converts recurring failures into governed symbolic edits of a process knowledge graph — without modifying foundation model weights.

<p align="center">
  <img src="Figures/anneal_system_arch.png" width="85%" alt="ANNEAL System Architecture"/>
</p>

## Key Results

Evaluated with GPT-4o-mini across 27 multi-seed runs (3 agents, 3 scenarios, 3 seeds each):

| Domain | ANNEAL | ReAct | Reflexion | Patches |
|--------|--------|-------|-----------|---------|
| Travel planning | 100.0±0.0% | 100.0±0.0% | 100.0±0.0% | 1.0±0.0 |
| Travel stochastic | 100.0±0.0% | 97.3±2.3% | 100.0±0.0% | 1.0±0.0 |
| E-commerce | 94.7±2.3% | 78.7±2.3% | **98.7±1.9%** | 2.7±0.6 |
| ITSM (untuned transfer) | 100.0±0.0% | — | — | 1.0±0.0 |

ANNEAL is the only system that commits persistent structural repairs. In recurring-failure stress tests, committed patches suppress holdout failures to **0%**, while ReAct and Reflexion retain 72–100% holdout failure rates despite high episodic SR.

## Quick Start

```bash
# Set up environment
conda activate hysym
export OPENAI_API_KEY="sk-..."

# Single evaluation run (default: e-commerce, 25 tasks)
python main.py --config config.yaml --mode eval

# Multi-seed reproducibility matrix (paper Tables 2–4)
python scripts/run_agent_matrix.py \
  --base-config config.yaml \
  --agents anneal react reflexion \
  --scenarios travel_planning ecommerce travel_stochastic \
  --seeds 7 13 31 --mode eval

# Aggregate results
python scripts/summarize_multiseed_metrics.py data/output_dir
```

All four domains (travel, e-commerce, travel stochastic, ITSM) are switchable inside `config.yaml`.

## Project Structure

```
├── main.py                          # Entry point
├── config.yaml                      # Unified configuration (all domains & providers)
├── src/
│   ├── constants.py                 # System name (single source of truth)
│   ├── core/
│   │   ├── system.py                # Main orchestrator
│   │   ├── planner.py               # HTN planner
│   │   └── executor.py              # Operator execution + verify/repair
│   ├── fdka/
│   │   ├── failure_classifier.py    # Patchability classification
│   │   ├── propose_edit.py          # Constrained patch synthesis (3-stage)
│   │   └── scoring.py               # Multi-dimensional scoring
│   ├── governance/
│   │   ├── guard.py                 # Value/causal guardrails
│   │   ├── canary.py                # Canary testing
│   │   ├── trust.py                 # Beta-Bernoulli trust scoring
│   │   └── provenance.py            # Provenance tracking + rollback sets
│   ├── metacognition/
│   │   ├── arbitrator.py            # S1/S2/Verify pathway selection
│   │   ├── signals.py               # Uncertainty + violation signals
│   │   └── reflection.py            # Threshold tuning
│   ├── knowledge/
│   │   ├── rule_pool.py             # Operator definitions + committed patches
│   │   ├── experience_pool.py       # Indexed failure traces
│   │   └── value_kg.py              # Deontic constraint knowledge graph
│   ├── scenarios/                   # Travel, e-commerce, ITSM, stochastic
│   └── baselines/                   # ReAct, Reflexion, LLM-Reflect, Static-NS
├── scripts/
│   ├── run_agent_matrix.py          # Multi-seed evaluation runner
│   └── summarize_multiseed_metrics.py
├── tests/                           # Unit and integration tests
└── data/knowledge/extracted/        # Domain knowledge graphs (input)
```

## Ablation Summary

| Ablation | Domain | Delta SR | Interpretation |
|----------|--------|----------|----------------|
| −FDKA | E-commerce | −26.7pp | Eliminates all structural repairs |
| −Arbitration | Travel stochastic | −4.7pp | Compound policy shifts overwhelm fast path |
| −Verify | E-commerce | +4.0pp | Residual constraint types |
| −Governance | Routine benchmark | 0.0pp | Low-risk patches pass silently |
| −Governance | Stress suite | 6/6 escalated | Auth-sensitive edits selectively blocked |

## Citation

```bibtex
@inproceedings{hakim2026anneal,
  title={{ANNEAL}: Adapting LLM Agents via Governed Symbolic Patch Learning},
  author={Hakim, Safayat Bin},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026}
}
```

## License

MIT License.
