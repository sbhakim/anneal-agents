# SelfEvolve: Governed Self-Evolution via Symbolic Knowledge Editing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Adaptive agents without model retraining.** SelfEvolve repairs symbolic process knowledge through governed self-editing, achieving perfect task success where static baselines plateau below 50%.

## Key Results

- **100% task success** on travel planning (vs. 30–49% baselines), **72%** on e-commerce cross-domain
- **9.0±5.6 tasks to adapt** (2 min vs. 4–8 GPU hours for fine-tuning)
- **1 patch per run, 100% acceptance rate** (surgical precision, zero rollbacks)
- **1250× cost reduction** ($0.04 vs. $50 per adaptation cycle)
- **0% terminal failures** despite 76% observed failures during recovery (robust iterative repair)

## Innovation

**Failure-Driven Knowledge Acquisition (FDKA)** converts execution failures into typed symbolic patches via constrained neurosymbolic generation, multi-dimensional scoring (plausibility, SMT consistency, counterfactual replay, risk), and multi-layered governance (value/causal guardrails, canary testing, cryptographic provenance). Foundation model weights remain frozen—adaptation occurs exclusively through auditable, reversible symbolic edits.

---

## Architecture

**Three-layer design:**

1. **Metacognitive Control**: Monitor–evaluate–regulate loop with uncertainty-driven arbitration (S1/S2 pathways), verify-before-act semantics, and reflection-based threshold tuning
2. **FDKA Pipeline**: Localization → constrained generation (closed JSON schema) → scoring (plausibility, SMT consistency, utility, risk) → guardrails (value/causal constraints) → canary testing → commit with provenance
3. **Knowledge Graphs**: Process (HTN operators), Value (deontic constraints), Causal (intervention effects), Experience (indexed failure traces)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (travel planning, 25 tasks)
python -m selfevolve.main

# Run with specific configuration
python -m selfevolve.main --config config.yaml

# Reproduce paper results
bash experiments/run_baselines.sh
bash experiments/run_ablations.sh
```

**Reproducibility:** All experiments use fixed seeds. Results directory contains evaluation outputs, patch commits, and provenance logs.

---

## ⚙️ Configuration

Key settings in `config.yaml`:
```yaml
fdka:
  propose_edit:
    llm_provider: "openai"      # Options: "openai", "deepseek", "transformers", "mock"
    model: "gpt-4o-mini"
    temperature: 0.3

metacognition:
  tau_u: 0.25                   # Uncertainty threshold
  tau_p: 0.20                   # Violation probability threshold

governance:
  gates:
    tau_impact: 0.6             # Risk escalation threshold
    tau_conf: 0.5               # Confidence gate threshold
```

---

## Empirical Validation

**Travel Planning** (50 tasks, 5 seeds, deterministic):

| System         | Success Rate | Terminal RFR | Time-to-Adapt | Patches | Acceptance |
|----------------|--------------|--------------|---------------|---------|------------|
| **SelfEvolve** | **100%***    | **0%***      | **9.0±5.6**   | **1.0** | **100%**   |
| Static-NS      | 49.2%        | 45.6%        | ∞             | 0       | —          |
| LLM-Reflect    | 48.8%        | 43.2%        | ∞             | 0       | —          |
| Verify-Only    | 30.0%        | 66.0%        | ∞             | 0       | —          |

*p < 0.001, Cohen's d = 4.2*

**Cross-Domain** (25 tasks, e-commerce): 72% SR confirms architectural portability.

**Observed vs. Terminal RFR:** 76% failures during recovery attempts (52% precondition violations, 48% tool errors), but 0% terminal failures—all tasks ultimately succeed through iterative self-repair.

### Additional Findings

- **Cross-model robustness**: 78–90% SR across GPT-4o-mini, DeepSeek, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B
- **Governance validation**: Zero rollbacks across 250 executions confirm defense-in-depth (scoring + guardrails + canary)
- **Hierarchy of learnability**: Operator-level failures adapt within 11–14 tasks; meta-planning failures require HTN-level edits (future work)

## Citation

If you use SelfEvolve in your research, please cite:

```bibtex
@article{hakim2025selfevolve,
  title={SelfEvolve: Governed Self-Evolution via Symbolic Knowledge Editing},
  author={Hakim, Safayat Bin and [Co-authors]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License & Contact

MIT License. For questions: safayat.b.hakim [at] gmail [dot] com

**Reproducibility artifacts**: Code, configurations, and evaluation scripts are provided for full reproducibility of paper results.