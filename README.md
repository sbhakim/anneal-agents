# SelfEvolve: Self-Evolving Neuro-Symbolic Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-selfevolve-blue)](https://github.com/sbhakim/selfevolve)

---

## 🎯 Overview

SelfEvolve is a **governed, self-evolving agent** that adapts to open-world failures **without model retraining**. By editing symbolic operators through Failure-Driven Knowledge Acquisition (FDKA), it achieves:

- ✅ **90% success rate** (vs. 47% for baselines)
- ✅ **4% repeat-failure rate** (92% reduction in recurring errors)
- ✅ **~15 tasks to adapt** (≈2 minutes vs. hours for fine-tuning)
- ✅ **1250× cost reduction** ($0.04 vs. $50 per adaptation)
- ✅ **Zero rollbacks** across 250 executions

### Key Innovation

Traditional LLM agents fail repeatedly or require expensive fine-tuning. SelfEvolve **learns from failures** by synthesizing, scoring, and deploying **verifiable symbolic patches** with multi-layered governance (provenance tracking, trust scoring, canary testing, human-in-the-loop gates).

---

## 🏗️ Architecture

![SelfEvolve System Architecture](Figures/SelfEvolve_Architecture.pdf)

**Core Components:**
- **Metacognition**: Budgeted S1/S2 arbitration with verify-before-act
- **FDKA Pipeline**: Constrained patch synthesis with multi-dimensional scoring
- **Governance**: Provenance tracking, trust scoring, deterministic rollback

---

## 🚀 Quick Start
```bash
# Quick demo (20 tasks, single seed)
python main.py --mode demo

# Full evaluation (50 tasks, 5 seeds)
python main.py --mode eval --config config.yaml
```

**Note:** See `experiments/` directory for scripts to reproduce baseline comparisons and ablation studies.

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

## 📊 Results

**Baseline Comparison** (50 tasks, travel planning domain):

| System         | Success Rate | Repeat Failures | Time-to-Adapt | Patches | Cost/Adapt |
|----------------|--------------|-----------------|---------------|---------|------------|
| **SelfEvolve** | **90.0%**    | **4.0%**        | **14.9±1.5**  | **6.0** | **$0.04**  |
| Static-NS      | 47.2%        | 46.8%           | ∞             | —       | —          |
| LLM-Reflect    | 47.2%        | 0.0%*           | ∞             | —       | —          |
| Verify-Only    | 47.2%        | 46.8%           | ∞             | —       | —          |

*\*Avoids repeats by abandoning novel tasks*

### Key Findings

- **Surgical precision**: Each patch resolves 7.5 tasks on average (root-cause repair)
- **Framework robustness**: Works across GPT-4o-mini, DeepSeek, Llama-3.1-8B, Qwen2.5-7B, Mistral-7B
- **Defense-in-depth**: Zero rollbacks validate multi-layered governance

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📧 Contact

**Safayat Bin Hakim** - shakim3@umbc.edu  
**Project**: https://github.com/sbhakim/selfevolve