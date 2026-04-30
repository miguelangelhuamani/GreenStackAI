# GreenStackAI — Green-Code Refactoring Agent

CS 498 AI Agents in the Wild | UIUC | Spring 2026  
Group S11: Miguel Angel Huamani, Aanya Singh Dhankhar, Haoming Qin, Santiago Martinez

---

## Overview

**GreenPyBench** is a reproducible benchmark for evaluating AI agents that optimize
Python programs for computational efficiency. The **Critic–Refiner Agent** operates
in a closed ReAct loop: a Critic interprets profiler output to diagnose bottlenecks
and propose algorithmic changes; a Refiner generates targeted rewrites; a Controller
validates correctness via PyTest and re-profiles to accept or roll back each candidate.

---

## Repository Structure

| File/Folder | Description |
|---|---|
| `agent_skeleton.py` | Main `RefactoringAgent` — combines Critic plans and Refiner rewrites into a self-testing loop |
| `benchmark_tasks.py` | 10 task specifications covering algorithmic complexity, data-structure, and memory inefficiencies |
| `harness.py` | Profiling harness: 7-run median wall-clock timing + `tracemalloc` peak RAM measurement |
| `llm_client.py` | LLM client supporting Anthropic and Tinker/OpenAI-compatible endpoints |
| `parsing_utils.py` | JSON parsing utilities for Critic output |
| `prompts.py` | Critic and Refiner prompt templates |
| `run_eval.py` | Unified evaluation runner: seeds all conditions, runs baselines and agent, writes `results/` |
| `baselines/` | Static rule, single-shot LLM, and one-pass profile-guided implementations |
| `tests/` | PyTest suite with fixed-seed correctness tests for all 10 tasks |
| `results/` | Pre-computed `results.csv` and `summary.csv` from the reported 3-trial run |

---

## Setup

### Requirements

Python 3.10+ is required. A clean virtual environment is strongly recommended.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### API Key Configuration

Final reported results were generated with:
- **Provider:** `tinker`
- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507`
- **Temperature:** 0.2 | **Max tokens:** 1400

**Tinker / OpenAI-compatible:**
```bash
export LLM_PROVIDER=tinker
export TINKER_API_KEY=your_tinker_key_here
export TINKER_BASE_URL=https://<your-host>/v1
export LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
```

**Anthropic:**
```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key_here
export LLM_MODEL=claude-3-5-sonnet-20241022
```

---

## Running the Benchmark

```bash
# Full evaluation — all systems, 3 trials (default)
python run_eval.py

# Low-cost mode — 1 trial
python run_eval.py --trials 1

# Baselines only (no LLM calls)
python run_eval.py --no-llm

# Run correctness tests
pytest tests/
```

---

## Reproducing Results

All input generators use a fixed random seed (`42`) for deterministic behavior.
To reproduce the exact numbers in the paper, run on **macOS or Linux** with a
consistent CPU configuration; Windows timing variability may cause small differences
in runtime speedup percentages. Results are written progressively to `results/results.csv`.

---

## Agent Architecture

The Critic–Refiner agent follows a ReAct loop with four components:

```
slow code ──profile──▶ Critic ──JSON plan──▶ Refiner ──rewrite──▶ Controller
                                                                       │
                                              accept or rollback ◀─ test + profile
```

1. **Critic** — receives source code and profiler metrics; returns a structured JSON
   plan with hotspots, root cause, proposed O(·) changes, and acceptance criteria.
   Does *not* write code.
2. **Refiner** — receives source + JSON plan; outputs a single corrected Python block.
3. **Controller** — runs PyTest (any failure → immediate rollback), then accepts only
   if VIS strictly improves. Two consecutive below-baseline profiling reads are required
   before rollback to avoid false rejections from cold-cache variance.
4. **Profiling Tool** — 7-run median wall-clock + `tracemalloc` peak RAM, isolated subprocess.

---

## Benchmark Tasks (GreenPyBench)

| # | Task | Inefficiency | Optimization | Difficulty |
|---|---|---|---|---|
| 1 | Duplicate Detection | O(n²) nested loop | Hash-based lookup | Medium |
| 2 | Linear Search | Full-list scan | Set membership / bisect | Easy |
| 3 | Loop Aggregation | Python loop over array | NumPy vectorization | Easy |
| 4 | Top-K Selection | Full sort for partial result | Heap-based selection | Medium |
| 5 | Nested-Loop Join | O(n·m) cross-join | Hash join | Hard |
| 6 | String Concatenation | Repeated `+` in loop | `str.join` | Easy |
| 7 | Generator Pipeline | Intermediate list allocation | Generator chaining | Medium |
| 8 | Memoization | Redundant recursive calls | LRU cache / DP | Medium |
| 9 | Data Structure Choice | List where set fits | Set/dict substitution | Medium |
| 10 | Multi-Pass Fusion | Three separate traversals | Single-pass fusion | Hard |

---

## Evaluation Metrics

- **Correctness pass rate** — fraction of tasks passing the full PyTest suite after rewrite
- **Runtime speedup (%)** — median of 7 measured runs vs. original, isolated subprocess
- **Peak RAM reduction (%)** — `tracemalloc` peak allocation reduction vs. original
- **VIS (Valid Improvement Score)** — composite score with correctness gate:

  `VIS = max(0, 0.7 × runtime_gain + 0.3 × memory_gain − quality_penalty)`

  Quality penalty of −0.25 applies if cyclomatic complexity increases >20% without
  achieving ≥10% speedup (measured via `radon`).
- **Regression rate** — fraction of tasks where VIS is negative (rewrite is worse than original)

---

## Results

Results from the 3-trial evaluation reported in the papers:

| System | Correctness (%) | RT Speedup (%) | RAM Reduction (%) | Mean VIS | Std |
|:---|---:|---:|---:|---:|---:|
| Baseline Slow | 100 | 0.00 | 0.00 | 0.00 | — |
| Static Rule | 100 | 25.6 | 0.1 | 18.32 | — |
| Single-Shot LLM | 10 | 19.9 | 0.0 | 13.92 | 1.84 |
| One-Pass Profile-Guided | 90 | −45.8 | 12.5 | 45.98 | 3.21 |
| **Critic–Refiner (ours)** | **90** | **64.2** | **22.1** | **52.08** | **1.47** |
| Reference Optimized | 100 | 82.1 | 41.6 | — | — |

Full per-task breakdown available in `results/summary.csv`.

---

## Citation

If you use GreenPyBench or this agent in your work, please cite:

```
Huamani, M.A., Dhankhar, A.S., Qin, H., and Martinez, S.
GreenPyBench: A benchmark for evaluating automated Python code efficiency optimization.
CS 498, UIUC, 2026.
```
