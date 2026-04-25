# GreenStackAI — Green-Code Refactoring Agent

CS 498 AI Agents in the Wild | UIUC | Spring 2026
Group S11: Miguel Huamani, Aanya Singh Dhankhar, Haoming Qin, Santiago Martinez

## Overview
GreenPyBench is a benchmark for evaluating automated performance optimization capabilities of AI agents on Python workloads. The implemented Critic-Refiner agent profiles target code using a harness, diagnoses structural hotspots, and synthesizes complexity-reducing refactoring via the ReAct framework to save energy and runtime.

## Repository Structure
- `agent_skeleton.py`: Main `RefactoringAgent` combining Critic plans and Refiner rewrites into a self-testing loop.
- `benchmark_tasks.py`: Houses 10 algorithmic inefficiencies (Task specs) covering memory scaling, complexity bloats, and nested scans.
- `harness.py`: High-fidelity OS-level metrics evaluation using `resource.getrusage()` mapping process peaks.
- `llm_client.py`: Configuration schemas linking native LLM payloads toward inference providers cleanly.
- `parsing_utils.py`: JSON abstraction utilities.
- `prompts.py`: Core logic directives mapped over ReAct developer instructions for profiling.
- `run_eval.py`: The unified execution suite generating seeded configurations, computing holistic multi-layer variants (0-shot, statically, via ReAct) storing logs to `results.csv`.
- `baselines/`: Folder containing static rules, single-shot models, and one-pass implementations for comparative benchmarking.
- `tests/`: Pytest suite holding fixed-seed tests enforcing zero-fault semantic correctness.

## Setup
### Requirements
Ensure you are using Python 3.10+. It is highly recommended to build within a clean venv.
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

### API Key
This project uses the Anthropic Claude API.
Set your key before running:
    export ANTHROPIC_API_KEY=your_key_here
Model used: claude-sonnet-4-5-20250929

## Running the Benchmark
### Run full evaluation (with LLM)
python run_eval.py

### Run without LLM (baselines only)
python run_eval.py --no-llm

### Run tests
pytest tests/

## Reproducing Results
Input datasets are enforced to be fully deterministic matching pseudo-random sequences statically seeded with `42` under the randomized Python standard logic per configuration structure. To reproduce exact result metrics, please run within macOS or Linux boundaries holding consistent CPU capabilities; CSV logs are populated progressively on successful metric runs spanning baselines automatically internally routing paths logically.

## Agent Architecture
The custom framework mimics sequential refinement implementing the generic sequential decision making (ReAct). The continuous iterative optimization loop executes logic in phases: Profile base execution → Critique identifying Big-O limits → Rewrite code block proposing a targeted replacement → Test candidate for execution regressions → Accept or Rollback measuring holistic VIS viability factors dynamically until optimization peaks.

## Benchmark (GreenPyBench)
1. **Duplicate Detection** — Replace O(N^2) list iteration with O(N) hash sets.
2. **Linear Search** — Remove loop scan iterations adopting nested hash dictionary keys mapping mapping direct returns.
3. **Loop Aggregation** — Offload Python scalar summation into bulk vectorized NumPy integrations.
4. **Top-K Selection** — Evict O(N log N) sorts mapping constrained partial allocations yielding O(N log K) partial sorts.
5. **Nested-Loop Join** — Transform nested O(NM) cross iterations replacing into O(N+M) hashed joins leveraging indexed hash sets natively.
6. **String Concatenation** — Modify destructive immutability over += aggregations building native C `.join` abstractions.
7. **Redundant List Materialisation** — Eliminate heavy RAM allocations adopting mapped generative pipeline chain structures.
8. **Memoization** — Augment branching recursively caching computed node tree responses recursively.
9. **Data Structure Choice** — Change repeated sequential iterations across sets matching sets over raw lists.
10. **Multi-Pass Reduction** — Aggregate sequential dataset parsers dynamically computing statistics inside one uniform iteration pass.

## Evaluation Metrics
- **Correctness pass rate**: Ratio of test iterations retaining equivalent logical functionality validating output matching identically.
- **Runtime speedup %**: End-to-end reduction tracked scaling `time.perf_counter` benchmarks against original logic mappings percentages.
- **RAM reduction %**: OS-level `getrusage` mapping reductions indicating byte scaling saved.
- **VIS (Value-Impact Score)**: Combines runtime gain, RAM reduction and penalty metrics evaluating cyclomatic logic adjustments quantitatively. Equation: `max(0, 0.7 * runtime_gain + 0.3 * memory_gain - quality_penalty)`.
- **Regression rate**: Negative VIS variations breaking application stability natively.

## Results
| Task | System | CorrRate | Speedup% | RAMred% | VIS |
|:-----|:-------|:---------|:---------|:--------|:----|
| ...  | ...    | ...      | ...      | ...     | ... |
