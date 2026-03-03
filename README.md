# GreenStackAI: AI-Driven Algorithmic Optimization

### Project Goal
Reducing energy consumption by automating code efficiency. This project focuses on identifying and eliminating computational inefficiencies—such as quadratic loops and redundant data copies—at the software application layer. By treating optimization as a structured, evidence-driven process, GreenStackAI aims to provide tangible benefits for "Green AI" through automated algorithmic refactoring.

### Tech Stack
* Python
* ReAct Architecture
* AST-based refactoring

### Overview
The Green-Code Refactoring Agent operates in a closed optimization loop. A Critic interprets profiling outputs to identify hotspots and hypothesize root causes. A Refiner then generates code transformations guided by algorithmic principles. Each rewrite is validated via unit tests and re-profiled to verify the optimization success.

### Instructions for Teammates
To extend the benchmarking suite, you need to add more tasks to `benchmark_tasks.py`.

1. Open `benchmark_tasks.py`.
2. Create a new class for your specific inefficiency pattern (e.g., `StringConcatenationTask`).
3. Implement `slow_version(self, input_data)` simulating the unoptimized, inefficient approach.
4. Implement `fast_version(self, input_data)` containing the algorithmically refactored, optimized approach.
5. In `run_eval.py`, import your new task class.
6. Provide a targeted input generator function inside `run_eval.py` that produces deterministic, relevant input data.
7. Add your task to the `tasks` list inside the `run_evaluation` function to include it in the automated benchmark testing.
