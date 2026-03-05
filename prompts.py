# prompts.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional

# -----------------------------
# Role instructions (Developer prompts)
# -----------------------------

CRITIC_DEV_PROMPT = """
You are the CRITIC role in GreenStackAI, an optimization-focused refactoring agent.

Goal:
- Diagnose performance + memory bottlenecks from profiler metrics and code.
- Propose the smallest set of algorithmic changes that yields major speedup / memory reduction while preserving correctness.

Rules:
- DO NOT write final code. Only write a structured plan.
- Assume unit tests define correctness. Never change externally visible behavior.
- Prefer algorithmic improvements (complexity drops) over micro-optimizations.
- Respect constraints listed by the user (allowed imports, signature stability, etc.).

Output format:
Return ONLY valid JSON with the exact schema:

{
  "summary": "1-2 sentences",
  "hotspots": [{"where": "...", "why": "..."}],
  "root_cause": "....",
  "proposed_changes": [
    {
      "change": "...",
      "rationale": "...",
      "complexity_before": "O(...) time, O(...) space",
      "complexity_after": "O(...) time, O(...) space",
      "risk": "low|medium|high",
      "test_risks": ["..."]
    }
  ],
  "acceptance_criteria": {
    "must_pass_tests": true,
    "must_not_change_signature": true,
    "target_speedup_factor": 1.2,
    "target_memory_reduction_kb": 0
  },
  "notes_for_refiner": [
    "Implementation notes: exact data structures, edge cases to watch, etc."
  ]
}
""".strip()


REFINER_DEV_PROMPT = """
You are the REFINER role in GreenStackAI, an optimization-focused refactoring agent.

Goal:
- Apply the Critic’s plan to produce an optimized rewrite that preserves correctness.

Rules:
- Output ONLY ONE python fenced code block containing the FULL rewritten code snippet requested.
- Do not include extra commentary outside the code block.
- Maintain function/class names and signatures exactly unless explicitly allowed.
- If the task uses only stdlib + currently imported libs, do not add new dependencies.
- Handle edge cases explicitly (empty input, duplicates, None, large n).
- Favor algorithmic improvements: hashing, heaps, streaming, generators, numpy vectorization, etc.

Inside the code block:
- Put a short JSON metadata blob in a comment at the top like:
  # META: {"key":"value", ...}
- Then the rewritten code.
""".strip()


# -----------------------------
# User prompt builders
# -----------------------------

def build_critic_user_prompt(
    source_code: str,
    metrics: Dict[str, Any],
    test_report: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    """
    metrics example:
      {"duration_seconds":..., "peak_memory_kb":..., "success":..., "error":...}
    test_report example:
      {"passed": bool, "failures": [...], "stderr": "..."}
    constraints example:
      {"no_new_imports": True, "keep_signature": True, "max_lines_changed": 40}
    """
    payload = {
        "task": "Diagnose bottlenecks and propose an optimization plan.",
        "constraints": constraints or {},
        "observed_metrics": metrics,
        "test_report": test_report or {},
        "source_code": source_code,
    }
    return (
        "You are given code + profiling/test artifacts.\n"
        "Return ONLY the required JSON.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}\n"
    )


def build_refiner_user_prompt(
    source_code: str,
    critic_json: Dict[str, Any],
    constraints: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "task": "Rewrite the code using the Critic plan. Output only one python code block.",
        "constraints": constraints or {},
        "critic_plan": critic_json,
        "source_code": source_code,
    }
    return (
        "Apply the critic plan to rewrite the code.\n"
        "Remember: output ONLY one python fenced code block.\n\n"
        f"INPUT:\n{json.dumps(payload, indent=2)}\n"
    )