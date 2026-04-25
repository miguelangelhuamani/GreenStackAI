import re
import tempfile
import subprocess
import os

try:
    from radon.complexity import cc_visit
except ImportError:
    pass

from prompts import (
    build_critic_user_prompt, 
    build_refiner_user_prompt, 
    CRITIC_DEV_PROMPT, 
    REFINER_DEV_PROMPT
)
from parsing_utils import parse_json_strict

class RefactoringAgent:
    """
    The main agent class responsible for autonomously refactoring inefficient code.
    Follows a ReAct loop to profile, critique, and edit code.
    """

    def __init__(self, llm_client, harness, max_iterations=5):
        self.llm_client = llm_client
        self.harness = harness
        self.max_iterations = max_iterations

    def run_pytest(self, code_string, task_name):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
            f.write(code_string)
            temp_name = f.name
        
        try:
            result = subprocess.run(
                ["pytest", temp_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired as e:
            return {
                "passed": False,
                "stdout": e.stdout.decode('utf-8') if e.stdout else "",
                "stderr": "Timeout expired"
            }
        finally:
            try:
                os.remove(temp_name)
            except OSError:
                pass

    def extract_code(self, refiner_response):
        match = re.search(r"```python\n(.*?)\n```", refiner_response, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        match_any = re.search(r"```(?:\w+)?\n(.*?)\n```", refiner_response, flags=re.DOTALL)
        if match_any:
            return match_any.group(1).strip()
        return None

    def plan_refactor(self, source_code, metrics):
        user_prompt = build_critic_user_prompt(source_code, metrics)
        response = self.llm_client.generate(CRITIC_DEV_PROMPT, user_prompt)
        try:
            return parse_json_strict(response)
        except Exception:
            return {"error": "parse_failed", "raw": response}

    def execute_cycle(self, source_code, slow_callable, input_data, task_name="task", constraints=None):
        baseline_metrics = self.harness.get_median_metrics(slow_callable, input_data)
        
        best_code = source_code
        best_metrics = baseline_metrics
        best_vis = 0.0
        
        for i in range(self.max_iterations):
            critic_plan = self.plan_refactor(best_code, best_metrics)
            if "error" in critic_plan:
                print(f"Iteration {i}: Critic failed to parse. Breaking.")
                break
                
            refiner_prompt = build_refiner_user_prompt(best_code, critic_plan, constraints)
            refiner_response = self.llm_client.generate(REFINER_DEV_PROMPT, refiner_prompt)
            candidate_code = self.extract_code(refiner_response)
            
            if not candidate_code:
                print(f"Iteration {i}: No code block extracted. Continuing.")
                continue
                
            candidate_namespace = {}
            import textwrap
            try:
                # Need to use a fresh namespace with imports if required
                exec(textwrap.dedent(candidate_code), candidate_namespace)
                candidate_func = candidate_namespace.get(slow_callable.__name__)
                
                # Fallback if name changed
                if not candidate_func:
                    funcs = [v for k,v in candidate_namespace.items() if callable(v) and not k.startswith('_')]
                    if funcs:
                        candidate_func = funcs[-1]
                
                if not candidate_func:
                    raise ValueError(f"Function {slow_callable.__name__} not found in rewritten code")
                    
                import inspect
                sig = inspect.signature(candidate_func)
                if len(sig.parameters) == 2:
                    wrapper = lambda data: candidate_func(None, data)
                else:
                    wrapper = candidate_func
                    
                new_metrics = self.harness.get_median_metrics(wrapper, input_data)
                
                if not new_metrics["success"]:
                    print(f"ROLLED BACK iteration {i}: execution failed - {new_metrics.get('error')}")
                    continue
                    
                baseline_duration = baseline_metrics["median_duration_seconds"]
                baseline_memory = baseline_metrics["median_peak_memory_kb"]
                
                candidate_duration = new_metrics["median_duration_seconds"]
                candidate_memory = new_metrics["median_peak_memory_kb"]
                
                runtime_gain = (baseline_duration - candidate_duration) / baseline_duration * 100 if baseline_duration else 0.0
                memory_gain = (baseline_memory - candidate_memory) / baseline_memory * 100 if baseline_memory else 0.0
                memory_gain = max(0.0, memory_gain)
                
                # calculate cyclomatic complexity
                try:
                    score_before = sum(b.complexity for b in cc_visit(best_code))
                    score_after = sum(b.complexity for b in cc_visit(candidate_code))
                except Exception:
                    score_before = 1
                    score_after = 1
                    
                lines_before = len(best_code.strip().split('\n'))
                lines_after = len(candidate_code.strip().split('\n'))
                
                cc_increased = score_after > score_before * 1.2 and score_before > 0
                lines_increased = lines_after > lines_before * 1.3 and lines_before > 0
                poor_gains = runtime_gain < 10.0 and memory_gain < 5.0
                
                quality_penalty = 0.25 if ((cc_increased or lines_increased) and poor_gains) else 0.0
                
                vis = max(0.0, 0.7 * runtime_gain + 0.3 * memory_gain - quality_penalty)
                
                if vis > best_vis:
                    print(f"ACCEPTED iteration {i}, VIS={vis:.2f}")
                    best_code = candidate_code
                    best_vis = vis
                    best_metrics = new_metrics
                else:
                    print(f"ROLLED BACK iteration {i}: vis ({vis:.2f}) <= best_vis ({best_vis:.2f})")
                    
            except Exception as e:
                print(f"ROLLED BACK iteration {i}: exception evaluating candidate - {e}")
                continue
                
        return {
            "final_code": best_code,
            "final_vis": best_vis,
            "final_metrics": best_metrics,
            "iterations": self.max_iterations
        }
