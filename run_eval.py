import random
import argparse
import os
import csv
import inspect
import time
from collections import defaultdict

from benchmark_tasks import (
    DuplicateDetectionTask, LinearSearchTask, LoopAggregationTask,
    TopKSelectionTask, NestedLoopJoinTask, StringConcatenationTask,
    RedundantListTask, MemoizationTask, DataStructureChoiceTask, MultiPassReductionTask
)
from harness import PerformanceEngine
from baselines.static_rule_baseline import run_static_baseline
from baselines.single_shot_baseline import run_single_shot_baseline
from baselines.one_pass_baseline import run_one_pass_baseline
from agent_skeleton import RefactoringAgent
from llm_client import LLMClient, LLMConfig

def generate_duplicate_data():
    random.seed(42)
    return [random.randint(0, 499) for _ in range(1000)] # Reduced from 5000 to 1000

def generate_search_data():
    random.seed(42)
    dataset = [(i, f"Val_{i}") for i in range(2000)]
    queries = [random.randint(0, 1999) for _ in range(500)]
    return (dataset, queries)

def generate_aggregation_data():
    random.seed(42)
    return [random.uniform(0.0, 100.0) for _ in range(500000)]

def generate_topk_data():
    random.seed(42)
    return ([random.randint(0, 100000) for _ in range(10000)], 50)

def generate_join_data():
    random.seed(42)
    left = [(random.randint(0, 800), f"L_{i}") for i in range(1000)]
    right = [(random.randint(200, 1000), f"R_{i}") for i in range(1000)]
    return (left, right)

def generate_string_data():
    random.seed(42)
    import string
    def rs(): return ''.join(random.choices(string.ascii_letters, k=5))
    return [rs() for _ in range(5000)]

def generate_redundant_data():
    random.seed(42)
    return list(range(200000))

def generate_memo_data():
    random.seed(42)
    return [25, 28, 30, 32, 35]

def generate_ds_choice_data():
    random.seed(42)
    universe = [random.randint(0, 10000) for _ in range(5000)] # Reduced from 50000 to 5000
    queries = [random.randint(0, 10000) for _ in range(1000)] # Reduced from 10000 to 1000
    return (universe, queries)

def generate_multipass_data():
    random.seed(42)
    return [random.uniform(0.0, 100.0) for _ in range(500000)]

def compute_vis(baseline_metrics, candidate_metrics, candidate_code, original_code):
    if candidate_code == original_code:
        return {"vis": 0.0, "runtime_gain_pct": 0.0, "memory_gain_pct": 0.0, "quality_penalty": 0.0}
        
    bg_rt = baseline_metrics["median_duration_seconds"]
    bg_mem = baseline_metrics["median_peak_memory_kb"]
    cg_rt = candidate_metrics["median_duration_seconds"]
    cg_mem = candidate_metrics["median_peak_memory_kb"]
    
    rt_gain = (bg_rt - cg_rt) / bg_rt * 100 if bg_rt else 0.0
    mem_gain = (bg_mem - cg_mem) / bg_mem * 100 if bg_mem else 0.0
    mem_gain = max(0.0, mem_gain)
    
    try:
        from radon.complexity import cc_visit
        cc_b = sum(b.complexity for b in cc_visit(original_code))
        cc_c = sum(b.complexity for b in cc_visit(candidate_code))
    except ImportError:
        cc_b, cc_c = 1, 1
        
    lb = len(original_code.strip().split('\n'))
    lc = len(candidate_code.strip().split('\n'))
    
    c_inc = cc_c > cc_b * 1.2 and cc_b > 0
    l_inc = lc > lb * 1.3 and lb > 0
    poor = rt_gain < 10.0 and mem_gain < 5.0
    
    qp = 0.25 if ((c_inc or l_inc) and poor) else 0.0
    vis = max(0.0, 0.7 * rt_gain + 0.3 * mem_gain - qp)
    return {"vis": vis, "runtime_gain_pct": rt_gain, "memory_gain_pct": mem_gain, "quality_penalty": qp}

def run_full_evaluation(llm_client=None, run_agent=True, run_baselines=True, n_trials=3, resume_from=None):
    harness = PerformanceEngine()
    
    tasks = [
        ("Duplicate Detection", DuplicateDetectionTask(), generate_duplicate_data()),
        ("Linear Search", LinearSearchTask(), generate_search_data()),
        ("Loop Aggregation", LoopAggregationTask(), generate_aggregation_data()),
        ("Top-K Selection", TopKSelectionTask(), generate_topk_data()),
        ("Nested-Loop Join", NestedLoopJoinTask(), generate_join_data()),
        ("String Concatenation", StringConcatenationTask(), generate_string_data()),
        ("Redundant List", RedundantListTask(), generate_redundant_data()),
        ("Memoization", MemoizationTask(), generate_memo_data()),
        ("Data Structure Choice", DataStructureChoiceTask(), generate_ds_choice_data()),
        ("Multi-Pass Reduction", MultiPassReductionTask(), generate_multipass_data())
    ]
    
    results = []
    
    # helper for tracking trials
    def record(sys_name, task_name, trial, metrics_data, c_code, o_code):
        if not metrics_data or not metrics_data.get("success"):
            results.append({
                "task_name": task_name, "system": sys_name, "trial": trial,
                "correctness": False, "runtime_speedup_pct": 0.0, "ram_reduction_pct": 0.0,
                "vis": 0.0, "regression": False
            })
            return
            
        if sys_name == "Baseline Slow":
            results.append({
                "task_name": task_name, "system": sys_name, "trial": trial,
                "correctness": True, "runtime_speedup_pct": 0.0, "ram_reduction_pct": 0.0,
                "vis": 0.0, "regression": False
            })
            return
            
        vis_res = compute_vis(bg_map[task_name], metrics_data, c_code, o_code)
        reg = vis_res["runtime_gain_pct"] < 0 or vis_res["memory_gain_pct"] < 0
        
        # Check correctness by comparing outputs
        bg_res = bg_map[task_name]["result"]
        cg_res = metrics_data["result"]
        try:
            if isinstance(bg_res, list) and bg_res and type(bg_res[0]) in (int, float, str):
                is_correct = sorted(bg_res) == sorted(cg_res)
            elif isinstance(bg_res, float):
                is_correct = abs(bg_res - cg_res) / max(1.0, abs(bg_res)) < 1e-4
            else:
                is_correct = bg_res == cg_res
        except Exception:
            is_correct = False
            
        print(f"[{task_name}] {sys_name} - Corr: {is_correct}")
        
        results.append({
            "task_name": task_name, "system": sys_name, "trial": trial,
            "correctness": is_correct, "runtime_speedup_pct": vis_res["runtime_gain_pct"],
            "ram_reduction_pct": vis_res["memory_gain_pct"], "vis": vis_res["vis"],
            "regression": reg
        })

    bg_map = {}
    
    task_names_list = [t[0] for t in tasks]
    
    for task_name, task_instance, input_data in tasks:
        if resume_from and resume_from in task_names_list:
            start_idx = task_names_list.index(resume_from)
            current_idx = task_names_list.index(task_name)
            if current_idx < start_idx:
                print(f"[Resume] Skipping {task_name}")
                continue
                
        print(f"--- Starting Task: {task_name} ---")
        import textwrap
        slow_src = textwrap.dedent(inspect.getsource(task_instance.slow_version))
        fast_src = textwrap.dedent(inspect.getsource(task_instance.fast_version))
        
        # 1) Original Slow Baseline
        bm = harness.get_median_metrics(task_instance.slow_version, input_data)
        bg_map[task_name] = bm
        record("Baseline Slow", task_name, 1, bm, slow_src, slow_src)
        
        # 2) Static Rule Baseline
        if run_baselines:
            sb_res = run_static_baseline(task_instance, input_data, harness)
            record("Static Rule", task_name, 1, sb_res.get("metrics"), 
                   sb_res.get("rewritten_code", slow_src), slow_src)
        
        # LLM dependent evaluations
        if llm_client is not None:
            # 3) Single-Shot Baseline
            if run_baselines:
                for t in range(n_trials):
                    ss_res = run_single_shot_baseline(slow_src, task_name, llm_client, harness, input_data)
                    record("Single-Shot LLM", task_name, t+1, ss_res.get("metrics"), 
                           ss_res.get("rewritten_code", slow_src) or slow_src, slow_src)
                    time.sleep(3)
            
            # 4) One-Pass Profile Guided
            if run_baselines:
                for t in range(n_trials):
                    op_res = run_one_pass_baseline(slow_src, task_name, task_instance.slow_version, input_data, llm_client, harness)
                    record("One-Pass Profile", task_name, t+1, op_res.get("metrics"), 
                           op_res.get("rewritten_code", slow_src) or slow_src, slow_src)
                    time.sleep(3)
            
            # 5) Full ReAct Agent
            if run_agent:
                agent = RefactoringAgent(llm_client, harness, max_iterations=3)
                for t in range(n_trials):
                    ag_res = agent.execute_cycle(slow_src, task_instance.slow_version, input_data, task_name)
                    record("Critic-Refiner Agent", task_name, t+1, ag_res.get("final_metrics"), 
                           ag_res.get("final_code", slow_src), slow_src)
                    time.sleep(3)

    # Save details
    os.makedirs('results', exist_ok=True)
    with open('results/results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_name", "system", "trial", "correctness", 
            "runtime_speedup_pct", "ram_reduction_pct", "vis", "regression"
        ])
        writer.writeheader()
        writer.writerows(results)

    # Aggregate summaries
    agg = defaultdict(lambda: {"hits": 0, "corr": 0, "rt": 0.0, "ram": 0.0, "vis": 0.0, "reg": 0})
    for r in results:
        key = (r["task_name"], r["system"])
        agg[key]["hits"] += 1
        if r["correctness"]: agg[key]["corr"] += 1
        agg[key]["rt"] += r["runtime_speedup_pct"]
        agg[key]["ram"] += r["ram_reduction_pct"]
        agg[key]["vis"] += r["vis"]
        if r["regression"]: agg[key]["reg"] += 1
        
    sum_res = []
    for (tname, sysname), v in agg.items():
        n = v["hits"]
        sum_res.append({
            "task_name": tname,
            "system": sysname,
            "correctness_rate": v["corr"] / n,
            "mean_runtime_speedup_pct": v["rt"] / n,
            "mean_ram_reduction_pct": v["ram"] / n,
            "mean_vis": v["vis"] / n,
            "regression_rate": v["reg"] / n
        })

    with open('results/summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_name", "system", "correctness_rate", "mean_runtime_speedup_pct", 
            "mean_ram_reduction_pct", "mean_vis", "regression_rate"
        ])
        writer.writeheader()
        writer.writerows(sum_res)

    print(f"{'Task':<22} | {'System':<20} | {'CorrRate':<8} | {'Speedup%':<8} | {'RAMred%':<8} | {'VIS':<6}")
    print("-" * 85)
    for row in sum_res:
        print(f"{row['task_name']:<22} | {row['system']:<20} | {row['correctness_rate']:<8.2f} | {row['mean_runtime_speedup_pct']:<8.2f} | {row['mean_ram_reduction_pct']:<8.2f} | {row['mean_vis']:<6.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GreenStackAI")
    parser.add_argument("--no-llm", action="store_true", help="Skip any LLM-based runs")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume evaluation from this task name (skips earlier tasks)")
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)
    
    llm = None
    if not args.no_llm:
        # Defaults to the env settings in llm_client
        llm = LLMClient(LLMConfig(provider="openai_responses", model="gpt-4o-mini", temperature=0.1))

    run_full_evaluation(
        llm_client=llm,
        run_agent=not args.no_llm,
        run_baselines=True,
        n_trials=3,
        resume_from=args.resume_from
    )
