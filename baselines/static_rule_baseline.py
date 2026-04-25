import functools
import heapq
import inspect
import re

def apply_static_rules(source_code: str) -> str:
    new_code = source_code
    
    # 1. result += word -> suggest "".join
    new_code = re.sub(
        r'(\w+)\s*\+=\s*(\w+)', 
        r'# SUGGESTED: "".join(...)\n        \1 += \2', 
        new_code
    )
    
    # 2. Add @functools.lru_cache(maxsize=None) before def fib(
    new_code = new_code.replace("def fib(", "@functools.lru_cache(maxsize=None)\n        def fib(")
    
    # 3. sorted(x, reverse=True)[:k] -> heapq.nlargest(k, x)
    new_code = re.sub(r'sorted\((.*?),\s*reverse=True\)\[:(.*?)\]', r'heapq.nlargest(\2, \1)', new_code)
    
    # 4. x in some_list -> iterate optimization
    new_code = new_code.replace("in universe_list)", "in set(universe_list))")
    
    # 5. sum([x for x in iterable]) -> sum(x for x in iterable)
    new_code = re.sub(r'sum\(\[(.*?)\]\)', r'sum(\1)', new_code)
    
    if new_code != source_code:
        return "import functools\nimport heapq\n\n" + new_code
    return source_code

def run_static_baseline(task_instance, input_data, harness):
    source_code = inspect.getsource(task_instance.slow_version)
    rewritten_code = apply_static_rules(source_code)
    
    if rewritten_code == source_code:
        return {"rewritten_code": source_code, "metrics": {"success": False}, "rule_applied": False}
        
    namespace = {}
    try:
        exec(rewritten_code, namespace)
        candidate_func = namespace.get("slow_version")
        if candidate_func is None:
             candidate_func = [v for k,v in namespace.items() if callable(v) and k != 'apply_static_rules'][-1]
        metrics = harness.get_median_metrics(candidate_func, input_data)
        return {"rewritten_code": rewritten_code, "metrics": metrics, "rule_applied": True}
    except Exception as e:
        return {"rewritten_code": rewritten_code, "metrics": {"success": False, "error": str(e)}, "rule_applied": True}
