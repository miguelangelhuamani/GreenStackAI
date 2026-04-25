import functools
import heapq
import inspect
import re

def apply_static_rules(source_code: str) -> str:
    new_code = source_code
    
    # 1. result += word -> suggest "".join
    new_code = re.sub(
        r'result_string\s*=\s*""\n\s*for index, word in enumerate\(input_data\):.*?return result_string', 
        r'return ",".join(input_data)', 
        new_code,
        flags=re.DOTALL
    )
    
    # 2. Add @functools.lru_cache(maxsize=None) before def fib(
    new_code = new_code.replace("    def fib(", "    @functools.lru_cache(maxsize=None)\n    def fib(")
    
    # 3. sorted(x, reverse=True)[:k] -> heapq.nlargest(k, x)
    new_code = re.sub(r'sorted\((.*?),\s*reverse=True\)\[:(.*?)\]', r'heapq.nlargest(\2, \1)', new_code)
    
    # 4. x in some_list -> iterate optimization
    if "set(" not in source_code:
        new_code = new_code.replace("not in duplicates:", "not in set(duplicates):")
        new_code = new_code.replace("in universe_list)", "in set(universe_list))")
    
    # 5. Loop aggregation -> NumPy
    new_code = re.sub(
        r'total_sum\s*=\s*0\.0\n\s*for number in input_data:\n\s*total_sum \+= number \* number\n\s*return total_sum',
        r'import numpy\n    return float(numpy.sum(numpy.array(input_data) ** 2))',
        new_code,
        flags=re.DOTALL
    )
    
    if new_code != source_code:
        return "import functools\nimport heapq\n\n" + new_code
    return source_code

def run_static_baseline(task_instance, input_data, harness):
    import textwrap
    source_code = textwrap.dedent(inspect.getsource(task_instance.slow_version))
    rewritten_code = apply_static_rules(source_code)
    
    task_name = task_instance.__class__.__name__
    print(f"[Static Rule DEBUG] {task_name} rewritten code:\n{rewritten_code[:300]}")
    
    if rewritten_code == source_code:
        metrics = harness.get_median_metrics(task_instance.slow_version, input_data)
        return {"rewritten_code": source_code, "metrics": metrics, "rule_applied": False}
        
    namespace = {}
    try:
        exec(rewritten_code, namespace)
        candidate_func = namespace.get("slow_version")
        if candidate_func is None:
             candidate_func = [v for k,v in namespace.items() if callable(v) and k != 'apply_static_rules'][-1]
             
        sig = inspect.signature(candidate_func)
        if len(sig.parameters) == 2:
            wrapper = lambda data: candidate_func(None, data)
        else:
            wrapper = candidate_func
            
        metrics = harness.get_median_metrics(wrapper, input_data)
        if metrics["success"]:
            b_metrics = harness.get_median_metrics(task_instance.slow_version, input_data)
            if metrics["median_duration_seconds"] > b_metrics["median_duration_seconds"] * 1.05:
                return {"rewritten_code": source_code, "metrics": b_metrics, "rule_applied": False}
                
        return {"rewritten_code": rewritten_code, "metrics": metrics, "rule_applied": True}
    except Exception as e:
        return {"rewritten_code": rewritten_code, "metrics": {"success": False, "error": str(e)}, "rule_applied": True}
