from prompts import REFINER_DEV_PROMPT
import re

def run_single_shot_baseline(source_code, task_name, llm_client, harness, input_data):
    user_prompt = f"Optimize the following Python function for runtime speed and memory efficiency. Return only a Python code block with the optimized version. Do not change the function signature or output behavior.\n\n{source_code}"
    
    response = llm_client.generate(REFINER_DEV_PROMPT, user_prompt)
    
    match = re.search(r"```python\n(.*?)\n```", response, flags=re.DOTALL)
    if not match:
        match = re.search(r"```(?:\w+)?\n(.*?)\n```", response, flags=re.DOTALL)
    
    extracted_code = match.group(1).strip() if match else None
    
    if not extracted_code:
        return {"rewritten_code": None, "metrics": None, "success": False}
        
    namespace = {}
    import textwrap
    try:
        exec(textwrap.dedent(extracted_code), namespace)
        funcs = [v for k,v in namespace.items() if callable(v) and not k.startswith('_')]
        candidate_func = funcs[-1] if funcs else None
        
        if candidate_func:
            import inspect
            sig = inspect.signature(candidate_func)
            if len(sig.parameters) == 2:
                wrapper = lambda data: candidate_func(None, data)
            else:
                wrapper = candidate_func
                
            metrics = harness.get_median_metrics(wrapper, input_data)
            return {"rewritten_code": extracted_code, "metrics": metrics, "success": metrics["success"]}
        return {"rewritten_code": extracted_code, "metrics": None, "success": False}
    except Exception:
        return {"rewritten_code": extracted_code, "metrics": None, "success": False}
