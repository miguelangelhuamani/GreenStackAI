from prompts import build_critic_user_prompt, build_refiner_user_prompt, CRITIC_DEV_PROMPT, REFINER_DEV_PROMPT
from parsing_utils import parse_json_strict
import re

def run_one_pass_baseline(source_code, task_name, slow_callable, input_data, llm_client, harness):
    baseline_metrics = harness.get_median_metrics(slow_callable, input_data)
    
    try:
        critic_prompt = build_critic_user_prompt(source_code, baseline_metrics)
        critic_raw = llm_client.generate(CRITIC_DEV_PROMPT, critic_prompt)
        critic_json = parse_json_strict(critic_raw)
        
        refiner_prompt = build_refiner_user_prompt(source_code, critic_json)
        refiner_raw = llm_client.generate(REFINER_DEV_PROMPT, refiner_prompt)
        
        match = re.search(r"```python\n(.*?)\n```", refiner_raw, flags=re.DOTALL)
        if not match:
            match = re.search(r"```(?:[a-zA-Z0-9]+)?\n(.*?)\n```", refiner_raw, flags=re.DOTALL)
        extracted_code = match.group(1).strip() if match else None
        
        if not extracted_code:
            return {"rewritten_code": None, "metrics": None, "success": False}
            
        namespace = {}
        import textwrap
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
        return {"rewritten_code": None, "metrics": None, "success": False}
