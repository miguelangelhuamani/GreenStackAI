from harness import PerformanceEngine
from benchmark_tasks import DuplicateDetectionTask, LinearSearchTask, LoopAggregationTask
import random

def generate_duplicate_data():
    base_list = list(range(1000))
    base_list.extend([10, 50, 100, 500, 999])
    random.shuffle(base_list)
    return base_list

def generate_search_data():
    dataset = [(index, f"Value_{index}") for index in range(2000)]
    queries = [random.randint(0, 1999) for _ in range(500)]
    return (dataset, queries)

def generate_aggregation_data():
    return [random.uniform(0.0, 100.0) for _ in range(500000)]

def run_evaluation():
    engine = PerformanceEngine()
    
    tasks = [
        ("Duplicate Detection", DuplicateDetectionTask(), generate_duplicate_data()),
        ("Linear Search", LinearSearchTask(), generate_search_data()),
        ("Loop Aggregation", LoopAggregationTask(), generate_aggregation_data())
    ]
    
    print(f"{'Task Name':<25} | {'Speedup Factor':<15} | {'Memory Saved (KB)':<18}")
    print("-" * 65)
    
    for task_name, task_instance, input_data in tasks:
        slow_metrics = engine.get_metrics(task_instance.slow_version, input_data)
        fast_metrics = engine.get_metrics(task_instance.fast_version, input_data)
        
        if not slow_metrics["success"] or not fast_metrics["success"]:
            print(f"{task_name:<25} | {'Error during execution':<36}")
            continue
            
        slow_duration = slow_metrics["duration_seconds"]
        fast_duration = fast_metrics["duration_seconds"]
        
        speedup_factor = slow_duration / fast_duration if fast_duration > 0 else 0
        memory_saved = slow_metrics["peak_memory_kb"] - fast_metrics["peak_memory_kb"]
        
        print(f"{task_name:<25} | {speedup_factor:<15.2f} | {memory_saved:<18.2f}")

if __name__ == "__main__":
    run_evaluation()
