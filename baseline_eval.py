import random

from benchmark_tasks import DuplicateDetectionTask, NestedLoopJoinTask, MemoizationTask
from harness import PerformanceEngine


def generate_duplicate_data():
    random.seed(42)
    return [random.randint(0, 499) for _ in range(1000)]


def generate_join_data():
    random.seed(42)
    left = [(random.randint(0, 800), f"L_{i}") for i in range(1000)]
    right = [(random.randint(200, 1000), f"R_{i}") for i in range(1000)]
    return (left, right)


def generate_memo_data():
    random.seed(42)
    return [25, 28, 30, 32, 35]


def duplicate_detection(input_data):
    seen_items = set()
    duplicates = set()
    for item in input_data:
        if item in seen_items:
            duplicates.add(item)
        else:
            seen_items.add(item)
    return list(duplicates)


def nested_loop_join(input_data):
    left_records, right_records = input_data
    right_index = {}
    for right_key, right_value in right_records:
        right_index.setdefault(right_key, []).append(right_value)

    joined_results = []
    for left_key, left_value in left_records:
        for right_value in right_index.get(left_key, []):
            joined_results.append((left_value, right_value))
    return joined_results


def memoization(input_data):
    cache = {0: 0, 1: 1}

    def fib(n):
        if n in cache:
            return cache[n]
        cache[n] = fib(n - 1) + fib(n - 2)
        return cache[n]

    return [fib(n) for n in input_data]


def is_correct(baseline_result, candidate_result):
    if isinstance(baseline_result, list):
        try:
            return sorted(baseline_result) == sorted(candidate_result)
        except Exception:
            return baseline_result == candidate_result
    return baseline_result == candidate_result


def speedup_pct(baseline_duration, candidate_duration):
    return (baseline_duration - candidate_duration) / baseline_duration * 100 if baseline_duration else 0.0


def main():
    harness = PerformanceEngine()
    rows = [
        ("Duplicate Detection", DuplicateDetectionTask().slow_version, generate_duplicate_data(), duplicate_detection),
        ("Nested-Loop Join", NestedLoopJoinTask().slow_version, generate_join_data(), nested_loop_join),
        ("Memoization", MemoizationTask().slow_version, generate_memo_data(), memoization),
    ]

    print(f"{'Task':<22} | {'Speedup %':>10} | {'Correct %':>9}")
    print("-" * 50)
    for task_name, slow_fn, input_data, candidate_fn in rows:
        baseline_metrics = harness.get_median_metrics(slow_fn, input_data)
        candidate_metrics = harness.get_median_metrics(candidate_fn, input_data)
        correct = is_correct(baseline_metrics["result"], candidate_metrics["result"])
        speedup = speedup_pct(
            baseline_metrics["median_duration_seconds"],
            candidate_metrics["median_duration_seconds"],
        )
        print(f"{task_name:<22} | {speedup:>10.1f} | {100.0 if correct else 0.0:>9.1f}")


if __name__ == "__main__":
    main()
