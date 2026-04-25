import pytest
from benchmark_tasks import (
    DuplicateDetectionTask, LinearSearchTask, LoopAggregationTask,
    TopKSelectionTask, NestedLoopJoinTask, StringConcatenationTask,
    RedundantListTask, MemoizationTask, DataStructureChoiceTask, MultiPassReductionTask
)

def test_duplicatedetection_correctness():
    task = DuplicateDetectionTask()
    input_data = [1, 2, 3, 2, 4, 1, 5]
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert sorted(slow) == sorted(fast)

def test_linearsearch_correctness():
    task = LinearSearchTask()
    input_data = ([(0,"a"),(1,"b"),(2,"c")], [1, 0, 2])
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert slow == fast

def test_loopaggregation_correctness():
    task = LoopAggregationTask()
    input_data = [1.0, 2.0, 3.0, 4.0]
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert abs(slow - fast) < 1e-6

def test_topkselection_correctness():
    task = TopKSelectionTask()
    input_data = ([5, 1, 8, 3, 9, 2, 7], 3)
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert sorted(slow) == sorted(fast)

def test_nestedloopjoin_correctness():
    task = NestedLoopJoinTask()
    input_data = ([(1,"a"),(2,"b"),(3,"c")], [(2,"x"),(3,"y"),(4,"z")])
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert sorted(slow) == sorted(fast)

def test_stringconcatenation_correctness():
    task = StringConcatenationTask()
    input_data = ["hello", "world", "foo"]
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert slow == fast

def test_redundantlist_correctness():
    task = RedundantListTask()
    input_data = list(range(20))
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert abs(slow - fast) < 1e-6

def test_memoization_correctness():
    task = MemoizationTask()
    input_data = [0, 1, 5, 10, 15]
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert slow == fast

def test_datastructurechoice_correctness():
    task = DataStructureChoiceTask()
    input_data = ([1,2,3,4,5], [3, 6, 1])
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    assert slow == fast

def test_multipassreduction_correctness():
    task = MultiPassReductionTask()
    input_data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    slow = task.slow_version(input_data)
    fast = task.fast_version(input_data)
    for s, f in zip(slow, fast):
        assert abs(s - f) < 1e-6
