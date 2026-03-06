import numpy
import heapq
from collections import deque
from functools import lru_cache
import itertools

class DuplicateDetectionTask:
    def slow_version(self, input_data):
        duplicates = []
        for primary_index in range(len(input_data)):
            for search_index in range(primary_index + 1, len(input_data)):
                current_item = input_data[primary_index]
                if current_item == input_data[search_index] and current_item not in duplicates:
                    duplicates.append(current_item)
        return duplicates

    def fast_version(self, input_data):
        seen_items = set()
        duplicates = set()
        for item in input_data:
            if item in seen_items:
                duplicates.add(item)
            else:
                seen_items.add(item)
        return list(duplicates)

class LinearSearchTask:
    def slow_version(self, input_data):
        dataset_pairs, query_keys = input_data
        search_results = []
        for target_key in query_keys:
            found_value = None
            for key, value in dataset_pairs:
                if key == target_key:
                    found_value = value
                    break
            search_results.append(found_value)
        return search_results

    def fast_version(self, input_data):
        dataset_pairs, query_keys = input_data
        dataset_dictionary = dict(dataset_pairs)
        search_results = []
        for target_key in query_keys:
            search_results.append(dataset_dictionary.get(target_key))
        return search_results

class LoopAggregationTask:
    def slow_version(self, input_data):
        total_sum = 0.0
        for number in input_data:
            total_sum += number * number
        return total_sum

    def fast_version(self, input_data):
        numpy_array = numpy.array(input_data)
        squared_array = numpy_array * numpy_array
        return numpy.sum(squared_array)

# Task 4: Top-K Selection — full sort vs. heap-based partial selection
class TopKSelectionTask:
    """
    Replace full O(n log n) sort with an O(n log k) heap-based selection.
    Returns the k largest elements (in any order).
    """

    def slow_version(self, input_data):
        numbers, k = input_data
        sorted_numbers = sorted(numbers, reverse=True)
        return sorted_numbers[:k]

    def fast_version(self, input_data):
        numbers, k = input_data
        return heapq.nlargest(k, numbers)


# Task 5: Nested-Loop Join — O(n*m) cross-join vs. O(n+m) hash join
class NestedLoopJoinTask:
    """
    Convert a nested-loop equi-join into a hash-based join.
    input_data = (left_records, right_records) where each record is (key, value).
    Returns a list of (left_value, right_value) pairs for matching keys.
    """

    def slow_version(self, input_data):
        left_records, right_records = input_data
        joined_results = []
        for left_key, left_value in left_records:
            for right_key, right_value in right_records:
                if left_key == right_key:
                    joined_results.append((left_value, right_value))
        return joined_results

    def fast_version(self, input_data):
        left_records, right_records = input_data
        right_index = {}
        for right_key, right_value in right_records:
            right_index.setdefault(right_key, []).append(right_value)
        joined_results = []
        for left_key, left_value in left_records:
            for right_value in right_index.get(left_key, []):
                joined_results.append((left_value, right_value))
        return joined_results


# Task 6: String Concatenation — repeated += vs. join
class StringConcatenationTask:
    """
    Replace O(n²) repeated string concatenation with a single O(n) join call.
    input_data is a list of strings to concatenate with a comma separator.
    """

    def slow_version(self, input_data):
        result_string = ""
        for index, word in enumerate(input_data):
            if index == 0:
                result_string += word
            else:
                result_string += "," + word
        return result_string

    def fast_version(self, input_data):
        return ",".join(input_data)


# Task 7: Redundant List Materialisation — intermediate list vs. generator pipeline
class RedundantListTask:
    """
    Eliminate intermediate list construction by chaining generators.
    Computes the sum of squares of all even numbers in input_data.
    """

    def slow_version(self, input_data):
        even_numbers = [x for x in input_data if x % 2 == 0]
        squared_numbers = [x * x for x in even_numbers]
        total = sum(squared_numbers)
        return total

    def fast_version(self, input_data):
        return sum(x * x for x in input_data if x % 2 == 0)


# Task 8: Memoization — repeated recursive computation vs. cached results
class MemoizationTask:
    """
    Introduce memoization to avoid redundant recursive Fibonacci computations.
    input_data is a list of n values for which fib(n) should be computed.
    """

    def slow_version(self, input_data):
        def fib(n):
            if n <= 1:
                return n
            return fib(n - 1) + fib(n - 2)

        return [fib(n) for n in input_data]

    def fast_version(self, input_data):
        cache = {}

        def fib(n):
            if n in cache:
                return cache[n]
            if n <= 1:
                return n
            cache[n] = fib(n - 1) + fib(n - 2)
            return cache[n]

        return [fib(n) for n in input_data]


# Task 9: Data Structure Choice — list membership test vs. set lookup
class DataStructureChoiceTask:
    """
    Replace O(n) list membership checks with O(1) set lookups.
    input_data = (universe_list, query_items).
    Returns a list of booleans indicating membership for each query item.
    """

    def slow_version(self, input_data):
        universe_list, query_items = input_data
        membership_results = []
        for item in query_items:
            membership_results.append(item in universe_list)
        return membership_results

    def fast_version(self, input_data):
        universe_list, query_items = input_data
        universe_set = set(universe_list)
        return [item in universe_set for item in query_items]


# Task 10: Multi-Pass Reduction — three separate traversals vs. single fused pass
class MultiPassReductionTask:
    """
    Fuse multiple passes over a dataset into a single traversal.
    Computes (minimum, maximum, mean) of input_data.
    """

    def slow_version(self, input_data):
        minimum_value = min(input_data)
        maximum_value = max(input_data)
        mean_value = sum(input_data) / len(input_data)
        return (minimum_value, maximum_value, mean_value)

    def fast_version(self, input_data):
        minimum_value = input_data[0]
        maximum_value = input_data[0]
        running_total = 0.0
        count = 0
        for number in input_data:
            if number < minimum_value:
                minimum_value = number
            if number > maximum_value:
                maximum_value = number
            running_total += number
            count += 1
        mean_value = running_total / count
        return (minimum_value, maximum_value, mean_value)