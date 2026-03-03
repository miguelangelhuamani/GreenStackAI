import numpy

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
