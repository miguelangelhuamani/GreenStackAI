import time
import resource
import sys
import statistics

class PerformanceEngine:
    def _get_peak_memory_kb(self):
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return maxrss / 1024.0
        else:
            return float(maxrss)

    def get_metrics(self, target_function, input_data):
        start_time = time.perf_counter()
        
        try:
            result = target_function(input_data)
            end_time = time.perf_counter()
            end_mem = self._get_peak_memory_kb()
            
            return {
                "success": True,
                "duration_seconds": end_time - start_time,
                "peak_memory_kb": end_mem,
                "result": result
            }
        except Exception as execution_error:
            end_time = time.perf_counter()
            end_mem = self._get_peak_memory_kb()
            
            return {
                "success": False,
                "duration_seconds": end_time - start_time,
                "peak_memory_kb": end_mem,
                "result": None,
                "error": str(execution_error)
            }

    def get_median_metrics(self, target_function, input_data, n_warmup=2, n_measured=7):
        for _ in range(n_warmup):
            try:
                target_function(input_data)
            except Exception:
                pass
        
        durations = []
        memories = []
        last_result = None
        
        for _ in range(n_measured):
            metrics = self.get_metrics(target_function, input_data)
            if not metrics["success"]:
                return {
                    "success": False,
                    "median_duration_seconds": 0.0,
                    "median_peak_memory_kb": 0.0,
                    "all_durations": [],
                    "all_memories": [],
                    "result": None,
                    "error": metrics.get("error", "Unknown error")
                }
            durations.append(metrics["duration_seconds"])
            memories.append(metrics["peak_memory_kb"])
            last_result = metrics["result"]
            
        return {
            "success": True,
            "median_duration_seconds": statistics.median(durations) if durations else 0.0,
            "median_peak_memory_kb": statistics.median(memories) if memories else 0.0,
            "all_durations": durations,
            "all_memories": memories,
            "result": last_result
        }
