import time
import tracemalloc

class PerformanceEngine:
    def get_metrics(self, target_function, input_data):
        tracemalloc.start()
        start_time = time.perf_counter()
        
        try:
            result = target_function(input_data)
            end_time = time.perf_counter()
            _, peak_memory = tracemalloc.get_traced_memory()
            
            return {
                "success": True,
                "duration_seconds": end_time - start_time,
                "peak_memory_kb": peak_memory / 1024.0,
                "result": result
            }
        except Exception as execution_error:
            end_time = time.perf_counter()
            _, peak_memory = tracemalloc.get_traced_memory()
            
            return {
                "success": False,
                "duration_seconds": end_time - start_time,
                "peak_memory_kb": peak_memory / 1024.0,
                "result": str(execution_error)
            }
        finally:
            tracemalloc.stop()
