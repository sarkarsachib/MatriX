"""
Performance benchmarks for MatriX components
Measures speed, throughput, and resource usage
"""
import time
import torch
import psutil
import gc
from typing import Callable, List, Dict, Any
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BenchmarkMetrics:
    """Collect and report benchmark metrics"""
    
    def __init__(self):
        """
        Initialize a BenchmarkMetrics collector.
        
        Creates an empty list to store individual benchmark result entries and prepares the instance for recording and exporting benchmark data.
        """
        self.results = []
    
    def record(self, name: str, duration: float, metrics: Dict[str, Any] = None):
        """
        Record a benchmark entry and store it in the instance's results list.
        
        Parameters:
            name (str): Human-readable identifier for the benchmark.
            duration (float): Measured duration of the benchmark in seconds.
            metrics (Dict[str, Any], optional): Additional key/value data to include with the result; keys are merged into the stored entry. A timestamp is automatically added to each recorded result.
        """
        result = {
            'name': name,
            'duration': duration,
            'timestamp': time.time()
        }
        if metrics:
            result.update(metrics)
        self.results.append(result)
    
    def save_to_file(self, filepath: str):
        """
        Write collected benchmark results to a JSON file.
        
        Serializes the stored results as pretty-printed JSON and writes them to the given filepath, overwriting the file if it already exists.
        
        Parameters:
        	filepath (str): Filesystem path where the JSON results will be written.
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics for all recorded benchmarks.
        
        Returns:
            summary (dict): A dictionary with the following keys:
                - total_benchmarks (int): Number of recorded benchmark entries.
                - total_duration (float): Sum of all benchmark durations (seconds).
                - average_duration (float): Mean duration per benchmark (seconds).
                - min_duration (float): Shortest benchmark duration (seconds).
                - max_duration (float): Longest benchmark duration (seconds).
                - benchmarks (List[dict]): The raw list of recorded benchmark entries.
            Returns an empty dict if no results have been recorded.
        """
        if not self.results:
            return {}
        
        durations = [r['duration'] for r in self.results]
        
        return {
            'total_benchmarks': len(self.results),
            'total_duration': sum(durations),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'benchmarks': self.results
        }


class NeuralCoreBenchmarks:
    """Benchmarks for Neural Core"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NeuralCoreBenchmarks instance by constructing a QuantumInspiredNeuralCore model on CPU using the provided configuration and placing the model into evaluation mode.
        
        Parameters:
            config (Dict[str, Any]): Configuration mapping containing:
                - 'vocab_size': vocabulary size for the model
                - 'd_model': model hidden dimensionality
                - 'num_heads': number of attention heads
                - 'num_layers': number of transformer layers
                - 'num_experts': number of mixture-of-experts (MoE) experts
                - 'top_k': top-k routing parameter for MoE
                - 'max_position_embeddings': maximum positional embedding length
        
        Side effects:
            Sets self.device to CPU and assigns the instantiated model to self.neural_core in evaluation mode.
        """
        from neural_core.quantum_inspired_neural_core import QuantumInspiredNeuralCore
        
        self.device = torch.device('cpu')
        self.neural_core = QuantumInspiredNeuralCore(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            num_experts=config['num_experts'],
            top_k=config['top_k'],
            max_position_embeddings=config['max_position_embeddings']
        ).to(self.device)
        self.neural_core.eval()
    
    def benchmark_forward_pass(self, batch_size: int, seq_len: int, iterations: int = 100) -> Dict[str, Any]:
        """
        Measure average forward-pass latency and token throughput for the neural core model.
        
        Parameters:
            batch_size (int): Number of samples in the input batch.
            seq_len (int): Sequence length (tokens) per sample.
            iterations (int): Number of measured forward-pass iterations (after warmup).
        
        Returns:
            dict: Benchmark results containing:
                - 'batch_size': the provided batch size.
                - 'seq_len': the provided sequence length.
                - 'iterations': the number of measured iterations.
                - 'duration_seconds': total elapsed time for the measured iterations.
                - 'tokens_per_second': throughput in tokens processed per second.
                - 'avg_forward_time_ms': average forward-pass time per iteration in milliseconds.
        """
        import torch
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Warmup
        for _ in range(10):
            _ = self.neural_core(input_ids)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            output = self.neural_core(input_ids)
        end_time = time.time()
        
        duration = end_time - start_time
        tokens_per_second = (batch_size * seq_len * iterations) / duration
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'iterations': iterations,
            'duration_seconds': duration,
            'tokens_per_second': tokens_per_second,
            'avg_forward_time_ms': (duration / iterations) * 1000
        }
    
    def benchmark_generation(self, batch_size: int, input_len: int, 
                       output_len: int, iterations: int = 50) -> Dict[str, Any]:
        """
                       Measure generation throughput of the neural core for the given batch and sequence sizes.
                       
                       Parameters:
                           batch_size (int): Number of sequences generated concurrently per iteration.
                           input_len (int): Length of the input prompt (in tokens) provided to the model.
                           output_len (int): Maximum number of tokens to generate per sequence.
                           iterations (int): Number of generation calls to time for the benchmark.
                       
                       Returns:
                           dict: Benchmark results containing:
                               - 'batch_size': the provided batch_size
                               - 'input_len': the provided input_len
                               - 'output_len': the provided output_len
                               - 'iterations': the provided iterations
                               - 'duration_seconds': total elapsed time for all iterations in seconds
                               - 'tokens_per_second': throughput computed as (batch_size * output_len * iterations) / duration_seconds
                               - 'avg_generation_time_ms': average generation time per iteration in milliseconds
                       """
        import torch
        input_ids = torch.randint(0, 1000, (batch_size, input_len))
        
        # Warmup
        for _ in range(5):
            _ = self.neural_core.generate(
                input_ids,
                max_length=output_len,
                temperature=0.8,
                top_k=10,
                top_p=0.9
            )
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            generated = self.neural_core.generate(
                input_ids,
                max_length=output_len,
                temperature=0.8,
                top_k=10,
                top_p=0.9
            )
        end_time = time.time()
        
        duration = end_time - start_time
        tokens_per_second = (batch_size * output_len * iterations) / duration
        
        return {
            'batch_size': batch_size,
            'input_len': input_len,
            'output_len': output_len,
            'iterations': iterations,
            'duration_seconds': duration,
            'tokens_per_second': tokens_per_second,
            'avg_generation_time_ms': (duration / iterations) * 1000
        }
    
    def benchmark_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Measure memory usage for a single forward pass of the neural core model.
        
        Parameters:
            batch_size (int): Number of samples in the input batch.
            seq_len (int): Sequence length for each input sample.
        
        Returns:
            dict: Benchmark results containing:
                - 'batch_size' (int): The provided batch size.
                - 'seq_len' (int): The provided sequence length.
                - 'current_memory_mb' (float): Memory currently used during the measurement in megabytes.
                - 'peak_memory_mb' (float): Peak memory recorded during the measurement in megabytes.
        """
        import torch
        import tracemalloc
        
        tracemalloc.start()
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        output = self.neural_core(input_ids)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'current_memory_mb': current / 1024 / 1024,
            'peak_memory_mb': peak / 1024 / 1024
        }


class MemorySystemBenchmarks:
    """Benchmarks for Memory System"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MemorySystemBenchmarks instance and create an InfiniteAdaptiveMemorySystem configured from `config`.
        
        Parameters:
            config (Dict[str, Any]): Configuration mapping with required keys:
                - d_model: model dimensionality used for embedding sizes.
                - ustm_capacity: capacity limit for the ultra short-term memory.
                - awm_capacity: capacity limit for the adaptive working memory.
                - ltkb_path: filesystem path for the long-term knowledge base storage.
        """
        from memory_system.infinite_adaptive_memory import InfiniteAdaptiveMemorySystem
        
        self.memory_system = InfiniteAdaptiveMemorySystem(
            d_model=config['d_model'],
            ustm_capacity=config['ustm_capacity'],
            awm_capacity=config['awm_capacity'],
            ltkb_path=config['ltkb_path']
        )
    
    def benchmark_ustm_operations(self, operations: int = 1000) -> Dict[str, Any]:
        """
        Measure throughput and average latency of Ultra Short-Term Memory (USTM) add_entry operations.
        
        Parameters:
            operations (int): Number of add_entry calls to execute during the benchmark.
        
        Returns:
            result (dict): Benchmark metrics containing:
                - operations (int): The number of operations performed.
                - duration_seconds (float): Total elapsed time for the benchmark run in seconds.
                - ops_per_second (float): Measured operations per second.
                - avg_op_time_us (float): Average time per operation in microseconds.
        """
        import time
        from memory_system.infinite_adaptive_memory import UltraShortTermMemory
        
        ustm = UltraShortTermMemory(capacity=100)
        
        # Warmup
        for _ in range(100):
            ustm.add_entry({'test': 'data'})
        
        # Benchmark
        start_time = time.time()
        for _ in range(operations):
            ustm.add_entry({'test': 'data', 'timestamp': time.time()})
        end_time = time.time()
        
        duration = end_time - start_time
        ops_per_second = operations / duration
        
        return {
            'operations': operations,
            'duration_seconds': duration,
            'ops_per_second': ops_per_second,
            'avg_op_time_us': (duration / operations) * 1000000
        }
    
    def benchmark_awm_operations(self, batch_size: int, iterations: int = 100) -> Dict[str, Any]:
        """
        Measure AWM (adaptive working memory) operation throughput and latency for a batch of query embeddings.
        
        Parameters:
            batch_size (int): Number of query embeddings in each batch used for the benchmark.
            iterations (int): Number of timed AWM calls to perform (after warmup).
        
        Returns:
            result (Dict[str, Any]): Dictionary containing:
                - 'batch_size': the provided batch_size.
                - 'iterations': the provided iterations.
                - 'duration_seconds': total elapsed time for the timed iterations.
                - 'ops_per_second': completed iterations divided by total elapsed time.
                - 'avg_op_time_ms': average time per operation in milliseconds.
        """
        import torch
        
        query_embeddings = torch.randn(batch_size, 1, self.memory_system.d_model)
        
        # Warmup
        for _ in range(10):
            _ = self.memory_system.awm(query_embeddings.unsqueeze(1))
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            output = self.memory_system.awm(query_embeddings.unsqueeze(1))
        end_time = time.time()
        
        duration = end_time - start_time
        ops_per_second = iterations / duration
        
        return {
            'batch_size': batch_size,
            'iterations': iterations,
            'duration_seconds': duration,
            'ops_per_second': ops_per_second,
            'avg_op_time_ms': (duration / iterations) * 1000
        }
    
    def benchmark_ltkb_operations(self, operations: int = 100) -> Dict[str, Any]:
        """
        Measure throughput of add and get operations on the LTKB store.
        
        Parameters:
            operations (int): Number of add and get operations to perform during each timed benchmark.
        
        Returns:
            result (Dict[str, Any]): Dictionary containing:
                - 'operations': the number of operations performed.
                - 'add_duration_seconds': total time spent performing add operations.
                - 'get_duration_seconds': total time spent performing get operations.
                - 'adds_per_second': measured add throughput (operations / second).
                - 'gets_per_second': measured get throughput (operations / second).
        """
        import time
        
        # Warmup
        for _ in range(10):
            self.memory_system.ltkb.add_concept(f'concept{_}', {'test': 'data'})
        
        # Benchmark adds
        start_time = time.time()
        for _ in range(operations):
            self.memory_system.ltkb.add_concept(f'concept{_}', {'test': 'data'})
        end_time = time.time()
        
        add_duration = end_time - start_time
        adds_per_second = operations / add_duration
        
        # Benchmark gets
        start_time = time.time()
        for _ in range(operations):
            _ = self.memory_system.ltkb.get_concept(f'concept{_}')
        end_time = time.time()
        
        get_duration = end_time - start_time
        gets_per_second = operations / get_duration
        
        return {
            'operations': operations,
            'add_duration_seconds': add_duration,
            'get_duration_seconds': get_duration,
            'adds_per_second': adds_per_second,
            'gets_per_second': gets_per_second
        }


class OutputEngineBenchmarks:
    """Benchmarks for Output Engine"""
    
    def __init__(self):
        """
        Initialize the OutputEngineBenchmarks instance and instantiate the OutputEngine used for benchmark operations.
        
        The created OutputEngine object is stored on `self.output_engine`.
        """
        from output_engine.output_system import OutputEngine
        self.output_engine = OutputEngine()
    
    def benchmark_text_generation(self, iterations: int = 1000) -> Dict[str, Any]:
        """
        Measure throughput and latency of the output engine's text generation.
        
        Parameters:
            iterations (int): Number of generation calls to perform during the timed benchmark (default 1000).
        
        Returns:
            results (Dict[str, Any]): Benchmark metrics including:
                - iterations: The number of timed generation iterations.
                - duration_seconds: Total elapsed time in seconds for the timed iterations.
                - ops_per_second: Completed generations per second.
                - avg_gen_time_us: Average generation time per call in microseconds.
        """
        import time
        
        self.output_engine.set_mode('text')
        test_data = {'content': 'Test response content', 'type': 'response'}
        
        # Warmup
        for _ in range(100):
            _ = self.output_engine.generate_response(test_data, 'text')
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            output = self.output_engine.generate_response(test_data, 'text')
        end_time = time.time()
        
        duration = end_time - start_time
        ops_per_second = iterations / duration
        
        return {
            'iterations': iterations,
            'duration_seconds': duration,
            'ops_per_second': ops_per_second,
            'avg_gen_time_us': (duration / iterations) * 1000000
        }
    
    def benchmark_code_generation(self, iterations: int = 1000) -> Dict[str, Any]:
        """
        Measure throughput and average latency for the output engine's code generation mode.
        
        Parameters:
            iterations (int): Number of code generation calls to perform during the timed benchmark.
        
        Returns:
            result (Dict[str, Any]): Dictionary containing:
                - iterations (int): The number of iterations executed.
                - duration_seconds (float): Total elapsed time for the timed iterations, in seconds.
                - ops_per_second (float): Completed generations per second.
                - avg_gen_time_us (float): Average generation time per call, in microseconds.
        """
        import time
        
        self.output_engine.set_mode('code')
        test_data = {
            'language': 'python',
            'code_type': 'function',
            'name': 'test_func',
            'body': 'pass'
        }
        
        # Warmup
        for _ in range(100):
            _ = self.output_engine.generate_response(test_data, 'code')
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            output = self.output_engine.generate_response(test_data, 'code')
        end_time = time.time()
        
        duration = end_time - start_time
        ops_per_second = iterations / duration
        
        return {
            'iterations': iterations,
            'duration_seconds': duration,
            'ops_per_second': ops_per_second,
            'avg_gen_time_us': (duration / iterations) * 1000000
        }
    
    def benchmark_mode_switching(self, iterations: int = 5000) -> Dict[str, Any]:
        """
        Measure throughput of switching the OutputEngine between predefined modes.
        
        Returns:
            dict: Benchmark results with the following keys:
                - iterations (int): Number of benchmark iterations performed.
                - modes_count (int): Number of distinct modes cycled.
                - duration_seconds (float): Total elapsed time of the benchmark in seconds.
                - switches_per_second (float): Number of mode switches performed per second.
                - avg_switch_time_us (float): Average time per mode switch in microseconds.
        """
        import time
        
        modes = ['text', 'code', 'audio', 'command']
        
        # Warmup
        for _ in range(100):
            for mode in modes:
                self.output_engine.set_mode(mode)
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            for mode in modes:
                self.output_engine.set_mode(mode)
        end_time = time.time()
        
        duration = end_time - start_time
        switches_per_second = (iterations * len(modes)) / duration
        
        return {
            'iterations': iterations,
            'modes_count': len(modes),
            'duration_seconds': duration,
            'switches_per_second': switches_per_second,
            'avg_switch_time_us': (duration / (iterations * len(modes))) * 1000000
        }


class ScalabilityBenchmarks:
    """Benchmarks for system scalability"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the benchmark runner and apply configuration overrides to the SathikAI instance.
        
        Parameters:
            config (Dict[str, Any]): Configuration overrides to merge into the created SathikAI instance's config.
        """
        from main import SathikAI
        
        self.sathik_ai = SathikAI(config_path=None)
        self.sathik_ai.config.update(config)
    
    def benchmark_batch_processing(self, batch_sizes: List[int], iterations: int = 10) -> Dict[str, Any]:
        """
        Run throughput benchmarks for processing queries across multiple batch sizes.
        
        Per batch size, performs a short warmup then measures processing time across `iterations` passes and computes throughput and average latency.
        
        Parameters:
            batch_sizes (List[int]): Sequence of batch sizes to benchmark; each value defines how many queries are processed per iteration.
            iterations (int): Number of iterations to run the timed benchmark for each batch size.
        
        Returns:
            Dict[int, Dict[str, Any]]: Mapping from each batch size to a metrics dictionary containing:
                - 'batch_size': the tested batch size
                - 'iterations': number of iterations performed
                - 'duration_seconds': total measured duration for the benchmark loop
                - 'queries_per_second': measured throughput (queries/sec)
                - 'avg_query_time_ms': average time per query in milliseconds
        """
        import time
        
        results = {}
        
        for batch_size in batch_sizes:
            queries = [f"Query {i}" for i in range(batch_size)]
            
            # Warmup
            for _ in range(3):
                for query in queries:
                    self.sathik_ai.process_query(
                        query=query,
                        user_id=f"user_{i}",
                        mode="trained",
                        submode="normal"
                    )
            
            # Benchmark
            start_time = time.time()
            for _ in range(iterations):
                for i, query in enumerate(queries):
                    self.sathik_ai.process_query(
                        query=query,
                        user_id=f"user_{i}",
                        mode="trained",
                        submode="normal"
                    )
            end_time = time.time()
            
            duration = end_time - start_time
            total_ops = batch_size * iterations
            queries_per_second = total_ops / duration
            
            results[batch_size] = {
                'batch_size': batch_size,
                'iterations': iterations,
                'duration_seconds': duration,
                'queries_per_second': queries_per_second,
                'avg_query_time_ms': (duration / total_ops) * 1000
            }
        
        return results


def run_performance_benchmarks(config: Dict[str, Any]):
    """
    Orchestrates and runs the full suite of performance benchmarks, records their results, saves them to a JSON file adjacent to this module, and returns an aggregate summary.
    
    Parameters:
        config (Dict[str, Any]): Configuration values passed to benchmarked components (examples: vocabulary size and model dimensions for the neural core, d_model/ustm/awm capacities and ltkb path for the memory system, and any component-specific settings required to initialize benchmarks).
    
    Returns:
        Dict[str, Any]: Summary of collected benchmarks including keys such as `total_benchmarks`, `total_duration`, `average_duration`, `min_duration`, `max_duration`, and the raw recorded results.
    """
    metrics = BenchmarkMetrics()
    
    # Neural Core benchmarks
    nc_benchmarks = NeuralCoreBenchmarks(config)
    
    metrics.record("Neural Core Forward Pass", 
                  nc_benchmarks.benchmark_forward_pass(batch_size=4, seq_len=256),
                  {'tokens_per_second': nc_benchmarks.benchmark_forward_pass(4, 256)['tokens_per_second']})
    
    metrics.record("Neural Core Generation",
                  nc_benchmarks.benchmark_generation(batch_size=2, input_len=10, output_len=64),
                  {'tokens_per_second': nc_benchmarks.benchmark_generation(2, 10, 64)['tokens_per_second']})
    
    metrics.record("Neural Core Memory Usage",
                  nc_benchmarks.benchmark_memory_usage(batch_size=8, seq_len=512),
                  {'peak_memory_mb': nc_benchmarks.benchmark_memory_usage(8, 512)['peak_memory_mb']})
    
    # Memory System benchmarks
    ms_benchmarks = MemorySystemBenchmarks(config)
    
    metrics.record("USTM Operations", ms_benchmarks.benchmark_ustm_operations(operations=10000))
    metrics.record("AWM Operations", ms_benchmarks.benchmark_awm_operations(batch_size=4, iterations=100))
    metrics.record("LTKB Operations", ms_benchmarks.benchmark_ltkb_operations(operations=100))
    
    # Output Engine benchmarks
    oe_benchmarks = OutputEngineBenchmarks()
    
    metrics.record("Text Generation", oe_benchmarks.benchmark_text_generation(iterations=5000))
    metrics.record("Code Generation", oe_benchmarks.benchmark_code_generation(iterations=5000))
    metrics.record("Mode Switching", oe_benchmarks.benchmark_mode_switching(iterations=10000))
    
    # Scalability benchmarks
    scal_benchmarks = ScalabilityBenchmarks(config)
    
    batch_results = scal_benchmarks.benchmark_batch_processing(batch_sizes=[1, 4, 8, 16], iterations=5)
    for batch_size, result in batch_results.items():
        metrics.record(f"Batch Processing (size={batch_size})", result)
    
    # Save results
    import json
    output_file = Path(__file__).parent / 'performance_baseline.json'
    metrics.save_to_file(str(output_file))
    
    summary = metrics.get_summary()
    print("\n=== Performance Benchmark Summary ===")
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Total duration: {summary['total_duration']:.2f}s")
    print(f"Average duration: {summary['average_duration']:.4f}s")
    print(f"Min duration: {summary['min_duration']:.4f}s")
    print(f"Max duration: {summary['max_duration']:.4f}s")
    print(f"\nResults saved to: {output_file}")
    
    return summary


if __name__ == '__main__':
    # Test configuration
    test_config = {
        'vocab_size': 10000,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'num_experts': 8,
        'top_k': 4,
        'max_position_embeddings': 1024,
        'ustm_capacity': 50,
        'awm_capacity': 100,
        'ltkb_path': '/tmp/benchmark_ltkb.json'
    }
    
    run_performance_benchmarks(test_config)