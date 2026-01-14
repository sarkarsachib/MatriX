"""
Unified Benchmark Runner for MatriX
Orchestrates running all benchmark suites and generating reports
"""
import sys
import argparse
import time
from pathlib import Path
import json
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BenchmarkRunner:
    """Run all benchmarks and generate comprehensive reports"""
    
    def __init__(self):
        """
        Initialize a BenchmarkRunner instance.
        
        Sets up an empty results dictionary and records the current time as the run start time for computing elapsed durations.
        """
        self.results = {}
        self.start_time = time.time()
    
    def run_performance_benchmarks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run performance benchmarks using the provided configuration.
        
        Parameters:
            config (Dict[str, Any]): Configuration values controlling benchmark parameters (e.g., model dimensions, memory capacities, file paths).
        
        Returns:
            Dict[str, Any]: A dictionary of performance benchmark results.
        
        Notes:
            Prints progress and elapsed time to stdout as benchmarks run.
        """
        from benchmarks.performance_benchmarks import run_performance_benchmarks
        
        print("\n" + "="*80)
        print("Running Performance Benchmarks")
        print("="*80)
        
        results = run_performance_benchmarks(config)
        
        duration = time.time() - self.start_time
        print(f"\nPerformance benchmarks completed in {duration:.2f}s")
        
        return results
    
    def run_quality_benchmarks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the quality benchmark suite using the provided configuration.
        
        @param config: Mapping of benchmark settings (e.g., dataset paths, sample sizes, thresholds) used to control the quality benchmark run.
        @returns: dict containing collected quality benchmark results, typically a mapping of metric names to values and nested sub-report structures.
        """
        from benchmarks.quality_benchmarks import run_quality_benchmarks
        
        print("\n" + "="*80)
        print("Running Quality Benchmarks")
        print("="*80)
        
        results = run_quality_benchmarks(config)
        
        duration = time.time() - self.start_time
        print(f"\nQuality benchmarks completed in {duration:.2f}s")
        
        return results
    
    def run_scalability_benchmarks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the scalability benchmark suite using the provided configuration.
        
        Parameters:
            config (Dict[str, Any]): Benchmark configuration options (e.g., batch sizes, concurrency levels, resource limits).
        
        Returns:
            Dict[str, Any]: Collected scalability benchmark results keyed by test identifiers, containing measured metrics.
        """
        from benchmarks.scalability.benchmarks import run_scalability_benchmarks
        
        print("\n" + "="*80)
        print("Running Scalability Benchmarks")
        print("="*80)
        
        results = run_scalability_benchmarks(config)
        
        duration = time.time() - self.start_time
        print(f"\nScalability benchmarks completed in {duration:.2f}s")
        
        return results
    
    def generate_report(self, results: Dict[str, Any], output_path: str):
        """
        Builds a consolidated benchmark report from provided results and writes it as JSON to the specified file path.
        
        Parameters:
            results (Dict[str, Any]): Mapping that may contain 'performance', 'quality', and 'scalability' keys with their respective benchmark data; missing keys will be represented as empty dictionaries in the report.
            output_path (str): Filesystem path where the JSON report will be written; parent directories will be created if necessary.
        
        Returns:
            Dict[str, Any]: The report dictionary containing:
                - `timestamp`: epoch time when the report was generated,
                - `total_duration_seconds`: seconds elapsed since the runner's start_time,
                - `performance`, `quality`, `scalability`: dictionaries with the corresponding results (empty if not provided).
        """
        report = {
            'timestamp': time.time(),
            'total_duration_seconds': time.time() - self.start_time,
            'performance': results.get('performance', {}),
            'quality': results.get('quality', {}),
            'scalability': results.get('scalability', {})
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nBenchmark report saved to: {output_file}")
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """
        Print a human-readable summary of the benchmark report to standard output.
        
        Parameters:
            report (Dict[str, Any]): Aggregated benchmark report containing at minimum
                the key `total_duration_seconds`. May also include `performance`,
                `quality`, and `scalability` sections with their respective metric
                dictionaries. The `performance` section may contain `neural_core`,
                `memory_system`, `output_engine`, and `direction_mode` sub-dictionaries.
                The `quality` section should contain percentage metrics (e.g.,
                `response_accuracy_percent`). The `scalability` section may contain
                batch keys prefixed with `batch_`, concurrency keys like `concurrent_1`,
                `concurrent_4`, `concurrent_8`, and optional `memory_growth` and
                `cache_effectiveness` entries.
        """
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        total_duration = report['total_duration_seconds']
        print(f"Total Duration: {total_duration:.2f}s")
        
        if 'performance' in report:
            perf = report['performance']
            print(f"\nPerformance Metrics:")
            
            if 'neural_core' in perf:
                nc = perf['neural_core']
                print(f"  Neural Core Forward: {nc.get('forward_pass_tokens_per_second', 'N/A'):.2f} tokens/sec")
                print(f"  Neural Core Generation: {nc.get('generation_tokens_per_second', 'N/A'):.2f} tokens/sec")
                print(f"  Neural Core Memory: {nc.get('peak_memory_mb', 'N/A'):.2f} MB")
            
            if 'memory_system' in perf:
                ms = perf['memory_system']
                print(f"  USTM: {ms.get('ustm_ops_per_second', 'N/A'):.2f} ops/sec")
                print(f"  AWM: {ms.get('awm_ops_per_second', 'N/A'):.2f} ops/sec")
                print(f"  LTKB: {ms.get('ltkb_adds_per_second', 'N/A'):.2f} adds/sec")
            
            if 'output_engine' in perf:
                oe = perf['output_engine']
                print(f"  Text: {oe.get('text_ops_per_second', 'N/A'):.2f} ops/sec")
                print(f"  Code: {oe.get('code_ops_per_second', 'N/A'):.2f} ops/sec")
                print(f"  Mode Switch: {oe.get('mode_switches_per_second', 'N/A'):.2f} switches/sec")
            
            if 'direction_mode' in perf:
                dm = perf['direction_mode']
                print(f"  Query Analysis: {dm.get('query_analysis_ms', 'N/A'):.2f} ms")
                print(f"  Search: {dm.get('search_ms', 'N/A'):.2f} ms")
                print(f"  Style: {dm.get('style_application_ms', 'N/A'):.2f} ms")
        
        if 'quality' in report:
            quality = report['quality']
            print(f"\nQuality Metrics:")
            print(f"  Response Accuracy: {quality.get('response_accuracy_percent', 'N/A'):.1f}%")
            print(f"  Confidence Calibration: {quality.get('confidence_calibration_percent', 'N/A'):.1f}%")
            print(f"  Citation Quality: {quality.get('citation_quality_percent', 'N/A'):.1f}%")
            print(f"  Knowledge Base Hit Rate: {quality.get('knowledge_hit_rate_percent', 'N/A'):.1f}%")
        
        if 'scalability' in report:
            scal = report['scalability']
            print(f"\nScalability Metrics:")
            print(f"  Batch Processing:")
            
            batch_sizes = sorted([k.replace('batch_', '') for k in scal.keys() if 'batch_' in k])
            for batch_size in batch_sizes:
                if f'batch_{batch_size}' in scal:
                    result = scal[f'batch_{batch_size}']
                    print(f"    Size {batch_size}: {result.get('queries_per_second', 'N/A'):.2f} queries/sec")
            
            if 'concurrent_1' in scal:
                result = scal['concurrent_1']
                print(f"  Concurrent (1 req/s): {result.get('requests_per_second', 'N/A'):.2f} req/sec")
                print(f"    Latency: {result.get('avg_request_time_ms', 'N/A'):.2f} ms")
            
            if 'concurrent_4' in scal:
                result = scal['concurrent_4']
                print(f"  Concurrent (4 req/s): {result.get('requests_per_second', 'N/A'):.2f} req/sec")
                print(f"    Latency: {result.get('avg_request_time_ms', 'N/A'):.2f} ms")
            
            if 'concurrent_8' in scal:
                result = scal['concurrent_8']
                print(f"  Concurrent (8 req/s): {result.get('requests_per_second', 'N/A'):.2f} req/sec")
                print(f"    Latency: {result.get('avg_request_time_ms', 'N/A'):.2f} ms")
            
            if 'memory_growth' in scal:
                mem = scal['memory_growth']
                print(f"  Memory Growth: {mem.get('memory_growth_mb', 'N/A'):.4f} MB per 1000 ops")
            
            if 'cache_effectiveness' in scal:
                cache = scal['cache_effectiveness']
                print(f"  Cache Effectiveness: {cache.get('speedup_percent', 'N/A'):.1f}%")
        
        print("="*80)
        print(f"\nReport Generated: {output_path}")


def main():
    """
    Run the CLI entry point that parses arguments, executes selected benchmark suites, and writes a JSON report.
    
    Parses command-line options to select benchmarks (performance, quality, scalability, or all), optionally loads and merges a JSON configuration file, instantiates BenchmarkRunner, runs the requested benchmark domains, generates a consolidated report written to the provided output path, and prints a completion banner.
    """
    parser = argparse.ArgumentParser(description='MatriX Benchmark Runner')
    parser.add_argument('--config', type=str, default='benchmark_config.json',
                       help='Configuration file for benchmarks')
    parser.add_argument('--output', type=str, default='benchmark_report.json',
                       help='Output file for benchmark report')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--quality', action='store_true',
                       help='Run quality benchmarks')
    parser.add_argument('--scalability', action='store_true',
                       help='Run scalability benchmarks')
    parser.add_argument('--all', action='store_true',
                       help='Run all benchmarks')
    
    args = parser.parse_args()
    
    if not any([args.performance, args.quality, args.scalability, args.all]):
        args.all = True
    
    # Default configuration
    config = {
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
    
    # Load custom config if provided
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    runner = BenchmarkRunner()
    
    if args.performance or args.all:
        runner.results['performance'] = runner.run_performance_benchmarks(config)
    
    if args.quality or args.all:
        runner.results['quality'] = runner.run_quality_benchmarks(config)
    
    if args.scalability or args.all:
        runner.results['scalability'] = runner.run_scalability_benchmarks(config)
    
    # Generate report
    runner.generate_report(runner.results, args.output)
    
    print("\n" + "="*80)
    print("BENCHMARK RUNNER COMPLETED")
    print("="*80)


if __name__ == '__main__':
    main()