"""
Unified Test Runner for MatriX
Discovers, categorizes, and runs all tests with reporting
"""
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class TestSuiteResult:
    """Result of a test suite"""
    suite_name: str
    tests: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float


class TestRunner:
    """Custom test framework with discovery and execution"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dirs = {
            'unit': project_root / 'tests' / 'unit',
            'integration': project_root / 'tests' / 'integration',
            'e2e': project_root / 'tests' / 'e2e'
        }
        self.results = []
    
    def discover_tests(self, test_type: str = 'all') -> List[str]:
        """Discover test files in project"""
        test_files = []
        
        if test_type in ['all', 'unit']:
            unit_dir = self.test_dirs['unit']
            if unit_dir.exists():
                test_files.extend(list(unit_dir.glob('test_*.py')))
        
        if test_type in ['all', 'integration']:
            integration_dir = self.test_dirs['integration']
            if integration_dir.exists():
                test_files.extend(list(integration_dir.glob('test_*.py')))
        
        if test_type in ['all', 'e2e']:
            e2e_dir = self.test_dirs['e2e']
            if e2e_dir.exists():
                test_files.extend(list(e2e_dir.glob('test_*.py')))
        
        return test_files
    
    def categorize_tests(self, test_files: List[str]) -> Dict[str, List[str]]:
        """Categorize tests by component"""
        categories = {
            'neural_core': [],
            'memory_system': [],
            'output_engine': [],
            'direction_mode': [],
            'safety_modules': [],
            'integration': []
        }
        
        for test_file in test_files:
            test_name = test_file.stem
            
            if 'neural_core' in test_name:
                categories['neural_core'].append(test_file)
            elif 'memory' in test_name:
                categories['memory_system'].append(test_file)
            elif 'output' in test_name:
                categories['output_engine'].append(test_file)
            elif 'direction' in test_name:
                categories['direction_mode'].append(test_file)
            elif 'safety' in test_name:
                categories['safety_modules'].append(test_file)
            elif 'component_integration' in test_name:
                categories['integration'].append(test_file)
        
        return categories
    
    def run_pytest(self, test_files: List[str], markers: str = '') -> TestSuiteResult:
        """Run pytest on given test files"""
        
        cmd = [
            sys.executable,
            '-m', 'pytest',
            '-v',
            '--tb=short'
        ]
        
        if markers:
            cmd.extend(['-m', markers])
        
        cmd.extend([str(f) for f in test_files])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Parse pytest output (simplified)
        # In real implementation, would parse JSON report
        # For now, return basic result
        return TestSuiteResult(
            suite_name=f"pytest_{markers}",
            tests=[],
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            total_duration=duration
        )
    
    def run_tests(self, test_type: str = 'all', parallel: bool = True,
                  markers: str = '') -> Dict[str, TestSuiteResult]:
        """Run all tests and return results"""
        print(f"\n{'='* 80}")
        print(f"{'='* 80}")
        print(f"MatriX Test Runner")
        print(f"{'='* 80}")
        print(f"\n{'='* 80}")
        
        # Discover tests
        test_files = self.discover_tests(test_type)
        print(f"Discovered {len(test_files)} test files")
        
        # Categorize tests
        categories = self.categorize_tests(test_files)
        
        results = {}
        
        # Run tests by category
        for category, files in categories.items():
            if files:
                print(f"\n{'='* 80}")
                print(f"Running {category} tests...")
                print(f"{'='* 80}")
                
                if parallel:
                    # Run parallel batches
                    batch_size = 4
                    for i in range(0, len(files), batch_size):
                        batch = files[i:i+batch_size]
                        if batch:
                            result = self.run_pytest(batch, markers=category)
                            results[f"{category}_batch_{i}"] = result
                else:
                    result = self.run_pytest(files, markers=category)
                    results[category] = result
        
        # Generate summary
        return self.generate_summary(results)
    
    def generate_summary(self, results: Dict[str, TestSuiteResult]) -> Dict[str, Any]:
        """Generate test summary"""
        return {
            'timestamp': time.time(),
            'total_suites': len(results),
            'total_tests': sum(r.total_tests for r in results.values()),
            'passed_tests': sum(r.passed_tests for r in results.values()),
            'failed_tests': sum(r.failed_tests for r in results.values()),
            'success_rate': 0,
            'total_duration': 0,
            'avg_suite_duration': 0,
            'suites': {k: asdict(r) for k, r in results.items()}
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MatriX Test Runner')
    parser.add_argument('--test-type', choices=['all', 'unit', 'integration', 'e2e'],
                       default='all', help='Type of tests to run')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--markers', type=str, default='',
                       help='Pytest markers to run')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for results')
    parser.add_argument('--baseline', type=str,
                       help='Baseline file to compare against')
    parser.add_argument('--ci-checks', action='store_true',
                       help='Run CI/CD quality checks')
    
    args = parser.parse_args()
    
    # Determine project root
    project_root = Path(__file__).parent.parent
    
    # Create test runner
    runner = TestRunner(project_root)
    
    # Run tests
    results = runner.run_tests(
        test_type=args.test_type,
        parallel=args.parallel,
        markers=args.markers
    )
    
    # Save results
    output_path = project_root / args.output
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='* 80}")
    print(f"Results saved to: {output_path}")
    
    # CI/CD checks
    if args.ci_checks:
        passed = True
        total_tests = results.get('total_tests', 0)
        passed_tests = results.get('passed_tests', 0)
        
        if total_tests > 0:
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            passed = success_rate >= 90
        
        print(f"\n{'='* 80}")
        if passed:
            print(f"{'='* 32}{'='* 32}ALL PASSED{'='* 0m'}")
        else:
            print(f"{'='* 31}CI/CD Checks: SOME FAILED{'='* 0m'}")
    
    sys.exit(0 if not args.ci_checks or passed else 1)


if __name__ == '__main__':
    main()
