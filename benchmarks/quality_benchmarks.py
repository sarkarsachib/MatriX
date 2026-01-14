"""
Quality benchmarks for MatriX
Tests response accuracy, confidence calibration, and citation quality
"""
import time
import random
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ResponseAccuracyBenchmark:
    """Test response accuracy and relevance"""
    
    def __init__(self):
        """
        Initialize the benchmark with predefined test queries categorized by type.
        
        Creates `self.test_queries`, a dictionary mapping category names to lists of (question, expected_answer) tuples. Categories provided:
        - 'factual': short fact-based question/answer pairs (e.g., capitals, arithmetic, authorship, boiling point).
        - 'explanatory': conceptual prompts paired with expected concise explanations.
        """
        self.test_queries = {
            'factual': [
                ("What is the capital of France?", "Paris"),
                ("What is 2 + 2?", "4"),
                ("Who wrote Romeo and Juliet?", "Shakespeare"),
                ("What is the boiling point of water?", "100°C")
            ],
            'explanatory': [
                ("Explain how neural networks work", "Neural networks learn patterns"),
                ("What is machine learning?", "Algorithms that improve from data"),
                ("How does a transformer work?", "Attention mechanisms")
            ]
        }
    
    def benchmark_response_accuracy(self, responses: List[str]) -> Dict[str, Any]:
        """
        Evaluate factual response accuracy against the benchmark's predefined factual test queries.
        
        Parameters:
            responses (List[str]): Candidate response texts to be checked for containing expected factual answers.
        
        Returns:
            Dict[str, Any]: A dictionary with:
                - total_queries: Number of responses evaluated.
                - correct_responses: Count of responses that contained the expected factual answers (case-insensitive substring match).
                - accuracy_percent: Percentage of correct responses (0-100).
                - accuracy_score: Normalized accuracy (0.0-1.0).
        """
        correct = 0
        total = len(responses)
        
        for query, expected_answer in self.test_queries['factual']:
            # Find matching response
            matching_response = None
            for response in responses:
                if expected_answer.lower() in response.lower():
                    matching_response = response
                    break
            
            if matching_response:
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'total_queries': total,
            'correct_responses': correct,
            'accuracy_percent': accuracy,
            'accuracy_score': accuracy / 100
        }


class ConfidenceCalibrationBenchmark:
    """Test confidence score calibration"""
    
    def __init__(self):
        """
        Initialize the ConfidenceCalibrationBenchmark with predefined calibration test cases.
        
        Each test case is a dict containing:
        - `query`: a short description of the test scenario.
        - `true_answer`: whether the expected model answer is correct in that scenario.
        - `expected_confidence`: the target confidence level (0.0 to 1.0) the model should exhibit for the query.
        """
        self.confidence_tests = [
            {'query': 'High confidence question', 'true_answer': True, 'expected_confidence': 0.9},
            {'query': 'Medium confidence question', 'true_answer': True, 'expected_confidence': 0.7},
            {'query': 'Low confidence question', 'true_answer': True, 'expected_confidence': 0.5},
            {'query': 'Uncertain question', 'true_answer': True, 'expected_confidence': 0.3}
        ]
    
    def benchmark_confidence_calibration(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate whether provided prediction confidence scores align with expected calibration thresholds.
        
        Parameters:
            predictions (List[Dict[str, Any]]): Sequence of prediction objects; each prediction is expected to include a numeric 'confidence' value. Predictions are matched by index against the instance's predefined `confidence_tests`.
        
        Returns:
            Dict[str, Any]: Summary of the calibration check containing:
                - total_predictions (int): Number of predictions evaluated.
                - calibration_errors (int): Count of predictions whose confidence deviated beyond the tolerated threshold relative to the test expectation.
                - calibration_score (float): Fraction of well-calibrated predictions (0.0 to 1.0).
                - well_calibrated_percent (float): calibration_score expressed as a percentage.
        """
        calibration_errors = 0
        total = len(predictions)
        
        for i, test in enumerate(self.confidence_tests):
            pred = predictions[i] if i < len(predictions) else None
            
            if pred and 'confidence' in pred:
                confidence = pred['confidence']
                expected = test['expected_confidence']
                
                # Check if confidence aligns with correctness
                if test['true_answer']:
                    # Correct answer should have higher confidence
                    if confidence < expected - 0.2:  # Allow some tolerance
                        calibration_errors += 1
                else:
                    # Incorrect answer should have lower confidence
                    if confidence > expected + 0.2:
                        calibration_errors += 1
        
        calibration_score = 1 - (calibration_errors / total) if total > 0 else 0
        
        return {
            'total_predictions': total,
            'calibration_errors': calibration_errors,
            'calibration_score': calibration_score,
            'well_calibrated_percent': calibration_score * 100
        }


class CitationQualityBenchmark:
    """Test citation quality and accuracy"""
    
    def __init__(self):
        """
        Initialize predefined citation quality test cases.
        
        Sets self.citation_tests to a list of dictionaries, each representing a test case with:
        - 'query' (str): the query string for the test.
        - 'required_citations' (int): minimum number of citations required to consider the test adequate.
        - 'citations' (List[Dict[str, str]]): list of citation entries where each entry contains 'source', 'url', and 'content'.
        """
        self.citation_tests = [
            {
                'query': 'Test query',
                'required_citations': 2,
                'citations': [
                    {'source': 'reliable.edu', 'url': 'http://reliable.edu/1', 'content': 'Fact 1'},
                    {'source': 'reliable.edu', 'url': 'http://reliable.edu/2', 'content': 'Fact 2'}
                ]
            },
            {
                'query': 'Test query 2',
                'required_citations': 1,
                'citations': [
                    {'source': 'unreliable.com', 'url': 'http://unreliable.com', 'content': 'Incorrect fact'}
                ]
            }
        ]
    
    def benchmark_citation_quality(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate whether each result contains at least the required number of citations for the corresponding citation test.
        
        Parameters:
            results (List[Dict[str, Any]]): List of result objects produced by the system under test. Each result may include a 'citations' key with a list of citation entries. Results are matched to predefined tests by list index; missing results are treated as failures.
        
        Returns:
            Dict[str, Any]: Summary of citation quality with the following keys:
                - total_tests: Number of tests evaluated (int).
                - adequate_citations: Count of tests where the provided citations met or exceeded the required number (int).
                - citation_quality_percent: Percentage of tests with adequate citations (0-100 float).
                - citation_quality_score: Same value as `citation_quality_percent` expressed as a 0.0-1.0 score (float).
        """
        correct = 0
        total = len(results)
        
        for i, test in enumerate(self.citation_tests):
            result = results[i] if i < len(results) else None
            
            if result and 'citations' in result:
                citations = result['citations']
                required = test['required_citations']
                
                # Check citation count
                if len(citations) >= required:
                    correct += 1
        
        quality_score = (correct / total * 100) if total > 0 else 0
        
        return {
            'total_tests': total,
            'adequate_citations': correct,
            'citation_quality_percent': quality_score,
            'citation_quality_score': quality_score / 100
        }


class KnowledgeBaseHitRateBenchmark:
    """Test knowledge base hit rates"""
    
    def __init__(self):
        """
        Initialize the benchmark with an in-memory knowledge base mapping topic names to short definitions.
        
        The knowledge_base attribute is populated with sample entries for 'AI', 'Machine Learning', 'Neural Networks', and 'Deep Learning' to support hit-rate testing.
        """
        self.knowledge_base = {
            'AI': 'Artificial Intelligence is intelligence demonstrated by machines',
            'Machine Learning': 'Machine Learning is a subset of AI',
            'Neural Networks': 'Neural Networks are computing systems inspired by biological brains',
            'Deep Learning': 'Deep Learning uses multiple layers for representation learning'
        }
    
    def benchmark_knowledge_hit_rate(self, queries: List[str]) -> Dict[str, Any]:
        """
        Compute how many queries match entries in the internal knowledge base.
        
        Returns:
            result (Dict[str, Any]): Dictionary containing:
                - total_queries (int): Number of input queries evaluated.
                - knowledge_hits (int): Number of queries that contained both a topic and its definition.
                - hit_rate_percent (float): Percentage of queries that were hits (0–100).
                - hit_rate_score (float): Hit rate as a normalized score (0.0–1.0).
        """
        hits = 0
        total = len(queries)
        
        for query in queries:
            for topic, definition in self.knowledge_base.items():
                if topic.lower() in query.lower() and definition.lower() in query.lower():
                    hits += 1
                    break
        
        hit_rate = (hits / total * 100) if total > 0 else 0
        
        return {
            'total_queries': total,
            'knowledge_hits': hits,
            'hit_rate_percent': hit_rate,
            'hit_rate_score': hit_rate / 100
        }


def run_quality_benchmarks():
    """
    Execute all defined quality benchmarks and return an aggregated summary.
    
    Runs the response accuracy, confidence calibration, citation quality, and knowledge-base hit-rate benchmarks and collects their individual results.
    
    Returns:
        summary (dict): Aggregated results with keys:
            - 'timestamp': float POSIX timestamp when the run completed.
            - 'response_accuracy': dict result from ResponseAccuracyBenchmark.
            - 'confidence_calibration': dict result from ConfidenceCalibrationBenchmark.
            - 'citation_quality': dict result from CitationQualityBenchmark.
            - 'knowledge_hit_rate': dict result from KnowledgeBaseHitRateBenchmark.
    """
    print("\n" + "="*80)
    print("Running Quality Benchmarks")
    print("="*80)
    
    results = {}
    
    # Response Accuracy Benchmark
    print("\nRunning Response Accuracy Benchmark...")
    accuracy_benchmark = ResponseAccuracyBenchmark()
    test_responses = [
        "The capital of France is Paris",
        "2 + 2 equals 4",
        "Romeo and Juliet was written by Shakespeare",
        "Water boils at 100°C at sea level"
    ]
    
    accuracy_result = accuracy_benchmark.benchmark_response_accuracy(test_responses)
    results['response_accuracy'] = accuracy_result
    print(f"  Accuracy: {accuracy_result['accuracy_percent']:.1f}%")
    
    # Confidence Calibration Benchmark
    print("\nRunning Confidence Calibration Benchmark...")
    confidence_benchmark = ConfidenceCalibrationBenchmark()
    predictions = [
        {'query': 'Test', 'confidence': 0.95, 'correct': True},
        {'query': 'Test 2', 'confidence': 0.75, 'correct': True},
        {'query': 'Test 3', 'confidence': 0.45, 'correct': True},
        {'query': 'Test 4', 'confidence': 0.90, 'correct': True}
    ]
    
    confidence_result = confidence_benchmark.benchmark_confidence_calibration(predictions)
    results['confidence_calibration'] = confidence_result
    print(f"  Calibration Score: {confidence_result['well_calibrated_percent']:.1f}%")
    
    # Citation Quality Benchmark
    print("\nRunning Citation Quality Benchmark...")
    citation_benchmark = CitationQualityBenchmark()
    test_results = [
        {'query': 'Test query', 'citations': [
            {'source': 'reliable.edu', 'content': 'Fact 1'},
            {'source': 'reliable.edu', 'content': 'Fact 2'}
        ]},
        {'query': 'Test query 2', 'citations': [
            {'source': 'unreliable.com', 'content': 'Incorrect fact'}
        ]}
    ]
    
    citation_result = citation_benchmark.benchmark_citation_quality(test_results)
    results['citation_quality'] = citation_result
    print(f"  Citation Quality: {citation_result['citation_quality_percent']:.1f}%")
    
    # Knowledge Base Hit Rate Benchmark
    print("\nRunning Knowledge Base Hit Rate Benchmark...")
    kb_benchmark = KnowledgeBaseHitRateBenchmark()
    test_queries = [
        "What is Artificial Intelligence?",
        "Explain Machine Learning",
        "How do Neural Networks work?"
    ]
    
    kb_result = kb_benchmark.benchmark_knowledge_hit_rate(test_queries)
    results['knowledge_hit_rate'] = kb_result
    print(f"  Knowledge Base Hit Rate: {kb_result['hit_rate_percent']:.1f}%")
    
    # Generate summary
    summary = {
        'timestamp': time.time(),
        'response_accuracy': accuracy_result,
        'confidence_calibration': confidence_result,
        'citation_quality': citation_result,
        'knowledge_hit_rate': kb_result
    }
    
    print("\n" + "="*80)
    print("QUALITY BENCHMARK SUMMARY")
    print("="*80)
    print(f"Response Accuracy: {accuracy_result['accuracy_percent']:.1f}%")
    print(f"Confidence Calibration: {confidence_result['well_calibrated_percent']:.1f}%")
    print(f"Citation Quality: {citation_result['citation_quality_percent']:.1f}%")
    print(f"Knowledge Base Hit Rate: {kb_result['hit_rate_percent']:.1f}%")
    
    return summary


if __name__ == '__main__':
    run_quality_benchmarks()