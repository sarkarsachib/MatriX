"""
Unit tests for Direction Mode components
Tests for query analyzer, search engine, answer generator, fact checker, etc.
"""
import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sathik_ai.direction_mode import (
    DirectionModeController,
    SubmodeStyle,
    ResponseStyler
)
from sathik_ai.direction_mode.query_analyzer import QueryAnalyzer
from sathik_ai.direction_mode.search_engine import SearchEngine
from sathik_ai.direction_mode.answer_generator import AnswerGenerator
from sathik_ai.direction_mode.fact_checker import FactChecker


class TestQueryAnalyzer:
    """Test Query Analyzer functionality"""
    
    @pytest.fixture
    def query_analyzer(self):
        """
        Provide a fresh QueryAnalyzer instance for tests.
        
        Returns:
            QueryAnalyzer: A newly constructed QueryAnalyzer instance.
        """
        return QueryAnalyzer()
    
    def test_initialization(self, query_analyzer):
        """Test query analyzer initializes correctly"""
        assert query_analyzer is not None
    
    def test_analyze_simple_query(self, query_analyzer):
        """Test analyzing simple query"""
        query = "What is artificial intelligence?"
        result = query_analyzer.analyze(query)
        
        assert 'query_type' in result
        assert 'keywords' in result
        assert 'intent' in result
        assert 'artificial' in result['keywords'] or 'intelligence' in result['keywords']
    
    def test_analyze_complex_query(self, query_analyzer):
        """Test analyzing complex query"""
        query = "Explain the relationship between quantum computing and machine learning"
        result = query_analyzer.analyze(query)
        
        assert 'query_type' in result
        assert len(result['keywords']) >= 2
        assert result['complexity'] > 0  # Should be complex
    
    def test_extract_keywords(self, query_analyzer):
        """Test keyword extraction"""
        query = "How does neural network training work with backpropagation?"
        result = query_analyzer.analyze(query)
        
        expected_keywords = ['neural', 'network', 'training', 'backpropagation']
        for keyword in expected_keywords:
            # Check that keywords are found (may be stemmed or partial)
            found = any(keyword.lower() in k.lower() for k in result['keywords'])
            assert found
    
    def test_identify_query_type(self, query_analyzer):
        """
        Verify that QueryAnalyzer correctly categorizes common query shapes.
        
        Asserts that a direct question is labeled as 'question' or 'definition', a comparison is labeled as 'comparison' or 'versus', and an instructional query is labeled as 'how-to' or 'instruction'.
        """
        # Question type
        q1 = query_analyzer.analyze("What is Python?")
        assert q1['query_type'] in ['question', 'definition']
        
        # Comparison type
        q2 = query_analyzer.analyze("Python vs JavaScript")
        assert q2['query_type'] in ['comparison', 'versus']
        
        # How-to type
        q3 = query_analyzer.analyze("How to create a neural network?")
        assert q3['query_type'] in ['how-to', 'instruction']


class TestSearchEngine:
    """Test Search Engine functionality"""
    
    @pytest.fixture
    def search_engine(self):
        """
        Provide a fresh SearchEngine instance for tests.
        
        Returns:
            SearchEngine: a new SearchEngine instance.
        """
        return SearchEngine()
    
    def test_initialization(self, search_engine):
        """Test search engine initializes correctly"""
        assert search_engine is not None
    
    @pytest.mark.asyncio
    async def test_search_basic_query(self, search_engine):
        """Test basic search query"""
        results = await search_engine.search("artificial intelligence", num_results=5)
        
        assert 'results' in results
        assert 'query' in results
        assert isinstance(results['results'], list)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self, search_engine):
        """Test search with source filters"""
        results = await search_engine.search(
            "machine learning",
            sources=['wikipedia.org', 'arxiv.org'],
            num_results=5
        )
        
        assert 'results' in results
        assert isinstance(results['results'], list)
    
    @pytest.mark.asyncio
    async def test_search_empty_query(self, search_engine):
        """Test search with empty query"""
        results = await search_engine.search("")
        
        assert 'results' in results
        assert len(results['results']) == 0
    
    def test_result_ranking(self, search_engine):
        """Test search result ranking"""
        # Mock results with different relevance scores
        mock_results = [
            {'url': 'url1.com', 'relevance': 0.9, 'content': 'content1'},
            {'url': 'url2.com', 'relevance': 0.7, 'content': 'content2'},
            {'url': 'url3.com', 'relevance': 0.95, 'content': 'content3'}
        ]
        
        # Results should be ranked by relevance
        ranked = search_engine._rank_results(mock_results)
        assert ranked[0]['relevance'] >= ranked[1]['relevance']
        assert ranked[0]['url'] == 'url3.com'  # Highest relevance


class TestAnswerGenerator:
    """Test Answer Generator functionality"""
    
    @pytest.fixture
    def answer_generator(self):
        """
        Provide a fresh AnswerGenerator instance for tests.
        
        Returns:
            AnswerGenerator: A new AnswerGenerator instance.
        """
        return AnswerGenerator()
    
    def test_initialization(self, answer_generator):
        """Test answer generator initializes correctly"""
        assert answer_generator is not None
    
    def test_generate_simple_answer(self, answer_generator):
        """Test generating simple answer"""
        sources = [
            {'content': 'AI is intelligence demonstrated by machines', 'source': 'source1.com'},
            {'content': 'Machine learning is subset of AI', 'source': 'source2.com'}
        ]
        answer = answer_generator.generate("What is AI?", sources)
        
        assert 'answer' in answer
        assert 'sources' in answer
        assert 'confidence' in answer
        assert len(answer['answer']) > 0
    
    def test_generate_with_citations(self, answer_generator):
        """Test generating answer with citations"""
        sources = [
            {'content': 'Fact 1', 'source': 'source1.com', 'url': 'http://source1.com'},
            {'content': 'Fact 2', 'source': 'source2.com', 'url': 'http://source2.com'}
        ]
        answer = answer_generator.generate("Question", sources, include_citations=True)
        
        assert 'citations' in answer
        assert len(answer['citations']) >= 1
    
    def test_generate_comprehensive_format(self, answer_generator):
        """
        Verify that generating an answer with format_type 'comprehensive' includes contextual content beyond the raw source.
        
        Asserts that the returned mapping contains an 'answer' key and that the generated answer text is longer than the provided source content, indicating inclusion of additional context.
        """
        sources = [{'content': 'Test content', 'source': 'source.com'}]
        answer = answer_generator.generate(
            "Question",
            sources,
            format_type='comprehensive'
        )
        
        assert 'answer' in answer
        # Comprehensive format should include context
        assert len(answer['answer']) > len(sources[0]['content'])
    
    def test_generate_summary_format(self, answer_generator):
        """Test generating summary format"""
        sources = [
            {'content': 'Point 1', 'source': 'source.com'},
            {'content': 'Point 2', 'source': 'source.com'}
        ]
        answer = answer_generator.generate("Question", sources, format_type='summary')
        
        assert 'answer' in answer
        # Summary should be concise
        assert len(answer['answer']) < 500  # Reasonable length limit
    
    def test_generate_bullet_points(self, answer_generator):
        """Test generating bullet point format"""
        sources = [
            {'content': 'Point 1', 'source': 'source.com'},
            {'content': 'Point 2', 'source': 'source.com'}
        ]
        answer = answer_generator.generate("Question", sources, format_type='bullet_points')
        
        assert 'answer' in answer
        # Bullet points typically use lists
        assert ('-' in answer['answer']) or ('*' in answer['answer']) or ('\n' in answer['answer'])


class TestFactChecker:
    """Test Fact Checker functionality"""
    
    @pytest.fixture
    def fact_checker(self):
        """
        Provide a new FactChecker instance for tests.
        
        Returns:
            FactChecker: A fresh FactChecker instance.
        """
        return FactChecker()
    
    def test_initialization(self, fact_checker):
        """Test fact checker initializes correctly"""
        assert fact_checker is not None
    
    def test_check_single_fact(self, fact_checker):
        """Test checking single fact"""
        fact = "The Earth is flat"
        result = fact_checker.check(fact)
        
        assert 'is_true' in result
        assert 'confidence' in result
        assert 'sources' in result
        # Earth is not flat
        assert result['is_true'] == False
    
    def test_check_multiple_facts(self, fact_checker):
        """Test checking multiple facts"""
        facts = [
            "The sun rises in the east",
            "Water boils at 100°C at sea level",
            "Humans have 5 senses"
        ]
        results = fact_checker.check_batch(facts)
        
        assert len(results) == 3
        for result in results:
            assert 'is_true' in result
            assert 'confidence' in result
    
    def test_consensus_detection(self, fact_checker):
        """
        Verifies that FactChecker marks a widely accepted statement with high confidence and includes consensus metadata.
        
        Asserts that a common factual statement (e.g., "Python is a programming language") yields a confidence greater than 0.8 and that the result contains a 'consensus' key.
        """
        fact = "Python is a programming language"
        result = fact_checker.check(fact)
        
        # This is widely accepted fact
        assert result['confidence'] > 0.8
        assert 'consensus' in result
    
    def test_source_verification(self, fact_checker):
        """Test source verification"""
        sources = [
            {'content': 'Claim', 'source': 'reliable-source.edu'},
            {'content': 'Claim', 'source': 'unreliable-source.com'}
        ]
        result = fact_checker.verify_sources(sources)
        
        assert 'verified_sources' in result
        assert 'unverified_sources' in result


class TestDirectionModeController:
    """Test Direction Mode Controller integration"""
    
    @pytest.fixture
    def controller_config(self):
        """
        Provide a test configuration dictionary for DirectionModeController containing API keys and a knowledge database path.
        
        Returns:
            config (dict): Configuration mapping with keys:
                - `google_api_key` (str): Test Google API key.
                - `google_cse_id` (str): Test Google Custom Search Engine ID.
                - `news_api_key` (str): Test News API key.
                - `knowledge_db_path` (str): Filesystem path to the test knowledge database.
        """
        return {
            'google_api_key': 'test_key',
            'google_cse_id': 'test_cse_id',
            'news_api_key': 'test_news_key',
            'knowledge_db_path': '/tmp/test_direction_mode.db'
        }
    
    @pytest.fixture
    def controller(self, controller_config):
        """
        Create a DirectionModeController configured with the provided settings.
        
        Parameters:
            controller_config (dict): Configuration values (e.g., API keys, knowledge DB path, and other options) used to initialize the controller.
        
        Returns:
            DirectionModeController: A new controller instance configured according to controller_config.
        """
        return DirectionModeController(controller_config)
    
    def test_initialization(self, controller, controller_config):
        """Test controller initializes correctly"""
        assert controller is not None
    
    @pytest.mark.asyncio
    async def test_process_query_normal_submode(self, controller):
        """Test processing query with normal submode"""
        result = await controller.process_query_direction_mode(
            query="What is AI?",
            user_id="test_user",
            submode="normal",
            format_type="comprehensive"
        )
        
        assert 'answer' in result
        assert 'status' in result
        assert result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_process_query_sugarcotted_submode(self, controller):
        """Test processing query with sugarcotted submode"""
        result = await controller.process_query_direction_mode(
            query="Explain quantum computing",
            user_id="test_user",
            submode="sugarcotted",
            format_type="summary"
        )
        
        assert 'answer' in result
        # Sugarcotted mode should have specific style
        assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_process_query_unhinged_submode(self, controller):
        """Test processing query with unhinged submode"""
        result = await controller.process_query_direction_mode(
            query="Tell me about deep learning",
            user_id="test_user",
            submode="unhinged",
            format_type="bullet_points"
        )
        
        assert 'answer' in result
        # Unhinged mode should have different style
        assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_process_query_reaper_submode(self, controller):
        """Test processing query with reaper submode"""
        result = await controller.process_query_direction_mode(
            query="What are neural networks?",
            user_id="test_user",
            submode="reaper",
            format_type="comprehensive"
        )
        
        assert 'answer' in result
        assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_process_query_hexagon_submode(self, controller):
        """Test processing query with hexagon (666) submode"""
        result = await controller.process_query_direction_mode(
            query="Explain machine learning",
            user_id="test_user",
            submode="666",
            format_type="comprehensive"
        )
        
        assert 'answer' in result
        assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, controller):
        """Test handling empty query"""
        result = await controller.process_query_direction_mode(
            query="",
            user_id="test_user",
            submode="normal"
        )
        
        # Should handle gracefully
        assert 'status' in result
        assert result['status'] in ['error', 'success']
    
    def test_mode_switching(self, controller):
        """Test mode switching"""
        # Test that all submodes are available
        submodes = ['normal', 'sugarcotted', 'unhinged', 'reaper', '666']
        for submode in submodes:
            assert submode in controller.available_submodes


class TestSubmodeStyles:
    """Test Submode Styles functionality"""
    
    @pytest.fixture
    def response_styler(self):
        """
        Provide a fresh ResponseStyler instance for use in tests.
        
        Returns:
            ResponseStyler: A new ResponseStyler instance.
        """
        return ResponseStyler()
    
    def test_initialization(self, response_styler):
        """Test response styler initializes correctly"""
        assert response_styler is not None
    
    def test_apply_normal_style(self, response_styler):
        """Test applying normal style"""
        content = "This is the base content"
        styled = response_styler.apply_style(content, "normal")
        
        assert 'This is the base content' in styled
        # Normal style should be professional and clear
    
    def test_apply_sugarcotted_style(self, response_styler):
        """Test applying sugarcotted style"""
        content = "Artificial intelligence is a field of computer science"
        styled = response_styler.apply_style(content, "sugarcotted")
        
        # Sugarcotted mode should add emphasis and enthusiasm
        assert isinstance(styled, str)
        assert len(styled) > 0
    
    def test_apply_unhinged_style(self, response_styler):
        """Test applying unhinged style"""
        content = "Neural networks learn from data"
        styled = response_styler.apply_style(content, "unhinged")
        
        # Unhinged mode should be more casual/direct
        assert isinstance(styled, str)
        assert len(styled) > 0
    
    def test_apply_reaper_style(self, response_styler):
        """Test applying reaper style"""
        content = "Deep learning uses multiple layers"
        styled = response_styler.apply_style(content, "reaper")
        
        # Reaper mode should be precise and technical
        assert isinstance(styled, str)
        assert len(styled) > 0
    
    def test_apply_hexagon_style(self, response_styler):
        """Test applying hexagon (666) style"""
        content = "Machine learning algorithms improve over time"
        styled = response_styler.apply_style(content, "666")
        
        # Hexagon mode should be advanced/complex
        assert isinstance(styled, str)
        assert len(styled) > 0
    
    def test_style_preserves_content(self, response_styler):
        """Test that styles preserve core content"""
        content = "The capital of France is Paris"
        submodes = ['normal', 'sugarcotted', 'unhinged', 'reaper', '666']
        
        for submode in submodes:
            styled = response_styler.apply_style(content, submode)
            # Core information should be preserved
            assert 'Paris' in styled or 'France' in styled
    
    def test_invalid_submode(self, response_styler):
        """Test handling invalid submode"""
        content = "Test content"
        styled = response_styler.apply_style(content, "invalid_submode")
        
        # Should fall back to normal or handle gracefully
        assert isinstance(styled, str)
        assert len(styled) > 0


class TestDirectionModePerformance:
    """Performance tests for direction mode"""
    
    @pytest.fixture
    def controller_config(self):
        """
        Provide a configuration dictionary for DirectionModeController unit tests.
        
        Returns:
            config (dict): Test configuration containing:
                - 'google_api_key' (str): placeholder API key for Google services.
                - 'google_cse_id' (str): placeholder Google Custom Search Engine ID.
                - 'knowledge_db_path' (str): filesystem path to the test knowledge database.
        """
        return {
            'google_api_key': 'test_key',
            'google_cse_id': 'test_cse_id',
            'knowledge_db_path': '/tmp/test_direction_mode.db'
        }
    
    @pytest.fixture
    def controller(self, controller_config):
        """
        Create a DirectionModeController configured with the provided settings.
        
        Parameters:
            controller_config (dict): Configuration values (e.g., API keys, knowledge DB path, and other options) used to initialize the controller.
        
        Returns:
            DirectionModeController: A new controller instance configured according to controller_config.
        """
        return DirectionModeController(controller_config)
    
    @pytest.mark.asyncio
    async def test_query_processing_speed(self, controller, benchmark):
        """Benchmark query processing speed"""
        queries = [
            "What is artificial intelligence?",
            "Explain quantum computing",
            "How do neural networks work?"
        ]
        
        with benchmark("Direction Mode Query Processing"):
            for query in queries:
                result = await controller.process_query_direction_mode(
                    query=query,
                    user_id="test_user",
                    submode="normal"
                )
                assert 'answer' in result
    
    @pytest.mark.asyncio
    async def test_submode_application_speed(self, controller, benchmark):
        """Benchmark submode application speed"""
        query = "Explain machine learning"
        submodes = ['normal', 'sugarcotted', 'unhinged', 'reaper', '666']
        
        with benchmark("Submode Application"):
            for submode in submodes:
                result = await controller.process_query_direction_mode(
                    query=query,
                    user_id="test_user",
                    submode=submode
                )
                assert 'answer' in result