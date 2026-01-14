"""
Direction Mode Controller
Main orchestrator for Direction Mode pipeline
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import time
import traceback

from .query_analyzer import QueryAnalyzer, QueryType
from .search_engine import SearchEngine, SearchResult
from .info_extractor import InformationExtractor, ExtractedFact
from .knowledge_store import KnowledgeStore
from .answer_generator import AnswerGenerator
from .fact_checker import FactChecker, ValidationResult
from .submode_styles import SubmodeStyle, ResponseStyler

# Import style processors
from .styles.sugarcotted import SugarcottedProcessor
from .styles.unhinged import UnhingedProcessor
from .styles.reaper import ReaperProcessor
from .styles.hexagon import HexagonProcessor

logger = logging.getLogger(__name__)

class DirectionModeController:
    """
    Main orchestrator for Direction Mode pipeline
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Direction Mode Controller
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer()
        self.search_engine = SearchEngine(self.config)
        self.info_extractor = InformationExtractor()
        self.knowledge_store = KnowledgeStore(
            db_path=self.config.get('knowledge_db_path', 'direction_mode_knowledge.db')
        )
        self.answer_generator = AnswerGenerator()
        self.fact_checker = FactChecker()
        self.response_styler = ResponseStyler()
        
        # Initialize style processors
        self.style_processors = {
            SubmodeStyle.NORMAL: None,
            SubmodeStyle.SUGARCOTTED: SugarcottedProcessor(),
            SubmodeStyle.UNHINGED: UnhingedProcessor(),
            SubmodeStyle.REAPER: ReaperProcessor(),
            SubmodeStyle.HEXAGON: HexagonProcessor()
        }
        
        # Performance metrics
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0.0,
            'cache_hit_rate': 0.0,
            'average_confidence': 0.0
        }
        
        logger.info("DirectionModeController initialized successfully")
    
    async def process_query_direction_mode(self, query: str, user_id: str = "default", 
                                        submode: str = "normal", 
                                        format_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Main pipeline for Direction Mode query processing
        
        Args:
            query: User query
            user_id: User identifier
            submode: Response style sub-mode
            format_type: Answer format type
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        start_time = time.time()
        query_id = None
        
        try:
            self.metrics['total_queries'] += 1
            
            logger.info(f"Processing Direction Mode query: {query[:50]}...")
            
            # Step 1: Check cache first
            cached_results = self.knowledge_store.retrieve_similar_queries(query, user_id, limit=3)
            if cached_results:
                best_match = cached_results[0]
                if best_match['similarity_score'] > 0.8:
                    logger.info(f"Using cached result (similarity: {best_match['similarity_score']:.2f})")
                    cached_response = await self._process_cached_result(
                        best_match, submode, format_type
                    )
                    cached_response['cache_hit'] = True
                    return cached_response
            
            # Step 2: Analyze query
            query_analysis = self.query_analyzer.analyze_query(query)
            logger.info(f"Query type: {query_analysis['query_type']}, confidence: {query_analysis['confidence']:.2f}")
            
            # Step 3: Multi-source search
            async with self.search_engine:
                search_results = await self.search_engine.search(
                    query=query,
                    max_results=10,
                    sources=query_analysis['optimal_sources']
                )
            
            if not search_results:
                logger.warning("No search results found")
                return self._generate_no_results_response(query, submode, format_type)
            
            logger.info(f"Found {len(search_results)} search results")
            
            # Step 4: Extract information
            extracted_facts = self.info_extractor.extract_information(search_results)
            if not extracted_facts:
                logger.warning("No facts extracted from search results")
                return self._generate_no_results_response(query, submode, format_type)
            
            logger.info(f"Extracted {len(extracted_facts)} facts")
            
            # Step 5: Validate facts
            validation_results = self.fact_checker.validate_facts(extracted_facts, query)
            valid_facts = [fact for fact, validation in zip(extracted_facts, validation_results) 
                          if validation.is_valid and validation.confidence > 0.3]
            
            if not valid_facts:
                logger.warning("No valid facts found after validation")
                return self._generate_low_confidence_response(query, submode, format_type)
            
            # Step 6: Generate key information
            key_information = self.info_extractor.extract_key_information(valid_facts, query)
            
            # Step 7: Generate answer
            answer_response = self.answer_generator.generate_answer(
                facts=valid_facts,
                query=query,
                format_type=format_type,
                key_information=key_information
            )
            
            # Step 8: Apply sub-mode styling
            styled_answer = self._apply_submode_styling(
                answer_response['answer'], submode
            )
            
            # Step 9: Compile final response
            final_response = {
                'query': query,
                'user_id': user_id,
                'mode': 'direction',
                'submode': submode,
                'answer': styled_answer,
                'confidence': answer_response['confidence'],
                'sources_used': answer_response['sources_used'],
                'facts_analyzed': answer_response['facts_analyzed'],
                'format': format_type,
                'citations': answer_response['citations'],
                'key_information': key_information,
                'validation_results': {
                    'total_facts': len(extracted_facts),
                    'valid_facts': len(valid_facts),
                    'average_confidence': sum(f.confidence for f in valid_facts) / len(valid_facts)
                },
                'query_analysis': {
                    'type': query_analysis['query_type'],
                    'confidence': query_analysis['confidence'],
                    'entities': query_analysis['entities']
                },
                'processing_time': time.time() - start_time,
                'cache_hit': False,
                'status': 'success',
                'timestamp': time.time()
            }
            
            # Step 10: Store in cache
            query_id = self._store_query_result(query, user_id, submode, final_response)
            
            # Update metrics
            self.metrics['successful_queries'] += 1
            self._update_metrics(time.time() - start_time, answer_response['confidence'])
            
            logger.info(f"Direction Mode query processed successfully in {final_response['processing_time']:.2f}s")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing Direction Mode query: {e}")
            logger.error(traceback.format_exc())
            
            self.metrics['failed_queries'] += 1
            
            return {
                'query': query,
                'user_id': user_id,
                'mode': 'direction',
                'submode': submode,
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'error': str(e),
                'status': 'error',
                'processing_time': time.time() - start_time,
                'timestamp': time.time()
            }
    
    async def _process_cached_result(self, cached_result: Dict[str, Any], submode: str, format_type: str) -> Dict[str, Any]:
        """
        Process a cached result with styling
        
        Args:
            cached_result: Cached result data
            submode: Response style sub-mode
            format_type: Answer format type
            
        Returns:
            Styled cached response
        """
        try:
            # Extract answer from cached result
            cached_answer = cached_result['results'].get('answer', 'No answer available')
            
            # Apply styling
            styled_answer = self._apply_submode_styling(cached_answer, submode)
            
            # Build response
            response = {
                'query': cached_result['query'],
                'user_id': 'cached',
                'mode': 'direction',
                'submode': submode,
                'answer': styled_answer,
                'confidence': cached_result['confidence'],
                'sources_used': cached_result['results'].get('sources_used', 0),
                'facts_analyzed': cached_result['results'].get('facts_analyzed', 0),
                'format': format_type,
                'citations': cached_result['results'].get('citations', []),
                'cache_hit': True,
                'cache_similarity': cached_result['similarity_score'],
                'cached_timestamp': cached_result['timestamp'],
                'status': 'success',
                'timestamp': time.time()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing cached result: {e}")
            # Fallback to generating a new response
            return await self.process_query_direction_mode(
                cached_result['query'], 'fallback', submode, format_type
            )
    
    def _apply_submode_styling(self, answer: str, submode: str) -> str:
        """
        Apply sub-mode styling to answer
        
        Args:
            answer: Original answer
            submode: Sub-mode style name
            
        Returns:
            Styled answer
        """
        try:
            # Convert string to enum
            style_enum = SubmodeStyle(submode.lower())
            
            # Apply styling
            styled = self.response_styler.apply_style(answer, style_enum)
            return styled
            
        except ValueError:
            logger.warning(f"Unknown submode: {submode}, using normal style")
            return answer
        except Exception as e:
            logger.error(f"Error applying styling: {e}")
            return answer
    
    def _store_query_result(self, query: str, user_id: str, submode: str, response: Dict[str, Any]) -> int:
        """
        Store query result in knowledge store
        
        Args:
            query: Original query
            user_id: User ID
            submode: Sub-mode used
            response: Complete response
            
        Returns:
            Query ID
        """
        try:
            # Prepare results for storage
            storage_results = {
                'answer': response['answer'],
                'confidence': response['confidence'],
                'sources_used': response['sources_used'],
                'facts_analyzed': response['facts_analyzed'],
                'citations': response.get('citations', []),
                'key_information': response.get('key_information', {})
            }
            
            # Store in knowledge store
            query_id = self.knowledge_store.store_query_result(
                query=query,
                user_id=user_id,
                results=storage_results,
                mode='direction',
                submode=submode
            )
            
            return query_id
            
        except Exception as e:
            logger.error(f"Error storing query result: {e}")
            return None
    
    def _generate_no_results_response(self, query: str, submode: str, format_type: str) -> Dict[str, Any]:
        """
        Generate response when no search results are found
        
        Args:
            query: Original query
            submode: Response sub-mode
            format_type: Answer format
            
        Returns:
            No results response
        """
        base_response = {
            'query': query,
            'user_id': 'system',
            'mode': 'direction',
            'submode': submode,
            'answer': f"I couldn't find any reliable information to answer your question about \"{query}\". This might be because the topic is very specific, recent, or not well-documented online.",
            'confidence': 0.0,
            'sources_used': 0,
            'facts_analyzed': 0,
            'format': format_type,
            'citations': [],
            'status': 'no_results',
            'timestamp': time.time()
        }
        
        # Apply styling
        styled_answer = self._apply_submode_styling(base_response['answer'], submode)
        base_response['answer'] = styled_answer
        
        return base_response
    
    def _generate_low_confidence_response(self, query: str, submode: str, format_type: str) -> Dict[str, Any]:
        """
        Generate response when low confidence results are found
        
        Args:
            query: Original query
            submode: Response sub-mode
            format_type: Answer format
            
        Returns:
            Low confidence response
        """
        base_response = {
            'query': query,
            'user_id': 'system',
            'mode': 'direction',
            'submode': submode,
            'answer': f"I found some information about \"{query}\", but the confidence is low. The available sources don't provide reliable or consistent information on this topic.",
            'confidence': 0.3,
            'sources_used': 0,
            'facts_analyzed': 0,
            'format': format_type,
            'citations': [],
            'status': 'low_confidence',
            'timestamp': time.time()
        }
        
        # Apply styling
        styled_answer = self._apply_submode_styling(base_response['answer'], submode)
        base_response['answer'] = styled_answer
        
        return base_response
    
    def _update_metrics(self, response_time: float, confidence: float):
        """
        Update performance metrics
        
        Args:
            response_time: Response processing time
            confidence: Response confidence score
        """
        # Update average response time
        total_queries = self.metrics['total_queries']
        current_avg = self.metrics['average_response_time']
        self.metrics['average_response_time'] = (
            (current_avg * (total_queries - 1) + response_time) / total_queries
        )
        
        # Update average confidence
        current_conf = self.metrics['average_confidence']
        self.metrics['average_confidence'] = (
            (current_conf * (total_queries - 1) + confidence) / total_queries
        )
        
        # Calculate cache hit rate
        # This would need to be tracked during processing
        # For now, we'll leave it as a placeholder
        cache_hits = getattr(self, '_cache_hits', 0)
        if total_queries > 0:
            self.metrics['cache_hit_rate'] = cache_hits / total_queries
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.metrics.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information
        
        Returns:
            Dictionary containing system status
        """
        try:
            # Get knowledge base stats
            kb_stats = self.knowledge_store.get_knowledge_base_stats()
            
            # Get available styles
            available_styles = {}
            for style in SubmodeStyle:
                style_info = self.response_styler.get_style_info(style)
                available_styles[style.value] = style_info
            
            # Get answer formats
            available_formats = self.answer_generator.get_available_formats()
            
            status = {
                'system': 'direction_mode',
                'status': 'operational',
                'version': '1.0.0',
                'components': {
                    'query_analyzer': 'operational',
                    'search_engine': 'operational',
                    'info_extractor': 'operational',
                    'knowledge_store': 'operational',
                    'answer_generator': 'operational',
                    'fact_checker': 'operational',
                    'response_styler': 'operational'
                },
                'metrics': self.metrics,
                'knowledge_base': kb_stats,
                'available_styles': available_styles,
                'available_formats': available_formats,
                'timestamp': time.time()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'system': 'direction_mode',
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def clear_cache(self, older_than_days: int = 30) -> int:
        """
        Clear old cached data
        
        Args:
            older_than_days: Remove entries older than this many days
            
        Returns:
            Number of entries removed
        """
        try:
            removed_count = self.knowledge_store.clear_cache(older_than_days)
            logger.info(f"Cleared {removed_count} old cache entries")
            return removed_count
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0
    
    async def search_knowledge_base(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search the knowledge base for concepts
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        try:
            results = self.knowledge_store.search_knowledge_base(search_term, limit)
            return results
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def get_available_submodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available sub-modes
        
        Returns:
            Dictionary of sub-mode information
        """
        submodes = {}
        for style in SubmodeStyle:
            style_info = self.response_styler.get_style_info(style)
            submodes[style.value] = style_info
        
        return submodes
    
    def get_available_formats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available answer formats
        
        Returns:
            Dictionary of format information
        """
        return self.answer_generator.get_available_formats()