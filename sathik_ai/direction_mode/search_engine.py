"""
Multi-Source Search Engine for Direction Mode
Integrates multiple search APIs with graceful fallbacks
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote
import json
import time

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Data class for search results"""
    title: str
    url: str
    snippet: str
    source: str
    relevance_score: float
    timestamp: float
    metadata: Dict[str, Any] = None

class SearchEngine:
    """
    Multi-source search engine with graceful fallbacks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the search engine
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config or {}
        self.session = None
        
        # API configurations
        self.google_api_key = self.config.get('google_api_key')
        self.google_cse_id = self.config.get('google_cse_id')
        self.news_api_key = self.config.get('news_api_key')
        
        # Rate limiting
        self.request_times = []
        self.max_requests_per_minute = 60
        
        logger.info("SearchEngine initialized with graceful fallbacks")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, max_results: int = 10, sources: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search using multiple sources with fallbacks
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            sources: List of sources to use (None for all available)
            
        Returns:
            List of SearchResult objects
        """
        if not self.session:
            async with SearchEngine(self.config) as engine:
                return await engine.search(query, max_results, sources)
        
        logger.info(f"Searching for: {query}")
        
        # Determine which sources to use
        if sources is None:
            sources = self._get_available_sources()
        
        # Perform parallel searches
        tasks = []
        for source in sources:
            if source in self._get_available_sources():
                task = self._search_source(source, query, max_results // len(sources) + 1)
                tasks.append(task)
        
        if not tasks:
            logger.warning("No search sources available")
            return []
        
        try:
            # Execute searches concurrently with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=30.0
            )
            
            # Combine and flatten results
            all_results = []
            for result_set in results:
                if isinstance(result_set, list):
                    all_results.extend(result_set)
                elif isinstance(result_set, Exception):
                    logger.warning(f"Search error: {result_set}")
            
            # Remove duplicates and rank
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query)
            
            # Limit results
            final_results = ranked_results[:max_results]
            
            logger.info(f"Found {len(final_results)} unique results for: {query}")
            return final_results
            
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for query: {query}")
            return []
        except Exception as e:
            logger.error(f"Search error for query '{query}': {e}")
            return []
    
    def _get_available_sources(self) -> List[str]:
        """
        Get list of available search sources based on configuration
        
        Returns:
            List of available source names
        """
        sources = []
        
        # Always available sources
        sources.extend(["duckduckgo", "wikipedia"])
        
        # Optional sources based on API keys
        if self.google_api_key and self.google_cse_id:
            sources.append("google")
        
        if self.news_api_key:
            sources.append("newsapi")
        
        sources.append("arxiv")  # Free ArXiv API
        
        return sources
    
    async def _search_source(self, source: str, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using a specific source
        
        Args:
            source: Source name
            query: Search query
            max_results: Maximum results from this source
            
        Returns:
            List of SearchResult objects
        """
        try:
            if source == "duckduckgo":
                return await self._search_duckduckgo(query, max_results)
            elif source == "google":
                return await self._search_google(query, max_results)
            elif source == "wikipedia":
                return await self._search_wikipedia(query, max_results)
            elif source == "newsapi":
                return await self._search_newsapi(query, max_results)
            elif source == "arxiv":
                return await self._search_arxiv(query, max_results)
            else:
                logger.warning(f"Unknown search source: {source}")
                return []
                
        except Exception as e:
            logger.warning(f"Error searching {source}: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using DuckDuckGo (no API key required)
        """
        try:
            # DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Extract results from various fields
                    for field in ['Abstract', 'RelatedTopics', 'Results']:
                        if field in data and data[field]:
                            if field == 'Abstract':
                                # Single result from abstract
                                result = SearchResult(
                                    title=data.get('Heading', ''),
                                    url=data.get('AbstractURL', ''),
                                    snippet=data.get('Abstract', ''),
                                    source='duckduckgo',
                                    relevance_score=0.8,
                                    timestamp=time.time()
                                )
                                results.append(result)
                            elif field == 'Results':
                                # List of results
                                for item in data[field]:
                                    if isinstance(item, dict) and 'Text' in item:
                                        result = SearchResult(
                                            title=item.get('Text', '')[:100] + '...',
                                            url=item.get('FirstURL', ''),
                                            snippet=item.get('Text', ''),
                                            source='duckduckgo',
                                            relevance_score=0.7,
                                            timestamp=time.time()
                                        )
                                        results.append(result)
                            elif field == 'RelatedTopics':
                                # Related topics
                                for topic in data[field]:
                                    if isinstance(topic, dict) and 'Text' in topic:
                                        result = SearchResult(
                                            title=topic.get('Text', '')[:100] + '...',
                                            url=topic.get('FirstURL', ''),
                                            snippet=topic.get('Text', ''),
                                            source='duckduckgo',
                                            relevance_score=0.6,
                                            timestamp=time.time()
                                        )
                                        results.append(result)
                    
                    return results[:max_results]
                else:
                    logger.warning(f"DuckDuckGo API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")
            return []
    
    async def _search_google(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using Google Custom Search API
        """
        if not self.google_api_key or not self.google_cse_id:
            return []
        
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': min(max_results, 10)  # Google API max is 10
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    if 'items' in data:
                        for item in data['items']:
                            result = SearchResult(
                                title=item.get('title', ''),
                                url=item.get('link', ''),
                                snippet=item.get('snippet', ''),
                                source='google',
                                relevance_score=0.9,
                                timestamp=time.time(),
                                metadata={'formattedUrl': item.get('formattedUrl', '')}
                            )
                            results.append(result)
                    
                    return results
                else:
                    logger.warning(f"Google API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.warning(f"Google search error: {e}")
            return []
    
    async def _search_wikipedia(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using Wikipedia API
        """
        try:
            # Search for Wikipedia pages
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(query.replace(' ', '_'))
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check if this is a valid Wikipedia page
                    if data.get('type') == 'standard':
                        result = SearchResult(
                            title=data.get('title', ''),
                            url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            snippet=data.get('extract', ''),
                            source='wikipedia',
                            relevance_score=0.95,  # High confidence for Wikipedia
                            timestamp=time.time(),
                            metadata={'description': data.get('description', '')}
                        )
                        return [result]
                
                # If direct search fails, try Wikipedia search API
                search_api_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': query,
                    'srlimit': max_results,
                    'origin': '*'
                }
                
                async with self.session.get(search_api_url, params=search_params) as search_response:
                    if search_response.status == 200:
                        search_data = await search_response.json()
                        results = []
                        
                        if 'query' in search_data and 'search' in search_data['query']:
                            for item in search_data['query']['search']:
                                result = SearchResult(
                                    title=item.get('title', ''),
                                    url=f"https://en.wikipedia.org/wiki/{quote(item.get('title', '').replace(' ', '_'))}",
                                    snippet=item.get('snippet', ''),
                                    source='wikipedia',
                                    relevance_score=0.9,
                                    timestamp=time.time(),
                                    metadata={'pageid': item.get('pageid', 0)}
                                )
                                results.append(result)
                        
                        return results
                    
                    return []
                    
        except Exception as e:
            logger.warning(f"Wikipedia search error: {e}")
            return []
    
    async def _search_newsapi(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using NewsAPI
        """
        if not self.news_api_key:
            return []
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'sortBy': 'relevancy',
                'pageSize': max_results,
                'language': 'en'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    if 'articles' in data:
                        for article in data['articles']:
                            result = SearchResult(
                                title=article.get('title', ''),
                                url=article.get('url', ''),
                                snippet=article.get('description', '') or article.get('content', ''),
                                source='newsapi',
                                relevance_score=0.8,
                                timestamp=time.time(),
                                metadata={
                                    'publishedAt': article.get('publishedAt', ''),
                                    'source': article.get('source', {}).get('name', ''),
                                    'author': article.get('author', '')
                                }
                            )
                            results.append(result)
                    
                    return results
                else:
                    logger.warning(f"NewsAPI returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.warning(f"NewsAPI search error: {e}")
            return []
    
    async def _search_arxiv(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search ArXiv for academic papers
        """
        try:
            # ArXiv API search
            search_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query.replace(" ", "+")}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance'
            }
            
            async with self.session.get(search_url, params=params) as response:
                if response.status == 200:
                    # ArXiv returns XML, we'll parse it simply
                    text = await response.text()
                    
                    # Simple XML parsing for ArXiv results
                    import re
                    
                    # Find all entry tags
                    entries = re.findall(r'<entry>(.*?)</entry>', text, re.DOTALL)
                    results = []
                    
                    for entry in entries:
                        # Extract title
                        title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                        title = title_match.group(1).strip() if title_match else ''
                        
                        # Extract summary
                        summary_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                        summary = summary_match.group(1).strip() if summary_match else ''
                        
                        # Extract link
                        link_match = re.search(r'<id>(.*?)</id>', entry)
                        link = link_match.group(1).strip() if link_match else ''
                        
                        if title and link:
                            result = SearchResult(
                                title=title,
                                url=link,
                                snippet=summary[:300] + '...' if len(summary) > 300 else summary,
                                source='arxiv',
                                relevance_score=0.9,  # High for academic papers
                                timestamp=time.time(),
                                metadata={'type': 'academic_paper'}
                            )
                            results.append(result)
                    
                    return results
                else:
                    logger.warning(f"ArXiv API returned status {response.status}")
                    return []
                    
        except Exception as e:
            logger.warning(f"ArXiv search error: {e}")
            return []
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results based on URL similarity
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated list of results
        """
        seen_urls = set()
        unique_results = []
        
        for result in results:
            # Normalize URL for comparison
            normalized_url = result.url.lower().strip('/')
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rank results by relevance score and query matching
        
        Args:
            results: List of search results
            query: Original search query
            
        Returns:
            Ranked list of results
        """
        query_lower = query.lower()
        
        for result in results:
            # Calculate additional relevance based on query matching
            title_score = 0
            snippet_score = 0
            
            # Title matching
            if query_lower in result.title.lower():
                title_score += 0.3
            
            # Snippet matching
            if query_lower in result.snippet.lower():
                snippet_score += 0.2
            
            # Word overlap
            query_words = set(query_lower.split())
            title_words = set(result.title.lower().split())
            snippet_words = set(result.snippet.lower().split())
            
            title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
            snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words) if query_words else 0
            
            # Update relevance score
            result.relevance_score += title_score + snippet_score + (title_overlap * 0.2) + (snippet_overlap * 0.1)
        
        # Sort by relevance score (descending)
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def get_source_reliability(self, source: str) -> float:
        """
        Get reliability score for a source
        
        Args:
            source: Source name
            
        Returns:
            Reliability score between 0 and 1
        """
        reliability_scores = {
            'wikipedia': 0.9,
            'arxiv': 0.95,
            'google': 0.8,
            'newsapi': 0.7,
            'duckduckgo': 0.6
        }
        
        return reliability_scores.get(source, 0.5)