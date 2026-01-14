"""
Query Analyzer for Direction Mode
Detects query types and determines data source requirements
"""

import re
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries that can be processed"""
    FACTUAL = "factual"
    CURRENT_EVENTS = "current_events"
    HOW_TO = "how_to"
    ACADEMIC = "academic"
    OPINION = "opinion"
    PRICE_INFO = "price_info"
    TECHNICAL = "technical"
    DEFINITION = "definition"


class QueryAnalyzer:
    """
    Analyzes queries to determine type, search terms, and data requirements
    """
    
    def __init__(self):
        # Define patterns for different query types
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what|who|when|where|why|how many|how much)\b',
                r'\b(is|are|was|were|does|do|did|can|will|should)\b.*\?',
                r'\b(definition|meaning|define)\b',
                r'\b(fact|facts|information about)\b'
            ],
            QueryType.CURRENT_EVENTS: [
                r'\b(latest|recent|current|today|yesterday|this week|this month|this year)\b',
                r'\b(news|breaking|just happened|recently)\b',
                r'\b(what happened|what\'s happening|what is happening)\b'
            ],
            QueryType.HOW_TO: [
                r'\b(how to|how do|how can|step by step|guide|tutorial)\b',
                r'\b(instruction|instructions|process|method)\b',
                r'\b(learn|study|master|understand)\b'
            ],
            QueryType.ACADEMIC: [
                r'\b(paper|research|study|analysis|study of|theory)\b',
                r'\b(arxiv|journal|academic|scholar|scientific)\b',
                r'\b(hypothesis|experiment|results|conclusion)\b'
            ],
            QueryType.OPINION: [
                r'\b(what do you think|your opinion|do you agree|should we|is it good)\b',
                r'\b(best|worst|better|worse|prefer)\b',
                r'\b(review|rating|recommend|suggest)\b'
            ],
            QueryType.PRICE_INFO: [
                r'\b(price|cost|how much|expensive|cheap|affordable)\b',
                r'\b(buy|purchase|order|sell|market)\b',
                r'\b(dollar|usd|eur|pound|¥|rupee)\b'
            ],
            QueryType.TECHNICAL: [
                r'\b(code|programming|software|algorithm|database|api)\b',
                r'\b(error|bug|issue|problem|solution)\b',
                r'\b(python|javascript|java|c\+\+|sql|html|css)\b'
            ],
            QueryType.DEFINITION: [
                r'\b(what is|what are|define|definition|meaning)\b',
                r'\b(explain|elaborate|describe)\b',
                r'\b(who is|what does)\b'
            ]
        }
        
        # Data freshness requirements by query type
        self.freshness_requirements = {
            QueryType.FACTUAL: "low",  # Facts don't change often
            QueryType.CURRENT_EVENTS: "high",  # Needs current data
            QueryType.HOW_TO: "medium",  # Methods can evolve
            QueryType.ACADEMIC: "low",  # Academic papers are stable
            QueryType.OPINION: "medium",  # Opinions can change
            QueryType.PRICE_INFO: "high",  # Prices change frequently
            QueryType.TECHNICAL: "medium",  # Tech evolves
            QueryType.DEFINITION: "low"  # Definitions are stable
        }
        
        # Optimal data sources by query type
        self.optimal_sources = {
            QueryType.FACTUAL: ["wikipedia", "general_search"],
            QueryType.CURRENT_EVENTS: ["news_search", "general_search"],
            QueryType.HOW_TO: ["stackoverflow", "general_search", "documentation"],
            QueryType.ACADEMIC: ["arxiv", "scholar_search", "wikipedia"],
            QueryType.OPINION: ["general_search", "reddit", "reviews"],
            QueryType.PRICE_INFO: ["general_search", "shopping_search"],
            QueryType.TECHNICAL: ["stackoverflow", "github", "documentation"],
            QueryType.DEFINITION: ["wikipedia", "general_search"]
        }
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine type, search terms, and requirements
        
        Args:
            query: The query string to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        query_lower = query.lower().strip()
        
        # Detect query type
        query_type = self._detect_query_type(query_lower)
        
        # Extract search terms and keywords
        search_terms = self._extract_search_terms(query_lower)
        
        # Determine freshness requirements
        freshness = self.freshness_requirements.get(query_type, "medium")
        
        # Select optimal data sources
        sources = self.optimal_sources.get(query_type, ["general_search"])
        
        # Extract entities and concepts
        entities = self._extract_entities(query_lower)
        
        # Determine confidence in analysis
        confidence = self._calculate_confidence(query_lower, query_type)
        
        analysis_result = {
            "query_type": query_type.value,
            "search_terms": search_terms,
            "freshness_requirement": freshness,
            "optimal_sources": sources,
            "entities": entities,
            "confidence": confidence,
            "original_query": query
        }
        
        logger.info(f"Query analyzed: {query_type.value} (confidence: {confidence:.2f})")
        return analysis_result
    
    def _detect_query_type(self, query: str) -> QueryType:
        """
        Detect the type of query based on patterns
        
        Args:
            query: Lowercase query string
            
        Returns:
            Detected QueryType
        """
        type_scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query))
                score += matches
            
            if score > 0:
                type_scores[query_type] = score
        
        if not type_scores:
            return QueryType.FACTUAL  # Default to factual
        
        # Return the type with highest score
        return max(type_scores, key=type_scores.get)
    
    def _extract_search_terms(self, query: str) -> List[str]:
        """
        Extract key search terms from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            List of search terms
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'what', 'who', 'when', 'where', 'why', 'how'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query)
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # If no terms found, use the original query
        if not search_terms:
            search_terms = [query]
        
        return search_terms
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities (people, places, organizations, etc.) from query
        
        Args:
            query: Lowercase query string
            
        Returns:
            Dictionary of entity types to lists of entities
        """
        entities = {
            'people': [],
            'places': [],
            'organizations': [],
            'technologies': [],
            'concepts': []
        }
        
        # Simple entity extraction patterns
        # This is basic - could be enhanced with proper NER
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Technology patterns
        tech_patterns = [
            r'\b(python|javascript|java|c\+\+|react|angular|vue|node\.js|django|flask)\b',
            r'\b(ai|ml|deep learning|machine learning|neural network|tensorflow|pytorch)\b',
            r'\b(api|rest|graphql|json|xml|sql|mongodb|postgresql)\b',
            r'\b(linux|windows|macos|ubuntu|debian|centos|docker|kubernetes)\b'
        ]
        
        # Place patterns
        place_patterns = [
            r'\b(usa|united states|america|uk|united kingdom|canada|australia|germany|france|italy|spain|japan|china|india|brazil|mexico)\b',
            r'\b(new york|london|paris|tokyo|beijing|delhi|mumbai|los angeles|chicago|houston|philadelphia|phoenix|san antonio|san diego|dallas|san jose)\b'
        ]
        
        # Organization patterns
        org_patterns = [
            r'\b(google|microsoft|apple|amazon|facebook|meta|netflix|tesla|nvidia|intel|amd|ibm|oracle|adobe|uber|airbnb|twitter|linkedin|youtube)\b',
            r'\b(university|college|institute|school|company|corporation|organization)\b'
        ]
        
        # Extract technologies
        for pattern in tech_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['technologies'].extend(matches)
        
        # Extract places
        for pattern in place_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['places'].extend(matches)
        
        # Extract organizations
        for pattern in org_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['organizations'].extend(matches)
        
        # Capitalized words might be people or concepts
        for word in capitalized:
            if word.lower() in entities['places'] or word.lower() in entities['organizations']:
                continue
            # If it's a title (like "Dr.", "Mr.", "Ms.")
            if word.lower() in ['dr', 'mr', 'mrs', 'ms', 'prof']:
                continue
            entities['people'].append(word)
        
        # Extract concepts (longer phrases)
        concept_patterns = [
            r'\b(climate change|artificial intelligence|machine learning|deep learning|quantum computing|blockchain|cryptocurrency)\b',
            r'\b(global warming|renewable energy|sustainable development|social media|digital transformation)\b'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities['concepts'].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _calculate_confidence(self, query: str, query_type: QueryType) -> float:
        """
        Calculate confidence in the analysis
        
        Args:
            query: The query string
            query_type: Detected query type
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.5
        
        # Increase confidence based on pattern matches
        patterns = self.query_patterns.get(query_type, [])
        match_count = 0
        
        for pattern in patterns:
            if re.search(pattern, query):
                match_count += 1
        
        # More pattern matches = higher confidence
        confidence = base_confidence + (match_count * 0.1)
        
        # Query length factor
        if len(query) > 10:
            confidence += 0.1
        
        # Question mark factor
        if '?' in query:
            confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return min(confidence, 1.0)
    
    def get_freshness_requirement(self, query_type: QueryType) -> str:
        """
        Get the freshness requirement for a query type
        
        Args:
            query_type: The query type
            
        Returns:
            Freshness requirement string
        """
        return self.freshness_requirements.get(query_type, "medium")
    
    def get_optimal_sources(self, query_type: QueryType) -> List[str]:
        """
        Get optimal data sources for a query type
        
        Args:
            query_type: The query type
            
        Returns:
            List of optimal source names
        """
        return self.optimal_sources.get(query_type, ["general_search"])