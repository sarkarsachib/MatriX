"""
Information Extractor for Direction Mode
Parses search results and extracts structured facts using NLP
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import html

logger = logging.getLogger(__name__)

@dataclass
class ExtractedFact:
    """Data class for extracted facts"""
    fact: str
    confidence: float
    source: str
    source_url: str
    context: str
    entities: List[str]
    timestamp: float

class InformationExtractor:
    """
    Extracts structured information from search results
    """
    
    def __init__(self):
        # Common fact patterns
        self.fact_patterns = {
            'definition': [
                r'(.*?) is (.*?)(?:\.|,|\n|$)',
                r'(.*?) are (.*?)(?:\.|,|\n|$)',
                r'The definition of (.*?) is (.*?)(?:\.|,|\n|$)',
                r'(.*?) means (.*?)(?:\.|,|\n|$)'
            ],
            'date_fact': [
                r'(\d{4}) saw (.*?)(?:\.|,|\n|$)',
                r'In (\d{4}), (.*?)(?:\.|,|\n|$)',
                r'(\d{4}) is when (.*?)(?:\.|,|\n|$)',
                r'Born in (\d{4}), (.*?)(?:\.|,|\n|$)'
            ],
            'location_fact': [
                r'(.*?) is located in (.*?)(?:\.|,|\n|$)',
                r'(.*?) is in (.*?)(?:\.|,|\n|$)',
                r'From (.*?), (.*?)(?:\.|,|\n|$)',
                r'Located in (.*?), (.*?)(?:\.|,|\n|$)'
            ],
            'quantitative_fact': [
                r'(\d+(?:\.\d+)?) (.*?)(?:\.|,|\n|$)',
                r'(.*?) (?:costs|prices?) (\$?\d+(?:\.\d+)?)(?:\.|,|\n|$)',
                r'(.*?) has (\d+) (.*?)(?:\.|,|\n|$)'
            ],
            'comparison_fact': [
                r'(.*?) is (?:better|worse|larger|smaller|older|younger) than (.*?)(?:\.|,|\n|$)',
                r'(.*?) versus (.*?)(?:\.|,|\n|$)',
                r'(.*?) compared to (.*?)(?:\.|,|\n|$)'
            ]
        }
        
        # Entity patterns for recognition
        self.entity_patterns = {
            'person': [
                r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # Full names
                r'\b(Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.) [A-Z][a-z]+\b',  # Titles + names
                r'\b(CEO|CTO|CFO-Founder-President|Leader|Director|Manager) [A-Z][a-z]+\b'  # Job titles + names
            ],
            'organization': [
                r'\b(Google|Microsoft|Apple|Amazon|Facebook|Meta|Netflix|Tesla|NVIDIA|Intel|AMD|IBM|Oracle|Adobe|Uber|Airbnb|Twitter|LinkedIn|YouTube)\b',
                r'\b(University|College|Institute|School) of [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
                r'\b([A-Z][a-z]+(?:&[A-Z][a-z]+)? (?:Inc|Corp|LLC|Ltd|Company))\b'
            ],
            'location': [
                r'\b(New York|London|Paris|Tokyo|Beijing|Delhi|Mumbai|Los Angeles|Chicago|Houston|Philadelphia|Phoenix|San Antonio|San Diego|Dallas|San Jose)\b',
                r'\b(USA|United States|UK|United Kingdom|Canada|Australia|Germany|France|Italy|Spain|Japan|China|India|Brazil|Mexico)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*, [A-Z][a-z]+)\b'  # City, State/Country
            ],
            'technology': [
                r'\b(Python|JavaScript|Java|C\+\+|React|Angular|Vue|Node\.js|Django|Flask)\b',
                r'\b(AI|ML|Deep Learning|Machine Learning|Neural Network|TensorFlow|PyTorch)\b',
                r'\b(API|REST|GraphQL|JSON|XML|SQL|MongoDB|PostgreSQL)\b',
                r'\b(Linux|Windows|macOS|Ubuntu|Debian|CentOS|Docker|Kubernetes)\b'
            ],
            'date': [
                r'\b(\d{4})\b',  # Years
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b \d{1,2}, \d{4}\b',
                r'\b\d{1,2}/\d{1,2}/\d{4}\b',  # MM/DD/YYYY
                r'\b\d{4}-\d{2}-\d{2}\b'  # YYYY-MM-DD
            ]
        }
        
        # Confidence scoring weights
        self.confidence_weights = {
            'pattern_match': 0.3,
            'entity_presence': 0.2,
            'source_reliability': 0.3,
            'text_length': 0.1,
            'position': 0.1
        }
    
    def extract_information(self, search_results: List) -> List[ExtractedFact]:
        """
        Extract structured information from search results
        
        Args:
            search_results: List of search result objects
            
        Returns:
            List of extracted facts
        """
        all_facts = []
        
        for result in search_results:
            try:
                # Extract facts from title and snippet
                title_facts = self._extract_facts_from_text(
                    result.title, result.source, result.url, "title"
                )
                snippet_facts = self._extract_facts_from_text(
                    result.snippet, result.source, result.url, "snippet"
                )
                
                all_facts.extend(title_facts)
                all_facts.extend(snippet_facts)
                
            except Exception as e:
                logger.warning(f"Error extracting facts from result: {e}")
                continue
        
        # Remove duplicates and rank by confidence
        unique_facts = self._deduplicate_facts(all_facts)
        ranked_facts = self._rank_facts(unique_facts)
        
        logger.info(f"Extracted {len(ranked_facts)} unique facts from {len(search_results)} results")
        return ranked_facts
    
    def _extract_facts_from_text(self, text: str, source: str, source_url: str, position: str) -> List[ExtractedFact]:
        """
        Extract facts from a piece of text
        
        Args:
            text: Text to extract facts from
            source: Source name
            source_url: Source URL
            position: Position in result (title, snippet)
            
        Returns:
            List of extracted facts
        """
        if not text or len(text.strip()) < 10:
            return []
        
        facts = []
        
        # Clean HTML and normalize text
        clean_text = html.unescape(text)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)  # Remove HTML tags
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Normalize whitespace
        
        # Extract facts using patterns
        for fact_type, patterns in self.fact_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, clean_text, re.IGNORECASE)
                for match in matches:
                    try:
                        fact_text = match.group(0).strip()
                        
                        # Skip very short or very long facts
                        if len(fact_text) < 10 or len(fact_text) > 200:
                            continue
                        
                        # Calculate confidence
                        confidence = self._calculate_fact_confidence(
                            fact_text, source, position, len(clean_text)
                        )
                        
                        # Extract entities from the fact
                        entities = self._extract_entities_from_text(fact_text)
                        
                        # Create context around the fact
                        start_pos = max(0, match.start() - 50)
                        end_pos = min(len(clean_text), match.end() + 50)
                        context = clean_text[start_pos:end_pos].strip()
                        
                        fact = ExtractedFact(
                            fact=fact_text,
                            confidence=confidence,
                            source=source,
                            source_url=source_url,
                            context=context,
                            entities=entities,
                            timestamp=time.time()
                        )
                        
                        facts.append(fact)
                        
                    except Exception as e:
                        logger.debug(f"Error processing match: {e}")
                        continue
        
        # If no pattern matches, try to extract sentence-level facts
        if not facts:
            sentences = re.split(r'[.!?]+', clean_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 150:
                    confidence = self._calculate_fact_confidence(
                        sentence, source, position, len(clean_text)
                    )
                    
                    entities = self._extract_entities_from_text(sentence)
                    
                    fact = ExtractedFact(
                        fact=sentence,
                        confidence=confidence,
                        source=source,
                        source_url=source_url,
                        context=clean_text,
                        entities=entities,
                        timestamp=time.time()
                    )
                    
                    facts.append(fact)
        
        return facts
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """
        Extract entities from text using patterns
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities.extend(matches)
        
        # Remove duplicates and return
        return list(set(entities))
    
    def _calculate_fact_confidence(self, fact_text: str, source: str, position: str, total_length: int) -> float:
        """
        Calculate confidence score for an extracted fact
        
        Args:
            fact_text: The extracted fact
            source: Source name
            position: Position in result (title, snippet)
            total_length: Total length of source text
            
        Returns:
            Confidence score between 0 and 1
        """
        confidence = 0.5  # Base confidence
        
        # Pattern matching boost
        for fact_type, patterns in self.fact_patterns.items():
            for pattern in patterns:
                if re.search(pattern, fact_text, re.IGNORECASE):
                    confidence += self.confidence_weights['pattern_match']
                    break
        
        # Entity presence boost
        entities = self._extract_entities_from_text(fact_text)
        if entities:
            confidence += self.confidence_weights['entity_presence']
        
        # Source reliability boost
        source_reliability = self._get_source_reliability(source)
        confidence += source_reliability * self.confidence_weights['source_reliability']
        
        # Text length optimization
        optimal_length = 50  # Sweet spot for fact length
        length_diff = abs(len(fact_text) - optimal_length)
        length_factor = max(0, 1 - (length_diff / 100))  # Decreases as length diverges
        confidence += length_factor * self.confidence_weights['text_length']
        
        # Position boost (titles are more reliable)
        if position == "title":
            confidence += self.confidence_weights['position']
        
        # Ensure confidence is between 0 and 1
        return min(confidence, 1.0)
    
    def _get_source_reliability(self, source: str) -> float:
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
    
    def _deduplicate_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """
        Remove duplicate facts based on content similarity
        
        Args:
            facts: List of extracted facts
            
        Returns:
            Deduplicated list of facts
        """
        seen_facts = set()
        unique_facts = []
        
        for fact in facts:
            # Create a normalized version for comparison
            normalized_fact = re.sub(r'[^\w\s]', '', fact.fact.lower())
            normalized_fact = re.sub(r'\s+', ' ', normalized_fact).strip()
            
            # Create a hash for quick comparison
            fact_hash = hash(normalized_fact)
            
            if fact_hash not in seen_facts:
                seen_facts.add(fact_hash)
                unique_facts.append(fact)
        
        return unique_facts
    
    def _rank_facts(self, facts: List[ExtractedFact]) -> List[ExtractedFact]:
        """
        Rank facts by confidence and relevance
        
        Args:
            facts: List of extracted facts
            
        Returns:
            Ranked list of facts
        """
        # Sort by confidence score (descending)
        return sorted(facts, key=lambda x: x.confidence, reverse=True)
    
    def extract_key_information(self, facts: List[ExtractedFact], query: str) -> Dict[str, Any]:
        """
        Extract key information relevant to the query
        
        Args:
            facts: List of extracted facts
            query: Original query
            
        Returns:
            Dictionary containing key information
        """
        key_info = {
            'main_facts': [],
            'definitions': [],
            'dates': [],
            'people': [],
            'places': [],
            'organizations': [],
            'quantitative_data': [],
            'sources': []
        }
        
        query_lower = query.lower()
        
        for fact in facts:
            # Categorize facts
            fact_lower = fact.fact.lower()
            
            # Check relevance to query
            query_words = set(query_lower.split())
            fact_words = set(fact_lower.split())
            overlap = len(query_words.intersection(fact_words))
            
            if overlap > 0:
                # Main facts - high relevance
                if fact.confidence > 0.7:
                    key_info['main_facts'].append({
                        'fact': fact.fact,
                        'confidence': fact.confidence,
                        'source': fact.source,
                        'url': fact.source_url
                    })
                
                # Categorize by type
                if any(word in fact_lower for word in ['is', 'are', 'means', 'definition']):
                    key_info['definitions'].append({
                        'term': fact.fact.split()[0] if fact.fact.split() else '',
                        'definition': fact.fact,
                        'confidence': fact.confidence,
                        'source': fact.source
                    })
                
                # Extract dates
                date_matches = re.findall(r'\b\d{4}\b', fact.fact)
                if date_matches:
                    key_info['dates'].extend(date_matches)
                
                # Extract people
                people = [entity for entity in fact.entities if self._is_person(entity)]
                key_info['people'].extend(people)
                
                # Extract places
                places = [entity for entity in fact.entities if self._is_place(entity)]
                key_info['places'].extend(places)
                
                # Extract organizations
                orgs = [entity for entity in fact.entities if self._is_organization(entity)]
                key_info['organizations'].extend(orgs)
                
                # Extract quantitative data
                quant_matches = re.findall(r'\d+(?:\.\d+)?', fact.fact)
                if quant_matches:
                    key_info['quantitative_data'].append({
                        'value': quant_matches[0],
                        'context': fact.fact,
                        'confidence': fact.confidence
                    })
            
            # Collect sources
            if fact.source not in key_info['sources']:
                key_info['sources'].append(fact.source)
        
        # Remove duplicates from lists
        for key in ['dates', 'people', 'places', 'organizations']:
            key_info[key] = list(set(key_info[key]))
        
        # Sort main facts by confidence
        key_info['main_facts'].sort(key=lambda x: x['confidence'], reverse=True)
        
        return key_info
    
    def _is_person(self, entity: str) -> bool:
        """Check if entity is likely a person"""
        return bool(re.match(r'\b([A-Z][a-z]+ [A-Z][a-z]+|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.)\b', entity))
    
    def _is_place(self, entity: str) -> bool:
        """Check if entity is likely a place"""
        # Common place patterns
        place_indicators = ['city', 'state', 'country', 'county', 'town', 'village']
        return any(indicator in entity.lower() for indicator in place_indicators) or \
               bool(re.match(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*, [A-Z][a-z]+)\b', entity))
    
    def _is_organization(self, entity: str) -> bool:
        """Check if entity is likely an organization"""
        org_indicators = ['inc', 'corp', 'llc', 'ltd', 'company', 'university', 'college', 'institute']
        return any(indicator in entity.lower() for indicator in org_indicators) or \
               entity.lower() in ['google', 'microsoft', 'apple', 'amazon', 'facebook', 'meta']