"""
Fact Checker & Validator for Direction Mode
Validates facts against sources and manages citations
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import re

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Data class for fact validation results"""
    fact: str
    is_valid: bool
    confidence: float
    contradictions: List[str]
    supporting_sources: List[str]
    validation_method: str
    timestamp: float

class FactChecker:
    """
    Validates facts against sources and manages citation tracking
    """
    
    def __init__(self):
        # Source reliability scores
        self.source_reliability = {
            'wikipedia': 0.9,
            'arxiv': 0.95,
            'google': 0.8,
            'newsapi': 0.7,
            'duckduckgo': 0.6,
            'stackoverflow': 0.8,
            'github': 0.8
        }
        
        # Contradiction patterns
        self.contradiction_patterns = [
            (r'\b(not|never|nothing|no one|nothing)\b', r'\b(yes|always|everyone|all)\b'),
            (r'\b(is|are|was|were)\b', r'\b(is not|are not|was not|were not)\b'),
            (r'\b(true|fact)\b', r'\b(false|myth|incorrect)\b'),
            (r'\b(correct|right|accurate)\b', r'\b(incorrect|wrong|inaccurate)\b')
        ]
        
        # Fact type validation patterns
        self.validation_patterns = {
            'definition': {
                'patterns': [
                    r'(.*?) is (.*?)(?:\.|,|\n|$)',
                    r'(.*?) are (.*?)(?:\.|,|\n|$)'
                ],
                'keywords': ['is', 'are', 'means', 'defined as', 'refers to']
            },
            'quantitative': {
                'patterns': [
                    r'(\d+(?:\.\d+)?) (.*?)(?:\.|,|\n|$)',
                    r'(.*?) (?:costs|prices?) (\$?\d+(?:\.\d+)?)'
                ],
                'keywords': ['number', 'count', 'amount', 'price', 'cost', 'total']
            },
            'date_fact': {
                'patterns': [
                    r'(\d{4})',
                    r'(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}'
                ],
                'keywords': ['year', 'date', 'when', 'occurred', 'happened']
            }
        }
    
    def validate_facts(self, facts: List, query: str) -> List[ValidationResult]:
        """
        Validate a list of facts
        
        Args:
            facts: List of extracted facts
            query: Original query for context
            
        Returns:
            List of validation results
        """
        validation_results = []
        
        for fact in facts:
            try:
                # Validate individual fact
                validation = self._validate_single_fact(fact, facts, query)
                validation_results.append(validation)
                
            except Exception as e:
                logger.warning(f"Error validating fact: {e}")
                # Create failed validation result
                failed_validation = ValidationResult(
                    fact=fact.fact if hasattr(fact, 'fact') else str(fact),
                    is_valid=False,
                    confidence=0.0,
                    contradictions=[],
                    supporting_sources=[],
                    validation_method='error',
                    timestamp=time.time()
                )
                validation_results.append(failed_validation)
        
        # Sort by confidence
        validation_results.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Validated {len(validation_results)} facts")
        return validation_results
    
    def _validate_single_fact(self, fact, all_facts: List, query: str) -> ValidationResult:
        """
        Validate a single fact
        
        Args:
            fact: Fact to validate
            all_facts: All facts for cross-validation
            query: Original query
            
        Returns:
            ValidationResult object
        """
        fact_text = fact.fact if hasattr(fact, 'fact') else str(fact)
        source = getattr(fact, 'source', 'unknown')
        source_url = getattr(fact, 'source_url', '')
        
        # Calculate base confidence
        confidence = getattr(fact, 'confidence', 0.5)
        
        # Validate fact structure
        structure_valid = self._validate_fact_structure(fact_text)
        
        # Check for contradictions with other facts
        contradictions = self._find_contradictions(fact_text, all_facts)
        
        # Find supporting sources
        supporting_sources = self._find_supporting_sources(fact_text, all_facts, source)
        
        # Determine validation method
        validation_method = self._determine_validation_method(fact_text)
        
        # Calculate final confidence
        final_confidence = self._calculate_validation_confidence(
            confidence, structure_valid, contradictions, supporting_sources
        )
        
        # Determine if fact is valid
        is_valid = (
            final_confidence > 0.4 and 
            len(contradictions) == 0 and
            structure_valid
        )
        
        return ValidationResult(
            fact=fact_text,
            is_valid=is_valid,
            confidence=final_confidence,
            contradictions=contradictions,
            supporting_sources=supporting_sources,
            validation_method=validation_method,
            timestamp=time.time()
        )
    
    def _validate_fact_structure(self, fact_text: str) -> bool:
        """
        Validate the structure of a fact
        
        Args:
            fact_text: Text of the fact
            
        Returns:
            Whether the fact has valid structure
        """
        if not fact_text or len(fact_text.strip()) < 5:
            return False
        
        # Check for minimum components
        words = fact_text.split()
        if len(words) < 3:
            return False
        
        # Check for valid patterns
        for fact_type, pattern_info in self.validation_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, fact_text, re.IGNORECASE):
                    return True
        
        # If no pattern matches, check for reasonable sentence structure
        if self._has_reasonable_sentence_structure(fact_text):
            return True
        
        return False
    
    def _has_reasonable_sentence_structure(self, text: str) -> bool:
        """
        Check if text has reasonable sentence structure
        
        Args:
            text: Text to check
            
        Returns:
            Whether structure is reasonable
        """
        # Basic checks for reasonable structure
        if not text.strip():
            return False
        
        # Check for proper capitalization
        if text[0].islower():
            return False
        
        # Check for excessive repetition
        words = text.lower().split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # If any word appears more than 30% of the time, likely spam
        max_freq = max(word_freq.values()) if word_freq else 0
        if len(words) > 5 and max_freq / len(words) > 0.3:
            return False
        
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c != ' ')
        if len(text) > 10 and special_chars / len(text) > 0.3:
            return False
        
        return True
    
    def _find_contradictions(self, fact_text: str, all_facts: List) -> List[str]:
        """
        Find contradictions with other facts
        
        Args:
            fact_text: Text of the fact to check
            all_facts: All facts for comparison
            
        Returns:
            List of contradictory statements
        """
        contradictions = []
        fact_words = set(fact_text.lower().split())
        
        for other_fact in all_facts:
            if hasattr(other_fact, 'fact') and other_fact.fact != fact_text:
                other_text = other_fact.fact.lower()
                other_words = set(other_text.split())
                
                # Check for word overlap (facts about same topic)
                overlap = len(fact_words.intersection(other_words))
                if overlap >= 2:  # At least 2 shared words
                    # Check for contradictory patterns
                    if self._are_contradictory(fact_text, other_text):
                        contradictions.append(other_text[:100] + "..." if len(other_text) > 100 else other_text)
        
        return contradictions[:3]  # Limit to top 3 contradictions
    
    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are contradictory
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Whether texts are contradictory
        """
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for pattern1, pattern2 in self.contradiction_patterns:
            has_pattern1_text1 = bool(re.search(pattern1, text1_lower))
            has_pattern2_text1 = bool(re.search(pattern2, text1_lower))
            has_pattern1_text2 = bool(re.search(pattern1, text2_lower))
            has_pattern2_text2 = bool(re.search(pattern2, text2_lower))
            
            # Check for contradictions
            if (has_pattern1_text1 and has_pattern2_text2) or (has_pattern2_text1 and has_pattern1_text2):
                return True
        
        # Check for direct contradictions
        contradiction_pairs = [
            ('is', 'is not'),
            ('are', 'are not'),
            ('true', 'false'),
            ('correct', 'incorrect'),
            ('yes', 'no'),
            ('all', 'none'),
            ('always', 'never')
        ]
        
        for word1, word2 in contradiction_pairs:
            if word1 in text1_lower and word2 in text2_lower:
                return True
            if word2 in text1_lower and word1 in text2_lower:
                return True
        
        return False
    
    def _find_supporting_sources(self, fact_text: str, all_facts: List, original_source: str) -> List[str]:
        """
        Find sources that support the fact
        
        Args:
            fact_text: Text of the fact
            all_facts: All facts for comparison
            original_source: Original source of the fact
            
        Returns:
            List of supporting source names
        """
        supporting_sources = []
        fact_words = set(fact_text.lower().split())
        
        for other_fact in all_facts:
            if hasattr(other_fact, 'fact') and other_fact.fact != fact_text:
                other_text = other_fact.fact.lower()
                other_words = set(other_text.split())
                
                # Check for similarity (supporting information)
                overlap = len(fact_words.intersection(other_words))
                if overlap >= 3:  # At least 3 shared words
                    source = getattr(other_fact, 'source', 'unknown')
                    if source != original_source and source not in supporting_sources:
                        supporting_sources.append(source)
        
        return supporting_sources[:3]  # Limit to top 3 supporting sources
    
    def _determine_validation_method(self, fact_text: str) -> str:
        """
        Determine what method was used to validate the fact
        
        Args:
            fact_text: Text of the fact
            
        Returns:
            Validation method description
        """
        # Check fact type
        for fact_type, pattern_info in self.validation_patterns.items():
            for pattern in pattern_info['patterns']:
                if re.search(pattern, fact_text, re.IGNORECASE):
                    return f"pattern_match_{fact_type}"
        
        # Check for numbers (quantitative validation)
        if re.search(r'\d+', fact_text):
            return "quantitative_validation"
        
        # Check for dates
        if re.search(r'\d{4}', fact_text):
            return "date_validation"
        
        # Default structure validation
        return "structure_validation"
    
    def _calculate_validation_confidence(self, base_confidence: float, structure_valid: bool, 
                                      contradictions: List[str], supporting_sources: List[str]) -> float:
        """
        Calculate final validation confidence
        
        Args:
            base_confidence: Base confidence from extraction
            structure_valid: Whether structure is valid
            contradictions: List of contradictions found
            supporting_sources: List of supporting sources
            
        Returns:
            Final validation confidence
        """
        confidence = base_confidence
        
        # Structure validation boost
        if structure_valid:
            confidence += 0.1
        
        # Contradiction penalty
        if contradictions:
            confidence -= len(contradictions) * 0.2
        
        # Supporting sources boost
        if supporting_sources:
            confidence += len(supporting_sources) * 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def get_fact_reliability_score(self, fact_text: str, source: str) -> float:
        """
        Get reliability score for a specific fact
        
        Args:
            fact_text: Text of the fact
            source: Source of the fact
            
        Returns:
            Reliability score between 0 and 1
        """
        # Base score from source reliability
        source_score = self.source_reliability.get(source.lower(), 0.5)
        
        # Boost for certain fact patterns
        pattern_boost = 0.0
        
        if re.search(r'\d{4}', fact_text):  # Has a year
            pattern_boost += 0.1
        
        if re.search(r'\b(is|are|was|were)\b', fact_text.lower()):  # Is/are statement
            pattern_boost += 0.05
        
        if len(fact_text.split()) >= 5:  # Reasonable length
            pattern_boost += 0.05
        
        final_score = min(1.0, source_score + pattern_boost)
        
        return final_score
    
    def generate_citation_info(self, facts: List) -> Dict[str, Any]:
        """
        Generate citation information from facts
        
        Args:
            facts: List of facts
            
        Returns:
            Dictionary with citation information
        """
        citations = []
        source_counts = {}
        
        for i, fact in enumerate(facts[:10], 1):  # Limit to top 10
            source = getattr(fact, 'source', 'unknown')
            source_url = getattr(fact, 'source_url', '')
            fact_text = fact.fact if hasattr(fact, 'fact') else str(fact)
            
            # Count sources
            source_counts[source] = source_counts.get(source, 0) + 1
            
            # Create citation
            citation = {
                'number': i,
                'source': source,
                'url': source_url,
                'fact_preview': fact_text[:100] + "..." if len(fact_text) > 100 else fact_text,
                'confidence': getattr(fact, 'confidence', 0.0)
            }
            citations.append(citation)
        
        return {
            'citations': citations,
            'source_distribution': source_counts,
            'total_sources': len(set(getattr(fact, 'source', 'unknown') for fact in facts)),
            'average_confidence': sum(getattr(fact, 'confidence', 0) for fact in facts) / len(facts) if facts else 0
        }
    
    def check_fact_consistency(self, facts: List) -> Dict[str, Any]:
        """
        Check consistency across multiple facts
        
        Args:
            facts: List of facts to check
            
        Returns:
            Dictionary with consistency analysis
        """
        consistency_report = {
            'overall_consistency': 1.0,
            'contradictions_found': 0,
            'consistent_facts': 0,
            'inconsistent_facts': 0,
            'consistency_score': 1.0
        }
        
        if not facts:
            return consistency_report
        
        # Find all contradictions
        all_contradictions = []
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if self._are_contradictory(
                    fact1.fact if hasattr(fact1, 'fact') else str(fact1),
                    fact2.fact if hasattr(fact2, 'fact') else str(fact2)
                ):
                    all_contradictions.append((i, j))
        
        consistency_report['contradictions_found'] = len(all_contradictions)
        
        # Calculate consistency score
        total_possible_pairs = len(facts) * (len(facts) - 1) // 2
        if total_possible_pairs > 0:
            consistency_ratio = 1.0 - (len(all_contradictions) / total_possible_pairs)
            consistency_report['consistency_score'] = consistency_ratio
        
        # Count consistent vs inconsistent facts
        contradictory_facts = set()
        for i, j in all_contradictions:
            contradictory_facts.add(i)
            contradictory_facts.add(j)
        
        consistency_report['inconsistent_facts'] = len(contradictory_facts)
        consistency_report['consistent_facts'] = len(facts) - len(contradictory_facts)
        
        return consistency_report