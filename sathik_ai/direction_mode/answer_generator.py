"""
Answer Generator for Direction Mode
Synthesizes answers from retrieved facts (NO neural network inference needed!)
"""

import re
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class AnswerFormat:
    """Data class for answer format options"""
    name: str
    description: str
    max_length: int
    include_citations: bool
    include_confidence: bool
    structure: str  # 'comprehensive', 'summary', 'bullet_points'

class AnswerGenerator:
    """
    Synthesizes answers from retrieved facts without neural network inference
    """
    
    def __init__(self):
        # Define answer formats
        self.answer_formats = {
            'comprehensive': AnswerFormat(
                name='comprehensive',
                description='Detailed answer with full context and citations',
                max_length=1000,
                include_citations=True,
                include_confidence=True,
                structure='comprehensive'
            ),
            'summary': AnswerFormat(
                name='summary',
                description='Brief, concise answer',
                max_length=300,
                include_citations=False,
                include_confidence=True,
                structure='summary'
            ),
            'bullet_points': AnswerFormat(
                name='bullet_points',
                description='Organized answer with bullet points',
                max_length=500,
                include_citations=True,
                include_confidence=True,
                structure='bullet_points'
            )
        }
        
        # Template patterns for different fact types
        self.fact_templates = {
            'definition': "Based on the information found, {subject} is {definition}.",
            'date_fact': "According to the sources, {event} occurred in {year}.",
            'location_fact': "The information indicates that {subject} is located in {location}.",
            'quantitative_fact': "The data shows that {quantity}.",
            'comparison_fact': "When comparing {item1} and {item2}, the sources indicate {comparison}."
        }
    
    def generate_answer(self, facts: List, query: str, format_type: str = 'comprehensive', 
                       key_information: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate an answer from extracted facts
        
        Args:
            facts: List of extracted facts
            query: Original query
            format_type: Answer format ('comprehensive', 'summary', 'bullet_points')
            key_information: Optional key information from extraction
            
        Returns:
            Dictionary containing the generated answer and metadata
        """
        if not facts:
            return self._generate_no_answer_response(query)
        
        # Select format
        format_config = self.answer_formats.get(format_type, self.answer_formats['comprehensive'])
        
        # Sort facts by confidence
        sorted_facts = sorted(facts, key=lambda x: x.confidence, reverse=True)
        
        # Generate answer based on format
        if format_config.structure == 'comprehensive':
            answer_text = self._generate_comprehensive_answer(sorted_facts, query, format_config)
        elif format_config.structure == 'summary':
            answer_text = self._generate_summary_answer(sorted_facts, query, format_config)
        elif format_config.structure == 'bullet_points':
            answer_text = self._generate_bullet_points_answer(sorted_facts, query, format_config)
        else:
            answer_text = self._generate_comprehensive_answer(sorted_facts, query, format_config)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(sorted_facts)
        
        # Generate citations
        citations = self._generate_citations(sorted_facts, format_config.include_citations)
        
        # Prepare response
        response = {
            'answer': answer_text,
            'confidence': overall_confidence,
            'citations': citations if format_config.include_citations else [],
            'sources_used': len(set(fact.source for fact in sorted_facts)),
            'facts_analyzed': len(sorted_facts),
            'format': format_type,
            'generated_at': time.time(),
            'query': query
        }
        
        # Add key information if provided
        if key_information:
            response['key_information'] = key_information
        
        logger.info(f"Generated {format_type} answer with {overall_confidence:.2f} confidence")
        return response
    
    def _generate_no_answer_response(self, query: str) -> Dict[str, Any]:
        """
        Generate response when no facts are available
        
        Args:
            query: Original query
            
        Returns:
            Response dictionary
        """
        return {
            'answer': f"I couldn't find reliable information to answer your question about \"{query}\". This might be because the topic is very specific, recent, or not well-documented online. You might want to try rephrasing your question or being more specific.",
            'confidence': 0.0,
            'citations': [],
            'sources_used': 0,
            'facts_analyzed': 0,
            'format': 'no_data',
            'generated_at': time.time(),
            'query': query
        }
    
    def _generate_comprehensive_answer(self, facts: List, query: str, format_config: AnswerFormat) -> str:
        """
        Generate a comprehensive answer
        
        Args:
            facts: List of sorted facts
            query: Original query
            format_config: Format configuration
            
        Returns:
            Comprehensive answer text
        """
        # Start with a direct answer based on top facts
        top_facts = facts[:3]  # Use top 3 most confident facts
        
        answer_parts = []
        
        # Main answer from top fact
        if top_facts:
            main_fact = top_facts[0]
            answer_parts.append(self._format_fact_as_sentence(main_fact))
        
        # Add supporting information from other top facts
        if len(top_facts) > 1:
            supporting_info = []
            for fact in top_facts[1:]:
                sentence = self._format_fact_as_sentence(fact)
                if sentence not in supporting_info:
                    supporting_info.append(sentence)
            
            if supporting_info:
                answer_parts.append("Additionally, " + " ".join(supporting_info))
        
        # Add context from key information if available
        context_additions = self._extract_contextual_information(facts, query)
        if context_additions:
            answer_parts.extend(context_additions)
        
        # Combine parts
        answer_text = " ".join(answer_parts)
        
        # Add confidence statement if requested
        if format_config.include_confidence:
            avg_confidence = sum(fact.confidence for fact in facts[:3]) / min(3, len(facts))
            confidence_text = self._format_confidence_statement(avg_confidence)
            answer_text += " " + confidence_text
        
        # Ensure answer doesn't exceed max length
        if len(answer_text) > format_config.max_length:
            answer_text = answer_text[:format_config.max_length-3] + "..."
        
        return answer_text
    
    def _generate_summary_answer(self, facts: List, query: str, format_config: AnswerFormat) -> str:
        """
        Generate a summary answer
        
        Args:
            facts: List of sorted facts
            query: Original query
            format_config: Format configuration
            
        Returns:
            Summary answer text
        """
        if not facts:
            return "I don't have enough information to provide a reliable answer."
        
        # Use the single most confident fact
        top_fact = facts[0]
        
        # Create a concise summary
        summary_parts = []
        
        # Direct answer
        summary_parts.append(self._format_fact_as_sentence(top_fact))
        
        # Add brief context if available
        if len(facts) > 1:
            second_fact = facts[1]
            context = self._extract_brief_context(second_fact)
            if context:
                summary_parts.append(context)
        
        summary_text = " ".join(summary_parts)
        
        # Add confidence if requested
        if format_config.include_confidence:
            confidence_text = self._format_confidence_statement(top_fact.confidence, brief=True)
            summary_text += " " + confidence_text
        
        # Ensure within length limit
        if len(summary_text) > format_config.max_length:
            summary_text = summary_text[:format_config.max_length-3] + "..."
        
        return summary_text
    
    def _generate_bullet_points_answer(self, facts: List, query: str, format_config: AnswerFormat) -> str:
        """
        Generate a bullet points answer
        
        Args:
            facts: List of sorted facts
            query: Original query
            format_config: Format configuration
            
        Returns:
            Bullet points answer text
        """
        if not facts:
            return "• No reliable information found for this query."
        
        # Select key facts for bullet points
        bullet_facts = facts[:5]  # Use top 5 facts
        
        bullet_points = []
        
        for i, fact in enumerate(bullet_facts, 1):
            # Format each fact as a bullet point
            bullet_text = self._format_fact_as_bullet_point(fact)
            bullet_points.append(f"{bullet_text}")
        
        # Join bullet points
        answer_text = "\n".join([f"• {point}" for point in bullet_points])
        
        # Add summary line if multiple facts
        if len(bullet_facts) > 1:
            avg_confidence = sum(fact.confidence for fact in bullet_facts) / len(bullet_facts)
            summary_line = f"\n• This information comes from {len(set(fact.source for fact in bullet_facts))} different sources."
            answer_text += summary_line
        
        return answer_text
    
    def _format_fact_as_sentence(self, fact) -> str:
        """
        Format a fact as a complete sentence
        
        Args:
            fact: Fact object
            
        Returns:
            Formatted sentence
        """
        fact_text = fact.fact.strip()
        
        # Capitalize first letter if needed
        if fact_text and fact_text[0].islower():
            fact_text = fact_text[0].upper() + fact_text[1:]
        
        # Ensure it ends with proper punctuation
        if fact_text and not fact_text.endswith(('.', '!', '?')):
            fact_text += '.'
        
        return fact_text
    
    def _format_fact_as_bullet_point(self, fact) -> str:
        """
        Format a fact as a bullet point
        
        Args:
            fact: Fact object
            
        Returns:
            Formatted bullet point
        """
        fact_text = fact.fact.strip()
        
        # Keep bullet points concise
        if len(fact_text) > 100:
            fact_text = fact_text[:97] + "..."
        
        return fact_text
    
    def _extract_contextual_information(self, facts: List, query: str) -> List[str]:
        """
        Extract additional contextual information from facts
        
        Args:
            facts: List of facts
            query: Original query
            
        Returns:
            List of contextual statements
        """
        context_info = []
        
        # Look for dates, people, places that add context
        for fact in facts[:3]:  # Check top 3 facts
            entities = getattr(fact, 'entities', [])
            
            # Add context about entities if relevant to query
            query_lower = query.lower()
            for entity in entities:
                if entity.lower() in query_lower:
                    context = f"This is related to {entity}."
                    if context not in context_info:
                        context_info.append(context)
                    break
        
        return context_info
    
    def _extract_brief_context(self, fact) -> str:
        """
        Extract brief context from a fact
        
        Args:
            fact: Fact object
            
        Returns:
            Brief contextual statement
        """
        # Keep it very brief for summary format
        entities = getattr(fact, 'entities', [])
        
        if entities:
            return f"Also mentioned: {', '.join(entities[:2])}."  # Max 2 entities
        
        # Fallback to a shortened version of the fact
        fact_text = fact.fact[:50]
        if len(fact.fact) > 50:
            fact_text += "..."
        return f"Additionally: {fact_text}."
    
    def _format_confidence_statement(self, confidence: float, brief: bool = False) -> str:
        """
        Format confidence as a readable statement
        
        Args:
            confidence: Confidence score (0-1)
            brief: Whether to use brief format
            
        Returns:
            Confidence statement
        """
        if brief:
            if confidence >= 0.8:
                return "(High confidence)"
            elif confidence >= 0.6:
                return "(Moderate confidence)"
            else:
                return "(Low confidence)"
        else:
            if confidence >= 0.8:
                return "This information has high confidence based on reliable sources."
            elif confidence >= 0.6:
                return "This information has moderate confidence based on the available sources."
            else:
                return "This information should be verified as it has lower confidence."
    
    def _calculate_overall_confidence(self, facts: List) -> float:
        """
        Calculate overall confidence from multiple facts
        
        Args:
            facts: List of facts
            
        Returns:
            Overall confidence score
        """
        if not facts:
            return 0.0
        
        # Weight by confidence and number of supporting sources
        total_weight = 0
        weighted_sum = 0
        
        for fact in facts:
            # Weight by confidence and source reliability
            source_weight = self._get_source_reliability_weight(fact.source)
            weight = fact.confidence * source_weight
            
            weighted_sum += weight
            total_weight += 1
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_source_reliability_weight(self, source: str) -> float:
        """
        Get reliability weight for a source
        
        Args:
            source: Source name
            
        Returns:
            Reliability weight (0-1)
        """
        reliability_weights = {
            'wikipedia': 1.0,
            'arxiv': 1.0,
            'google': 0.9,
            'newsapi': 0.8,
            'duckduckgo': 0.7
        }
        
        return reliability_weights.get(source, 0.5)
    
    def _generate_citations(self, facts: List, include_citations: bool) -> List[Dict[str, Any]]:
        """
        Generate citation information for facts
        
        Args:
            facts: List of facts
            include_citations: Whether to include citations
            
        Returns:
            List of citation dictionaries
        """
        if not include_citations:
            return []
        
        citations = []
        
        for i, fact in enumerate(facts[:5], 1):  # Max 5 citations
            citation = {
                'number': i,
                'source': fact.source,
                'url': getattr(fact, 'source_url', ''),
                'fact': fact.fact[:100] + "..." if len(fact.fact) > 100 else fact.fact,
                'confidence': round(fact.confidence, 2)
            }
            citations.append(citation)
        
        return citations
    
    def get_available_formats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available answer formats
        
        Returns:
            Dictionary of format information
        """
        formats_info = {}
        
        for format_name, format_config in self.answer_formats.items():
            formats_info[format_name] = {
                'name': format_config.name,
                'description': format_config.description,
                'max_length': format_config.max_length,
                'includes_citations': format_config.include_citations,
                'includes_confidence': format_config.include_confidence
            }
        
        return formats_info