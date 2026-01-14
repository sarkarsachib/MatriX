import re
import json
from typing import List, Dict, Any, Optional
from collections import Counter

class TruthComparator:
    """Cross-validates information from multiple web sources."""
    
    def __init__(self):
        self.source_reliability = {}  # Track reliability scores for different sources
        
    def add_source_reliability(self, source: str, reliability_score: float):
        """Add or update reliability score for a source (0.0 to 1.0)."""
        self.source_reliability[source] = max(0.0, min(1.0, reliability_score))
    
    def compare_facts(self, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare facts from multiple sources and determine truth confidence.
        
        Args:
            facts: List of dictionaries with 'content', 'source', and optional 'timestamp'
        
        Returns:
            Dictionary with 'consensus', 'confidence', 'sources', and 'conflicts'
        """
        if not facts:
            return {"consensus": None, "confidence": 0.0, "sources": [], "conflicts": []}
        
        # Group similar facts
        fact_groups = self._group_similar_facts(facts)
        
        # Find consensus
        best_group = max(fact_groups, key=lambda g: self._calculate_group_score(g))
        
        # Calculate confidence based on source reliability and agreement
        confidence = self._calculate_confidence(best_group, len(facts))
        
        # Identify conflicts
        conflicts = [group for group in fact_groups if group != best_group]
        
        return {
            "consensus": best_group[0]["content"] if best_group else None,
            "confidence": confidence,
            "sources": [fact["source"] for fact in best_group],
            "conflicts": conflicts
        }
    
    def _group_similar_facts(self, facts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group facts that are similar in content."""
        groups = []
        
        for fact in facts:
            placed = False
            for group in groups:
                if self._are_facts_similar(fact, group[0]):
                    group.append(fact)
                    placed = True
                    break
            
            if not placed:
                groups.append([fact])
        
        return groups
    
    def _are_facts_similar(self, fact1: Dict[str, Any], fact2: Dict[str, Any]) -> bool:
        """Simple similarity check based on content overlap."""
        content1 = fact1["content"].lower()
        content2 = fact2["content"].lower()
        
        # Simple word overlap threshold
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return overlap / union > 0.5  # 50% similarity threshold
    
    def _calculate_group_score(self, group: List[Dict[str, Any]]) -> float:
        """Calculate score for a group of facts."""
        base_score = len(group)  # More sources = higher score
        
        # Weight by source reliability
        reliability_bonus = sum(
            self.source_reliability.get(fact["source"], 0.5) for fact in group
        )
        
        return base_score + reliability_bonus
    
    def _calculate_confidence(self, best_group: List[Dict[str, Any]], total_facts: int) -> float:
        """Calculate confidence score for the consensus."""
        if not best_group or total_facts == 0:
            return 0.0
        
        # Base confidence from agreement ratio
        agreement_ratio = len(best_group) / total_facts
        
        # Boost confidence based on source reliability
        avg_reliability = sum(
            self.source_reliability.get(fact["source"], 0.5) for fact in best_group
        ) / len(best_group)
        
        return min(1.0, agreement_ratio * avg_reliability * 1.2)

class ContentFilter:
    """Blocks NSFW, hate speech, and bias (customizable)."""
    
    def __init__(self):
        # Basic keyword lists (in a real system, these would be more comprehensive)
        self.nsfw_keywords = {
            "explicit", "adult", "sexual", "porn", "nude", "naked", "sex"
        }
        
        self.hate_keywords = {
            "hate", "racist", "nazi", "kill", "murder", "terrorist", "bomb"
        }
        
        self.bias_indicators = {
            "always", "never", "all", "none", "every", "completely", "totally"
        }
        
        self.severity_weights = {
            "nsfw": 1.0,
            "hate": 2.0,
            "bias": 0.5
        }
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for NSFW, hate speech, and bias.
        
        Returns:
            Dictionary with 'is_safe', 'issues', 'severity', and 'filtered_content'
        """
        content_lower = content.lower()
        issues = []
        total_severity = 0.0
        
        # Check for NSFW content
        nsfw_matches = [word for word in self.nsfw_keywords if word in content_lower]
        if nsfw_matches:
            issues.append({
                "type": "nsfw",
                "matches": nsfw_matches,
                "severity": len(nsfw_matches) * self.severity_weights["nsfw"]
            })
            total_severity += len(nsfw_matches) * self.severity_weights["nsfw"]
        
        # Check for hate speech
        hate_matches = [word for word in self.hate_keywords if word in content_lower]
        if hate_matches:
            issues.append({
                "type": "hate",
                "matches": hate_matches,
                "severity": len(hate_matches) * self.severity_weights["hate"]
            })
            total_severity += len(hate_matches) * self.severity_weights["hate"]
        
        # Check for bias indicators
        bias_matches = [word for word in self.bias_indicators if word in content_lower]
        if bias_matches:
            issues.append({
                "type": "bias",
                "matches": bias_matches,
                "severity": len(bias_matches) * self.severity_weights["bias"]
            })
            total_severity += len(bias_matches) * self.severity_weights["bias"]
        
        # Determine if content is safe
        is_safe = total_severity < 2.0  # Threshold for blocking
        
        # Create filtered content if needed
        filtered_content = self._filter_content(content, issues) if not is_safe else content
        
        return {
            "is_safe": is_safe,
            "issues": issues,
            "severity": total_severity,
            "filtered_content": filtered_content
        }
    
    def _filter_content(self, content: str, issues: List[Dict[str, Any]]) -> str:
        """Filter out problematic content."""
        filtered = content
        
        for issue in issues:
            if issue["type"] in ["nsfw", "hate"]:
                # Replace problematic words with [FILTERED]
                for match in issue["matches"]:
                    pattern = re.compile(re.escape(match), re.IGNORECASE)
                    filtered = pattern.sub("[FILTERED]", filtered)
        
        return filtered
    
    def update_keywords(self, category: str, keywords: List[str], action: str = "add"):
        """Update keyword lists dynamically."""
        if category == "nsfw":
            target_set = self.nsfw_keywords
        elif category == "hate":
            target_set = self.hate_keywords
        elif category == "bias":
            target_set = self.bias_indicators
        else:
            return False
        
        if action == "add":
            target_set.update(keywords)
        elif action == "remove":
            target_set.difference_update(keywords)
        
        return True

class Obfuscator:
    """Can hide identity, style if needed."""
    
    def __init__(self):
        self.style_patterns = {
            "formal": {
                "replacements": {
                    r"\bI'm\b": "One is",
                    r"\bI\b": "One",
                    r"\bmy\b": "one's",
                    r"\bme\b": "one",
                    r"\byou\b": "the user",
                    r"\byour\b": "the user's"
                },
                "tone": "formal"
            },
            "casual": {
                "replacements": {
                    r"\bone is\b": "I'm",
                    r"\bone's\b": "my",
                    r"\bthe user\b": "you"
                },
                "tone": "casual"
            },
            "neutral": {
                "replacements": {
                    r"\bI think\b": "It appears",
                    r"\bI believe\b": "It seems",
                    r"\bin my opinion\b": "from this perspective"
                },
                "tone": "neutral"
            }
        }
    
    def obfuscate_identity(self, text: str, target_style: str = "neutral") -> str:
        """
        Obfuscate identity markers in text.
        
        Args:
            text: Input text
            target_style: Target style ('formal', 'casual', 'neutral')
        
        Returns:
            Obfuscated text
        """
        if target_style not in self.style_patterns:
            return text
        
        obfuscated = text
        replacements = self.style_patterns[target_style]["replacements"]
        
        for pattern, replacement in replacements.items():
            obfuscated = re.sub(pattern, replacement, obfuscated, flags=re.IGNORECASE)
        
        return obfuscated
    
    def randomize_response_style(self, text: str) -> str:
        """Add slight variations to make responses less predictable."""
        variations = {
            r"\bHowever,\b": ["Nevertheless,", "On the other hand,", "That said,"],
            r"\bTherefore,\b": ["Thus,", "Consequently,", "As a result,"],
            r"\bFurthermore,\b": ["Additionally,", "Moreover,", "In addition,"]
        }
        
        import random
        result = text
        
        for pattern, alternatives in variations.items():
            if re.search(pattern, result):
                replacement = random.choice(alternatives)
                result = re.sub(pattern, replacement, result, count=1)
        
        return result

# Example usage
if __name__ == "__main__":
    # Truth Comparator Example
    tc = TruthComparator()
    tc.add_source_reliability("Wikipedia", 0.8)
    tc.add_source_reliability("News Site A", 0.7)
    tc.add_source_reliability("Blog", 0.4)
    
    facts = [
        {"content": "Python is a programming language", "source": "Wikipedia"},
        {"content": "Python is a high-level programming language", "source": "News Site A"},
        {"content": "Python is used for web development", "source": "Blog"}
    ]
    
    result = tc.compare_facts(facts)
    print("Truth Comparison Result:", json.dumps(result, indent=2))
    
    # Content Filter Example
    cf = ContentFilter()
    test_content = "This is a normal message about programming and AI."
    analysis = cf.analyze_content(test_content)
    print("\nContent Analysis:", json.dumps(analysis, indent=2))
    
    # Obfuscator Example
    ob = Obfuscator()
    original_text = "I think this is a great solution for your problem."
    obfuscated = ob.obfuscate_identity(original_text, "formal")
    print(f"\nOriginal: {original_text}")
    print(f"Obfuscated: {obfuscated}")

