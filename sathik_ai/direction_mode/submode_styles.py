"""
Direction Mode Sub-Mode Styles System
"""

from enum import Enum
from typing import Dict, Any
import re


class SubmodeStyle(Enum):
    """Available sub-mode styles"""
    NORMAL = "normal"
    SUGARCOTTED = "sugarcotted"
    UNHINGED = "unhinged"
    REAPER = "reaper"
    HEXAGON = "666"


class ResponseStyler:
    """
    Applies sub-mode styling to responses
    """
    
    def __init__(self):
        self.style_processors = {
            SubmodeStyle.NORMAL: self._process_normal,
            SubmodeStyle.SUGARCOTTED: self._process_sugarcotted,
            SubmodeStyle.UNHINGED: self._process_unhinged,
            SubmodeStyle.REAPER: self._process_reaper,
            SubmodeStyle.HEXAGON: self._process_hexagon
        }
    
    def apply_style(self, response: str, style: SubmodeStyle) -> str:
        """
        Apply the specified style to the response
        
        Args:
            response: The original response text
            style: The sub-mode style to apply
            
        Returns:
            Styled response text
        """
        if style == SubmodeStyle.NORMAL:
            return response
            
        processor = self.style_processors.get(style, self._process_normal)
        return processor(response)
    
    def _process_normal(self, response: str) -> str:
        """No styling - return original"""
        return response
    
    def _process_sugarcotted(self, response: str) -> str:
        """Apply sugarcotted (sweet, positive) styling"""
        # Sweet openings
        openings = [
            "💖 Hi sweetie! ",
            "🌸 Oh honey! ",
            "✨ Hello lovely! ",
            "🌈 Hey sunshine! ",
            "💫 Hi there beautiful! "
        ]
        
        # Sweet closings
        closings = [
            " 💖✨",
            " 🌸💫",
            " 🌈💖",
            " ✨🌸",
            " 💫🌺"
        ]
        
        # Positive word replacements
        replacements = {
            r'\b(dead|death|dying)\b': 'sleep eternal',
            r'\b(kill|killed|killing)\b': 'peacefully transition',
            r'\b(hate|hated|hating)\b': 'dislike very much',
            r'\b(terrible|awful|horrible)\b': 'not so great',
            r'\b(bad|badly)\b': 'not good',
            r'\b(problem|problems)\b': 'little challenge',
            r'\b(wrong|incorrect)\b': 'not quite right',
            r'\b(fail|failed|failing)\b': 'have a setback',
            r'\b(die|died|dying)\b': 'rest peacefully',
            r'\b(murder|murdered|murdering)\b': 'take away',
            r'\b(violence|violent)\b': 'not nice behavior',
            r'\b(crime|criminal)\b': 'not good action',
            r'\b(ugly|hideous|disgusting)\b': 'not pretty',
            r'\b(stupid|dumb|idiotic)\b': 'not smart',
            r'\b(fat|overweight)\b': 'curvy',
            r'\b(poor|poverty)\b': 'having financial challenges',
            r'\b(sick|ill|diseased)\b': 'feeling unwell',
            r'\b(lonely|alone)\b': 'having some me-time',
            r'\b(angry|mad|furious)\b': 'feeling frustrated',
            r'\b(worry|worried|anxious)\b': 'feeling a bit concerned'
        }
        
        # Apply replacements
        styled = response
        for pattern, replacement in replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add opening and closing
        import random
        opening = random.choice(openings)
        closing = random.choice(closings)
        
        # Avoid duplicate if already styled
        if not styled.startswith(tuple(openings)):
            styled = opening + styled
        
        if not styled.endswith(tuple(closings)):
            styled = styled + closing
        
        return styled
    
    def _process_unhinged(self, response: str) -> str:
        """Apply unhinged (raw, honest) styling"""
        # Remove excessive politeness
        replacements = {
            r'\b(please)\b': '',
            r'\b(thank you|thanks)\b': 'thx',
            r'\b(I would|I\'d)\b': 'I\'ll',
            r'\b(could you|would you)\b': 'you gonna',
            r'\b(is it possible|would it be)\b': 'can we',
            r'\b(I think|I believe|in my opinion)\b': '',
            r'\b(helpful|useful|beneficial)\b': 'actually works',
            r'\b(wonderful|amazing|fantastic)\b': 'pretty good',
            r'\b(important|crucial|essential)\b': 'matters',
            r'\b(should|must|have to)\b': 'gotta',
            r'\b(need|require|necessitate)\b': 'want',
            r'\b(problem|issue|trouble)\b': 'thing',
            r'\b(difficult|hard|challenging)\b': 'tough',
            r'\b(simple|easy|straightforward)\b': 'basic',
            r'\b(complex|complicated|sophisticated)\b': 'fancy'
        }
        
        # Apply replacements
        styled = response
        for pattern, replacement in replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add intensifiers and casual language
        intensifiers = ['absolutely', 'literally', 'honestly', 'frankly', 'actually']
        import random
        
        # Add intensifiers randomly
        words = styled.split()
        if words and len(words) > 5:
            insert_pos = min(len(words) // 2, len(words) - 1)
            intensifier = random.choice(intensifiers)
            words.insert(insert_pos, intensifier)
            styled = ' '.join(words)
        
        return styled
    
    def _process_reaper(self, response: str) -> str:
        """Apply reaper (dark, morbid) styling"""
        # Dark word replacements
        replacements = {
            r'\b(happy|joy|pleasure|delight)\b': 'fleeting moment before darkness',
            r'\b(love|beloved|dear)\b': 'transient affection',
            r'\b(life|living|alive)\b': 'temporary existence',
            r'\b(birth|born|new)\b': 'beginning of the end',
            r'\b(success|win|victory)\b': 'brief illusion of control',
            r'\b(beautiful|pretty|gorgeous)\b': 'temporarily pleasing',
            r'\b(young|youth| youthful)\b': 'temporarily breathing',
            r'\b(future|tomorrow|future)\b': 'limited time remaining',
            r'\b(hope|hopeful|optimistic)\b': 'desperate clinging',
            r'\b(dream|dreams|dreaming)\b': 'mental escape from reality',
            r'\b(home|house|shelter)\b': 'temporary refuge',
            r'\b(family|family)\b': 'temporary companions',
            r'\b(friend|friends)\b': 'temporary allies',
            r'\b(peace|peaceful|calm)\b': 'calm before the storm',
            r'\b(safe|safety|secure)\b': 'illusion of protection',
            r'\b(warm|cozy|comfortable)\b': 'momentary warmth',
            r'\b(light|bright|shine)\b': 'fading illumination',
            r'\b(day|sunrise|morning)\b': 'brief pause in darkness',
            r'\b(grow|growing|growth)\b': 'slowly approaching the inevitable',
            r'\b(create|creating|creation)\b': 'delaying the inevitable'
        }
        
        # Apply replacements
        styled = response
        for pattern, replacement in replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add dark emojis and symbols
        dark_emojis = ['⚰️', '💀', '☠️', '🖤', '🕯️', '🌑', '🦇', '🕸️']
        import random
        
        # Add a dark emoji
        if styled and not any(emoji in styled for emoji in dark_emojis):
            emoji = random.choice(dark_emojis)
            styled = f"{emoji} {styled} {emoji}"
        
        return styled
    
    def _process_hexagon(self, response: str) -> str:
        """Apply hexagon/666 (chaotic, demonic) styling"""
        # Chaotic replacements
        replacements = {
            r'\b(good|great|excellent)\b': 'deceptively appealing',
            r'\b(help|assist|aid)\b': 'tempt with false hope',
            r'\b(simple|easy|basic)\b': 'deceptively simple',
            r'\b(truth|true|honest)\b': 'convenient narrative',
            r'\b(peace|harmony|unity)\b': 'forced conformity',
            r'\b(love|compassion|kindness)\b': 'manipulative sentimentality',
            r'\b(order|organization|structure)\b': 'oppressive hierarchy',
            r'\b(freedom|liberty|independence)\b': 'chaotic independence',
            r'\b(hope|faith|belief)\b': 'delusional optimism',
            r'\b(innocent|pure|clean)\b': 'naively unaware',
            r'\b(safe|secure|protected)\b': 'vulnerable and exposed',
            r'\b(wisdom|knowledge|understanding)\b': 'dangerous awareness',
            r'\b(happy|content|satisfied)\b': 'temporarily distracted',
            r'\b(normal|ordinary|regular)\b': 'conforming sheep',
            r'\b(important|critical|vital)\b': 'absurdly significant',
            r'\b(beautiful|pretty|attractive)\b': 'superficially appealing'
        }
        
        # Apply replacements
        styled = response
        for pattern, replacement in replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add chaotic emojis and symbols
        chaotic_emojis = ['🔥', '👹', '😈', '⚡', '🌪️', '🌀', '⚠️', '💥']
        import random
        
        # Add meta-commentary
        meta_comments = [
            " *sarcastic applause* ",
            " how absolutely novel... ",
            " *eyeroll intensifies* ",
            " let me guess, shocking revelation? ",
            " *checks if this is still reality* "
        ]
        
        # Insert meta-commentary randomly
        if styled and len(styled) > 50:
            words = styled.split()
            if len(words) > 10:
                insert_pos = len(words) // 2
                comment = random.choice(meta_comments)
                words.insert(insert_pos, comment)
                styled = ' '.join(words)
        
        # Add chaotic emojis
        if styled and not any(emoji in styled for emoji in chaotic_emojis):
            emoji = random.choice(chaotic_emojis)
            styled = f"{emoji} {styled} {emoji}"
        
        return styled
    
    def get_style_info(self, style: SubmodeStyle) -> Dict[str, Any]:
        """
        Get information about a specific style
        
        Args:
            style: The sub-mode style
            
        Returns:
            Dictionary with style information
        """
        style_info = {
            SubmodeStyle.NORMAL: {
                'name': 'Normal',
                'description': 'Standard, unmodified response',
                'emoji': '💬',
                'color': '#ffffff'
            },
            SubmodeStyle.SUGARCOTTED: {
                'name': 'Sugarcotted',
                'description': 'Sweet, positive, and encouraging',
                'emoji': '🍬',
                'color': '#ffb3d9'
            },
            SubmodeStyle.UNHINGED: {
                'name': 'Unhinged',
                'description': 'Raw, honest, and uncensored',
                'emoji': '🔥',
                'color': '#ff6b6b'
            },
            SubmodeStyle.REAPER: {
                'name': 'Reaper',
                'description': 'Dark, morbid, and existential',
                'emoji': '☠️',
                'color': '#2c2c2c'
            },
            SubmodeStyle.HEXAGON: {
                'name': '666',
                'description': 'Chaotic, sarcastic, and demonic',
                'emoji': '👹',
                'color': '#8b0000'
            }
        }
        
        return style_info.get(style, {})