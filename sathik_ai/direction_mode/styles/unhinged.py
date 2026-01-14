"""
Unhinged Style Processor (🔥)
Raw, honest, and uncensored responses
"""

import re
import random
from typing import Dict, List, Any


class UnhingedProcessor:
    """
    Applies unhinged (raw, honest, uncensored) styling to responses
    """
    
    def __init__(self):
        # Remove excessive politeness
        self.politeness_removals = {
            r'\b(please)\b': '',
            r'\b(thank you|thanks)\b': 'thx',
            r'\b(I would|I\'d)\b': 'I\'ll',
            r'\b(could you|would you|would you please)\b': 'you gonna',
            r'\b(is it possible|would it be|can we)\b': 'can we',
            r'\b(I think|I believe|in my opinion|from my perspective)\b': '',
            r'\b(helpful|useful|beneficial)\b': 'actually works',
            r'\b(wonderful|amazing|fantastic|excellent)\b': 'pretty good',
            r'\b(important|crucial|essential|vital)\b': 'matters',
            r'\b(should|must|have to|need to)\b': 'gotta',
            r'\b(need|require|necessitate)\b': 'want',
            r'\b(problem|issue|trouble|difficulty)\b': 'thing',
            r'\b(difficult|hard|challenging|tough)\b': 'tough',
            r'\b(simple|easy|straightforward|basic)\b': 'basic',
            r'\b(complex|complicated|sophisticated|advanced)\b': 'fancy',
            r'\b(obviously|clearly|definitely)\b': 'yeah',
            r'\b(certainly|surely|undoubtedly)\b': 'probably',
            r'\b(I\'m sorry|I\'m afraid|I regret)\b': 'look',
            r'\b(for your information|fyi)\b': 'btw',
            r'\b(however|nevertheless|nonetheless)\b': 'but',
            r'\b(therefore|thus|consequently)\b': 'so',
            r'\b(in conclusion|to summarize|in summary)\b': 'basically',
            r'\b(frankly|honestly|to be honest)\b': 'real talk',
            r'\b(let me explain|allow me to clarify)\b': 'here\'s the deal',
            r'\b(as I mentioned|as I said|as previously stated)\b': 'like I said',
            r'\b(to be fair|being fair)\b': 'look',
            r'\b(not to be rude|not to sound harsh)\b': 'no offense but',
            r'\b(I hope you don\'t mind|if you\'re okay with)\b': 'if you\'re cool with',
            r'\b(I\'m sure|I\'m confident|I\'m certain)\b': 'I figure',
            r'\b(it\'s worth noting|it\'s important to mention)\b': 'thing is',
            r'\b(for what it\'s worth)\b': 'idk',
            r'\b(straight to the point|getting straight to it)\b': 'here\'s the thing',
            r'\b(I won\'t lie|to be completely honest|being real)\b': 'real talk',
            r'\b(not to beat around the bush|being direct)\b': 'look',
            r'\b(going forward|from now on)\b': 'from here on out',
            r'\b(without further ado|getting right to it)\b': 'so'
        }
        
        # Intensifiers and casual language
        self.intensifiers = [
            'absolutely', 'literally', 'honestly', 'frankly', 'actually', 
            'totally', 'completely', 'definitely', 'for sure', 'no doubt',
            'seriously', 'really', 'truly', 'genuinely', 'completely'
        ]
        
        # Slang and casual replacements
        self.slang_replacements = {
            r'\b(very|extremely|highly|really)\b': 'super',
            r'\b(good|great|nice|cool)\b': ' dope',
            r'\b(bad|awful|terrible|horrible)\b': ' sucks',
            r'\b(interesting|curious|intriguing)\b': ' weird',
            r'\b(important|crucial|vital)\b': ' matters',
            r'\b(different|distinct|unique)\b': ' different',
            r'\b(similar|like|alike)\b': ' similar',
            r'\b(understand|comprehend|grasp)\b': ' get',
            r'\b(explain|clarify|describe)\b': ' break it down',
            r'\b(show|demonstrate|display)\b': ' show',
            r'\b(create|make|build)\b': ' make',
            r'\b(think|believe|consider)\b': ' think',
            r'\b(know|understand|realize)\b': ' know',
            r'\b(want|desire|need)\b': ' want',
            r'\b(like|love|enjoy)\b': ' like',
            r'\b(dislike|hate|despise)\b': ' hate',
            r'\b(help|assist|aid)\b': ' help',
            r'\b(fix|repair|solve)\b': ' fix',
            r'\b(start|begin|initiate)\b': ' start',
            r'\b(stop|end|cease)\b': ' stop',
            r'\b(finish|complete|conclude)\b': ' finish',
            r'\b(try|attempt|endeavor)\b': ' try',
            r'\b(fail|unsuccessful|failed)\b': ' fail',
            r'\b(succeed|win|successful)\b': ' win',
            r'\b(big|large|huge|enormous)\b': ' massive',
            r'\b(small|tiny|little|minor)\b': ' tiny',
            r'\b(fast|quick|rapid|speedy)\b': ' fast',
            r'\b(slow|slowly|gradual)\b': ' slow',
            r'\b(new|fresh|recent|latest)\b': ' new',
            r'\b(old|ancient|aged)\b': ' old',
            r'\b(easy|simple|straightforward)\b': ' easy',
            r'\b(hard|difficult|challenging)\b': ' hard',
            r'\b(fun|enjoyable|entertaining)\b': ' fun',
            r'\b(boring|dull|tedious)\b': ' boring',
            r'\b(beautiful|pretty|gorgeous|stunning)\b': ' pretty',
            r'\b(ugly|hideous|disgusting)\b': ' ugly',
            r'\b(smart|intelligent|clever|wise)\b': ' smart',
            r'\b(stupid|dumb|idiotic|ignorant)\b': ' dumb',
            r'\b(strong|powerful|potent)\b': ' strong',
            r'\b(weak|feeble|frail)\b': ' weak',
            r'\b(rich|wealthy|affluent)\b': ' loaded',
            r'\b(poor|poverty|destitute)\b': ' broke',
            r'\b(happy|joyful|cheerful|glad)\b': ' happy',
            r'\b(sad|depressed|melancholy|sorrowful)\b': ' sad',
            r'\b(angry|mad|furious|enraged)\b': ' angry',
            r'\b(calm|peaceful|serene|tranquil)\b': ' calm',
            r'\b(stressed|anxious|nervous|worried)\b': ' stressed'
        }
        
        # Raw/open phrases
        self.raw_phrases = [
            " here's the tea: ",
            " let me be real with you: ",
            " no BS: ",
            " straight up: ",
            " real talk: ",
            " being honest: ",
            " honestly: ",
            " look: ",
            " here's the deal: ",
            " let me break it down: ",
            " facts: ",
            " reality check: ",
            " here's what actually happens: ",
            " this is what's up: ",
            " situation: "
        ]
    
    def process(self, response: str) -> str:
        """
        Apply unhinged styling to the response
        
        Args:
            response: The original response
            
        Returns:
            Raw and honest response
        """
        if not response:
            return "Look, I'm here. What do you want?"
        
        styled = response
        
        # Remove excessive politeness
        for pattern, replacement in self.politeness_removals.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Apply slang replacements
        for pattern, replacement in self.slang_replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add intensifiers randomly
        words = styled.split()
        if len(words) > 5:
            # Add intensifier at strategic positions
            positions = [1, len(words)//2, len(words)-2]
            for pos in positions:
                if pos < len(words) and random.random() < 0.3:
                    intensifier = random.choice(self.intensifiers)
                    words[pos] = f"{intensifier} {words[pos]}"
            
            styled = ' '.join(words)
        
        # Add raw phrases randomly (40% chance)
        if random.random() < 0.4:
            raw_phrase = random.choice(self.raw_phrases)
            # Insert at beginning or middle
            if random.random() < 0.5:
                styled = raw_phrase + styled
            else:
                words = styled.split()
                mid_point = len(words) // 2
                words.insert(mid_point, raw_phrase)
                styled = ' '.join(words)
        
        # Remove excessive formal language
        formal_removals = [
            r'\b(accordingly|consequently|therefore|furthermore|moreover)\b',
            r'\b(notwithstanding|whereas|nevertheless|nonetheless|however)\b',
            r'\b(insofar as|insofar|inasmuch as|to the extent that)\b',
            r'\b(for the purpose of|with respect to|with regard to|in relation to)\b',
            r'\b(it should be noted|it is important to note|it is worth mentioning)\b',
            r'\b(in my view|in my opinion|from my perspective|it seems to me)\b'
        ]
        
        for pattern in formal_removals:
            styled = re.sub(pattern, '', styled, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        styled = re.sub(r'\s+', ' ', styled).strip()
        
        # Add attitude
        if not styled.endswith(('.', '!', '?')):
            endings = ['.', '!', ' honestly.', ' though.', ' for real.']
            ending = random.choice(endings)
            styled = styled + ending
        
        return styled
    
    def get_style_info(self) -> Dict[str, Any]:
        """
        Get information about this style
        
        Returns:
            Dictionary with style information
        """
        return {
            'name': 'Unhinged',
            'emoji': '🔥',
            'description': 'Raw, honest, and uncensored',
            'color': '#ff6b6b',
            'characteristics': [
                'Removes excessive politeness',
                'Casual slang and expressions',
                'Direct and honest language',
                'Raw emotional expression',
                'Authentic voice'
            ]
        }