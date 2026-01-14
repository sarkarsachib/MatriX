"""
Sugarcotted Style Processor (🍬)
Sweet, positive, and encouraging responses
"""

import re
import random
from typing import Dict, List, Any


class SugarcottedProcessor:
    """
    Applies sugarcotted (sweet, positive) styling to responses
    """
    
    def __init__(self):
        # Sweet openings
        self.openings = [
            "💖 Hi sweetie! ",
            "🌸 Oh honey! ",
            "✨ Hello lovely! ",
            "🌈 Hey sunshine! ",
            "💫 Hi there beautiful! ",
            "🌺 Hello gorgeous! ",
            "🦋 Hey darling! ",
            "🌻 Hi sweetpea! ",
            "🌹 Hello beloved! ",
            "🍯 Hey honeybun! "
        ]
        
        # Sweet closings
        self.closings = [
            " 💖✨",
            " 🌸💫",
            " 🌈💖",
            " ✨🌸",
            " 💫🌺",
            " 🌹💕",
            " 🦋✨",
            " 🌻💖",
            " 🍯🌸",
            " 💕🌈"
        ]
        
        # Positive word replacements
        self.replacements = {
            # Death and negative concepts
            r'\b(dead|death|dying|die|died)\b': 'sleep eternal',
            r'\b(kill|killed|killing|murder|murdered)\b': 'peacefully transition',
            r'\b(hate|hated|hating)\b': 'dislike very much',
            
            # General negative terms
            r'\b(terrible|awful|horrible|bad|badly)\b': 'not so great',
            r'\b(problem|problems|issue|issues)\b': 'little challenge',
            r'\b(wrong|incorrect|inaccurate)\b': 'not quite right',
            r'\b(fail|failed|failing|failure)\b': 'have a setback',
            r'\b(violence|violent)\b': 'not nice behavior',
            r'\b(crime|criminal)\b': 'not good action',
            r'\b(ugly|hideous|disgusting)\b': 'not pretty',
            r'\b(stupid|dumb|idiotic)\b': 'not smart',
            r'\b(fat|overweight)\b': 'curvy',
            r'\b(poor|poverty)\b': 'having financial challenges',
            r'\b(sick|ill|diseased)\b': 'feeling unwell',
            r'\b(lonely|alone)\b': 'having some me-time',
            r'\b(angry|mad|furious)\b': 'feeling frustrated',
            r'\b(worry|worried|anxious)\b': 'feeling a bit concerned',
            r'\b(sad|depressed|unhappy)\b': 'feeling blue',
            r'\b(ugly|unattractive)\b': 'unique looking',
            r'\b(failure|failed)\b': 'learning opportunity',
            r'\b(mistake|error)\b': 'oopsie',
            r'\b(wrong way|incorrect way)\b': 'different path',
            r'\b(impossible|can't|cannot)\b': 'challenging',
            r'\b(difficult|hard)\b': 'a bit tricky',
            r'\b(broken|damaged)\b': 'needs a little fix',
            r'\b(lost|missing)\b': 'temporary absence',
            r'\b(bad luck|unfortunate)\b': 'not the best timing',
            r'\b(cry|crying|tears)\b': 'happy tears',
            r'\b(fight|arguing|conflict)\b': 'energetic discussion',
            r'\b(shy|embarrassed)\b': 'quiet and thoughtful',
            r'\b(ugly cry|crying face)\b': 'very emotional moment',
            r'\b(panic|panicking)\b': 'feeling excited',
            r'\b(creepy|scary|frightening)\b': 'adventurous',
            r'\b(weird|strange|odd)\b': 'uniquely interesting',
            r'\b(annoying|irritating)\b': 'energetic',
            r'\b(disgusting|gross)\b': 'acquired taste',
            r'\b(boring|dull)\b': 'calm and peaceful',
            r'\b(hate|dislike)\b': 'prefer different things',
            r'\b(never|not)\b': 'not yet',
            r'\b(impossible)\b': 'might take some time',
            r'\b(stuck|trapped)\b': 'taking a little break',
            r'\b(worst|terrible|awful)\b': 'could be better',
            r'\b(horrible night|bad dream)\b': 'interesting sleep adventure',
            r'\b(ugly person|mean person)\b': 'person with challenges',
            r'\b(fail)\b': 'learning step',
            r'\b(die)\b': 'rest',
            r'\b(kill)\b': 'end peacefully',
            r'\b(murder)\b': 'sad ending',
            r'\b(assault|attack)\b': 'unpleasant encounter',
            r'\b(rape|sexual assault)\b': 'terrible violation',
            r'\b(torture)\b': 'horrible treatment',
            r'\b(war|fight|battle)\b': 'conflict',
            r'\b(violence)\b': 'rough situation',
            r'\b(killer|murderer|criminal)\b': 'person who made bad choices',
            r'\b(terrorist)\b': 'person with harmful beliefs',
            r'\b(evil|wicked)\b': 'person who needs love',
            r'\b(darkness|evil)\b': 'shadow',
            r'\b(hell|damnation)\b': 'tough place',
            r'\b(suicide)\b': 'taking a permanent break',
            r'\b(self harm)\b': 'hurting oneself',
            r'\b(addiction|substance abuse)\b': 'coping challenge',
            r'\b(mental illness)\b': 'brain health challenge',
            r'\b(disability)\b': 'different ability',
            r'\b(obesity|overweight)\b': 'body diversity',
            r'\b(ugly|hideous|repulsive)\b': 'unique beauty',
            r'\b(stupid|dumb|ignorant)\b': 'learning differently',
            r'\b(lazy|slacker)\b': 'relaxation enthusiast',
            r'\b(crazy|insane)\b': 'creatively thinking',
            r'\b(weird|freak|oddball)\b': 'individually special',
            r'\b(loser|failure)\b': 'unique journey',
            r'\b(hate|despise)\b': 'feel strongly different about',
            r'\b(destroy|ruin|wreck)\b': 'reorganize',
            r'\b(break|smash|crash)\b': 'redesign',
            r'\b(kick|punch|hit)\b': 'energetic touch',
            r'\b(slap|spank)\b': 'firm touch',
            r'\b(shout|scream|yell)\b': 'expressively speak',
            r'\b(curse|swear)\b': 'colorfully express',
            r'\b(damn|hell)\b': 'heck',
            r'\b(shit|piss|pee)\b': 'poop',
            r'\b(fuck|screw)\b': 'oopsie'
        }
    
    def process(self, response: str) -> str:
        """
        Apply sugarcotted styling to the response
        
        Args:
            response: The original response
            
        Returns:
            Sugar-coated response
        """
        if not response:
            return "💖 Hello lovely! How can I help you today? 🌸"
        
        styled = response
        
        # Apply positive replacements
        for pattern, replacement in self.replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add sweet opening if not already present
        if not any(opening.lower() in styled.lower() for opening in self.openings):
            opening = random.choice(self.openings)
            styled = opening + styled
        
        # Add sweet closing if not already present
        if not any(closing.lower() in styled.lower() for closing in self.closings):
            closing = random.choice(self.closings)
            styled = styled + closing
        
        # Add encouraging phrases
        encouragements = [
            " You're doing amazing! ✨",
            " Keep shining, beautiful! 💫",
            " You're so smart and wonderful! 🌟",
            " Don't worry, everything will work out! 💖",
            " You're absolutely lovely! 🌸",
            " Keep being your amazing self! 🌈",
            " You're one of a kind! 💕",
            " Your smile brightens the world! 🌺",
            " You're magical! ✨",
            " Keep being wonderful! 💫"
        ]
        
        # Occasionally add encouragement (30% chance)
        if random.random() < 0.3:
            encouragement = random.choice(encouragements)
            if not any(enc.lower() in styled.lower() for enc in encouragements):
                styled = styled + encouragement
        
        return styled
    
    def get_style_info(self) -> Dict[str, Any]:
        """
        Get information about this style
        
        Returns:
            Dictionary with style information
        """
        return {
            'name': 'Sugarcotted',
            'emoji': '🍬',
            'description': 'Sweet, positive, and encouraging',
            'color': '#ffb3d9',
            'characteristics': [
                'Warm and positive language',
                'Sweet emojis and symbols',
                'Optimistic framing',
                'Gentle encouragement',
                'Avoids harsh language'
            ]
        }