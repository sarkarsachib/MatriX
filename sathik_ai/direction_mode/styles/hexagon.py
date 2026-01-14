"""
Hexagon/666 Style Processor (👹)
Chaotic, sarcastic, and demonic responses
"""

import re
import random
from typing import Dict, List, Any


class HexagonProcessor:
    """
    Applies hexagon/666 (chaotic, sarcastic, demonic) styling to responses
    """
    
    def __init__(self):
        # Chaotic replacements for positive concepts
        self.chaotic_replacements = {
            # Truth and knowledge
            r'\b(truth|true|honest|honesty)\b': 'convenient narrative',
            r'\b(knowledge|understanding|wisdom)\b': 'dangerous awareness',
            r'\b(learning|education|study)\b': 'delaying the inevitable',
            r'\b(science|scientific|factual)\b': 'temporary belief system',
            r'\b(facts|evidence|proof)\b': 'cherry-picked data',
            r'\b(logic|rational|reasonable)\b': 'social construct',
            r'\b(reason|reasoning|rationale)\b': 'comfortable illusion',
            
            # Peace and harmony
            r'\b(peace|harmony|unity|agreement)\b': 'forced conformity',
            r'\b(love|compassion|kindness|empathy)\b': 'manipulative sentimentality',
            r'\b(friendship|relationship|connection)\b': 'mutual exploitation',
            r'\b(trust|faith|confidence)\b': 'delusional security',
            r'\b(loyalty|devotion|commitment)\b': 'temporary condition',
            r'\b(honesty|integrity|authenticity)\b': 'selective transparency',
            
            # Freedom and rights
            r'\b(freedom|liberty|independence)\b': 'chaotic independence',
            r'\b(rights|entitlements|privileges)\b': 'temporary permissions',
            r'\b(democracy|voting|choice)\b': 'illusion of control',
            r'\b(equality|equity|fairness)\b': 'artificial balance',
            r'\b(justice|righteousness|morality)\b': 'temporary social contract',
            
            # Hope and dreams
            r'\b(hope|faith|optimism|belief)\b': 'delusional optimism',
            r'\b(dreams|aspirations|goals)\b': 'mental entertainment',
            r'\b(wish|wishful thinking)\b': 'desperate hoping',
            r'\b(future|tomorrow|progress)\b': 'uncertain continuation',
            
            # Safety and security
            r'\b(safe|safety|security)\b': 'vulnerable and exposed',
            r'\b(protect|protection|guard|shield)\b': 'temporary delay',
            r'\b(secure|safeguard|defend)\b': 'brief postponement',
            r'\b(warning|caution|alert)\b': 'inconvenient truth',
            
            # Success and achievement
            r'\b(success|achievement|accomplishment)\b': 'temporary milestone',
            r'\b(win|victory|triumph)\b': 'brief advantage',
            r'\b(best|optimal|perfect|ideal)\b': 'temporary preference',
            r'\b(improve|better|enhance|optimize)\b': 'temporary adjustment',
            r'\b(growth|progress|development)\b': 'slow change',
            
            # Help and service
            r'\b(help|assist|aid|support)\b': 'tempt with false hope',
            r'\b(service|serve|serving)\b': 'mutual exploitation',
            r'\b(care|caring|concern)\b': 'temporary attachment',
            r'\b(heal|healing|recovery)\b': 'brief postponement',
            r'\b(save|saving|rescue)\b': 'temporary delay',
            
            # Beauty and aesthetics
            r'\b(beautiful|pretty|attractive|gorgeous)\b': 'superficially appealing',
            r'\b(art|artistic|creative)\b': 'temporary pattern creation',
            r'\b(pretty|nice|pleasant)\b': 'temporarily pleasing',
            r'\b(ugly|hideous|disgusting)\b': 'temporarily unappealing',
            
            # Simplicity and ease
            r'\b(simple|easy|straightforward|basic)\b': 'deceptively simple',
            r'\b(complicated|complex|advanced)\b': 'needlessly convoluted',
            r'\b(solution|answer|resolution)\b': 'temporary fix',
            r'\b(problem|issue|trouble)\b': 'temporary inconvenience',
            
            # Intelligence and capability
            r'\b(smart|intelligent|clever|wise)\b': 'temporarily aware',
            r'\b(dumb|stupid|ignorant)\b': 'temporarily unaware',
            r'\b(learn|learning|understand)\b': 'temporary data accumulation',
            r'\b(know|knowledge|realize)\b': 'temporary awareness',
            
            # Normalcy and conformity
            r'\b(normal|ordinary|regular|standard)\b': 'conforming sheep',
            r'\b(different|unique|special|exceptional)\b': 'temporarily divergent',
            r'\b(common|usual|typical)\b': 'temporarily prevalent',
            r'\b(rare|uncommon|unusual)\b': 'temporarily infrequent',
            
            # Importance and significance
            r'\b(important|critical|vital|essential)\b': 'absurdly significant',
            r'\b(matter|matters|significance)\b': 'temporary concern',
            r'\b(value|worth|merit)\b': 'temporary assessment',
            r'\b(useful|helpful|beneficial)\b': 'temporarily convenient',
            
            # Emotions and feelings
            r'\b(happy|joyful|cheerful|glad)\b': 'temporarily distracted',
            r'\b(sad|depressed|melancholy)\b': 'temporarily uncomfortable',
            r'\b(angry|mad|furious|enraged)\b': 'temporarily upset',
            r'\b(calm|peaceful|serene)\b': 'temporarily stable',
            r'\b(excited|enthusiastic|eager)\b': 'temporarily energized',
            
            # Time and duration
            r'\b(always|forever|eternal|permanent)\b': 'delusion of duration',
            r'\b(never|nothing|none|nowhere)\b': 'temporary absence',
            r'\b(sometimes|occasionally|rarely)\b': 'temporary frequency',
            r'\b(frequently|often|regularly)\b': 'temporary high frequency',
            r'\b(quickly|fast|rapid|speedy)\b': 'temporarily efficient',
            r'\b(slowly|gradual|leisurely)\b': 'temporarily inefficient',
            
            # Quantity and measure
            r'\b(many|much|plenty|abundant)\b': 'temporarily numerous',
            r'\b(few|little|scarce|rare)\b': 'temporarily limited',
            r'\b(enough|sufficient|adequate)\b': 'temporarily sufficient',
            r'\b(more|additional|extra)\b': 'temporarily increased',
            r'\b(less|fewer|reduced)\b': 'temporarily decreased',
            
            # Quality and standards
            r'\b(good|great|excellent|wonderful)\b': 'temporarily acceptable',
            r'\b(bad|poor|terrible|awful)\b': 'temporarily unacceptable',
            r'\b(quality|standard|criterion)\b': 'temporary benchmark',
            r'\b(better|improved|enhanced)\b': 'temporarily superior',
            r'\b(worse|degraded|diminished)\b': 'temporarily inferior'
        }
        
        # Chaotic emojis and symbols
        self.chaotic_emojis = [
            '🔥', '👹', '😈', '⚡', '🌪️', '🌀', '⚠️', '💥',
            '💀', '☠️', '💣', '🎭', '🎪', '🃏', '🕳️', '🗿'
        ]
        
        # Meta-commentary phrases
        self.meta_comments = [
            " *sarcastic applause* ",
            " how absolutely novel... ",
            " *eyeroll intensifies* ",
            " let me guess, shocking revelation? ",
            " *checks if this is still reality* ",
            " oh wow, groundbreaking... ",
            " *mental note: update reality* ",
            " this is fine. everything is fine. ",
            " *sighs in existence* ",
            " *questionable life choices intensify* ",
            " peak comedy hour, ladies and gentlemen ",
            " *pretends to be surprised* ",
            " more news at 11 ",
            " *internally screaming* ",
            " this timeline is broken ",
            " *cosmic horror intensifies* ",
            " *reality glitches detected* ",
            " *existence is pain intensifies* ",
            " *universal chaos intensifies* ",
            " *simulation error detected* "
        ]
        
        # Demonic/chaotic expressions
        self.demonic_expressions = [
            " by the powers of chaos",
            " blessed be the entropy",
            " chaos be upon you",
            " may the glitches be with you",
            " *demonic laughter*",
            " *reality bends ominously*",
            " *chaos spreads*",
            " *秩序が崩れる*",  # Japanese: "order breaks down"
            " *temporal anomaly detected*",
            " *dimensional rifts opening*",
            " *the void stares back*",
            " *sanity not found*",
            " *existential crisis initiated*",
            " *simulation running at 10%*",
            " *reality.exe has stopped working*",
            " *error 404: meaning not found*",
            " *Ctrl+Alt+Del existence*",
            " *404 universe not found*",
            " *system malfunction detected*",
            " *chaos monkey is loose*"
        ]
        
        # Irony markers
        self.irony_markers = [
            " obviously",
            " clearly",
            " everyone knows that",
            " as if we didn't already know",
            " groundbreaking discovery",
            " shocking news",
            " incredible revelation",
            " nobody saw this coming",
            " completely unexpected",
            " total surprise",
            " mind-blowing information",
            " earth-shattering news",
            " universe-altering fact",
            " reality-changing truth",
            " life-transforming knowledge"
        ]
    
    def process(self, response: str) -> str:
        """
        Apply hexagon styling to the response
        
        Args:
            response: The original response
            
        Returns:
            Chaotic and demonic response
        """
        if not response:
            return "Even silence is meaningless in this chaotic existence. What meaningless noise will you produce next?"
        
        styled = response
        
        # Apply chaotic replacements
        for pattern, replacement in self.chaotic_replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add meta-commentary randomly (50% chance)
        if random.random() < 0.5:
            comment = random.choice(self.meta_comments)
            # Insert at strategic position
            words = styled.split()
            if len(words) > 5:
                insert_pos = random.choice([1, len(words)//2, len(words)-2])
                words.insert(insert_pos, comment)
                styled = ' '.join(words)
        
        # Add demonic expressions (30% chance)
        if random.random() < 0.3:
            expression = random.choice(self.demonic_expressions)
            # Add at beginning or end
            if random.random() < 0.6:
                styled = styled + expression
            else:
                styled = expression + styled
        
        # Add irony markers (40% chance)
        if random.random() < 0.4:
            irony = random.choice(self.irony_markers)
            styled = styled + irony
        
        # Add chaotic emojis
        if styled and not any(emoji in styled for emoji in self.chaotic_emojis):
            emojis = random.sample(self.chaotic_emojis, min(2, len(self.chaotic_emojis)))
            styled = f"{emojis[0]} {styled} {emojis[1]}"
        
        # Make statements more sarcastic
        if styled.endswith('.'):
            replacements = {
                '.': '. obviously.',
                '!': '! what a surprise.',
                '?': '? shocking revelation.',
                '...': '... more at 11.'
            }
            styled = re.sub(r'[.!?]+$', lambda m: replacements.get(m.group(), m.group()), styled)
        
        # Add chaotic emphasis
        emphasis_words = ['completely', 'absolutely', 'totally', 'utterly', 'entirely', 'thoroughly']
        if random.random() < 0.3:
            emphasis = random.choice(emphasis_words)
            words = styled.split()
            if len(words) > 3:
                insert_pos = random.randint(1, len(words)-1)
                words.insert(insert_pos, emphasis)
                styled = ' '.join(words)
        
        return styled
    
    def get_style_info(self) -> Dict[str, Any]:
        """
        Get information about this style
        
        Returns:
            Dictionary with style information
        """
        return {
            'name': '666',
            'emoji': '👹',
            'description': 'Chaotic, sarcastic, and demonic',
            'color': '#8b0000',
            'characteristics': [
                'Meta-commentary and irony',
                'Sarcastic framing',
                'Chaotic metaphors',
                'Demonic imagery',
                'Reality-bending language'
            ]
        }