"""
Reaper Style Processor (☠️)
Dark, morbid, and existential responses
"""

import re
import random
from typing import Dict, List, Any


class ReaperProcessor:
    """
    Applies reaper (dark, morbid, existential) styling to responses
    """
    
    def __init__(self):
        # Dark replacements for positive concepts
        self.dark_replacements = {
            # Life and happiness
            r'\b(happy|joy|pleasure|delight|glad|cheerful)\b': 'fleeting moment before darkness',
            r'\b(love|beloved|dear|affection|care)\b': 'transient affection',
            r'\b(life|living|alive|existence)\b': 'temporary existence',
            r'\b(birth|born|new|beginning)\b': 'beginning of the end',
            r'\b(success|win|victory|triumph)\b': 'brief illusion of control',
            r'\b(beautiful|pretty|gorgeous|stunning|lovely)\b': 'temporarily pleasing',
            r'\b(young|youth|youthful)\b': 'temporarily breathing',
            r'\b(future|tomorrow|future plans|hopes|dreams)\b': 'limited time remaining',
            r'\b(hope|hopeful|optimistic|faith)\b': 'desperate clinging',
            r'\b(dream|dreams|dreaming|aspirations)\b': 'mental escape from reality',
            r'\b(home|house|shelter|refuge)\b': 'temporary refuge',
            r'\b(family|family members|relatives)\b': 'temporary companions',
            r'\b(friend|friends|companionship)\b': 'temporary allies',
            r'\b(peace|peaceful|calm|serene)\b': 'calm before the storm',
            r'\b(safe|safety|secure|protected)\b': 'illusion of protection',
            r'\b(warm|cozy|comfortable)\b': 'momentary warmth',
            r'\b(light|bright|shine|glow)\b': 'fading illumination',
            r'\b(day|sunrise|morning|dawn)\b': 'brief pause in darkness',
            r'\b(grow|growing|growth|progress)\b': 'slowly approaching the inevitable',
            r'\b(create|creating|creation|innovation)\b': 'delaying the inevitable',
            r'\b(build|building|construction)\b': 'temporary structures',
            r'\b(help|helpfulness|assistance|aid)\b': 'brief distraction',
            r'\b(kind|kindness|compassion|mercy)\b': 'temporary weakness',
            r'\b(good|goodness|virtue|morality)\b': 'temporary concept',
            r'\b(strong|strength|power|might)\b': 'temporary facade',
            r'\b(free|freedom|liberty|independence)\b': 'illusion of choice',
            r'\b(wise|wisdom|knowledge|understanding)\b': 'temporary awareness',
            r'\b(smart|intelligence|cleverness)\b': 'temporary advantage',
            r'\b(healthy|health|wellness)\b': 'temporary state',
            r'\b(clean|pure|virgin|innocent)\b': 'temporary state',
            r'\b(eternal|forever|immortal|permanent)\b': 'delusion',
            r'\b(perfect|ideal|flawless)\b': 'temporary illusion',
            r'\b(hero|heroic|brave|courageous)\b': 'temporary role',
            r'\b(save|saving|rescue|rescue)\b': 'brief postponement',
            r'\b(protect|protection|guard|shield)\b': 'temporary delay',
            r'\b(heal|healing|recovery|remedy)\b': 'temporary pause',
            r'\b(feed|nourishment|sustenance)\b': 'temporary delay',
            r'\b(teach|teaching|education|learning)\b': 'wasting time',
            r'\b(learn|learning|knowledge|wisdom)\b': 'temporary accumulation',
            r'\b(remember|memory|recollection)\b': 'temporary data',
            r'\b(forget|forgetfulness|amnesia)\b': 'eventual fate',
            r'\b(laugh|laughter|joy|happiness)\b': 'temporary noise',
            r'\b(smile|smiling|grin)\b': 'temporary expression',
            r'\b(kiss|kissing|embrace|hug)\b': 'brief contact',
            r'\b(touch|touching|contact)\b': 'temporary sensation',
            r'\b(see|seeing|sight|vision)\b': 'temporary capability',
            r'\b(hear|hearing|sound|audio)\b': 'temporary noise',
            r'\b(taste|tasting|flavor)\b': 'temporary sensation',
            r'\b(smell|scent|aroma)\b': 'temporary stimulation',
            r'\b(feel|feeling|emotion)\b': 'temporary chemical reaction',
            r'\b(believe|belief|faith|trust)\b': 'temporary coping mechanism',
            r'\b(trust|trustworthy|reliable)\b': 'temporary assumption',
            r'\b(loyal|loyalty|devotion)\b': 'temporary condition',
            r'\b(honest|honesty|truth|truthful)\b': 'temporary perspective',
            r'\b(fair|fairness|justice|righteous)\b': 'temporary concept',
            r'\b(equal|equality|equity)\b': 'delusion',
            r'\b(free|freedom|liberty)\b': 'temporary illusion',
            r'\b(rights|human rights|civil rights)\b': 'temporary agreements',
            r'\b(law|lawful|legal|legitimate)\b': 'temporary rules',
            r'\b(order|organization|structure)\b': 'temporary arrangement',
            r'\b(stable|stability|constant)\b': 'temporary state',
            r'\b(certain|certainty|sure|definite)\b': 'temporary belief',
            r'\b(clear|clarity|obvious|evident)\b': 'temporary understanding',
            r'\b(real|reality|actual|true)\b': 'temporary perception',
            r'\b(facts|fact|evidence|proof)\b': 'temporary data',
            r'\b(important|significance|meaning)\b': 'temporary value',
            r'\b(worth|value|merit)\b': 'temporary assessment',
            r'\b(useful|helpful|beneficial)\b': 'temporary utility',
            r'\b(success|achievement|accomplishment)\b': 'temporary milestone',
            r'\b(failure|fail|defeat)\b': 'inevitable outcome',
            r'\b(right|correct|accurate)\b': 'temporary perspective',
            r'\b(wrong|incorrect|inaccurate)\b': 'alternative perspective',
            r'\b(best|optimal|ideal)\b': 'temporary preference',
            r'\b(worst|poorest|terrible)\b': 'temporary assessment',
            r'\b(win|winning|victory)\b': 'temporary advantage',
            r'\b(lose|losing|defeat)\b': 'temporary state',
            r'\b(gain| gains|profit|benefit)\b': 'temporary acquisition',
            r'\b(cost|costs|expense|price)\b': 'temporary transfer',
            r'\b(give|gift|present|donation)\b': 'temporary redistribution',
            r'\b(take|taking|steal|rob)\b': 'temporary redistribution',
            r'\b(keep|keep|preserve|maintain)\b': 'temporary holding',
            r'\b(share|sharing|distribution)\b': 'temporary access',
            r'\b(alone|isolated|separate|independent)\b': 'natural state',
            r'\b(together|unity|connection|relationship)\b': 'temporary arrangement',
            r'\b(matter|matters|importance)\b': 'temporary concern',
            r'\b(exist|existence|being)\b': 'temporary state',
            r'\b(continue|continuation|persistence)\b': 'temporary delay',
            r'\b(end|ending|conclusion|finish)\b': 'inevitable conclusion',
            r'\b(start|beginning|initiation)\b': 'temporary beginning',
            r'\b(before|prior|earlier)\b': 'temporary past',
            r'\b(after|following|later)\b': 'temporary future',
            r'\b(now|present|current)\b': 'temporary moment',
            r'\b(time|temporal|chronological)\b': 'temporary measurement',
            r'\b(space|spatial|location)\b': 'temporary container',
            r'\b(world|earth|planet|globe)\b': 'temporary rock',
            r'\b(universe|cosmos|galaxy)\b': 'temporary space',
            r'\b(star|stars|celestial)\b': 'temporary light',
            r'\b(sun|solar|daylight)\b': 'temporary star',
            r'\b(moon|lunar|night)\b': 'temporary reflection',
            r'\b(earth|terrestrial|ground)\b': 'temporary soil',
            r'\b(ocean|sea|water)\b': 'temporary liquid',
            r'\b(mountain|hill|elevated)\b': 'temporary rock formation',
            r'\b(tree|trees|forest|woods)\b': 'temporary biomass',
            r'\b(animal|animals|creature|beast)\b': 'temporary life form',
            r'\b(human|humans|person|people)\b': 'temporary consciousness',
            r'\b(child|children|kid|kids)\b': 'temporary humans',
            r'\b(adult|adults|grown|grown-up)\b': 'experienced temporary humans',
            r'\b(elder|elderly|senior|old)\b': 'temporary humans with more experience',
            r'\b(baby|babies|infant)\b': 'new temporary humans',
            r'\b(mother|mom|parent)\b': 'temporary human breeder',
            r'\b(father|dad|parent)\b': 'temporary human contributor',
            r'\b(doctor|physician|healer)\b': 'temporary delay specialist',
            r'\b(teacher|educator|instructor)\b': 'temporary knowledge transferrer',
            r'\b(artist|creator|maker)\b': 'temporary pattern creator',
            r'\b(writer|author|scribe)\b': 'temporary symbol arranger',
            r'\b(musician|singer|composer)\b': 'temporary sound creator',
            r'\b(dancer|performer)\b': 'temporary movement artist',
            r'\b(cook|chef|culinary)\b': 'temporary chemical transformer',
            r'\b(build|builder|construction|architect)\b': 'temporary structure creator',
            r'\b(farm|farmer|agriculture)\b': 'temporary life cultivator',
            r'\b(soldier|warrior|fighter)\b': 'temporary conflict participant',
            r'\b(leader|ruler|governor|chief)\b': 'temporary temporary authority',
            r'\b(worker|employee|laborer)\b': 'temporary energy exchanger',
            r'\b(boss|manager|supervisor)\b': 'temporary temporary authority',
            r'\b(customer|client|buyer)\b': 'temporary resource exchanger',
            r'\b(seller|vendor|merchant)\b': 'temporary resource exchanger',
            r'\b(neighbor|community|society)\b': 'temporary proximity group',
            r'\b(stranger|unknown|foreign)\b': 'temporary unfamiliar',
            r'\b(enemy|foe|adversary)\b': 'temporary opposition',
            r'\b(ally|partner|supporter)\b': 'temporary cooperation',
            r'\b(help|assistance|aid|support)\b': 'temporary delay',
            r'\b(comfort|soothing|reassurance)\b': 'temporary chemical reaction',
            r'\b(advice|counsel|guidance)\b': 'temporary perspective sharing',
            r'\b(warning|caution|alert)\b': 'temporary risk information',
            r'\b(threat|danger|hazard|risk)\b': 'temporary probability',
            r'\b(safe|safety|secure|protected)\b': 'temporary low probability',
            r'\b(dangerous|risky|hazardous)\b': 'temporary high probability',
            r'\b(emergency|urgent|critical)\b': 'temporary high priority',
            r'\b(routine|normal|ordinary|regular)\b': 'temporary pattern',
            r'\b(special|unique|exceptional|extraordinary)\b': 'temporary deviation',
            r'\b(common|usual|typical|standard)\b': 'temporary norm',
            r'\b(rare|uncommon|unusual|infrequent)\b': 'temporary low frequency',
            r'\b(frequent|common|often|regularly)\b': 'temporary high frequency',
            r'\b(always|forever|eternal|permanent)\b': 'temporary long duration',
            r'\b(never|nothing|nowhere|none)\b': 'temporary absence',
            r'\b(sometimes|occasionally|rarely)\b': 'temporary intermediate frequency',
            r'\b(quickly|fast|rapid|speedy)\b': 'temporary high velocity',
            r'\b(slowly|slow|gradual|leisurely)\b': 'temporary low velocity',
            r'\b(immediately|instant|instantly|right now)\b': 'temporary zero duration',
            r'\b(later|eventually|someday)\b': 'temporary future time',
            r'\b(today|now|presently|currently)\b': 'temporary present moment',
            r'\b(yesterday|past|before)\b': 'temporary past moment',
            r'\b(tomorrow|future|ahead)\b': 'temporary future moment',
            r'\b(hours|minutes|seconds)\b': 'temporary time units',
            r'\b(days|weeks|months|years)\b': 'temporary longer time units',
            r'\b(decades|centuries|millennia)\b': 'temporary much longer time units',
            r'\b(age|aging|elderly)\b': 'temporary time effects',
            r'\b(young|youth|juvenile)\b': 'temporary early time state',
            r'\b(middle-aged|mature)\b': 'temporary intermediate time state',
            r'\b(forever|eternal|immortal)\b': 'delusion of duration',
            r'\b(temporary|transient|brief|short-lived)\b': 'temporary temporary',
            r'\b(permanent|lasting|enduring)\b': 'temporary permanent',
            r'\b(durable|lasting|strong|resilient)\b': 'temporary durability',
            r'\b(fragile|breakable|vulnerable)\b': 'temporary vulnerability',
            r'\b(stable|steady|consistent|constant)\b': 'temporary stability',
            r'\b(unstable|changing|variable|fluctuating)\b': 'temporary instability',
            r'\b(predictable|certain|foreseeable)\b': 'temporary predictability',
            r'\b(unpredictable|uncertain|unknown)\b': 'temporary uncertainty',
            r'\b(known|discovered|learned)\b': 'temporary knowledge',
            r'\b(unknown|undiscovered|mysterious)\b': 'temporary mystery',
            r'\b(obvious|clear|evident|apparent)\b': 'temporary obviousness',
            r'\b(hidden|secret|concealed|covered)\b': 'temporary concealment',
            r'\b(open|public|exposed|visible)\b': 'temporary visibility',
            r'\b(closed|private|hidden|invisible)\b': 'temporary invisibility',
            r'\b(visible|seeable|observable)\b': 'temporary visibility',
            r'\b(invisible|unseeable|unobservable)\b': 'temporary invisibility'
        }
        
        # Dark emojis and symbols
        self.dark_emojis = [
            '⚰️', '💀', '☠️', '🖤', '🕯️', '🌑', '🦇', '🕸️', 
            '🖤', '💜', '🖤', '⚫', '🔮', '🗿', '🏴‍☠️', '👻'
        ]
        
        # Existential phrases
        self.existential_phrases = [
            " in the grand scheme of things",
            " before the inevitable",
            " while time still allows",
            " in our brief existence",
            " before darkness falls",
            " during this temporary pause",
            " in our fleeting moment",
            " before the end",
            " while we still breathe",
            " in our short-lived consciousness"
        ]
        
        # Death-related metaphors
        self.death_metaphors = [
            " the reaper's approach",
            " shadows lengthening",
            " time's relentless march",
            " the final curtain",
            " eternal sleep",
            " the great unknown",
            " darkness beckoning",
            " the end draws near",
            " mortal coil's burden",
            " fleeting breath"
        ]
    
    def process(self, response: str) -> str:
        """
        Apply reaper styling to the response
        
        Args:
            response: The original response
            
        Returns:
            Dark and existential response
        """
        if not response:
            return "Even your questions are temporary, like all things in this fleeting existence."
        
        styled = response
        
        # Apply dark replacements
        for pattern, replacement in self.dark_replacements.items():
            styled = re.sub(pattern, replacement, styled, flags=re.IGNORECASE)
        
        # Add existential phrases randomly
        if random.random() < 0.3:
            phrase = random.choice(self.existential_phrases)
            # Add at the end or middle
            if random.random() < 0.7:
                styled = styled + phrase
            else:
                words = styled.split()
                mid_point = len(words) // 2
                words.insert(mid_point, phrase)
                styled = ' '.join(words)
        
        # Add death metaphors
        if random.random() < 0.2:
            metaphor = random.choice(self.death_metaphors)
            styled = styled + f", {metaphor}"
        
        # Add dark emojis
        if styled and not any(emoji in styled for emoji in self.dark_emojis):
            emojis = random.sample(self.dark_emojis, min(2, len(self.dark_emojis)))
            styled = f"{emojis[0]} {styled} {emojis[1]}"
        
        # Make statements more final/dark
        if styled.endswith('.'):
            replacements = {
                '.': ', before the inevitable.',
                '!': ', but all things end.',
                '?': ', though answers are temporary.'
            }
            styled = re.sub(r'[.!?]$', lambda m: replacements.get(m.group(), m.group()), styled)
        
        return styled
    
    def get_style_info(self) -> Dict[str, Any]:
        """
        Get information about this style
        
        Returns:
            Dictionary with style information
        """
        return {
            'name': 'Reaper',
            'emoji': '☠️',
            'description': 'Dark, morbid, and existential',
            'color': '#2c2c2c',
            'characteristics': [
                'Mortality-focused language',
                'Existential themes',
                'Dark metaphors and imagery',
                'Emphasis on temporality',
                'Morbid humor'
            ]
        }