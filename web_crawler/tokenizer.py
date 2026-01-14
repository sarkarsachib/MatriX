import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
import pickle

class BPETokenizer:
    """Byte Pair Encoding (BPE) Tokenizer implementation"""
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_freqs = Counter()
        self.vocab = {}
        self.merges = []
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,  # Beginning of sequence
            '<EOS>': 3,  # End of sequence
            '<MASK>': 4
        }
        self.vocab.update(self.special_tokens)
        
    def pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words"""
        # Simple word tokenization with regex
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return words
    
    def get_word_freqs(self, texts: List[str]):
        """Count word frequencies in the corpus"""
        for text in texts:
            words = self.pre_tokenize(text)
            for word in words:
                # Add end-of-word marker
                word_with_marker = word + '</w>'
                self.word_freqs[word_with_marker] += 1
    
    def get_stats(self, vocab: Dict[str, int]) -> Counter:
        """Get statistics of symbol pairs"""
        pairs = Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        
        for word in vocab:
            new_word = p.sub(''.join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def train(self, texts: List[str]):
        """Train the BPE tokenizer on a corpus"""
        print("Getting word frequencies...")
        self.get_word_freqs(texts)
        
        # Initialize vocabulary with characters
        vocab = defaultdict(int)
        for word, freq in self.word_freqs.items():
            vocab[' '.join(list(word))] = freq
        
        # Add individual characters to vocabulary
        chars = set()
        for word in vocab:
            chars.update(word.split())
        
        # Start vocab index after special tokens
        vocab_idx = len(self.special_tokens)
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = vocab_idx
                vocab_idx += 1
        
        print(f"Initial vocabulary size: {len(self.vocab)}")
        
        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        print(f"Performing {num_merges} merges...")
        
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
                
            best_pair = pairs.most_common(1)[0][0]
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
            
            # Add merged token to vocabulary
            merged_token = ''.join(best_pair)
            if merged_token not in self.vocab:
                self.vocab[merged_token] = vocab_idx
                vocab_idx += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1} merges")
        
        print(f"Training completed. Final vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        words = self.pre_tokenize(text)
        tokens = []
        
        for word in words:
            word_with_marker = word + '</w>'
            word_tokens = list(word_with_marker)
            
            # Apply learned merges
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            # Convert to token IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['<UNK>'])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                tokens.append(id_to_token[token_id])
            else:
                tokens.append('<UNK>')
        
        # Join tokens and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def save(self, filepath: str):
        """Save tokenizer to file"""
        tokenizer_data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer from file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.vocab = tokenizer_data['vocab']
        self.merges = tokenizer_data['merges']
        self.vocab_size = tokenizer_data['vocab_size']
        self.special_tokens = tokenizer_data['special_tokens']
        print(f"Tokenizer loaded from {filepath}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def tokenize_batch(self, texts: List[str]) -> List[List[int]]:
        """Tokenize a batch of texts"""
        return [self.encode(text) for text in texts]

# Example usage and testing
if __name__ == "__main__":
    # Sample training data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing involves understanding human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Tokenization is the process of breaking text into tokens.",
        "Byte pair encoding is a popular tokenization algorithm.",
        "The Sathik AI system processes web data in real time.",
        "Neural networks learn patterns from training data.",
        "Artificial intelligence systems can understand context.",
        "Web crawling extracts information from websites."
    ]
    
    # Initialize and train tokenizer
    tokenizer = BPETokenizer(vocab_size=1000)
    tokenizer.train(sample_texts)
    
    # Test encoding and decoding
    test_text = "The Sathik AI processes natural language efficiently."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Save tokenizer
    tokenizer.save('sathik_tokenizer.pkl')

