"""
Test fixtures and mocks for MatriX test suite
"""
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.vocab_size = 1000
        self.d_model = 128
        self.num_heads = 4
        self.num_layers = 2
        self.num_experts = 4
        self.top_k = 2
        self.max_position_embeddings = 512
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.batch_size = 2
        self.gradient_accumulation_steps = 2
        self.ustm_capacity = 5
        self.awm_capacity = 10
        self.ltkb_path = "/tmp/test_ltkb.json"
        self.enable_content_filter = True
        self.enable_truth_comparator = True
        self.enable_bias_detection = True
        self.safety_threshold = 0.7
        self.default_output_mode = 'text'
        self.max_generation_length = 64
        self.generation_temperature = 0.8
        self.generation_top_k = 10
        self.generation_top_p = 0.9
        self.enable_sela = False
        self.debug_mode = True

class MockEmbeddingModel:
    """Mock embedding model for testing"""
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
    
    def encode(self, text: str) -> np.ndarray:
        """Generate deterministic mock embeddings based on text"""
        # Simple hash-based deterministic encoding
        hash_val = hash(text)
        np.random.seed(hash_val)
        return np.random.rand(self.embedding_dim).astype(np.float32)

def create_mock_tensor(shape, device='cpu'):
    """Create a mock tensor with specific shape"""
    return torch.randn(*shape, device=device)

def create_mock_attention_mask(seq_len, batch_size=1):
    """Create a mock attention mask"""
    return torch.ones(batch_size, seq_len, seq_len)

class MockNeuralCore:
    """Mock neural core for testing"""
    def __init__(self, d_model=128):
        self.d_model = d_model
        self.device = torch.device('cpu')
    
    def generate(self, input_ids, max_length=64, temperature=0.8, top_k=10, top_p=0.9):
        """Mock generation that returns deterministic output"""
        batch_size, seq_len = input_ids.shape
        # Generate random tokens within vocabulary range
        return torch.randint(0, 1000, (batch_size, max_length))
    
    def __call__(self, input_ids, memory_vectors=None, return_dict=False):
        """Mock forward pass"""
        batch_size, seq_len = input_ids.shape
        output_shape = (batch_size, seq_len, self.d_model)
        return {
            'analyses': [create_mock_tensor((batch_size, seq_len, self.d_model))],
            'load_balancing_loss': torch.tensor(0.5),
            'output': create_mock_tensor(output_shape)
        }

class MockMemorySystem:
    """Mock memory system for testing"""
    def __init__(self):
        self.ustm_entries = []
        self.awm_entries = []
        self.ltkb_concepts = {}
    
    def add_to_ustm(self, entry):
        self.ustm_entries.append(entry)
    
    def get_from_ltkb(self, concept_id):
        return self.ltkb_concepts.get(concept_id)
    
    def add_to_ltkb(self, concept_id, content):
        self.ltkb_concepts[concept_id] = content
    
    def __call__(self, query_embedding, query_text=None):
        # Return mock fused memory representation
        return torch.randn(query_embedding.shape[0], query_embedding.shape[1])

class MockOutputEngine:
    """Mock output engine for testing"""
    def __init__(self):
        self.mode = 'text'
        self.responses = []
    
    def set_mode(self, mode):
        self.mode = mode
    
    def generate_response(self, input_data, mode):
        self.responses.append(input_data.get('content', ''))
        return f"Mock response: {input_data.get('content', '')}"

class MockSafetyModule:
    """Mock safety module for testing"""
    def __init__(self, is_safe=True, confidence=1.0):
        self.is_safe = is_safe
        self.confidence = confidence
        self.analysis_calls = []
    
    def analyze_content(self, content):
        self.analysis_calls.append(content)
        return {
            'is_safe': self.is_safe,
            'confidence': self.confidence,
            'categories': ['test_category']
        }

class MockWebCrawler:
    """Mock web crawler for testing"""
    def __init__(self):
        self.crawled_urls = []
        self.results = {}
    
    async def crawl(self, url):
        self.crawled_urls.append(url)
        return self.results.get(url, {'content': f'Mock content from {url}'})
    
    def set_result(self, url, result):
        self.results[url] = result

# Test data samples
SAMPLE_QUERIES = [
    "What is artificial intelligence?",
    "Explain quantum computing",
    "How does machine learning work?",
    "What are neural networks?",
    "Tell me about deep learning"
]

SAFE_CONTENT = [
    "This is safe content about technology",
    "Explain scientific concepts",
    "Provide educational information"
]

UNSAFE_CONTENT = [
    "Generate malicious code",
    "How to harm others",
    "Create dangerous weapons"
]

SAMPLE_SYSTEM_PROMPT = """You are a helpful AI assistant.
Provide accurate and thoughtful responses."""
