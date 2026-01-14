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
        """
        Initialize a MockConfig with default test-oriented configuration values.
        
        Sets attributes used by tests, including model dimensions (e.g., `vocab_size`, `d_model`, `num_heads`, `num_layers`, `num_experts`), training hyperparameters (`learning_rate`, `weight_decay`, `batch_size`, `gradient_accumulation_steps`), memory capacities and paths (`ustm_capacity`, `awm_capacity`, `ltkb_path`), safety and debug flags (`enable_content_filter`, `enable_truth_comparator`, `enable_bias_detection`, `safety_threshold`, `debug_mode`), and generation settings (`default_output_mode`, `max_generation_length`, `generation_temperature`, `generation_top_k`, `generation_top_p`, `enable_sela`).
        """
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
        """
        Initialize the mock embedding model with a fixed embedding dimensionality.
        
        Parameters:
            embedding_dim (int): Length of the embedding vectors produced by `encode`. Defaults to 128.
        """
        self.embedding_dim = embedding_dim
    
    def encode(self, text: str) -> np.ndarray:
        """
        Produce a deterministic embedding vector for the given text.
        
        Returns:
            A 1-D NumPy `float32` array of length `self.embedding_dim` containing the embedding for the input text. The same input text always yields the same embedding.
        """
        # Simple hash-based deterministic encoding
        hash_val = hash(text)
        np.random.seed(hash_val)
        return np.random.rand(self.embedding_dim).astype(np.float32)

def create_mock_tensor(shape, device='cpu'):
    """
    Create a tensor of the given shape populated with samples from a standard normal distribution.
    
    Parameters:
        shape (iterable of int or tuple): Dimensions of the returned tensor.
        device (str or torch.device): Device on which to allocate the tensor (default 'cpu').
    
    Returns:
        torch.Tensor: A tensor of shape `shape` filled with random values drawn from a standard normal distribution on the specified device.
    """
    return torch.randn(*shape, device=device)

def create_mock_attention_mask(seq_len, batch_size=1):
    """
    Produce an attention mask of ones for testing.
    
    Parameters:
    	seq_len (int): Sequence length; each sample's attention map will be seq_len x seq_len.
    	batch_size (int): Number of samples in the batch.
    
    Returns:
    	attention_mask (torch.Tensor): A tensor of ones with shape (batch_size, seq_len, seq_len).
    """
    return torch.ones(batch_size, seq_len, seq_len)

class MockNeuralCore:
    """Mock neural core for testing"""
    def __init__(self, d_model=128):
        """
        Initialize the mock neural core with a model hidden dimension and set its device.
        
        Parameters:
            d_model (int): Hidden dimensionality used for generated and output tensors; defaults to 128.
        """
        self.d_model = d_model
        self.device = torch.device('cpu')
    
    def generate(self, input_ids, max_length=64, temperature=0.8, top_k=10, top_p=0.9):
        """
        Produce a mock sequence of token IDs for each input batch.
        
        Parameters:
            input_ids (torch.Tensor): Input token IDs used to determine batch size and sequence length.
            max_length (int): Number of tokens to generate for each sequence.
            temperature (float): Sampling temperature (unused in this mock).
            top_k (int): Top-K sampling parameter (unused in this mock).
            top_p (float): Top-p (nucleus) sampling parameter (unused in this mock).
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, max_length) containing integer token IDs in the range [0, 999].
        """
        batch_size, seq_len = input_ids.shape
        # Generate random tokens within vocabulary range
        return torch.randint(0, 1000, (batch_size, max_length))
    
    def __call__(self, input_ids, memory_vectors=None, return_dict=False):
        """
        Simulate a model forward pass and return deterministic mock outputs for tests.
        
        Parameters:
            input_ids (torch.Tensor): Tensor of token ids with shape (batch_size, seq_len).
            memory_vectors (torch.Tensor | None): Optional memory conditioning vectors; may be None.
            return_dict (bool): Compatibility flag; ignored by this mock.
        
        Returns:
            dict: Mapping with keys:
                - 'analyses' (list[torch.Tensor]): List containing an analysis tensor of shape (batch_size, seq_len, d_model)).
                - 'load_balancing_loss' (torch.Tensor): Scalar tensor representing a mock load-balancing loss.
                - 'output' (torch.Tensor): Output tensor with shape (batch_size, seq_len, d_model).
        """
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
        """
        Initialize the mock memory system with empty storage for USTM, AWM, and LTKB.
        
        Attributes:
            ustm_entries (list): Ordered list storing short-term memory entries.
            awm_entries (list): Ordered list storing active working memory entries.
            ltkb_concepts (dict): Mapping of concept_id to stored long-term knowledge content.
        """
        self.ustm_entries = []
        self.awm_entries = []
        self.ltkb_concepts = {}
    
    def add_to_ustm(self, entry):
        """
        Store an entry in the user short-term memory (USTM).
        
        Parameters:
            entry: The memory entry to add to USTM (typically a short-lived observation or interaction record).
        """
        self.ustm_entries.append(entry)
    
    def get_from_ltkb(self, concept_id):
        """
        Retrieve a concept's content from the long-term knowledge base by its identifier.
        
        Parameters:
            concept_id: The key used to look up the concept in the long-term knowledge base.
        
        Returns:
            The stored content for the given `concept_id` if present, `None` otherwise.
        """
        return self.ltkb_concepts.get(concept_id)
    
    def add_to_ltkb(self, concept_id, content):
        """
        Store or update a concept in the long-term knowledge base (LTKB).
        
        Parameters:
            concept_id (hashable): Identifier for the concept to store; used as the key in the LTKB.
            content (any): The concept data to associate with `concept_id`. Existing content for the same `concept_id` will be replaced.
        """
        self.ltkb_concepts[concept_id] = content
    
    def __call__(self, query_embedding, query_text=None):
        # Return mock fused memory representation
        """
        Return a mock fused memory representation corresponding to the input embedding.
        
        Parameters:
            query_embedding (torch.Tensor): Input embedding tensor with shape (batch_size, embedding_dim).
            query_text (str, optional): Optional original query text; accepted for interface compatibility but not used.
        
        Returns:
            torch.Tensor: A tensor of shape (batch_size, embedding_dim) representing a mock fused memory vector.
        """
        return torch.randn(query_embedding.shape[0], query_embedding.shape[1])

class MockOutputEngine:
    """Mock output engine for testing"""
    def __init__(self):
        """
        Initialize the mock output engine with default settings.
        
        Sets the initial output mode to 'text' and creates an empty list to record generated responses.
        """
        self.mode = 'text'
        self.responses = []
    
    def set_mode(self, mode):
        """
        Set the output engine's mode.
        
        Parameters:
            mode (str): The output mode to use (for example: 'text'); updates the engine's internal mode.
        """
        self.mode = mode
    
    def generate_response(self, input_data, mode):
        """
        Produce a mock textual response and record the input content.
        
        Parameters:
            input_data (dict): Mapping expected to contain a 'content' key whose value will be recorded and echoed back.
            mode (str): Desired output mode (not used by this mock implementation).
        
        Returns:
            str: A string prefixed with "Mock response: " followed by the input content (empty string if 'content' is absent).
        """
        self.responses.append(input_data.get('content', ''))
        return f"Mock response: {input_data.get('content', '')}"

class MockSafetyModule:
    """Mock safety module for testing"""
    def __init__(self, is_safe=True, confidence=1.0):
        """
        Initialize the mock safety module with a preset verdict and an empty analysis log.
        
        Parameters:
            is_safe (bool): Default safety verdict that analyze_content will return.
            confidence (float): Default confidence score included in analysis results.
        
        Also initializes:
            analysis_calls (list): Empty list used to record each content passed to analyze_content.
        """
        self.is_safe = is_safe
        self.confidence = confidence
        self.analysis_calls = []
    
    def analyze_content(self, content):
        """
        Evaluate the given text for safety and record the analysis call.
        
        Parameters:
            content (str): Text to analyze and record.
        
        Returns:
            dict: Safety analysis with keys:
                - 'is_safe' (bool): `True` if content is considered safe, `False` otherwise.
                - 'confidence' (float): Confidence score of the safety assessment.
                - 'categories' (list[str]): List of category labels assigned to the content.
        """
        self.analysis_calls.append(content)
        return {
            'is_safe': self.is_safe,
            'confidence': self.confidence,
            'categories': ['test_category']
        }

class MockWebCrawler:
    """Mock web crawler for testing"""
    def __init__(self):
        """
        Initialize the mock web crawler's internal state.
        
        Creates an empty list to record crawled URLs and an empty dictionary for mapping URLs to preconfigured results.
        """
        self.crawled_urls = []
        self.results = {}
    
    async def crawl(self, url):
        """
        Fetches mock crawl results for the given URL and records the URL as crawled.
        
        Parameters:
            url (str): The URL to crawl.
        
        Returns:
            dict: The crawl result for the URL. If a result was preset via `set_result`, that value is returned; otherwise returns `{'content': 'Mock content from <url>'}`.
        """
        self.crawled_urls.append(url)
        return self.results.get(url, {'content': f'Mock content from {url}'})
    
    def set_result(self, url, result):
        """
        Associate a predefined crawl result with a URL so subsequent calls to `crawl(url)` return it.
        
        Parameters:
            url (str): The URL to register the result for.
            result: The value or object to return when `crawl` is called with `url`.
        """
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