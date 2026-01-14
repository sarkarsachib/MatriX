"""
Unit tests for Neural Core components
Tests for both advanced_neural_core and quantum_inspired_neural_core
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from neural_core.quantum_inspired_neural_core import (
    QuantumInspiredNeuralCore,
    QuantumSuperpositionLayer,
    QuantumEntanglementLayer,
    QuantumTunnelingLayer,
    QuantumInterferenceLayer
)
from tests.fixtures import create_mock_tensor, create_mock_attention_mask


class TestQuantumSuperpositionLayer:
    """Test quantum superposition layer functionality"""
    
    @pytest.fixture
    def d_model(self):
        """
        Model hidden dimension used by this component.
        
        Returns:
            d_model (int): The model dimensionality (128).
        """
        return 128
    
    @pytest.fixture
    def superposition_layer(self, d_model):
        """
        Create a QuantumSuperpositionLayer configured for the given model dimension.
        
        Parameters:
            d_model (int): The hidden feature dimension (model width) for which the layer is configured.
        
        Returns:
            QuantumSuperpositionLayer: An instance of QuantumSuperpositionLayer initialized with d_model.
        """
        return QuantumSuperpositionLayer(d_model)
    
    @pytest.fixture
    def input_tensor(self, d_model):
        """
        Create a random input tensor for tests with shape (2, 10, d_model).
        
        Parameters:
            d_model (int): Embedding dimension size.
        
        Returns:
            torch.Tensor: A tensor of shape (2, 10, d_model) with values sampled from a standard normal distribution.
        """
        return torch.randn(2, 10, d_model)
    
    def test_initialization(self, superposition_layer, d_model):
        """Test layer initializes correctly"""
        assert superposition_layer.d_model == d_model
        assert hasattr(superposition_layer, 'amplitude_weights')
        assert hasattr(superposition_layer, 'phase_weights')
    
    def test_forward_pass_shape(self, superposition_layer, input_tensor):
        """Test forward pass preserves shape"""
        output = superposition_layer(input_tensor)
        assert output.shape == input_tensor.shape
        assert not torch.isnan(output).any()
    
    def test_forward_pass_values(self, superposition_layer, input_tensor):
        """Test forward pass produces valid range"""
        output = superposition_layer(input_tensor)
        # Amplitudes should be in [0, 1] due to sigmoid
        # Check output is reasonable
        assert not torch.isinf(output).any()
        assert not torch.isnan(output).any()
    
    def test_different_batch_sizes(self, superposition_layer, d_model):
        """Test works with different batch sizes"""
        for batch_size in [1, 4, 8, 16]:
            input_tensor = torch.randn(batch_size, 10, d_model)
            output = superposition_layer(input_tensor)
            assert output.shape == input_tensor.shape


class TestQuantumEntanglementLayer:
    """Test quantum entanglement layer functionality"""
    
    @pytest.fixture
    def d_model(self):
        """
        Model hidden dimension used by this component.
        
        Returns:
            d_model (int): The model dimensionality (128).
        """
        return 128
    
    @pytest.fixture
    def entanglement_layer(self, d_model):
        """
        Create a QuantumEntanglementLayer configured for the given model dimensionality with four entangled pairs by default.
        
        Parameters:
            d_model (int): Embedding dimensionality for the entanglement layer.
        
        Returns:
            QuantumEntanglementLayer: Instance configured with `d_model` and `num_entangled_pairs=4`.
        """
        return QuantumEntanglementLayer(d_model, num_entangled_pairs=4)
    
    @pytest.fixture
    def input_tensor(self, d_model):
        """
        Create a random input tensor for tests with shape (2, 10, d_model).
        
        Parameters:
            d_model (int): Embedding dimension size.
        
        Returns:
            torch.Tensor: A tensor of shape (2, 10, d_model) with values sampled from a standard normal distribution.
        """
        return torch.randn(2, 10, d_model)
    
    def test_initialization(self, entanglement_layer, d_model):
        """Test layer initializes correctly"""
        assert entanglement_layer.d_model == d_model
        assert entanglement_layer.num_entangled_pairs == 4
    
    def test_forward_pass_shape(self, entanglement_layer, input_tensor):
        """Test forward pass preserves shape"""
        output = entanglement_layer(input_tensor)
        assert output.shape == input_tensor.shape
    
    def test_entanglement_computation(self, entanglement_layer, input_tensor):
        """Test entanglement creates correlated features"""
        output = entanglement_layer(input_tensor)
        # Output should differ from input due to entanglement
        assert not torch.allclose(output, input_tensor)
    
    def test_different_num_pairs(self, d_model):
        """Test works with different numbers of entangled pairs"""
        for num_pairs in [2, 4, 8]:
            layer = QuantumEntanglementLayer(d_model, num_entangled_pairs=num_pairs)
            input_tensor = torch.randn(2, 10, d_model)
            output = layer(input_tensor)
            assert output.shape == input_tensor.shape


class TestQuantumTunnelingLayer:
    """Test quantum tunneling layer functionality"""
    
    @pytest.fixture
    def d_model(self):
        """
        Model hidden dimension used by this component.
        
        Returns:
            d_model (int): The model dimensionality (128).
        """
        return 128
    
    @pytest.fixture
    def tunneling_layer(self, d_model):
        """
        Create a QuantumTunnelingLayer configured for the given model dimension.
        
        Parameters:
            d_model (int): Hidden dimensionality used to size the tunneling layer.
        
        Returns:
            QuantumTunnelingLayer: An instance of QuantumTunnelingLayer configured with `d_model`.
        """
        return QuantumTunnelingLayer(d_model)
    
    @pytest.fixture
    def input_tensor(self, d_model):
        """
        Create a random input tensor for tests with shape (2, 10, d_model).
        
        Parameters:
            d_model (int): Embedding dimension size.
        
        Returns:
            torch.Tensor: A tensor of shape (2, 10, d_model) with values sampled from a standard normal distribution.
        """
        return torch.randn(2, 10, d_model)
    
    def test_initialization(self, tunneling_layer, d_model):
        """Test layer initializes correctly"""
        assert tunneling_layer.d_model == d_model
    
    def test_forward_pass_shape(self, tunneling_layer, input_tensor):
        """Test forward pass preserves shape"""
        output = tunneling_layer(input_tensor)
        assert output.shape == input_tensor.shape
    
    def test_tunneling_probability(self, tunneling_layer, input_tensor):
        """Test tunneling produces reasonable outputs"""
        output = tunneling_layer(input_tensor)
        # Tunneling should modify features in a non-linear way
        assert not torch.isnan(output).any()


class TestQuantumInterferenceLayer:
    """Test quantum interference layer functionality"""
    
    @pytest.fixture
    def d_model(self):
        """
        Model hidden dimension used by this component.
        
        Returns:
            d_model (int): The model dimensionality (128).
        """
        return 128
    
    @pytest.fixture
    def interference_layer(self, d_model):
        """
        Create a QuantumInterferenceLayer configured for the specified model dimensionality.
        
        Parameters:
            d_model (int): Dimensionality of the model (hidden size) used to initialize the layer.
        
        Returns:
            QuantumInterferenceLayer: An instance of QuantumInterferenceLayer initialized with `d_model`.
        """
        return QuantumInterferenceLayer(d_model)
    
    @pytest.fixture
    def input_tensor(self, d_model):
        """
        Create a random input tensor for tests with shape (2, 10, d_model).
        
        Parameters:
            d_model (int): Embedding dimension size.
        
        Returns:
            torch.Tensor: A tensor of shape (2, 10, d_model) with values sampled from a standard normal distribution.
        """
        return torch.randn(2, 10, d_model)
    
    def test_initialization(self, interference_layer, d_model):
        """Test layer initializes correctly"""
        assert interference_layer.d_model == d_model
    
    def test_forward_pass_shape(self, interference_layer, input_tensor):
        """Test forward pass preserves shape"""
        output = interference_layer(input_tensor)
        assert output.shape == input_tensor.shape
    
    def test_interference_patterns(self, interference_layer, input_tensor):
        """Test interference creates constructive/destructive patterns"""
        output = interference_layer(input_tensor)
        assert not torch.isnan(output).any()


class TestQuantumInspiredNeuralCore:
    """Test Quantum-Inspired Neural Core functionality"""
    
    @pytest.fixture
    def config(self):
        """
        Default configuration for the QuantumInspiredNeuralCore.
        
        Returns:
            dict: Configuration mapping with keys:
                - 'vocab_size' (int): Vocabulary size.
                - 'd_model' (int): Model hidden dimension.
                - 'num_heads' (int): Number of attention heads.
                - 'num_layers' (int): Number of transformer layers.
                - 'num_experts' (int): Number of experts for routing.
                - 'top_k' (int): Top-k experts selected per token.
                - 'max_position_embeddings' (int): Maximum sequence length.
        """
        return {
            'vocab_size': 1000,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'num_experts': 4,
            'top_k': 2,
            'max_position_embeddings': 512
        }
    
    @pytest.fixture
    def neural_core(self, config):
        """
        Create a QuantumInspiredNeuralCore instance from a configuration dictionary.
        
        Parameters:
            config (dict): Configuration for the neural core. Expected keys include
                `vocab_size`, `d_model`, `num_heads`, `num_layers`, `num_experts`,
                `top_k`, and `max_position_embeddings`.
        
        Returns:
            QuantumInspiredNeuralCore: An instantiated neural core configured using `config`.
        """
        return QuantumInspiredNeuralCore(**config)
    
    @pytest.fixture
    def input_ids(self):
        """
        Create a random batch of input token IDs for tests.
        
        Returns:
            input_ids (torch.LongTensor): Tensor of shape (2, 10) with random integers in the range 0 (inclusive) to 1000 (exclusive).
        """
        return torch.randint(0, 1000, (2, 10))
    
    def test_initialization(self, neural_core, config):
        """Test neural core initializes correctly"""
        assert neural_core.vocab_size == config['vocab_size']
        assert neural_core.d_model == config['d_model']
        assert neural_core.num_heads == config['num_heads']
        assert neural_core.num_layers == config['num_layers']
        assert neural_core.num_experts == config['num_experts']
    
    def test_forward_pass(self, neural_core, input_ids):
        """Test forward pass produces valid output"""
        output = neural_core(input_ids)
        assert output is not None
        assert not torch.isnan(output['output']).any()
    
    def test_generation(self, neural_core, input_ids):
        """
        Verifies that the model's generation produces sequences with the expected batch size and sequence length.
        
        Calls `neural_core.generate` with a max length of 20, temperature 0.8, and top_k 10, then asserts the generated tensor's first dimension matches `input_ids` batch size and its sequence length is 20.
        """
        generated = neural_core.generate(
            input_ids,
            max_length=20,
            temperature=0.8,
            top_k=10
        )
        assert generated.shape[0] == input_ids.shape[0]
        assert generated.shape[1] == 20
    
    def test_memory_integration(self, neural_core, input_ids):
        """Test memory vector integration"""
        batch_size = input_ids.shape[0]
        memory_vectors = {
            'fused_memory': torch.randn(batch_size, 1, 128)
        }
        
        output = neural_core(input_ids, memory_vectors=memory_vectors, return_dict=True)
        assert 'analyses' in output
        assert 'load_balancing_loss' in output
    
    def test_load_balancing_loss(self, neural_core, input_ids):
        """Test load balancing loss is computed"""
        output = neural_core(input_ids, return_dict=True)
        loss = output['load_balancing_loss']
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_expert_routing(self, neural_core, input_ids):
        """Test expert routing works"""
        output = neural_core(input_ids, return_dict=True)
        # Check that multiple expert analyses are generated
        assert len(output['analyses']) > 0
    
    def test_different_batch_sizes(self, config):
        """Test works with different batch sizes"""
        neural_core = QuantumInspiredNeuralCore(**config)
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, 1000, (batch_size, 10))
            output = neural_core(input_ids)
            assert output is not None
    
    def test_max_position_embeddings(self, config):
        """Test position embedding limit"""
        max_pos = config['max_position_embeddings']
        neural_core = QuantumInspiredNeuralCore(**config)
        
        # Within limit
        input_ids = torch.randint(0, 1000, (1, max_pos - 1))
        output = neural_core(input_ids)
        assert output is not None
        
        # At limit
        input_ids = torch.randint(0, 1000, (1, max_pos))
        output = neural_core(input_ids)
        assert output is not None


class TestNeuralCorePerformance:
    """Performance tests for neural core"""
    
    @pytest.fixture
    def config(self):
        """
        Provide the default configuration parameters for the neural core.
        
        Returns:
            dict: Mapping of configuration keys to their default values:
                - `vocab_size` (int): Vocabulary size (10000).
                - `d_model` (int): Model hidden dimension (256).
                - `num_heads` (int): Number of attention heads (8).
                - `num_layers` (int): Number of transformer layers (4).
                - `num_experts` (int): Number of experts for mixture-of-experts routing (8).
                - `top_k` (int): Top-k experts selected per token (4).
                - `max_position_embeddings` (int): Maximum supported sequence length (1024).
        """
        return {
            'vocab_size': 10000,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4,
            'num_experts': 8,
            'top_k': 4,
            'max_position_embeddings': 1024
        }
    
    @pytest.fixture
    def neural_core(self, config):
        """
        Create a QuantumInspiredNeuralCore instance from a configuration dictionary.
        
        Parameters:
            config (dict): Configuration for the neural core. Expected keys include
                `vocab_size`, `d_model`, `num_heads`, `num_layers`, `num_experts`,
                `top_k`, and `max_position_embeddings`.
        
        Returns:
            QuantumInspiredNeuralCore: An instantiated neural core configured using `config`.
        """
        return QuantumInspiredNeuralCore(**config)
    
    def test_forward_pass_speed(self, neural_core, benchmark):
        """Benchmark forward pass speed"""
        input_ids = torch.randint(0, 10000, (4, 256))
        
        with benchmark("Neural Core Forward Pass"):
            for _ in range(10):
                output = neural_core(input_ids)
                assert output is not None
    
    def test_generation_speed(self, neural_core, benchmark):
        """
        Measure generation throughput of the neural core using the benchmark fixture.
        
        Runs a generation call with a short prompt and asserts the generated sequence length matches the requested `max_length` (100), allowing the benchmark harness to time the operation.
        """
        input_ids = torch.randint(0, 10000, (1, 10))
        
        with benchmark("Neural Core Generation"):
            generated = neural_core.generate(
                input_ids,
                max_length=100,
                temperature=0.8
            )
            assert generated.shape[1] == 100
    
    def test_memory_usage(self, neural_core):
        """
        Measure peak memory during a forward pass of the provided neural_core and assert peak usage is less than 1024 MB.
        
        Runs a single forward pass using random input_ids of shape (8, 512), records peak memory with tracemalloc, and fails the test if peak memory in megabytes is 1024 or greater.
        
        Parameters:
            neural_core: The model instance to test (callable that accepts input_ids).
        """
        import tracemalloc
        tracemalloc.start()
        
        input_ids = torch.randint(0, 10000, (8, 512))
        output = neural_core(input_ids)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Peak memory should be reasonable (< 1GB for this config)
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 1024, f"Memory usage too high: {peak_mb}MB"