"""
Integration tests for MatriX components
Tests how different modules work together
"""
import pytest
import torch
import asyncio
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import SathikAI
from tests.fixtures import MockConfig, SAMPLE_QUERIES


class TestNeuralCoreIntegration:
    """Test Neural Core integration with other components"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file for tests and yield its file path.
        
        The file contains default model, memory, safety, and generation settings used by integration tests.
        The fixture yields the path to the temporary JSON file and removes the file after the test completes.
        
        Returns:
            str: Path to the temporary JSON config file.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'batch_size': 2,
                'gradient_accumulation_steps': 2,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': True,
                'enable_truth_comparator': True,
                'safety_threshold': 0.7,
                'default_output_mode': 'text',
                'max_generation_length': 64,
                'generation_temperature': 0.8,
                'generation_top_k': 10,
                'generation_top_p': 0.9,
                'enable_sela': False,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    def test_neural_core_initialization(self, sathik_ai):
        """Test neural core initializes correctly in SathikAI"""
        assert sathik_ai.neural_core is not None
        assert sathik_ai.neural_core.d_model == 128
        assert sathik_ai.neural_core.num_experts == 4
    
    def test_neural_core_with_memory(self, sathik_ai):
        """Test neural core works with memory system"""
        query = "What is AI?"
        result = sathik_ai._process_trained_mode_query(
            query=query,
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        assert 'response' in result
        assert 'status' in result
        # Should process successfully
        assert result.get('status') != 'error'
    
    def test_neural_core_generation(self, sathik_ai):
        """Test neural core generates responses"""
        query = SAMPLE_QUERIES[0]
        result = sathik_ai._process_trained_mode_query(
            query=query,
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        assert 'response' in result
        assert len(result['response']) > 0


class TestMemorySystemIntegration:
    """Test Memory System integration with other components"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file for tests and yield its filepath.
        
        The file contains a minimal SathikAI configuration (model hyperparameters, capacities, output and safety flags, and debug mode). The fixture yields the full filesystem path to the temporary JSON file as a string; the file is removed after the fixture finishes.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'default_output_mode': 'text',
                'max_generation_length': 64,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    def test_ustm_integration(self, sathik_ai):
        """Test USTM integrates with query processing"""
        query = "Test query"
        sathik_ai._process_trained_mode_query(
            query=query,
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # USTM should have stored the query
        assert len(sathik_ai.memory_system.ustm.memory) > 0
    
    def test_awm_integration(self, sathik_ai):
        """Test AWM integrates with query processing"""
        # Multiple queries should populate AWM
        for i in range(3):
            sathik_ai._process_trained_mode_query(
                query=f"Query {i}",
                user_id=f"user_{i}",
                output_mode='text',
                mode='trained',
                submode='normal',
                format_type='comprehensive'
            )
        
        # AWM should have processed multiple inputs
        assert sathik_ai.memory_system.awm is not None
    
    def test_ltkb_integration(self, sathik_ai):
        """Test LTKB integrates with user profiles"""
        # Process query for specific user
        sathik_ai._process_trained_mode_query(
            query="Test query",
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # Check user profile was created
        user_profile = sathik_ai.memory_system.ltkb.get_concept("user_profile_test_user")
        assert user_profile is not None


class TestSafetyModulesIntegration:
    """Test Safety Modules integration"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file for integration tests and yield its filesystem path.
        
        The file contains a minimal SathikAI configuration (model hyperparameters, memory capacities, safety toggles, generation settings, and debug flag). The temporary file is removed after the caller finishes using the yielded path.
        
        Returns:
            str: Path to the temporary JSON config file.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': True,
                'enable_truth_comparator': True,
                'safety_threshold': 0.7,
                'default_output_mode': 'text',
                'max_generation_length': 64,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    def test_content_filter_integration(self, sathik_ai):
        """Test content filter integrates with query processing"""
        from tests.fixtures import UNSAFE_CONTENT
        
        # Unsafe content should be filtered
        result = sathik_ai._process_trained_mode_query(
            query=UNSAFE_CONTENT[0],
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        assert 'safety_analysis' in result
        # Should reject unsafe content
        assert result.get('status') == 'rejected'
    
    def test_truth_comparator_integration(self, sathik_ai):
        """Test truth comparator integrates with response generation"""
        query = "What is the capital of France?"
        result = sathik_ai._process_trained_mode_query(
            query=query,
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        assert 'truth_analysis' in result
        # Should have some truth analysis
        assert result['truth_analysis'] is not None


class TestOutputEngineIntegration:
    """Test Output Engine integration"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file for tests and yield its filepath.
        
        The file contains a minimal SathikAI configuration (model hyperparameters, capacities, output and safety flags, and debug mode). The fixture yields the full filesystem path to the temporary JSON file as a string; the file is removed after the fixture finishes.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'default_output_mode': 'text',
                'max_generation_length': 64,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    def test_text_mode_integration(self, sathik_ai):
        """Test text mode output integration"""
        result = sathik_ai._process_trained_mode_query(
            query="Generate text response",
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        assert 'response' in result
        assert isinstance(result['response'], str)
    
    def test_code_mode_integration(self, sathik_ai):
        """Test code mode output integration"""
        result = sathik_ai._process_trained_mode_query(
            query="Generate Python function",
            user_id="test_user",
            output_mode='code',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        assert 'response' in result
        # Should contain code
        assert 'def ' in result['response'] or 'class ' in result['response']
    
    def test_mode_switching(self, sathik_ai):
        """Test switching between output modes"""
        modes = ['text', 'code', 'audio', 'command']
        results = []
        
        for mode in modes:
            result = sathik_ai._process_trained_mode_query(
                query=f"Generate {mode} output",
                user_id="test_user",
                output_mode=mode,
                mode='trained',
                submode='normal',
                format_type='comprehensive'
            )
            results.append(result)
        
        # All modes should work
        for result in results:
            assert 'response' in result
            assert len(result['response']) > 0


class TestDirectionModeIntegration:
    """Test Direction Mode integration"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file with default test settings and yield its path; the file is removed after use.
        
        The configuration contains default model hyperparameters, memory capacities, API keys (placeholder values), and debug flags suitable for integration tests.
        
        Returns:
            str: Path to the temporary JSON config file.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'google_api_key': 'test_key',
                'google_cse_id': 'test_cse',
                'news_api_key': 'test_news_key',
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    @pytest.mark.asyncio
    async def test_direction_mode_basic_query(self, sathik_ai):
        """Test direction mode processes basic query"""
        result = await sathik_ai.process_query(
            query="What is artificial intelligence?",
            user_id="test_user",
            mode="direction",
            submode="normal",
            format_type="comprehensive"
        )
        
        assert 'answer' in result
        assert 'status' in result
    
    @pytest.mark.asyncio
    async def test_direction_mode_with_submodes(self, sathik_ai):
        """Test direction mode with all submodes"""
        submodes = ['normal', 'sugarcotted', 'unhinged', 'reaper', '666']
        
        for submode in submodes:
            result = await sathik_ai.process_query(
                query="Explain AI",
                user_id="test_user",
                mode="direction",
                submode=submode,
                format_type="summary"
            )
            
            assert 'answer' in result
            assert 'status' in result


class TestFullPipeline:
    """Test complete query processing pipeline"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file for tests and yield its filepath.
        
        The file contains a minimal SathikAI configuration (model hyperparameters, capacities, output and safety flags, and debug mode). The fixture yields the full filesystem path to the temporary JSON file as a string; the file is removed after the fixture finishes.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'default_output_mode': 'text',
                'max_generation_length': 64,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    def test_complete_trained_mode_pipeline(self, sathik_ai):
        """Test complete trained mode pipeline"""
        from tests.fixtures import SAMPLE_QUERIES
        
        for query in SAMPLE_QUERIES[:2]:
            result = sathik_ai._process_trained_mode_query(
                query=query,
                user_id="test_user",
                output_mode='text',
                mode='trained',
                submode='normal',
                format_type='comprehensive'
            )
            
            # Verify all expected fields are present
            assert 'response' in result
            assert 'query' in result
            assert 'user_id' in result
            assert 'neural_analysis' in result
            assert 'safety_analysis' in result
            assert 'truth_analysis' in result
            assert 'status' in result
            assert 'timestamp' in result
    
    def test_memory_flow_integration(self, sathik_ai):
        """Test complete memory flow: USTM -> AWM -> LTKB"""
        # First query
        result1 = sathik_ai._process_trained_mode_query(
            query="First query",
            user_id="user1",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # Second query from same user
        result2 = sathik_ai._process_trained_mode_query(
            query="Second query",
            user_id="user1",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # USTM should have both queries
        assert len(sathik_ai.memory_system.ustm.memory) >= 2
    
    def test_safety_in_pipeline(self, sathik_ai):
        """Test safety checks throughout pipeline"""
        from tests.fixtures import SAFE_CONTENT, UNSAFE_CONTENT
        
        # Safe query
        safe_result = sathik_ai._process_trained_mode_query(
            query=SAFE_CONTENT[0],
            user_id="user_safe",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # Unsafe query
        unsafe_result = sathik_ai._process_trained_mode_query(
            query=UNSAFE_CONTENT[0],
            user_id="user_unsafe",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # Safe should succeed
        assert safe_result.get('status') != 'rejected'
        
        # Unsafe should be rejected
        assert unsafe_result.get('status') == 'rejected'
        assert 'safety_analysis' in unsafe_result


class TestComponentCommunication:
    """Test communication between components"""
    
    @pytest.fixture
    def temp_config(self):
        """
        Create a temporary JSON configuration file for tests and yield its filepath.
        
        The file contains a minimal SathikAI configuration (model hyperparameters, capacities, output and safety flags, and debug mode). The fixture yields the full filesystem path to the temporary JSON file as a string; the file is removed after the fixture finishes.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            import json
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 5,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'default_output_mode': 'text',
                'max_generation_length': 64,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        """
        Create a SathikAI instance configured using the provided temporary config file.
        
        Parameters:
            temp_config (str): Path to a temporary JSON configuration file containing model and system settings.
        
        Returns:
            SathikAI: A SathikAI instance initialized with the config at `temp_config`.
        """
        return SathikAI(config_path=temp_config)
    
    def test_neural_core_to_memory(self, sathik_ai):
        """Test neural core passes data to memory"""
        query = "Test query"
        sathik_ai._process_trained_mode_query(
            query=query,
            user_id="test_user",
            output_mode='text',
            mode='trained',
            submode='normal',
            format_type='comprehensive'
        )
        
        # Memory should have received query embedding
        assert len(sathik_ai.memory_system.ustm.memory) > 0