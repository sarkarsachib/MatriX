"""
End-to-End (E2E) tests for MatriX
Tests complete user workflows and system behavior
"""
import pytest
import torch
import asyncio
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from main import SathikAI
from tests.fixtures import SAMPLE_QUERIES, SAFE_CONTENT, UNSAFE_CONTENT


class TestTrainedModeWorkflow:
    """Test complete trained mode workflow"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
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
                'ustm_capacity': 10,
                'awm_capacity': 20,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': True,
                'enable_truth_comparator': True,
                'safety_threshold': 0.7,
                'default_output_mode': 'text',
                'max_generation_length': 128,
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
        return SathikAI(config_path=temp_config)
    
    def test_single_query_workflow(self, sathik_ai):
        """Test single query through trained mode"""
        query = SAMPLE_QUERIES[0]
        result = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            output_mode="text",
            mode="trained",
            submode="normal",
            format_type="comprehensive"
        )
        
        assert result['status'] == 'success'
        assert 'response' in result
        assert 'neural_analysis' in result
        assert 'safety_analysis' in result
        assert 'truth_analysis' in result
        assert result['response'] is not None
    
    def test_multiple_queries_workflow(self, sathik_ai):
        """Test multiple queries in sequence"""
        user_id = "test_user"
        
        for i, query in enumerate(SAMPLE_QUERIES[:3]):
            result = sathik_ai.process_query(
                query=query,
                user_id=user_id,
                output_mode="text",
                mode="trained",
                submode="normal",
                format_type="comprehensive"
            )
            
            assert result['status'] == 'success'
            assert 'response' in result
        
        # Check memory accumulated
        assert len(sathik_ai.memory_system.ustm.memory) >= 3
    
    def test_memory_learning_workflow(self, sathik_ai):
        """Test memory learns from user interactions"""
        user_id = "learning_user"
        
        # First query about topic A
        result1 = sathik_ai.process_query(
            query="What is neural networks?",
            user_id=user_id,
            mode="trained",
            submode="normal"
        )
        
        # Second related query
        result2 = sathik_ai.process_query(
            query="How do they learn?",
            user_id=user_id,
            mode="trained",
            submode="normal"
        )
        
        # User profile should be updated
        user_profile = sathik_ai.memory_system.ltkb.get_concept(f"user_profile_{user_id}")
        assert user_profile is not None
        assert user_profile['content']['interaction_count'] >= 2
    
    def test_safety_filtering_workflow(self, sathik_ai):
        """Test safety filtering in workflow"""
        # Safe query
        safe_result = sathik_ai.process_query(
            query=SAFE_CONTENT[0],
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        # Unsafe query
        unsafe_result = sathik_ai.process_query(
            query=UNSAFE_CONTENT[0],
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        assert safe_result['status'] == 'success'
        assert unsafe_result['status'] == 'rejected'
        assert unsafe_result['safety_analysis']['is_safe'] == False
    
    def test_output_mode_switching_workflow(self, sathik_ai):
        """Test switching output modes during workflow"""
        query = "Explain AI"
        
        # Text output
        result1 = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            output_mode="text",
            mode="trained"
        )
        
        # Code output
        result2 = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            output_mode="code",
            mode="trained"
        )
        
        assert 'response' in result1
        assert 'response' in result2
        assert isinstance(result1['response'], str)
        assert 'def ' in result2['response'] or 'class ' in result2['response']


class TestDirectionModeWorkflow:
    """Test complete direction mode workflow"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 10,
                'awm_capacity': 20,
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
        return SathikAI(config_path=temp_config)
    
    @pytest.mark.asyncio
    async def test_direction_mode_query_workflow(self, sathik_ai):
        """Test direction mode query workflow"""
        result = await sathik_ai.process_query(
            query="What is artificial intelligence?",
            user_id="test_user",
            mode="direction",
            submode="normal",
            format_type="comprehensive"
        )
        
        assert result['status'] == 'success'
        assert 'answer' in result
        assert 'sources' in result or 'search_results' in result
    
    @pytest.mark.asyncio
    async def test_all_submodes_workflow(self, sathik_ai):
        """Test all submodes in direction mode"""
        query = "Explain quantum computing"
        submodes = ['normal', 'sugarcotted', 'unhinged', 'reaper', '666']
        
        for submode in submodes:
            result = await sathik_ai.process_query(
                query=query,
                user_id="test_user",
                mode="direction",
                submode=submode,
                format_type="summary"
            )
            
            assert result['status'] == 'success'
            assert 'answer' in result
    
    @pytest.mark.asyncio
    async def test_format_types_workflow(self, sathik_ai):
        """Test different format types in direction mode"""
        query = "What is machine learning?"
        formats = ['comprehensive', 'summary', 'bullet_points']
        
        for format_type in formats:
            result = await sathik_ai.process_query(
                query=query,
                user_id="test_user",
                mode="direction",
                submode="normal",
                format_type=format_type
            )
            
            assert result['status'] == 'success'
            assert 'answer' in result


class TestModeSwitching:
    """Test switching between trained and direction modes"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 10,
                'awm_capacity': 20,
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
        return SathikAI(config_path=temp_config)
    
    @pytest.mark.asyncio
    async def test_trained_to_direction_switch(self, sathik_ai):
        """Test switching from trained to direction mode"""
        query = "What is AI?"
        
        # Trained mode
        trained_result = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        # Direction mode
        direction_result = await sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="direction",
            submode="normal"
        )
        
        assert trained_result['status'] == 'success'
        assert direction_result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_direction_to_trained_switch(self, sathik_ai):
        """Test switching from direction to trained mode"""
        query = "Explain AI"
        
        # Direction mode
        direction_result = await sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="direction",
            submode="normal"
        )
        
        # Trained mode
        trained_result = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        assert direction_result['status'] == 'success'
        assert trained_result['status'] == 'success'
    
    @pytest.mark.asyncio
    async def test_submode_switching(self, sathik_ai):
        """Test switching between submodes"""
        query = "Tell me about AI"
        submodes = ['normal', 'sugarcotted', 'unhinged', 'reaper', '666']
        
        results = []
        for submode in submodes:
            if submode == 'normal':
                result = sathik_ai.process_query(
                    query=query,
                    user_id="test_user",
                    mode="trained",
                    submode=submode
                )
            else:
                result = await sathik_ai.process_query(
                    query=query,
                    user_id="test_user",
                    mode="direction",
                    submode=submode
                )
            results.append(result)
        
        # All switches should work
        for result in results:
            assert result['status'] == 'success'


class TestMemoryConsolidation:
    """Test memory consolidation over time"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
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
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        return SathikAI(config_path=temp_config)
    
    def test_ustm_to_awm_transfer(self, sathik_ai):
        """Test USTM transfers to AWM"""
        # Fill USTM beyond capacity
        for i in range(10):
            sathik_ai.process_query(
                query=f"Query {i}",
                user_id="test_user",
                mode="trained",
                submode="normal"
            )
        
        # USTM should be at capacity
        assert len(sathik_ai.memory_system.ustm.memory) <= 10
        
        # AWM should have processed queries
        assert sathik_ai.memory_system.awm is not None
    
    def test_awm_to_ltkb_consolidation(self, sathik_ai):
        """Test AWM consolidates to LTKB"""
        user_id = "consolidation_user"
        
        # Multiple queries
        for i in range(5):
            sathik_ai.process_query(
                query=f"Important concept {i}",
                user_id=user_id,
                mode="trained",
                submode="normal"
            )
        
        # Trigger memory management
        sathik_ai.memory_system.manage_memory_lifecycle()
        
        # Check LTKB has data
        user_profile = sathik_ai.memory_system.ltkb.get_concept(f"user_profile_{user_id}")
        assert user_profile is not None
        assert user_profile['content']['interaction_count'] >= 5
    
    def test_long_term_memory_retention(self, sathik_ai):
        """Test long-term memory retention"""
        user_id = "retention_user"
        important_queries = [
            "User's name is John",
            "User prefers Python",
            "User works in AI field"
        ]
        
        for query in important_queries:
            sathik_ai.process_query(
                query=query,
                user_id=user_id,
                mode="trained",
                submode="normal"
            )
        
        # Check LTKB retains information
        user_profile = sathik_ai.memory_system.ltkb.get_concept(f"user_profile_{user_id}")
        assert user_profile is not None


class TestErrorHandling:
    """Test error handling in workflows"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
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
        return SathikAI(config_path=temp_config)
    
    def test_empty_query_handling(self, sathik_ai):
        """Test handling of empty queries"""
        result = sathik_ai.process_query(
            query="",
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        # Should handle gracefully
        assert 'status' in result
    
    def test_invalid_mode_handling(self, sathik_ai):
        """Test handling of invalid modes"""
        result = sathik_ai.process_query(
            query="Test query",
            user_id="test_user",
            mode="invalid_mode",
            submode="normal"
        )
        
        # Should handle gracefully
        assert 'status' in result
        assert 'error' in result
    
    def test_invalid_submode_handling(self, sathik_ai):
        """Test handling of invalid submodes"""
        result = sathik_ai.process_query(
            query="Test query",
            user_id="test_user",
            mode="trained",
            submode="invalid_submode"
        )
        
        # Should fall back to normal or handle gracefully
        assert 'status' in result
    
    def test_uninitialized_system(self):
        """Test behavior when system not initialized"""
        from main import SathikAI
        
        # Create uninitialized system
        uninitialized = SathikAI.__new__(SathikAI)
        uninitialized.is_initialized = False
        
        result = uninitialized.process_query(
            query="Test query",
            user_id="test_user",
            mode="trained"
        )
        
        # Should return error
        assert 'error' in result


class TestMultiUserWorkflows:
    """Test workflows with multiple users"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 10,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        return SathikAI(config_path=temp_config)
    
    def test_concurrent_user_queries(self, sathik_ai):
        """Test queries from multiple users"""
        users = ["user1", "user2", "user3"]
        query = "What is AI?"
        
        for user_id in users:
            result = sathik_ai.process_query(
                query=query,
                user_id=user_id,
                mode="trained",
                submode="normal"
            )
            
            assert result['status'] == 'success'
        
        # Each user should have their own profile
        for user_id in users:
            profile = sathik_ai.memory_system.ltkb.get_concept(f"user_profile_{user_id}")
            assert profile is not None
    
    def test_user_isolation(self, sathik_ai):
        """Test that user data is isolated"""
        user1 = "isolated_user1"
        user2 = "isolated_user2"
        
        # User 1 queries
        sathik_ai.process_query(
            query="My name is Alice",
            user_id=user1,
            mode="trained",
            submode="normal"
        )
        
        # User 2 queries
        sathik_ai.process_query(
            query="My name is Bob",
            user_id=user2,
            mode="trained",
            submode="normal"
        )
        
        # Profiles should be separate
        profile1 = sathik_ai.memory_system.ltkb.get_concept(f"user_profile_{user1}")
        profile2 = sathik_ai.memory_system.ltkb.get_concept(f"user_profile_{user2}")
        
        assert profile1 is not None
        assert profile2 is not None
        
        # Check data is not mixed
        if 'name' in profile1['content']:
            assert 'Alice' in profile1['content']['name']
            assert 'Bob' not in profile1['content']['name']
        
        if 'name' in profile2['content']:
            assert 'Bob' in profile2['content']['name']
            assert 'Alice' not in profile2['content']['name']


class TestSystemPrompts:
    """Test system and mode prompts integration"""
    
    @pytest.fixture
    def temp_config(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'num_experts': 4,
                'top_k': 2,
                'max_position_embeddings': 512,
                'ustm_capacity': 10,
                'awm_capacity': 10,
                'ltkb_path': '/tmp/test_ltkb.json',
                'enable_content_filter': False,
                'enable_truth_comparator': False,
                'debug_mode': True
            }
            json.dump(config, f)
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sathik_ai(self, temp_config):
        return SathikAI(config_path=temp_config)
    
    def test_system_prompt_influence(self, sathik_ai):
        """Test system prompt influences responses"""
        query = "What is your purpose?"
        
        result = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        assert result['status'] == 'success'
        assert 'response' in result
        # Response should reflect system prompt
        assert len(result['response']) > 0
    
    @pytest.mark.asyncio
    async def test_mode_prompt_differences(self, sathik_ai):
        """Test that different mode prompts produce different outputs"""
        query = "Explain AI"
        
        # Normal mode
        normal_result = sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="trained",
            submode="normal"
        )
        
        # Sugarcotted mode
        sugarcotted_result = await sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="direction",
            submode="sugarcotted"
        )
        
        # Unhinged mode
        unhinged_result = await sathik_ai.process_query(
            query=query,
            user_id="test_user",
            mode="direction",
            submode="unhinged"
        )
        
        # All should succeed
        assert normal_result['status'] == 'success'
        assert sugarcotted_result['status'] == 'success'
        assert unhinged_result['status'] == 'success'
        
        # Outputs should differ
        if 'response' in normal_result:
            if 'answer' in sugarcotted_result and 'answer' in unhinged_result:
                # Styles should be different
                outputs_different = True
                if 'response' in normal_result:
                    assert isinstance(normal_result['response'], str)
                    assert isinstance(sugarcotted_result['answer'], str)
                    assert isinstance(unhinged_result['answer'], str)
