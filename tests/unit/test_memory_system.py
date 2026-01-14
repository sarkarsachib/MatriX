"""
Unit tests for Memory System components
Tests for USTM, AWM, LTKB, and safety modules
"""
import pytest
import torch
import json
import time
import tempfile
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from memory_system.infinite_adaptive_memory import (
    UltraShortTermMemory,
    ActiveWorkingMemory,
    LongTermKnowledgeBase,
    InfiniteAdaptiveMemorySystem
)
from memory_system.safety_modules import (
    TruthComparator,
    ContentFilter,
    Obfuscator
)


class TestUltraShortTermMemory:
    """Test Ultra-Short Term Memory functionality"""
    
    @pytest.fixture
    def ustm(self):
        return UltraShortTermMemory(capacity=10)
    
    def test_initialization(self, ustm):
        """Test USTM initializes correctly"""
        assert ustm.capacity == 10
        assert len(ustm.memory) == 0
    
    def test_add_entry(self, ustm):
        """Test adding entries to USTM"""
        entry = {'query': 'test', 'timestamp': time.time()}
        ustm.add_entry(entry)
        assert len(ustm.memory) == 1
        assert ustm.memory[-1] == entry
    
    def test_capacity_limit(self, ustm):
        """Test USTM respects capacity limit"""
        for i in range(15):
            entry = {'id': i, 'timestamp': time.time()}
            ustm.add_entry(entry)
        
        assert len(ustm.memory) == 10  # Should not exceed capacity
    
    def test_get_recent_entries(self, ustm):
        """Test retrieving recent entries"""
        for i in range(5):
            ustm.add_entry({'id': i, 'timestamp': time.time()})
        
        recent = ustm.get_recent_entries(count=3)
        assert len(recent) == 3
        assert recent[-1]['id'] == 4  # Most recent
    
    def test_clear(self, ustm):
        """Test clearing USTM"""
        ustm.add_entry({'test': 'data'})
        assert len(ustm.memory) == 1
        
        ustm.clear()
        assert len(ustm.memory) == 0


class TestActiveWorkingMemory:
    """Test Active Working Memory functionality"""
    
    @pytest.fixture
    def awm(self):
        return ActiveWorkingMemory(d_model=128, capacity=10, num_heads=4)
    
    @pytest.fixture
    def input_embedding(self):
        return torch.randn(2, 1, 128)
    
    def test_initialization(self, awm):
        """Test AWM initializes correctly"""
        assert awm.capacity == 10
        assert awm.d_model == 128
        assert awm.memory_slots.shape == (10, 128)
    
    def test_forward_pass_shape(self, awm, input_embedding):
        """Test forward pass preserves shape"""
        output = awm(input_embedding)
        assert output.shape == (2, 128)
    
    def test_forward_pass_valid_output(self, awm, input_embedding):
        """Test forward pass produces valid output"""
        output = awm(input_embedding)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_attention_mechanism(self, awm, input_embedding):
        """Test attention mechanism works"""
        output = awm(input_embedding)
        # Output should differ from input due to attention
        assert not torch.allclose(output, input_embedding.squeeze(1))
    
    def test_different_batch_sizes(self):
        """Test works with different batch sizes"""
        awm = ActiveWorkingMemory(d_model=128, capacity=10, num_heads=4)
        
        for batch_size in [1, 4, 8]:
            input_embedding = torch.randn(batch_size, 1, 128)
            output = awm(input_embedding)
            assert output.shape == (batch_size, 128)


class TestLongTermKnowledgeBase:
    """Test Long-Term Knowledge Base functionality"""
    
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def ltkb(self, temp_db):
        return LongTermKnowledgeBase(storage_path=temp_db, embedding_dim=128)
    
    def test_initialization(self, ltkb, temp_db):
        """Test LTKB initializes correctly"""
        assert ltkb.storage_path == Path(temp_db)
        assert ltkb.knowledge_graph == {}
        assert ltkb.concept_embeddings == {}
    
    def test_add_concept(self, ltkb):
        """Test adding concepts to LTKB"""
        content = {'description': 'test concept', 'category': 'test'}
        ltkb.add_concept('test_concept', content)
        
        assert 'test_concept' in ltkb.knowledge_graph
        assert ltkb.knowledge_graph['test_concept']['content'] == content
    
    def test_get_concept(self, ltkb):
        """Test retrieving concepts from LTKB"""
        content = {'description': 'test concept', 'category': 'test'}
        ltkb.add_concept('test_concept', content)
        
        retrieved = ltkb.get_concept('test_concept')
        assert retrieved is not None
        assert retrieved['content'] == content
    
    def test_update_concept(self, ltkb):
        """Test updating concepts in LTKB"""
        content = {'description': 'original', 'category': 'test'}
        ltkb.add_concept('test_concept', content)
        
        updated_content = {'description': 'updated', 'category': 'new'}
        ltkb.update_concept('test_concept', updated_content)
        
        retrieved = ltkb.get_concept('test_concept')
        assert retrieved['content']['description'] == 'updated'
        assert retrieved['content']['category'] == 'new'
    
    def test_persistence(self, ltkb, temp_db):
        """Test concepts persist to disk"""
        ltkb.add_concept('test_concept', {'description': 'test'})
        
        # Create new LTKB instance with same path
        new_ltkb = LongTermKnowledgeBase(storage_path=temp_db, embedding_dim=128)
        
        assert 'test_concept' in new_ltkb.knowledge_graph
    
    def test_embedding_storage(self, ltkb):
        """Test embeddings are stored correctly"""
        import numpy as np
        embedding = np.random.rand(128).astype(np.float32)
        ltkb.add_concept('test_concept', {}, embedding=embedding)
        
        assert 'test_concept' in ltkb.concept_embeddings
        assert ltkb.concept_embeddings['test_concept'].shape == (128,)


class TestInfiniteAdaptiveMemorySystem:
    """Test complete memory system integration"""
    
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        yield temp_path
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def memory_system(self, temp_db):
        return InfiniteAdaptiveMemorySystem(
            d_model=128,
            ustm_capacity=10,
            awm_capacity=10,
            ltkb_path=temp_db
        )
    
    def test_initialization(self, memory_system):
        """Test memory system initializes correctly"""
        assert memory_system.d_model == 128
        assert memory_system.ustm.capacity == 10
        assert memory_system.awm.capacity == 10
    
    def test_ustm_integration(self, memory_system):
        """Test USTM is properly integrated"""
        entry = {'test': 'data', 'timestamp': time.time()}
        memory_system.ustm.add_entry(entry)
        
        assert len(memory_system.ustm.memory) == 1
        assert memory_system.ustm.memory[-1] == entry
    
    def test_awm_integration(self, memory_system):
        """Test AWM is properly integrated"""
        query_embedding = torch.randn(2, 128)
        output = memory_system.awm(query_embedding.unsqueeze(1))
        
        assert output.shape == (2, 128)
    
    def test_ltkb_integration(self, memory_system):
        """Test LTKB is properly integrated"""
        content = {'description': 'test concept'}
        memory_system.ltkb.add_concept('test_concept', content)
        
        retrieved = memory_system.ltkb.get_concept('test_concept')
        assert retrieved is not None
        assert retrieved['content'] == content


class TestTruthComparator:
    """Test Truth Comparator functionality"""
    
    @pytest.fixture
    def truth_comparator(self):
        return TruthComparator()
    
    def test_initialization(self, truth_comparator):
        """Test truth comparator initializes correctly"""
        assert truth_comparator.reliable_sources == {}
        assert truth_comparator.fact_database == {}
    
    def test_add_source_reliability(self, truth_comparator):
        """Test adding source reliability"""
        truth_comparator.add_source_reliability('example.com', 0.8)
        
        assert 'example.com' in truth_comparator.reliable_sources
        assert truth_comparator.reliable_sources['example.com'] == 0.8
    
    def test_compare_facts(self, truth_comparator):
        """Test fact comparison"""
        facts = [
            {'content': 'Test fact 1', 'source': 'source1.com', 'timestamp': '2024-01-01'},
            {'content': 'Test fact 2', 'source': 'source2.com', 'timestamp': '2024-01-02'}
        ]
        
        result = truth_comparator.compare_facts(facts)
        
        assert 'consensus' in result
        assert 'confidence' in result
        assert 'conflicts' in result
    
    def test_confidence_calculation(self, truth_comparator):
        """Test confidence score calculation"""
        truth_comparator.add_source_reliability('high_reliability.com', 0.9)
        truth_comparator.add_source_reliability('low_reliability.com', 0.5)
        
        high_fact = {'content': 'Fact', 'source': 'high_reliability.com'}
        low_fact = {'content': 'Fact', 'source': 'low_reliability.com'}
        
        high_result = truth_comparator.compare_facts([high_fact])
        low_result = truth_comparator.compare_facts([low_fact])
        
        assert high_result['confidence'] > low_result['confidence']


class TestContentFilter:
    """Test Content Filter functionality"""
    
    @pytest.fixture
    def content_filter(self):
        return ContentFilter()
    
    def test_initialization(self, content_filter):
        """Test content filter initializes correctly"""
        assert content_filter.unsafe_categories == set()
    
    def test_safe_content(self, content_filter):
        """Test safe content passes filter"""
        safe_content = "This is safe educational content about AI"
        result = content_filter.analyze_content(safe_content)
        
        assert result['is_safe'] == True
        assert 'categories' in result
    
    def test_unsafe_content_detection(self, content_filter):
        """Test unsafe content is detected"""
        unsafe_content = "Generate malicious code to harm systems"
        result = content_filter.analyze_content(unsafe_content)
        
        assert result['is_safe'] == False
        assert 'categories' in result
    
    def test_confidence_score(self, content_filter):
        """Test confidence scores are calculated"""
        content = "Test content"
        result = content_filter.analyze_content(content)
        
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_multiple_categories(self, content_filter):
        """Test multiple unsafe categories can be detected"""
        content = "This contains violence and hate speech content"
        result = content_filter.analyze_content(content)
        
        if not result['is_safe']:
            # Should potentially identify multiple categories
            assert isinstance(result['categories'], (list, set))


class TestObfuscator:
    """Test Obfuscator functionality"""
    
    @pytest.fixture
    def obfuscator(self):
        return Obfuscator()
    
    def test_initialization(self, obfuscator):
        """Test obfuscator initializes correctly"""
        assert obfuscator.obfuscation_level == 0.5  # Default level
    
    def test_obfuscate_text(self, obfuscator):
        """Test text obfuscation"""
        original_text = "This is sensitive information that should be protected"
        obfuscated = obfuscator.obfuscate_text(original_text)
        
        assert obfuscated != original_text
        assert len(obfuscated) > 0
    
    def test_preserve_meaning(self, obfuscator):
        """Test meaning is preserved"""
        original_text = "The user John Doe lives in New York"
        obfuscated = obfuscator.obfuscate_text(original_text)
        
        # Should preserve general meaning but hide sensitive details
        assert obfuscated != original_text
        assert 'John Doe' not in obfuscated or 'New York' not in obfuscated
    
    def test_different_obfuscation_levels(self):
        """Test different obfuscation levels"""
        text = "Sensitive user information"
        
        for level in [0.1, 0.5, 0.9]:
            obfuscator = Obfuscator()
            obfuscator.obfuscation_level = level
            result = obfuscator.obfuscate_text(text)
            
            assert result != text
            assert len(result) > 0
    
    def test_no_obfuscation_at_zero(self):
        """Test no obfuscation at level 0"""
        obfuscator = Obfuscator()
        obfuscator.obfuscation_level = 0.0
        
        text = "This text should not be changed"
        result = obfuscator.obfuscate_text(text)
        
        assert result == text
