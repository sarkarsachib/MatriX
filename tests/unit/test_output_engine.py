"""
Unit tests for Output Engine components
Tests for all output modalities
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from output_engine.output_system import (
    OutputMode,
    TextMode,
    CodeMode,
    AudioMode,
    CommandMode,
    OutputEngine
)


class TestTextMode:
    """Test Text Mode functionality"""
    
    @pytest.fixture
    def text_mode(self):
        return TextMode()
    
    def test_initialization(self, text_mode):
        """Test text mode initializes correctly"""
        assert text_mode.response_templates is not None
        assert 'default' in text_mode.response_templates
    
    def test_generate_output_default(self, text_mode):
        """Test generating default output"""
        input_data = {'content': 'Test response content', 'type': 'default'}
        output = text_mode.generate_output(input_data)
        
        assert 'Test response content' in output
        assert isinstance(output, str)
    
    def test_generate_output_explanation(self, text_mode):
        """Test generating explanation output"""
        input_data = {'content': 'AI explanation', 'type': 'explanation'}
        output = text_mode.generate_output(input_data)
        
        assert 'Based on information provided' in output
        assert 'AI explanation' in output
    
    def test_generate_output_answer(self, text_mode):
        """Test generating answer output"""
        input_data = {'content': '42', 'type': 'answer'}
        output = text_mode.generate_output(input_data)
        
        assert 'The answer is' in output
        assert '42' in output
    
    def test_generate_output_summary(self, text_mode):
        """Test generating summary output"""
        input_data = {'content': 'Summary content', 'type': 'summary'}
        output = text_mode.generate_output(input_data)
        
        assert 'Summary:' in output
        assert 'Summary content' in output
    
    def test_generate_output_empty_content(self, text_mode):
        """Test handling empty content"""
        input_data = {'content': '', 'type': 'default'}
        output = text_mode.generate_output(input_data)
        
        assert isinstance(output, str)
        assert len(output) > 0


class TestCodeMode:
    """Test Code Mode functionality"""
    
    @pytest.fixture
    def code_mode(self):
        return CodeMode()
    
    def test_initialization(self, code_mode):
        """Test code mode initializes correctly"""
        assert code_mode.language_templates is not None
        assert 'python' in code_mode.language_templates
        assert 'javascript' in code_mode.language_templates
        assert 'html' in code_mode.language_templates
    
    def test_python_function_generation(self, code_mode):
        """Test generating Python function"""
        input_data = {
            'language': 'python',
            'code_type': 'function',
            'name': 'add',
            'params': 'a, b',
            'docstring': 'Add two numbers',
            'body': 'return a + b'
        }
        output = code_mode.generate_output(input_data)
        
        assert 'def add(a, b):' in output
        assert 'Add two numbers' in output
        assert 'return a + b' in output
    
    def test_python_class_generation(self, code_mode):
        """Test generating Python class"""
        input_data = {
            'language': 'python',
            'code_type': 'class',
            'name': 'Calculator',
            'docstring': 'A simple calculator class',
            'body': 'pass'
        }
        output = code_mode.generate_output(input_data)
        
        assert 'class Calculator:' in output
        assert 'A simple calculator class' in output
    
    def test_javascript_function_generation(self, code_mode):
        """Test generating JavaScript function"""
        input_data = {
            'language': 'javascript',
            'code_type': 'function',
            'name': 'multiply',
            'params': 'x, y',
            'docstring': 'Multiply two numbers',
            'body': 'return x * y'
        }
        output = code_mode.generate_output(input_data)
        
        assert 'function multiply(x, y)' in output
        assert 'Multiply two numbers' in output
    
    def test_html_generation(self, code_mode):
        """Test generating HTML"""
        input_data = {
            'language': 'html',
            'code_type': 'page',
            'title': 'Test Page',
            'body': '<h1>Hello</h1>'
        }
        output = code_mode.generate_output(input_data)
        
        assert '<!DOCTYPE html>' in output
        assert '<title>Test Page</title>' in output
        assert '<h1>Hello</h1>' in output
    
    def test_unsupported_language(self, code_mode):
        """Test handling unsupported language"""
        input_data = {
            'language': 'unsupported',
            'code_type': 'function',
            'content': 'test'
        }
        output = code_mode.generate_output(input_data)
        
        assert 'Unsupported language' in output
    
    def test_unsupported_code_type(self, code_mode):
        """Test handling unsupported code type"""
        input_data = {
            'language': 'python',
            'code_type': 'unsupported',
            'content': 'test'
        }
        output = code_mode.generate_output(input_data)
        
        assert 'Unsupported code type' in output


class TestAudioMode:
    """Test Audio Mode functionality"""
    
    @pytest.fixture
    def audio_mode(self):
        return AudioMode()
    
    def test_initialization(self, audio_mode):
        """Test audio mode initializes correctly"""
        assert audio_mode.voice_settings is not None
        assert 'speed' in audio_mode.voice_settings
        assert 'pitch' in audio_mode.voice_settings
        assert 'volume' in audio_mode.voice_settings
    
    def test_generate_output_with_default_settings(self, audio_mode):
        """Test generating output with default settings"""
        input_data = {
            'content': 'Hello world',
            'output_file': 'test_audio'
        }
        output = audio_mode.generate_output(input_data)
        
        assert 'Audio generated' in output or 'TTS' in output
        assert isinstance(output, str)
    
    def test_custom_output_file(self, audio_mode):
        """Test custom output file name"""
        input_data = {
            'content': 'Test',
            'output_file': 'custom_audio'
        }
        output = audio_mode.generate_output(input_data)
        
        assert 'custom_audio' in output
    
    def test_voice_settings_preserved(self, audio_mode):
        """Test voice settings are preserved"""
        audio_mode.voice_settings = {
            'speed': 1.5,
            'pitch': 1.2,
            'volume': 0.8
        }
        
        assert audio_mode.voice_settings['speed'] == 1.5
        assert audio_mode.voice_settings['pitch'] == 1.2
        assert audio_mode.voice_settings['volume'] == 0.8


class TestCommandMode:
    """Test Command Mode functionality"""
    
    @pytest.fixture
    def command_mode(self):
        return CommandMode()
    
    def test_initialization(self, command_mode):
        """Test command mode initializes correctly"""
        assert command_mode.command_templates is not None
    
    def test_generate_shell_command(self, command_mode):
        """Test generating shell command"""
        input_data = {
            'command_type': 'shell',
            'command': 'ls -la'
        }
        output = command_mode.generate_output(input_data)
        
        assert 'ls -la' in output
        assert isinstance(output, str)
    
    def test_generate_python_command(self, command_mode):
        """Test generating Python command"""
        input_data = {
            'command_type': 'python',
            'command': 'script.py --arg1 value1'
        }
        output = command_mode.generate_output(input_data)
        
        assert 'python' in output
        assert 'script.py' in output


class TestOutputEngine:
    """Test Output Engine integration"""
    
    @pytest.fixture
    def output_engine(self):
        return OutputEngine()
    
    def test_initialization(self, output_engine):
        """Test output engine initializes correctly"""
        assert output_engine.current_mode is not None
        assert 'text' in output_engine.modes
    
    def test_set_mode(self, output_engine):
        """Test setting output mode"""
        output_engine.set_mode('code')
        assert output_engine.current_mode.__class__.__name__ == 'CodeMode'
        
        output_engine.set_mode('text')
        assert output_engine.current_mode.__class__.__name__ == 'TextMode'
    
    def test_generate_response_text(self, output_engine):
        """Test generating text response"""
        output_engine.set_mode('text')
        input_data = {'content': 'Test content', 'type': 'response'}
        output = output_engine.generate_response(input_data, 'text')
        
        assert 'Test content' in output
        assert isinstance(output, str)
    
    def test_generate_response_code(self, output_engine):
        """Test generating code response"""
        output_engine.set_mode('code')
        input_data = {
            'language': 'python',
            'code_type': 'function',
            'name': 'test',
            'body': 'pass'
        }
        output = output_engine.generate_response(input_data, 'code')
        
        assert 'def test' in output or 'class test' in output
        assert isinstance(output, str)
    
    def test_switch_modes(self, output_engine):
        """Test switching between modes"""
        output_engine.set_mode('text')
        text_output = output_engine.generate_response({'content': 'test'}, 'text')
        
        output_engine.set_mode('code')
        code_output = output_engine.generate_response({
            'language': 'python',
            'code_type': 'function',
            'name': 'func',
            'body': 'pass'
        }, 'code')
        
        assert 'test' in text_output
        assert 'def func' in code_output
    
    def test_invalid_mode(self, output_engine):
        """Test handling invalid mode"""
        result = output_engine.set_mode('invalid_mode')
        assert result is False
    
    def test_all_available_modes(self, output_engine):
        """Test all modes are available"""
        available_modes = ['text', 'code', 'audio', 'command']
        for mode in available_modes:
            assert mode in output_engine.modes


class TestOutputModePerformance:
    """Performance tests for output engine"""
    
    @pytest.fixture
    def output_engine(self):
        return OutputEngine()
    
    def test_text_generation_throughput(self, output_engine, benchmark):
        """Benchmark text generation throughput"""
        output_engine.set_mode('text')
        
        with benchmark("Text Generation"):
            for i in range(100):
                output = output_engine.generate_response(
                    {'content': f'Test message {i}'},
                    'text'
                )
                assert len(output) > 0
    
    def test_code_generation_throughput(self, output_engine, benchmark):
        """Benchmark code generation throughput"""
        output_engine.set_mode('code')
        
        with benchmark("Code Generation"):
            for i in range(100):
                output = output_engine.generate_response({
                    'language': 'python',
                    'code_type': 'function',
                    'name': f'func_{i}',
                    'body': 'pass'
                }, 'code')
                assert 'def' in output
    
    def test_mode_switching_speed(self, output_engine, benchmark):
        """Benchmark mode switching speed"""
        with benchmark("Mode Switching"):
            for _ in range(1000):
                output_engine.set_mode('text')
                output_engine.set_mode('code')
                output_engine.set_mode('audio')
                output_engine.set_mode('command')
