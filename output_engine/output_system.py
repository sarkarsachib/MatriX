import json
import subprocess
import sys
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

class OutputMode(ABC):
    """Abstract base class for different output modes."""
    
    @abstractmethod
    def generate_output(self, input_data: Dict[str, Any]) -> str:
        pass

class TextMode(OutputMode):
    """Generates text responses."""
    
    def __init__(self):
        self.response_templates = {
            "explanation": "Based on the information provided, {content}",
            "answer": "The answer is: {content}",
            "summary": "Summary: {content}",
            "default": "{content}"
        }
    
    def generate_output(self, input_data: Dict[str, Any]) -> str:
        content = input_data.get("content", "")
        output_type = input_data.get("type", "default")
        
        template = self.response_templates.get(output_type, self.response_templates["default"])
        return template.format(content=content)

class CodeMode(OutputMode):
    """Generates code in various programming languages."""
    
    def __init__(self):
        self.language_templates = {
            "python": {
                "function": "def {name}({params}):\n    \"\"\"{docstring}\"\"\"\n    {body}",
                "class": "class {name}:\n    \"\"\"{docstring}\"\"\"\n    \n    def __init__(self):\n        {body}",
                "script": "#!/usr/bin/env python3\n\"\"\"{docstring}\"\"\"\n\n{body}"
            },
            "javascript": {
                "function": "function {name}({params}) {{\n    // {docstring}\n    {body}\n}}",
                "class": "class {name} {{\n    // {docstring}\n    constructor() {{\n        {body}\n    }}\n}}",
                "script": "// {docstring}\n\n{body}"
            },
            "html": {
                "page": "<!DOCTYPE html>\n<html>\n<head>\n    <title>{title}</title>\n</head>\n<body>\n    {body}\n</body>\n</html>"
            }
        }
    
    def generate_output(self, input_data: Dict[str, Any]) -> str:
        language = input_data.get("language", "python")
        code_type = input_data.get("code_type", "function")
        
        if language not in self.language_templates:
            return f"# Unsupported language: {language}\n{input_data.get('content', '')}"
        
        template_dict = self.language_templates[language]
        if code_type not in template_dict:
            return f"# Unsupported code type: {code_type}\n{input_data.get('content', '')}"
        
        template = template_dict[code_type]
        
        # Fill in template parameters
        params = {
            "name": input_data.get("name", "example"),
            "params": input_data.get("params", ""),
            "docstring": input_data.get("docstring", "Generated code"),
            "body": input_data.get("body", "pass"),
            "title": input_data.get("title", "Generated Page"),
            "content": input_data.get("content", "")
        }
        
        try:
            return template.format(**params)
        except KeyError as e:
            return f"# Error formatting template: {e}\n{input_data.get('content', '')}"

class AudioMode(OutputMode):
    """Generates audio using text-to-speech."""
    
    def __init__(self):
        self.voice_settings = {
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0
        }
    
    def generate_output(self, input_data: Dict[str, Any]) -> str:
        text = input_data.get("content", "")
        output_file = input_data.get("output_file", "output.wav")
        
        # Simulate TTS generation (in a real implementation, this would use actual TTS)
        tts_command = f"echo 'TTS: {text}' > {output_file}.txt"
        
        try:
            subprocess.run(tts_command, shell=True, check=True)
            return f"Audio generated and saved to {output_file}.txt (simulated)"
        except subprocess.CalledProcessError:
            return "Error: Failed to generate audio"

class CommandMode(OutputMode):
    """Generates system commands."""
    
    def __init__(self):
        self.safe_commands = {
            "list_files": "ls -la",
            "show_date": "date",
            "show_uptime": "uptime",
            "disk_usage": "df -h",
            "memory_usage": "free -h"
        }
    
    def generate_output(self, input_data: Dict[str, Any]) -> str:
        command_type = input_data.get("command_type", "")
        
        if command_type in self.safe_commands:
            return self.safe_commands[command_type]
        else:
            # For safety, don't execute arbitrary commands
            return f"# Command type '{command_type}' not recognized or not safe"

class OutputEngine:
    """Main output engine that coordinates different output modes."""
    
    def __init__(self):
        self.modes = {
            "text": TextMode(),
            "code": CodeMode(),
            "audio": AudioMode(),
            "command": CommandMode()
        }
        self.current_mode = "text"
    
    def set_mode(self, mode: str) -> bool:
        """Set the current output mode."""
        if mode in self.modes:
            self.current_mode = mode
            return True
        return False
    
    def generate_response(self, input_data: Dict[str, Any], mode: Optional[str] = None) -> str:
        """Generate response using the specified or current mode."""
        target_mode = mode or self.current_mode
        
        if target_mode not in self.modes:
            return f"Error: Unknown output mode '{target_mode}'"
        
        try:
            return self.modes[target_mode].generate_output(input_data)
        except Exception as e:
            return f"Error generating output: {str(e)}"
    
    def get_available_modes(self) -> List[str]:
        """Get list of available output modes."""
        return list(self.modes.keys())
    
    def process_multi_modal_request(self, requests: List[Dict[str, Any]]) -> Dict[str, str]:
        """Process multiple output requests in different modes."""
        responses = {}
        
        for i, request in enumerate(requests):
            mode = request.get("mode", self.current_mode)
            response = self.generate_response(request, mode)
            responses[f"response_{i}_{mode}"] = response
        
        return responses

class TerminalInterface:
    """Terminal mode interface for command-line interaction."""
    
    def __init__(self, output_engine: OutputEngine):
        self.output_engine = output_engine
        self.running = False
    
    def start(self):
        """Start the terminal interface."""
        self.running = True
        print("🔥 Sathik AI Terminal Interface")
        print("Type 'help' for commands, 'quit' to exit")
        
        while self.running:
            try:
                user_input = input("\nsathik> ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    self.running = False
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower().startswith('mode '):
                    mode = user_input[5:].strip()
                    if self.output_engine.set_mode(mode):
                        print(f"Mode set to: {mode}")
                    else:
                        print(f"Unknown mode: {mode}")
                        print(f"Available modes: {', '.join(self.output_engine.get_available_modes())}")
                else:
                    # Process user input
                    input_data = {
                        "content": user_input,
                        "type": "default"
                    }
                    response = self.output_engine.generate_response(input_data)
                    print(f"\n{response}")
                    
            except KeyboardInterrupt:
                self.running = False
                print("\nGoodbye!")
            except Exception as e:
                print(f"Error: {e}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
Available commands:
- help: Show this help message
- quit/exit: Exit the terminal
- mode <mode>: Change output mode
- Available modes: text, code, audio, command

Current mode: {}
        """.format(self.output_engine.current_mode)
        print(help_text)

# Example usage
if __name__ == "__main__":
    # Initialize output engine
    engine = OutputEngine()
    
    # Test different modes
    print("=== Testing Text Mode ===")
    text_response = engine.generate_response({
        "content": "Artificial Intelligence is transforming the world.",
        "type": "explanation"
    }, "text")
    print(text_response)
    
    print("\n=== Testing Code Mode ===")
    code_response = engine.generate_response({
        "language": "python",
        "code_type": "function",
        "name": "hello_world",
        "params": "name",
        "docstring": "Greets a person by name",
        "body": "return f'Hello, {name}!'"
    }, "code")
    print(code_response)
    
    print("\n=== Testing Command Mode ===")
    command_response = engine.generate_response({
        "command_type": "list_files"
    }, "command")
    print(command_response)
    
    # Test multi-modal
    print("\n=== Testing Multi-Modal ===")
    multi_requests = [
        {"mode": "text", "content": "This is a text response", "type": "answer"},
        {"mode": "code", "language": "javascript", "code_type": "function", "name": "test", "body": "console.log('test');"}
    ]
    multi_responses = engine.process_multi_modal_request(multi_requests)
    for key, response in multi_responses.items():
        print(f"{key}:\n{response}\n")
    
    # Uncomment to start terminal interface
    # terminal = TerminalInterface(engine)
    # terminal.start()

