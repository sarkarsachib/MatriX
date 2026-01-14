# Sathik AI Direction Mode

Comprehensive RAG-based query processing system with multiple response styles and FastAPI web interface.

## 🔥 Features

### Core Direction Mode
- **Multi-Source Search**: Google, Wikipedia, DuckDuckGo, ArXiv, NewsAPI
- **Intelligent Fact Extraction**: NLP-powered information parsing
- **Fact Validation**: Confidence scoring and source reliability
- **Citation Tracking**: Complete source attribution
- **Knowledge Caching**: SQLite-based persistent storage
- **Real-time Processing**: Async search and retrieval

### Response Style Sub-Modes
1. **🍬 Sugarcotted**: Sweet, positive, encouraging responses
2. **🔥 Unhinged**: Raw, honest, uncensored responses  
3. **☠️ Reaper**: Dark, morbid, existential responses
4. **👹 666**: Chaotic, sarcastic, demonic responses
5. **💬 Normal**: Standard, unmodified responses

### System Integration
- **Dual Mode Operation**: Direction Mode (RAG) + Trained Mode (Neural)
- **Terminal Interface**: Enhanced CLI with mode switching
- **FastAPI Web UI**: Modern React frontend with real-time styling
- **REST API**: Complete API for integration
- **Performance Metrics**: Real-time statistics and monitoring

## 🏗️ Architecture

```
Sathik AI Direction Mode
├── Core Pipeline
│   ├── Query Analyzer      # Detect query type and requirements
│   ├── Search Engine       # Multi-source parallel search
│   ├── Info Extractor     # Parse and extract facts
│   ├── Fact Checker       # Validate and score facts
│   ├── Answer Generator   # Synthesize responses
│   └── Knowledge Store    # SQLite caching layer
│
├── Response Styling
│   ├── Submode Styles     # Style processing engine
│   ├── Sugarcotted       # Sweet positive styling
│   ├── Unhinged          # Raw honest styling
│   ├── Reaper            # Dark existential styling
│   └── Hexagon           # Chaotic demonic styling
│
├── Integration Layer
│   ├── Direction Controller # Main orchestrator
│   ├── Mode Router        # Trained/Direction routing
│   └── Response Styler    # Style application
│
└── Interfaces
    ├── Terminal Interface # Enhanced CLI
    ├── FastAPI Backend   # REST API
    └── React Frontend    # Web UI
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd sathik-ai

# Install Python dependencies
pip install -r requirements_direction_mode.txt

# Install frontend dependencies
cd web_ui && npm install
```

### 2. Configuration (Optional)

Edit `config_direction_mode.py`:

```python
DIRECTION_MODE_CONFIG = {
    'google_api_key': 'your-google-api-key',
    'google_cse_id': 'your-google-cse-id',
    'news_api_key': 'your-news-api-key',
    # ... other settings
}
```

### 3. Run the System

```bash
# Terminal interface
python main.py --mode terminal

# FastAPI server
cd api && uvicorn main:app --reload

# Development (both)
python main.py --mode terminal  # Terminal 1
cd web_ui && npm run dev        # Terminal 2
```

### 4. Access Interfaces

- **Terminal**: Run `python main.py --mode terminal`
- **Web UI**: Open `http://localhost:3000`
- **API Docs**: Visit `http://localhost:8000/docs`

## 📖 Usage Guide

### Terminal Interface

```bash
# Start terminal interface
python main.py --mode terminal

# Available commands:
mode direction         # Switch to Direction Mode
mode trained          # Switch to Trained Mode
submode sugarcotted  # Change response style
status               # Show system status
clear_cache         # Clear knowledge cache
stats               # Show statistics
help                # Show all commands

# Ask questions:
What is artificial intelligence?
Explain quantum computing
Latest news about space exploration
```

### Web Interface

1. **Select Mode**: Choose between Direction Mode (🔍) or Trained Mode (🧠)
2. **Choose Style**: Select sub-mode (Normal, Sugarcotted, Unhinged, Reaper, 666)
3. **Format**: Choose answer format (Comprehensive, Summary, Bullet Points)
4. **Submit**: Enter query and receive styled response
5. **Review**: Check citations, confidence, and sources

### API Usage

```python
import requests

# Submit query
response = requests.post('http://localhost:8000/query', json={
    'query': 'What is machine learning?',
    'mode': 'direction',
    'submode': 'sugarcotted',
    'format_type': 'comprehensive'
})

print(response.json())
```

## 🔧 API Reference

### Core Endpoints

#### POST /query
Submit a query for processing

**Request:**
```json
{
  "query": "What is artificial intelligence?",
  "mode": "direction",
  "submode": "sugarcotted", 
  "format_type": "comprehensive",
  "user_id": "user123"
}
```

**Response:**
```json
{
  "query": "What is artificial intelligence?",
  "answer": "💖 Hi sweetie! Artificial intelligence is...",
  "confidence": 0.85,
  "sources_used": 5,
  "citations": [...],
  "processing_time": 2.3,
  "status": "success"
}
```

#### GET /modes
Get available modes, sub-modes, and formats

**Response:**
```json
{
  "modes": [...],
  "submodes": {...},
  "formats": {...}
}
```

#### GET /status
Get system status and metrics

#### POST /clear-cache
Clear old cache entries

### Query Modes

- **direction**: RAG-based search and retrieval
- **trained**: Neural network inference

### Sub-modes

- **normal**: Standard response
- **sugarcotted**: Sweet and positive
- **unhinged**: Raw and honest  
- **reaper**: Dark and existential
- **666**: Chaotic and demonic

### Answer Formats

- **comprehensive**: Detailed answer with citations
- **summary**: Brief concise answer
- **bullet_points**: Organized bullet points

## 🎨 Response Styles

### Sugarcotted (🍬)
Sweet, positive, and encouraging responses
- Warm greetings and closings
- Positive word replacements
- Optimistic framing
- Gentle emojis and encouragement

### Unhinged (🔥) 
Raw, honest, and uncensored responses
- Removes excessive politeness
- Casual slang and expressions
- Direct honest language
- Authentic emotional expression

### Reaper (☠️)
Dark, morbid, and existential responses
- Mortality-focused language
- Existential themes
- Dark metaphors and imagery
- Emphasis on temporality

### 666 (👹)
Chaotic, sarcastic, and demonic responses
- Meta-commentary and irony
- Sarcastic framing
- Chaotic metaphors
- Reality-bending language

## 📊 Performance

### Metrics Tracked
- Total queries processed
- Success/failure rates
- Cache hit rates
- Average confidence scores
- Response times
- Source reliability

### Optimization Features
- Intelligent caching (80% similarity threshold)
- Parallel search execution
- Source reliability scoring
- Fact validation and deduplication
- Async processing throughout

## 🛠️ Configuration

### Search APIs
```python
DIRECTION_MODE_CONFIG = {
    'google_api_key': 'your-key',
    'google_cse_id': 'your-cse-id', 
    'news_api_key': 'your-key',
    # DuckDuckGo and Wikipedia don't require keys
}
```

### Styling Options
```python
SUBMODE_CONFIG = {
    'sugarcotted': {'enabled': True, 'intensity': 0.8},
    'unhinged': {'enabled': True, 'intensity': 0.7},
    'reaper': {'enabled': True, 'intensity': 0.6},
    'hexagon': {'enabled': True, 'intensity': 0.8}
}
```

### Performance Settings
```python
DIRECTION_MODE_CONFIG = {
    'max_search_results': 10,
    'search_timeout_seconds': 30,
    'min_confidence_threshold': 0.3,
    'cache_similarity_threshold': 0.8,
    'parallel_search': True
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements_direction_mode.txt
   ```

2. **API Key Errors**
   - Check `config_direction_mode.py`
   - Verify environment variables
   - Ensure keys have proper permissions

3. **Port Conflicts**
   ```bash
   # Change ports in config
   FASTAPI_CONFIG['port'] = 8001
   ```

4. **CORS Errors**
   ```python
   # Update security settings
   ALLOWED_ORIGINS = ['http://localhost:3000']
   ```

### Debug Commands

```bash
# Test imports
python -c "from sathik_ai.direction_mode import *; print('OK')"

# Run with debug logging
python main.py --mode terminal --debug

# Check system status
curl http://localhost:8000/status
```

### Log Files
- Backend logs: `direction_mode.log`
- Sathik logs: `sathik_ai.log`
- Browser console: Web UI errors

## 🔮 Advanced Usage

### Custom Styling
Create custom sub-modes by extending the style processors:

```python
class CustomProcessor:
    def process(self, response: str) -> str:
        # Your custom processing logic
        return processed_response

# Register with ResponseStyler
response_styler.style_processors['custom'] = CustomProcessor()
```

### Knowledge Base Queries
```python
# Search existing knowledge
results = await direction_mode.search_knowledge_base("machine learning")

# Get statistics  
stats = direction_mode.knowledge_store.get_knowledge_base_stats()

# Clear old cache
direction_mode.clear_cache(older_than_days=7)
```

### Performance Monitoring
```python
# Get real-time metrics
metrics = direction_mode.get_performance_metrics()

# System status
status = direction_mode.get_system_status()
```

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements_dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black sathik_ai/
flake8 sathik_ai/

# Type checking
mypy sathik_ai/
```

## 📄 License

This project is licensed under the CC0 1.0 Universal License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with FastAPI, React, and TypeScript
- Powered by multiple search APIs and NLP
- Inspired by modern RAG architectures
- Designed for extensibility and performance

---

**Sathik AI Direction Mode** - Where Search Meets Style 🔥