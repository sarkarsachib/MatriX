# Sathik AI Direction Mode API Reference

Complete API documentation for the Sathik AI Direction Mode REST API.

## Base URL

```
http://localhost:8000
```

## Authentication

API authentication is optional. When enabled, include your API key in the request headers:

```http
X-API-Key: your-api-key-here
Authorization: Bearer your-api-key-here
```

## Content Type

All API requests and responses use JSON:

```http
Content-Type: application/json
Accept: application/json
```

## Rate Limiting

Default rate limits:
- **100 requests per hour** per API key
- **60 requests per minute** per IP address
- **Custom limits** can be configured

Rate limit headers included in responses:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Requests remaining
- `X-RateLimit-Reset`: Reset timestamp

## Error Handling

### Error Response Format

```json
{
  "error": "Error message",
  "details": "Additional error details",
  "timestamp": 1640995200
}
```

### HTTP Status Codes

- `200` - Success
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `429` - Too Many Requests
- `500` - Internal Server Error
- `503` - Service Unavailable

---

## Endpoints

### 1. Submit Query

Process a query using Direction Mode or Trained Mode.

#### POST `/query`

**Request Body:**

```json
{
  "query": "What is artificial intelligence?",
  "user_id": "user123",
  "mode": "direction",
  "submode": "sugarcotted",
  "format_type": "comprehensive",
  "output_mode": "text"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | The query to process |
| `user_id` | string | No | User identifier (default: "default") |
| `mode` | string | No | Processing mode (default: "direction") |
| `submode` | string | No | Response sub-mode style (default: "normal") |
| `format_type` | string | No | Answer format (default: "comprehensive") |
| `output_mode` | string | No | Output mode (for trained mode only) |

**Valid Values:**

- `mode`: `"trained"`, `"direction"`
- `submode`: `"normal"`, `"sugarcotted"`, `"unhinged"`, `"reaper"`, `"666"`
- `format_type`: `"comprehensive"`, `"summary"`, `"bullet_points"`
- `output_mode`: `"text"`, `"code"`, `"audio"`, `"command"`

**Success Response (200):**

```json
{
  "query": "What is artificial intelligence?",
  "user_id": "user123",
  "mode": "direction",
  "submode": "sugarcotted",
  "answer": "💖 Hi sweetie! Artificial intelligence is a fascinating field...",
  "confidence": 0.85,
  "sources_used": 5,
  "facts_analyzed": 12,
  "format": "comprehensive",
  "citations": [
    {
      "number": 1,
      "source": "wikipedia",
      "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
      "fact": "Artificial intelligence (AI) is intelligence demonstrated by machines...",
      "confidence": 0.9
    }
  ],
  "key_information": {
    "main_facts": [...],
    "definitions": [...],
    "people": ["Alan Turing", "John McCarthy"],
    "places": [],
    "organizations": ["MIT", "Stanford"],
    "dates": ["1956", "2023"],
    "quantitative_data": [...],
    "sources": ["wikipedia", "google", "arxiv"]
  },
  "query_analysis": {
    "type": "factual",
    "confidence": 0.9,
    "entities": {
      "concepts": ["artificial intelligence", "machine learning"],
      "technologies": ["AI", "neural networks"]
    }
  },
  "validation_results": {
    "total_facts": 12,
    "valid_facts": 10,
    "average_confidence": 0.82
  },
  "processing_time": 2.34,
  "cache_hit": false,
  "status": "success",
  "timestamp": 1640995200
}
```

**Error Response (400):**

```json
{
  "error": "Validation failed",
  "details": "Query parameter is required and cannot be empty",
  "timestamp": 1640995200
}
```

---

### 2. Get Available Modes

Retrieve available processing modes, sub-modes, and formats.

#### GET `/modes`

**Success Response (200):**

```json
{
  "modes": [
    {
      "name": "trained",
      "description": "Neural network inference",
      "available": true
    },
    {
      "name": "direction", 
      "description": "RAG-based search and retrieval",
      "available": true
    }
  ],
  "submodes": {
    "normal": {
      "name": "Normal",
      "description": "Standard, unmodified response",
      "emoji": "💬",
      "color": "#ffffff",
      "characteristics": [
        "Standard response format",
        "No style modifications"
      ]
    },
    "sugarcotted": {
      "name": "Sugarcotted",
      "description": "Sweet, positive, and encouraging",
      "emoji": "🍬",
      "color": "#ffb3d9",
      "characteristics": [
        "Warm and positive language",
        "Sweet emojis and symbols",
        "Optimistic framing"
      ]
    }
  },
  "formats": {
    "comprehensive": {
      "name": "comprehensive",
      "description": "Detailed answer with full context and citations",
      "max_length": 1000,
      "includes_citations": true,
      "includes_confidence": true
    }
  }
}
```

---

### 3. Get System Status

Retrieve current system status and performance metrics.

#### GET `/status`

**Success Response (200):**

```json
{
  "system": "direction_mode",
  "status": "operational",
  "version": "1.0.0",
  "components": {
    "query_analyzer": "operational",
    "search_engine": "operational", 
    "info_extractor": "operational",
    "knowledge_store": "operational",
    "answer_generator": "operational",
    "fact_checker": "operational",
    "response_styler": "operational"
  },
  "metrics": {
    "total_queries": 1250,
    "successful_queries": 1198,
    "failed_queries": 52,
    "average_response_time": 1.85,
    "cache_hit_rate": 0.23,
    "average_confidence": 0.78
  },
  "knowledge_base": {
    "total_queries": 1198,
    "total_facts": 4562,
    "total_concepts": 892,
    "recent_queries_24h": 45,
    "average_confidence": 0.78,
    "top_sources": [
      {"source": "wikipedia", "count": 1250},
      {"source": "google", "count": 987},
      {"source": "arxiv", "count": 234}
    ],
    "popular_concepts": [
      {"concept": "artificial intelligence", "popularity": 4.2},
      {"concept": "machine learning", "popularity": 3.8}
    ],
    "database_size_mb": 12.5
  },
  "available_styles": {...},
  "available_formats": {...},
  "timestamp": 1640995200
}
```

---

### 4. Get Knowledge Base Statistics

Retrieve detailed knowledge base statistics.

#### GET `/stats`

**Success Response (200):**

```json
{
  "total_queries": 1198,
  "total_facts": 4562,
  "total_concepts": 892,
  "recent_queries_24h": 45,
  "average_confidence": 0.78,
  "top_sources": [
    {
      "source": "wikipedia",
      "count": 1250
    },
    {
      "source": "google", 
      "count": 987
    }
  ],
  "popular_concepts": [
    {
      "concept": "artificial intelligence",
      "popularity": 4.2,
      "last_accessed": 1640995200
    }
  ],
  "database_size_mb": 12.5
}
```

---

### 5. Clear Cache

Remove old cache entries from the knowledge base.

#### POST `/clear-cache`

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `older_than_days` | integer | No | Remove entries older than this many days (default: 30) |

**Request Example:**

```http
POST /clear-cache?older_than_days=7
```

**Success Response (200):**

```json
{
  "removed_entries": 45,
  "message": "Cleared 45 old cache entries"
}
```

---

### 6. Search Knowledge Base

Search the cached knowledge base for concepts.

#### POST `/search-knowledge`

**Request Body:**

```json
{
  "search_term": "machine learning",
  "limit": 10
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `search_term` | string | Yes | Term to search for |
| `limit` | integer | No | Maximum results (default: 10, max: 100) |

**Success Response (200):**

```json
{
  "results": [
    {
      "concept": "machine learning",
      "definition": "A subset of artificial intelligence that focuses on algorithms",
      "popularity": 4.5,
      "last_accessed": 1640995200
    }
  ],
  "total_results": 1,
  "search_term": "machine learning"
}
```

---

### 7. Health Check

Check system health and component status.

#### GET `/health`

**Success Response (200):**

```json
{
  "status": "healthy",
  "timestamp": 1640995200,
  "version": "1.0.0",
  "components": {
    "direction_mode": "healthy",
    "neural_core": "healthy",
    "memory_system": "healthy"
  }
}
```

---

### 8. Root Endpoint

Basic API information.

#### GET `/`

**Success Response (200):**

```json
{
  "message": "Sathik AI Direction Mode API",
  "version": "1.0.0", 
  "status": "operational"
}
```

---

## Response Objects

### QueryResponse

Complete response object for query processing.

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | Original query |
| `user_id` | string | User identifier |
| `mode` | string | Processing mode used |
| `submode` | string | Sub-mode style applied |
| `answer` | string | Generated response |
| `confidence` | float | Overall confidence score (0-1) |
| `sources_used` | integer | Number of sources consulted |
| `facts_analyzed` | integer | Number of facts processed |
| `format` | string | Answer format used |
| `citations` | array | Source citations |
| `key_information` | object | Extracted key information |
| `query_analysis` | object | Query type and analysis |
| `validation_results` | object | Fact validation results |
| `processing_time` | float | Processing time in seconds |
| `cache_hit` | boolean | Whether response was cached |
| `status` | string | Response status |
| `timestamp` | float | Response timestamp |

### CitationInfo

Citation information for sources.

| Field | Type | Description |
|-------|------|-------------|
| `number` | integer | Citation number |
| `source` | string | Source name |
| `url` | string | Source URL |
| `fact` | string | Extracted fact |
| `confidence` | float | Confidence score (0-1) |

### KeyInformation

Extracted key information from the query.

| Field | Type | Description |
|-------|------|-------------|
| `main_facts` | array | Primary facts found |
| `definitions` | array | Definitions extracted |
| `dates` | array | Dates mentioned |
| `people` | array | People mentioned |
| `places` | array | Places mentioned |
| `organizations` | array | Organizations mentioned |
| `quantitative_data` | array | Numbers and measurements |
| `sources` | array | Sources used |

---

## Examples

### cURL Examples

#### Submit Query

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "mode": "direction",
    "submode": "sugarcotted",
    "format_type": "comprehensive"
  }'
```

#### Get System Status

```bash
curl -X GET "http://localhost:8000/status"
```

#### Search Knowledge Base

```bash
curl -X POST "http://localhost:8000/search-knowledge" \
  -H "Content-Type: application/json" \
  -d '{
    "search_term": "artificial intelligence",
    "limit": 5
  }'
```

### Python Examples

#### Using requests

```python
import requests

# Submit query
response = requests.post('http://localhost:8000/query', json={
    'query': 'What is quantum computing?',
    'mode': 'direction',
    'submode': 'unhinged',
    'format_type': 'summary'
})

print(response.json())
```

#### Using httpx

```python
import httpx

async def query_sathik():
    async with httpx.AsyncClient() as client:
        response = await client.post('http://localhost:8000/query', json={
            'query': 'Explain blockchain technology',
            'mode': 'direction',
            'submode': 'normal',
            'format_type': 'bullet_points'
        })
        return response.json()

# Usage
import asyncio
result = asyncio.run(query_sathik())
print(result)
```

### JavaScript Examples

#### Using fetch

```javascript
const submitQuery = async () => {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: 'What is deep learning?',
      mode: 'direction',
      submode: 'reaper',
      format_type: 'comprehensive'
    })
  });
  
  const result = await response.json();
  console.log(result);
};
```

#### Using axios

```javascript
const axios = require('axios');

const getStatus = async () => {
  try {
    const response = await axios.get('http://localhost:8000/status');
    console.log(response.data);
  } catch (error) {
    console.error('Error:', error.message);
  }
};
```

---

## SDKs and Libraries

### Python SDK

```python
from sathik_ai_api import SathikAIClient

client = SathikAIClient(base_url='http://localhost:8000')

# Submit query
response = client.query(
    query='What is neural networks?',
    mode='direction',
    submode='sugarcotted'
)

print(response.answer)
```

### JavaScript SDK

```javascript
import { SathikAIClient } from 'sathik-ai-api';

const client = new SathikAIClient({
  baseURL: 'http://localhost:8000'
});

// Submit query
const response = await client.query({
  query: 'What is cryptography?',
  mode: 'direction',
  submode: 'unhinged'
});

console.log(response.answer);
```

---

## WebSocket API

Real-time query processing with WebSocket support.

### Connect

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Submit Query

```json
{
  "type": "query",
  "data": {
    "query": "What is natural language processing?",
    "mode": "direction",
    "submode": "sugarcotted",
    "format_type": "comprehensive"
  }
}
```

### Receive Response

```json
{
  "type": "response",
  "data": {
    "query": "What is natural language processing?",
    "answer": "💖 Hi sweetie! Natural language processing is...",
    "status": "success",
    "confidence": 0.87
  }
}
```

---

## Webhooks

Configure webhooks for real-time notifications.

### Webhook Events

- `query.completed` - Query processing completed
- `query.failed` - Query processing failed  
- `cache.cleared` - Cache was cleared
- `system.error` - System error occurred

### Webhook Payload

```json
{
  "event": "query.completed",
  "timestamp": 1640995200,
  "data": {
    "query_id": "uuid-123",
    "user_id": "user123",
    "mode": "direction",
    "status": "success",
    "processing_time": 2.1
  }
}
```

---

## Testing

### Test Endpoints

Use these endpoints for testing:

```bash
# Test basic connectivity
curl http://localhost:8000/

# Test health check
curl http://localhost:8000/health

# Test with sample query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello world", "mode": "direction"}'
```

### Load Testing

```bash
# Install artillery
npm install -g artillery

# Run load test
artillery run load-test.yml
```

**load-test.yml:**
```yaml
config:
  target: 'http://localhost:8000'
  phases:
    - duration: 60
      arrivalRate: 10

scenarios:
  - name: "Query submission"
    weight: 100
    flow:
      - post:
          url: "/query"
          json:
            query: "What is {{ $randomString() }}?"
            mode: "direction"
            submode: "normal"
```

---

## SDK Development

### Python SDK Structure

```
sathik_ai_api/
├── __init__.py
├── client.py          # Main client class
├── models.py          # Request/response models
├── exceptions.py      # Custom exceptions
└── utils.py          # Utility functions
```

### JavaScript SDK Structure

```
sathik-ai-api/
├── index.js           # Main export
├── client.js          # Client class
├── models.js          # TypeScript interfaces
└── utils.js           # Utility functions
```

---

## Changelog

### Version 1.0.0 (Current)

**Added:**
- Complete Direction Mode implementation
- Multi-source search engine
- 4 response style sub-modes
- FastAPI REST API
- React web frontend
- Knowledge base with caching
- Fact extraction and validation
- Terminal interface integration
- Docker deployment support

**Breaking Changes:**
- None (initial release)

---

## Support

For API support and questions:

- **Documentation**: [API Reference](./API_REFERENCE.md)
- **GitHub Issues**: [Create an issue]
- **Email**: [support@sathik.ai]
- **Discord**: [Join our server]

---

*Last updated: January 2024*