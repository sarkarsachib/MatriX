"""
Configuration for Sathik AI Direction Mode
Contains settings for Direction Mode functionality
"""

# Direction Mode Configuration
DIRECTION_MODE_CONFIG = {
    # Search API Configuration
    'google_api_key': None,  # Set your Google API key here
    'google_cse_id': None,   # Set your Google Custom Search Engine ID
    'news_api_key': None,    # Set your NewsAPI key here
    
    # Knowledge Store Configuration
    'knowledge_db_path': 'direction_mode_knowledge.db',
    'cache_expiration_days': 30,
    'max_cache_size_mb': 100,
    
    # Search Configuration
    'max_search_results': 10,
    'search_timeout_seconds': 30,
    'min_confidence_threshold': 0.3,
    
    # Fact Extraction Configuration
    'enable_fact_extraction': True,
    'enable_fact_validation': True,
    'min_fact_length': 10,
    'max_fact_length': 200,
    
    # Response Generation Configuration
    'default_format': 'comprehensive',
    'enable_citations': True,
    'max_citations': 5,
    
    # Style Configuration
    'default_submode': 'normal',
    'enable_styling': True,
    
    # Performance Configuration
    'enable_caching': True,
    'cache_similarity_threshold': 0.8,
    'parallel_search': True,
    
    # Source Reliability Configuration
    'source_reliability': {
        'wikipedia': 0.9,
        'arxiv': 0.95,
        'google': 0.8,
        'newsapi': 0.7,
        'duckduckgo': 0.6,
        'stackoverflow': 0.8,
        'github': 0.8
    },
    
    # Rate Limiting Configuration
    'rate_limit_enabled': True,
    'max_requests_per_minute': 60,
    'requests_per_source': {
        'google': 10,
        'newsapi': 100,
        'duckduckgo': 1000,
        'wikipedia': 1000,
        'arxiv': 1000
    }
}

# FastAPI Configuration
FASTAPI_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'reload': True,
    'log_level': 'info',
    'title': 'Sathik AI Direction Mode API',
    'description': 'RAG-based query processing with multiple response styles',
    'version': '1.0.0'
}

# Security Configuration
SECURITY_CONFIG = {
    'cors_origins': ['*'],
    'require_api_key': False,
    'api_key': None,
    'rate_limit_enabled': True,
    'max_requests_per_hour': 1000
}

# Web UI Configuration
WEB_UI_CONFIG = {
    'api_base_url': 'http://localhost:8000',
    'request_timeout': 30000,
    'enable_analytics': False,
    'theme': 'dark'
}

# Sub-mode Style Configuration
SUBMODE_CONFIG = {
    'sugarcotted': {
        'enabled': True,
        'intensity': 0.8,
        'emojis': ['💖', '🌸', '✨', '🌈', '💫'],
        'positive_replacements': True
    },
    'unhinged': {
        'enabled': True,
        'intensity': 0.7,
        'remove_politeness': True,
        'casual_language': True
    },
    'reaper': {
        'enabled': True,
        'intensity': 0.6,
        'dark_themes': True,
        'mortality_focus': True
    },
    'hexagon': {
        'enabled': True,
        'intensity': 0.8,
        'chaotic_elements': True,
        'meta_commentary': True
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'direction_mode.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}