"""
Security and CORS configuration for FastAPI
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional


def setup_cors(app: FastAPI):
    """
    Setup CORS middleware for the FastAPI application
    
    Args:
        app: FastAPI application instance
    """
    # Get CORS settings from environment variables
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    allowed_methods = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
    allowed_headers = os.getenv("ALLOWED_HEADERS", "*,Content-Type,Authorization,X-Requested-With").split(",")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in allowed_origins],
        allow_credentials=True,
        allow_methods=[method.strip() for method in allowed_methods],
        allow_headers=[header.strip() for header in allowed_headers],
        expose_headers=["*"],
        max_age=86400,  # 24 hours
    )


def validate_api_key(request: Request, api_key: Optional[str] = None) -> bool:
    """
    Validate API key if required
    
    Args:
        request: FastAPI request object
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check if API key validation is enabled
    require_auth = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    
    if not require_auth:
        return True
    
    # Get expected API key from environment
    expected_api_key = os.getenv("API_KEY")
    
    if not expected_api_key:
        return True  # No API key configured, allow access
    
    # Check API key in headers
    provided_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
    
    return provided_key == expected_api_key


class SecurityConfig:
    """Security configuration settings"""
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # API Key settings
    REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
    API_KEY = os.getenv("API_KEY")
    
    # CORS settings
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ALLOWED_METHODS = os.getenv("ALLOWED_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(",")
    ALLOWED_HEADERS = os.getenv("ALLOWED_HEADERS", "*,Content-Type,Authorization,X-Requested-With").split(",")
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    }


def add_security_headers(app: FastAPI):
    """
    Add security headers to all responses
    
    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def add_security_headers_middleware(request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        for header, value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header] = value
        
        return response


def setup_security(app: FastAPI):
    """
    Setup all security configurations
    
    Args:
        app: FastAPI application instance
    """
    # Setup CORS
    setup_cors(app)
    
    # Add security headers
    add_security_headers(app)