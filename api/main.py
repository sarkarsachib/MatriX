"""
FastAPI application for Sathik AI Direction Mode
"""

import asyncio
import logging
from typing import Dict, Any, Optional
import os
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.models import *
from api.security import setup_cors, validate_api_key

# Import Sathik AI Direction Mode components
import sys
sys.path.append('/home/engine/project')

from main import SathikAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sathik AI Direction Mode API",
    description="RAG-based query processing with multiple response styles",
    version="1.0.0"
)

# Setup CORS
setup_cors(app)

# Global Sathik AI instance
sathik_instance: Optional[SathikAI] = None


@app.on_event("startup")
async def startup_event():
    """Initialize Sathik AI system on startup"""
    global sathik_instance
    try:
        logger.info("Initializing Sathik AI Direction Mode...")
        sathik_instance = SathikAI()
        logger.info("Sathik AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Sathik AI: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global sathik_instance
    if sathik_instance:
        try:
            # Save system state
            sathik_instance.save_system_state()
            logger.info("System state saved")
        except Exception as e:
            logger.error(f"Error saving system state: {e}")


def get_sathik_ai() -> SathikAI:
    """Dependency to get Sathik AI instance"""
    if sathik_instance is None:
        raise HTTPException(status_code=503, detail="Sathik AI not initialized")
    return sathik_instance


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Sathik AI Direction Mode API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(sathik: SathikAI = Depends(get_sathik_ai)):
    """Health check endpoint"""
    try:
        # Check Direction Mode status
        direction_status = sathik.direction_mode.get_system_status()
        
        # Check component health
        components = {
            "direction_mode": "healthy" if direction_status.get("status") == "operational" else "unhealthy",
            "neural_core": "healthy" if sathik.is_initialized else "unhealthy",
            "memory_system": "healthy" if sathik.memory_system else "unhealthy"
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=asyncio.get_event_loop().time(),
            version="1.0.0",
            components=components
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    sathik: SathikAI = Depends(get_sathik_ai)
):
    """Process a query using Direction Mode or Trained Mode"""
    try:
        logger.info(f"Processing query: {request.query[:50]}...")
        
        # Process query using Sathik AI
        response_data = sathik.process_query(
            query=request.query,
            user_id=request.user_id,
            mode=request.mode.value,
            submode=request.submode.value,
            format_type=request.format_type.value
        )
        
        # Convert to response model
        if response_data.get('status') == 'success':
            return QueryResponse(
                query=response_data['query'],
                user_id=response_data['user_id'],
                mode=response_data['mode'],
                submode=response_data['submode'],
                answer=response_data.get('answer', response_data.get('response', '')),
                confidence=response_data.get('confidence', 0.0),
                sources_used=response_data.get('sources_used', 0),
                facts_analyzed=response_data.get('facts_analyzed', 0),
                format=response_data.get('format', 'comprehensive'),
                citations=[
                    CitationInfo(
                        number=cit['number'],
                        source=cit['source'],
                        url=cit['url'],
                        fact=cit['fact'],
                        confidence=cit['confidence']
                    )
                    for cit in response_data.get('citations', [])
                ],
                key_information=KeyInformation(**response_data.get('key_information', {})) if response_data.get('key_information') else None,
                query_analysis=QueryAnalysis(**response_data.get('query_analysis', {})) if response_data.get('query_analysis') else None,
                validation_results=ValidationResults(**response_data.get('validation_results', {})) if response_data.get('validation_results') else None,
                processing_time=response_data.get('processing_time', 0.0),
                cache_hit=response_data.get('cache_hit', False),
                status=response_data['status'],
                timestamp=response_data['timestamp']
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=response_data.get('error', 'Query processing failed')
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/modes", response_model=ModesResponse)
async def get_available_modes(sathik: SathikAI = Depends(get_sathik_ai)):
    """Get available modes, sub-modes, and formats"""
    try:
        # Get sub-modes
        submodes = sathik.direction_mode.get_available_submodes()
        
        # Get formats
        formats = sathik.direction_mode.get_available_formats()
        
        # Convert to response format
        submodes_info = {}
        for submode_key, submode_data in submodes.items():
            submodes_info[submode_key] = SubmodeInfo(
                name=submode_data['name'],
                description=submode_data['description'],
                emoji=submode_data['emoji'],
                color=submode_data['color'],
                characteristics=submode_data.get('characteristics', [])
            )
        
        formats_info = {}
        for format_key, format_data in formats.items():
            formats_info[format_key] = FormatInfo(
                name=format_data['name'],
                description=format_data['description'],
                max_length=format_data['max_length'],
                includes_citations=format_data['includes_citations'],
                includes_confidence=format_data['includes_confidence']
            )
        
        return ModesResponse(
            modes=[
                ModeInfo(name="trained", description="Neural network inference", available=True),
                ModeInfo(name="direction", description="RAG-based search and retrieval", available=True)
            ],
            submodes=submodes_info,
            formats=formats_info
        )
        
    except Exception as e:
        logger.error(f"Error getting modes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get modes: {str(e)}")


@app.get("/submodes", response_model=Dict[str, SubmodeInfo])
async def get_available_submodes(sathik: SathikAI = Depends(get_sathik_ai)):
    """Get available sub-modes"""
    try:
        submodes = sathik.direction_mode.get_available_submodes()
        
        response = {}
        for submode_key, submode_data in submodes.items():
            response[submode_key] = SubmodeInfo(
                name=submode_data['name'],
                description=submode_data['description'],
                emoji=submode_data['emoji'],
                color=submode_data['color'],
                characteristics=submode_data.get('characteristics', [])
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting sub-modes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sub-modes: {str(e)}")


@app.get("/stats", response_model=KnowledgeBaseStats)
async def get_knowledge_base_stats(sathik: SathikAI = Depends(get_sathik_ai)):
    """Get knowledge base statistics"""
    try:
        stats = sathik.direction_mode.knowledge_store.get_knowledge_base_stats()
        
        return KnowledgeBaseStats(
            total_queries=stats['total_queries'],
            total_facts=stats['total_facts'],
            total_concepts=stats['total_concepts'],
            recent_queries_24h=stats['recent_queries_24h'],
            average_confidence=stats['average_confidence'],
            top_sources=stats['top_sources'],
            popular_concepts=stats['popular_concepts'],
            database_size_mb=stats['database_size_mb']
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/status", response_model=DirectionModeStatus)
async def get_system_status(sathik: SathikAI = Depends(get_sathik_ai)):
    """Get Direction Mode system status"""
    try:
        status = sathik.direction_mode.get_system_status()
        kb_stats = sathik.direction_mode.knowledge_store.get_knowledge_base_stats()
        
        return DirectionModeStatus(
            system=status['system'],
            status=status['status'],
            version=status['version'],
            components=status['components'],
            metrics=status['metrics'],
            knowledge_base=KnowledgeBaseStats(
                total_queries=kb_stats['total_queries'],
                total_facts=kb_stats['total_facts'],
                total_concepts=kb_stats['total_concepts'],
                recent_queries_24h=kb_stats['recent_queries_24h'],
                average_confidence=kb_stats['average_confidence'],
                top_sources=kb_stats['top_sources'],
                popular_concepts=kb_stats['popular_concepts'],
                database_size_mb=kb_stats['database_size_mb']
            ),
            available_styles=status['available_styles'],
            available_formats=status['available_formats'],
            timestamp=status['timestamp']
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/clear-cache", response_model=CacheClearResponse)
async def clear_cache(
    older_than_days: int = 30,
    sathik: SathikAI = Depends(get_sathik_ai)
):
    """Clear Direction Mode cache"""
    try:
        removed_count = sathik.direction_mode.clear_cache(older_than_days)
        
        return CacheClearResponse(
            removed_entries=removed_count,
            message=f"Cleared {removed_count} old cache entries"
        )
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.post("/search-knowledge", response_model=SearchResponse)
async def search_knowledge_base(
    request: SearchRequest,
    sathik: SathikAI = Depends(get_sathik_ai)
):
    """Search the knowledge base"""
    try:
        results = await sathik.direction_mode.search_knowledge_base(
            request.search_term,
            request.limit
        )
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                concept=result['concept'],
                definition=result['definition'],
                popularity=result['popularity'],
                last_accessed=result['last_accessed']
            ))
        
        return SearchResponse(
            results=search_results,
            total_results=len(search_results),
            search_term=request.search_term
        )
        
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": asyncio.get_event_loop().time()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": asyncio.get_event_loop().time()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )