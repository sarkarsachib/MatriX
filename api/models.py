"""
Pydantic models for FastAPI endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class QueryMode(str, Enum):
    """Query processing modes"""
    TRAINED = "trained"
    DIRECTION = "direction"


class SubmodeStyle(str, Enum):
    """Response sub-mode styles"""
    NORMAL = "normal"
    SUGARCOTTED = "sugarcotted"
    UNHINGED = "unhinged"
    REAPER = "reaper"
    HEXAGON = "666"


class AnswerFormat(str, Enum):
    """Answer format types"""
    COMPREHENSIVE = "comprehensive"
    SUMMARY = "summary"
    BULLET_POINTS = "bullet_points"


class QueryRequest(BaseModel):
    """Request model for query processing"""
    query: str = Field(..., description="The query to process", min_length=1)
    user_id: str = Field("default", description="User identifier")
    mode: QueryMode = Field(QueryMode.DIRECTION, description="Processing mode")
    submode: SubmodeStyle = Field(SubmodeStyle.NORMAL, description="Response sub-mode style")
    format_type: AnswerFormat = Field(AnswerFormat.COMPREHENSIVE, description="Answer format")
    output_mode: Optional[str] = Field(None, description="Output mode (for trained mode)")


class CitationInfo(BaseModel):
    """Citation information"""
    number: int = Field(..., description="Citation number")
    source: str = Field(..., description="Source name")
    url: str = Field(..., description="Source URL")
    fact: str = Field(..., description="Fact from source")
    confidence: float = Field(..., description="Confidence score")


class SourceInfo(BaseModel):
    """Source information"""
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    snippet: str = Field(..., description="Content snippet")
    confidence: float = Field(..., description="Source confidence")


class QueryAnalysis(BaseModel):
    """Query analysis results"""
    type: str = Field(..., description="Query type")
    confidence: float = Field(..., description="Analysis confidence")
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted entities")


class ValidationResults(BaseModel):
    """Fact validation results"""
    total_facts: int = Field(..., description="Total facts analyzed")
    valid_facts: int = Field(..., description="Number of valid facts")
    average_confidence: float = Field(..., description="Average confidence score")


class KeyInformation(BaseModel):
    """Key information extracted from query"""
    main_facts: List[Dict[str, Any]] = Field(default_factory=list, description="Main facts")
    definitions: List[Dict[str, Any]] = Field(default_factory=list, description="Definitions found")
    dates: List[str] = Field(default_factory=list, description="Dates mentioned")
    people: List[str] = Field(default_factory=list, description="People mentioned")
    places: List[str] = Field(default_factory=list, description="Places mentioned")
    organizations: List[str] = Field(default_factory=list, description="Organizations mentioned")
    quantitative_data: List[Dict[str, Any]] = Field(default_factory=list, description="Quantitative information")
    sources: List[str] = Field(default_factory=list, description="Sources used")


class QueryResponse(BaseModel):
    """Response model for query processing"""
    query: str = Field(..., description="Original query")
    user_id: str = Field(..., description="User identifier")
    mode: str = Field(..., description="Processing mode used")
    submode: str = Field(..., description="Sub-mode style used")
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Overall confidence score")
    sources_used: int = Field(..., description="Number of sources used")
    facts_analyzed: int = Field(..., description="Number of facts analyzed")
    format: str = Field(..., description="Answer format used")
    citations: List[CitationInfo] = Field(default_factory=list, description="Citations")
    key_information: Optional[KeyInformation] = Field(None, description="Extracted key information")
    query_analysis: Optional[QueryAnalysis] = Field(None, description="Query analysis results")
    validation_results: Optional[ValidationResults] = Field(None, description="Validation results")
    processing_time: float = Field(..., description="Processing time in seconds")
    cache_hit: bool = Field(False, description="Whether response was from cache")
    status: str = Field(..., description="Response status")
    timestamp: float = Field(..., description="Response timestamp")
    error: Optional[str] = Field(None, description="Error message if any")


class SystemStatus(BaseModel):
    """System status information"""
    system: str = Field(..., description="System name")
    status: str = Field(..., description="System status")
    version: str = Field(..., description="Version information")
    components: Dict[str, str] = Field(..., description="Component statuses")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    timestamp: float = Field(..., description="Status timestamp")


class KnowledgeBaseStats(BaseModel):
    """Knowledge base statistics"""
    total_queries: int = Field(..., description="Total queries in database")
    total_facts: int = Field(..., description="Total facts stored")
    total_concepts: int = Field(..., description="Total concepts in knowledge base")
    recent_queries_24h: int = Field(..., description="Recent queries in last 24 hours")
    average_confidence: float = Field(..., description="Average confidence score")
    top_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Top sources by usage")
    popular_concepts: List[Dict[str, Any]] = Field(default_factory=list, description="Most popular concepts")
    database_size_mb: float = Field(..., description="Database size in MB")


class DirectionModeStatus(BaseModel):
    """Direction Mode specific status"""
    system: str = Field(..., description="System name")
    status: str = Field(..., description="Operational status")
    version: str = Field(..., description="Version")
    components: Dict[str, str] = Field(..., description="Component statuses")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    knowledge_base: KnowledgeBaseStats = Field(..., description="Knowledge base stats")
    available_styles: Dict[str, Dict[str, Any]] = Field(..., description="Available response styles")
    available_formats: Dict[str, Dict[str, Any]] = Field(..., description="Available answer formats")
    timestamp: float = Field(..., description="Status timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    timestamp: float = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component health")


class CacheClearResponse(BaseModel):
    """Cache clear response"""
    removed_entries: int = Field(..., description="Number of entries removed")
    message: str = Field(..., description="Status message")


class SearchRequest(BaseModel):
    """Search request model"""
    search_term: str = Field(..., description="Term to search for")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100)


class SearchResult(BaseModel):
    """Search result model"""
    concept: str = Field(..., description="Concept name")
    definition: str = Field(..., description="Concept definition")
    popularity: float = Field(..., description="Concept popularity score")
    last_accessed: float = Field(..., description="Last access timestamp")


class SearchResponse(BaseModel):
    """Search response model"""
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_term: str = Field(..., description="Search term used")


class ModeInfo(BaseModel):
    """Mode information"""
    name: str = Field(..., description="Mode name")
    description: str = Field(..., description="Mode description")
    available: bool = Field(..., description="Whether mode is available")


class SubmodeInfo(BaseModel):
    """Sub-mode information"""
    name: str = Field(..., description="Sub-mode name")
    description: str = Field(..., description="Sub-mode description")
    emoji: str = Field(..., description="Sub-mode emoji")
    color: str = Field(..., description="Sub-mode color")
    characteristics: List[str] = Field(..., description="Style characteristics")


class FormatInfo(BaseModel):
    """Format information"""
    name: str = Field(..., description="Format name")
    description: str = Field(..., description="Format description")
    max_length: int = Field(..., description="Maximum response length")
    includes_citations: bool = Field(..., description="Whether format includes citations")
    includes_confidence: bool = Field(..., description="Whether format includes confidence score")


class ModesResponse(BaseModel):
    """Available modes response"""
    modes: List[ModeInfo] = Field(..., description="Available processing modes")
    submodes: Dict[str, SubmodeInfo] = Field(..., description="Available sub-modes")
    formats: Dict[str, FormatInfo] = Field(..., description="Available answer formats")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Error details")
    timestamp: float = Field(..., description="Error timestamp")