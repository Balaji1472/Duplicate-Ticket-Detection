
"""
Enhanced Pydantic models for API request/response validation.
Added support for multilingual, clustering, and feedback features.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class SupportedLanguage(str, Enum):
    """Supported languages for multilingual processing"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    HINDI = "hi"
    CHINESE = "zh"
    TAMIL = "ta"

class TicketInput(BaseModel):
    """Model for single ticket input with language support"""
    ticket_id: str = Field(..., description="Unique identifier for the ticket")
    text: str = Field(..., min_length=5, description="Ticket description text")
    language: Optional[str] = Field(None, description="Language code (auto-detected if not provided)")

class BatchTicketInput(BaseModel):
    """Model for batch ticket input"""
    tickets: List[TicketInput] = Field(..., min_items=1, max_items=100, description="List of tickets to check")

class DuplicateMatch(BaseModel):
    """Model for duplicate match result"""
    ticket_id: str = Field(..., description="ID of the matching ticket")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between 0 and 1")
    text: str = Field(..., description="Text of the matching ticket")
    language: Optional[str] = Field(None, description="Detected language of the matching ticket")

class DuplicateResult(BaseModel):
    """Model for duplicate detection result"""
    ticket_id: str = Field(..., description="ID of the input ticket")
    is_duplicate: bool = Field(..., description="Whether duplicates were found")
    matches: List[DuplicateMatch] = Field(default=[], description="List of duplicate matches")
    processed_text: Optional[str] = Field(None, description="Preprocessed text used for comparison")
    detected_language: Optional[str] = Field(None, description="Auto-detected language")
    cluster_id: Optional[int] = Field(None, description="Cluster ID if clustering is enabled")

class BatchDuplicateResult(BaseModel):
    """Model for batch duplicate detection result"""
    results: List[DuplicateResult] = Field(..., description="List of duplicate detection results")
    total_tickets: int = Field(..., description="Total number of tickets processed")
    duplicates_found: int = Field(..., description="Number of tickets with duplicates")
    languages_detected: Dict[str, int] = Field(default={}, description="Count of tickets per language")

# New models for bonus features

class FeedbackInput(BaseModel):
    """Model for user feedback on duplicate detection"""
    ticket_id: str = Field(..., description="ID of the ticket that was checked")
    matched_ticket_id: Optional[str] = Field(None, description="ID of the ticket it was matched with")
    is_correct_duplicate: bool = Field(..., description="Whether the duplicate detection was correct")
    user_comment: Optional[str] = Field(None, max_length=500, description="Optional user comment")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="User confidence in their feedback")

class FeedbackResponse(BaseModel):
    """Model for feedback response"""
    message: str = Field(..., description="Confirmation message")
    feedback_id: str = Field(..., description="Unique feedback identifier")
    total_feedback_count: int = Field(..., description="Total feedback entries collected")

class ClusterInfo(BaseModel):
    """Model for ticket cluster information"""
    cluster_id: int = Field(..., description="Cluster identifier")
    ticket_count: int = Field(..., description="Number of tickets in this cluster")
    representative_text: str = Field(..., description="Representative text for the cluster")
    avg_similarity: float = Field(..., description="Average similarity within cluster")

class ClusterResult(BaseModel):
    """Model for clustering results"""
    total_clusters: int = Field(..., description="Total number of clusters found")
    clusters: List[ClusterInfo] = Field(..., description="List of cluster information")
    unclustered_tickets: int = Field(..., description="Number of tickets not assigned to clusters")

class TicketCluster(BaseModel):
    """Model for individual ticket cluster assignment"""
    ticket_id: str = Field(..., description="Ticket ID")
    cluster_id: int = Field(..., description="Assigned cluster ID")
    distance_to_centroid: float = Field(..., description="Distance to cluster centroid")

class LanguageStats(BaseModel):
    """Model for language statistics"""
    language_code: str = Field(..., description="Language code")
    language_name: str = Field(..., description="Language name")
    ticket_count: int = Field(..., description="Number of tickets in this language")
    avg_confidence: float = Field(..., description="Average language detection confidence")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code for programmatic handling")

class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the embedding model is loaded")
    features_enabled: Dict[str, bool] = Field(..., description="Status of bonus features")
    supported_languages: List[str] = Field(..., description="List of supported languages")

class StatsResponse(BaseModel):
    """Enhanced model for system statistics"""
    total_tickets: int = Field(..., description="Total number of tickets")
    model_name: str = Field(..., description="Embedding model name")
    similarity_threshold: float = Field(..., description="Similarity threshold")
    embedding_dimension: int = Field(..., description="Embedding vector dimension")
    language_distribution: List[LanguageStats] = Field(default=[], description="Distribution of languages")
    clustering_stats: Optional[ClusterResult] = Field(None, description="Clustering statistics if enabled")
    feedback_count: int = Field(default=0, description="Total feedback entries received")
