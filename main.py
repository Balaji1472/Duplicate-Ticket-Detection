
"""
Enhanced FastAPI application for duplicate ticket detection with bonus features.
Includes multilingual support, clustering, and active learning feedback.
"""

from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
import logging
from typing import List, Optional
import uvicorn

from src.config import settings
from src.models import (
    TicketInput, 
    BatchTicketInput, 
    DuplicateResult, 
    BatchDuplicateResult,
    FeedbackInput,
    FeedbackResponse,
    ClusterResult,
    ErrorResponse,
    HealthResponse,
    StatsResponse
)
from src.ticket_service import EnhancedTicketService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced lifespan management with comprehensive initialization"""
    global ticket_service
    try:
        logger.info("🚀 Starting Enhanced Duplicate Ticket Detection API...")
        logger.info(f"Features enabled: Multilingual={settings.auto_detect_language}, "
                   f"Clustering={settings.clustering_enabled}, Feedback={settings.feedback_enabled}")
        
        ticket_service = EnhancedTicketService()
        logger.info("✅ API startup completed successfully")
        yield
        logger.info("🔄 Shutting down API...")
    except Exception as e:
        logger.error(f"❌ Failed during lifespan: {e}")
        raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Duplicate Ticket Detection API",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service
ticket_service = None

# Root and health endpoints
@app.get("/", response_model=dict)
async def root():
    """Enhanced root endpoint with feature overview"""
    return {
        "message": "Enhanced Duplicate Ticket Detection API",
        "version": settings.api_version,
        "features": {
            "multilingual_support": settings.auto_detect_language,
            "clustering": settings.clustering_enabled,
            "active_learning": settings.feedback_enabled
        },
        "supported_languages": settings.supported_languages,
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "stats": "/stats",
            "check_duplicate": "/check-duplicate",
            "batch_check": "/check-batch-duplicates",
            "add_ticket": "/add-ticket",
            "feedback": "/feedback",
            "clustering": "/clustering"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with feature status"""
    try:
        model_loaded = ticket_service is not None and ticket_service.embedding_model is not None
        features_enabled = {
            "multilingual_support": settings.auto_detect_language,
            "clustering": settings.clustering_enabled,
            "feedback_collection": settings.feedback_enabled
        }
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            version=settings.api_version,
            model_loaded=model_loaded,
            features_enabled=features_enabled,
            supported_languages=settings.supported_languages
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.api_version,
            model_loaded=False,
            features_enabled={},
            supported_languages=[]
        )

# Core duplicate detection endpoints
@app.post("/check-duplicate", response_model=DuplicateResult)
async def check_duplicate(ticket: TicketInput):
    """
    Enhanced duplicate checking with multilingual support and clustering.
    
    - **ticket_id**: Unique identifier for the ticket
    - **text**: The ticket description text
    - **language**: Optional language code (auto-detected if not provided)
    
    Returns duplicate matches with similarity scores, detected language, and cluster assignment.
    """
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        result = ticket_service.check_single_duplicate(ticket)
        logger.info(f"Processed duplicate check for ticket: {ticket.ticket_id} "
                   f"(language: {result.detected_language}, cluster: {result.cluster_id})")
        return result
        
    except Exception as e:
        logger.error(f"Error checking duplicate for ticket {ticket.ticket_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/check-batch-duplicates", response_model=BatchDuplicateResult)
async def check_batch_duplicates(batch_input: BatchTicketInput):
    """
    Enhanced batch duplicate checking with language statistics.
    
    - **tickets**: List of tickets to check (max 100 tickets per batch)
    
    Returns duplicate results for each ticket with language distribution analysis.
    """
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        results = ticket_service.check_batch_duplicates(batch_input.tickets)
        duplicates_found = sum(1 for r in results if r.is_duplicate)
        
        # Calculate language distribution
        languages_detected = {}
        for result in results:
            if result.detected_language:
                lang = result.detected_language
                languages_detected[lang] = languages_detected.get(lang, 0) + 1
        
        logger.info(f"Processed batch of {len(batch_input.tickets)} tickets, "
                   f"found {duplicates_found} duplicates, languages: {languages_detected}")
        
        return BatchDuplicateResult(
            results=results,
            total_tickets=len(batch_input.tickets),
            duplicates_found=duplicates_found,
            languages_detected=languages_detected
        )
        
    except Exception as e:
        logger.error(f"Error processing batch duplicates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/add-ticket", response_model=dict)
async def add_ticket(ticket: TicketInput):
    """
    Add a new ticket with enhanced language detection and clustering.
    
    - **ticket_id**: Unique identifier for the ticket
    - **text**: The ticket description text
    - **language**: Optional language code (auto-detected if not provided)
    
    This endpoint expands the database with language detection and cluster assignment.
    """
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        # Check if ticket already exists
        existing_ids = [t["ticket_id"] for t in ticket_service.existing_tickets]
        if ticket.ticket_id in existing_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ticket {ticket.ticket_id} already exists"
            )
        
        success = ticket_service.add_ticket(ticket)
        
        if success:
            # Get the added ticket details
            added_ticket = next(t for t in ticket_service.existing_tickets if t["ticket_id"] == ticket.ticket_id)
            logger.info(f"Successfully added new ticket: {ticket.ticket_id}")
            
            return {
                "message": f"Ticket {ticket.ticket_id} added successfully",
                "total_tickets": len(ticket_service.existing_tickets),
                "detected_language": added_ticket.get("language"),
                "cluster_id": added_ticket.get("cluster_id")
            }
        else:
            logger.error(f"Failed to add ticket {ticket.ticket_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add ticket due to internal error"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error adding ticket {ticket.ticket_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Feedback endpoints
@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackInput):
    """
    Submit feedback on duplicate detection results for active learning.
    
    - **ticket_id**: ID of the ticket that was checked
    - **matched_ticket_id**: ID of the ticket it was matched with (if any)
    - **is_correct_duplicate**: Whether the duplicate detection was correct
    - **user_comment**: Optional user comment
    - **confidence_score**: User's confidence in their feedback (0.0-1.0)
    
    Helps improve the model through active learning.
    """
    try:
        if not settings.feedback_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Feedback collection is disabled"
            )
        
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        feedback_id = ticket_service.submit_feedback(
            feedback.ticket_id,
            feedback.matched_ticket_id,
            feedback.is_correct_duplicate,
            feedback.user_comment,
            feedback.confidence_score
        )
        
        total_feedback = len(ticket_service.feedback_service.feedback_data)
        
        logger.info(f"Received feedback for ticket {feedback.ticket_id}: {feedback.is_correct_duplicate}")
        
        return FeedbackResponse(
            message="Feedback submitted successfully",
            feedback_id=feedback_id,
            total_feedback_count=total_feedback
        )
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/feedback/stats", response_model=dict)
async def get_feedback_stats():
    """Get feedback statistics"""
    try:
        if not settings.feedback_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Feedback collection is disabled"
            )
        
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        stats = ticket_service.feedback_service.get_feedback_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Clustering endpoints
@app.get("/clustering", response_model=dict)
async def get_clustering_info():
    """Get current clustering information"""
    try:
        if not settings.clustering_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Clustering is disabled"
            )
        
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        cluster_results = ticket_service.get_clustering_results()
        
        if cluster_results is None:
            return {"message": "No clustering results available", "clusters": []}
        
        return cluster_results
        
    except Exception as e:
        logger.error(f"Error getting clustering info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/clustering/recluster", response_model=dict)
async def recluster_tickets(method: str = Query("kmeans", description="Clustering method")):
    """Force re-clustering of all tickets"""
    try:
        if not settings.clustering_enabled:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Clustering is disabled"
            )
        
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        results = ticket_service.recluster_tickets(method)
        logger.info(f"Re-clustering completed with method: {method}")
        
        return {
            "message": "Tickets re-clustered successfully",
            "method": method,
            "results": results
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error re-clustering tickets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Statistics endpoints
@app.get("/stats", response_model=dict)
async def get_stats():
    """Get comprehensive system statistics"""
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        stats = ticket_service.get_enhanced_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/stats/languages", response_model=dict)
async def get_language_stats():
    """Get language distribution statistics"""
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        language_stats = ticket_service.get_language_statistics()
        return {
            "supported_languages": settings.supported_languages,
            "language_distribution": [stat.dict() for stat in language_stats],
            "total_tickets": len(ticket_service.existing_tickets)
        }
        
    except Exception as e:
        logger.error(f"Error getting language stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Ticket management endpoints
@app.get("/tickets", response_model=dict)
async def get_all_tickets(limit: int = Query(100, ge=1, le=1000)):
    """Get all tickets with optional limit"""
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        tickets = ticket_service.existing_tickets[:limit]
        return {
            "tickets": tickets,
            "total_count": len(ticket_service.existing_tickets),
            "returned_count": len(tickets)
        }
        
    except Exception as e:
        logger.error(f"Error getting tickets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/tickets/{ticket_id}", response_model=dict)
async def get_ticket(ticket_id: str):
    """Get a specific ticket by ID"""
    try:
        if ticket_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )
        
        ticket = next((t for t in ticket_service.existing_tickets if t["ticket_id"] == ticket_id), None)
        
        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Ticket {ticket_id} not found"
            )
        
        return ticket
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting ticket {ticket_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )