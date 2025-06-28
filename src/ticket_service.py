
"""
Enhanced service layer for duplicate ticket detection with bonus features.
Integrates multilingual support, clustering, and active learning.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import logging

from .embedding import EmbeddingModel
from .preprocessing import preprocess_text
from .data_loader import load_tickets
from .config import settings
from .models import TicketInput, DuplicateResult, DuplicateMatch, LanguageStats
from .language_detector import LanguageDetector
from .clustering_service import TicketClusteringService
from .feedback_service import FeedbackService

logger = logging.getLogger(__name__)

class EnhancedTicketService:
    """Enhanced service for handling multilingual duplicate ticket detection with clustering and feedback"""
    
    def __init__(self):
        self.embedding_model = None
        self.existing_tickets = []
        self.existing_embeddings = None
        self.language_detector = LanguageDetector()
        self.clustering_service = TicketClusteringService(
            min_cluster_size=settings.min_cluster_size,
            max_clusters=settings.max_clusters
        )
        self.feedback_service = FeedbackService(settings.feedback_storage_path)
        self.cluster_results = None
        self._load_model()
        self._load_existing_tickets()
        self._initialize_clustering()
    
    def _load_model(self):
        """Load the multilingual embedding model"""
        try:
            logger.info(f"Loading multilingual embedding model: {settings.embedding_model_name}")
            self.embedding_model = EmbeddingModel(settings.embedding_model_name)
            logger.info("Multilingual embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _load_existing_tickets(self):
        """Load existing tickets and generate their embeddings with language detection"""
        try:
            logger.info(f"Loading existing tickets from: {settings.data_path}")
            self.existing_tickets = load_tickets(settings.data_path)
            
            if not self.existing_tickets:
                logger.warning("No existing tickets found")
                self.existing_embeddings = torch.empty(0, 384)  # Empty tensor
                return
            
            # Process tickets with language detection
            existing_texts = []
            for i, ticket in enumerate(self.existing_tickets):
                # Detect language if not specified
                if 'language' not in ticket or not ticket['language']:
                    lang_code, confidence = self.language_detector.detect_language(ticket['text'])
                    ticket['language'] = lang_code
                    ticket['language_confidence'] = confidence
                
                # Preprocess text
                processed_text = preprocess_text(ticket['text'])
                existing_texts.append(processed_text)
            
            # Generate embeddings
            self.existing_embeddings = self.embedding_model.embed_batch(existing_texts)
            
            logger.info(f"Loaded {len(self.existing_tickets)} existing tickets with language detection")
            
        except Exception as e:
            logger.error(f"Failed to load existing tickets: {e}")
            self.existing_tickets = []
            self.existing_embeddings = torch.empty(0, 384)
    
    def _initialize_clustering(self):
        """Initialize clustering if enabled and sufficient data exists"""
        if not settings.clustering_enabled or len(self.existing_tickets) < settings.min_cluster_size:
            logger.info("Clustering disabled or insufficient data for clustering")
            return
        
        try:
            logger.info("Initializing ticket clustering...")
            ticket_ids = [t['ticket_id'] for t in self.existing_tickets]
            ticket_texts = [t['text'] for t in self.existing_tickets]
            
            self.cluster_results = self.clustering_service.cluster_tickets(
                self.existing_embeddings, ticket_ids, ticket_texts
            )
            
            # Update tickets with cluster assignments
            cluster_assignments = self.cluster_results.get('cluster_assignments', {})
            for ticket in self.existing_tickets:
                ticket['cluster_id'] = cluster_assignments.get(ticket['ticket_id'])
            
            logger.info(f"Clustering initialized: {self.cluster_results['total_clusters']} clusters created")
            
        except Exception as e:
            logger.error(f"Failed to initialize clustering: {e}")
            self.cluster_results = None
    
    def check_single_duplicate(self, ticket: TicketInput) -> DuplicateResult:
        """Check if a single ticket is a duplicate with enhanced features"""
        try:
            # Language detection
            detected_language = None
            language_confidence = 0.0
            
            if settings.auto_detect_language:
                if ticket.language:
                    detected_language = ticket.language
                    language_confidence = 1.0  # User-provided language
                else:
                    detected_language, language_confidence = self.language_detector.detect_language(ticket.text)
            
            # Preprocess the input text
            processed_text = preprocess_text(ticket.text)
            
            # Generate embedding for the new ticket
            new_embedding = self.embedding_model.embed_text(processed_text)
            
            matches = []
            is_duplicate = False
            cluster_id = None
            
            # Get cluster assignment if clustering is enabled
            if settings.clustering_enabled and self.cluster_results:
                cluster_id = self.clustering_service.get_cluster_for_new_ticket(new_embedding)
            
            if len(self.existing_tickets) > 0:
                # Calculate similarities with existing tickets
                similarities = cosine_similarity(
                    new_embedding.numpy().reshape(1, -1),
                    self.existing_embeddings.numpy()
                )[0]
                
                # Find matches above threshold
                for i, similarity in enumerate(similarities):
                    if similarity >= settings.similarity_threshold:
                        existing_ticket = self.existing_tickets[i]
                        matches.append(DuplicateMatch(
                            ticket_id=existing_ticket["ticket_id"],
                            similarity_score=round(float(similarity), 4),
                            text=existing_ticket["text"],
                            language=existing_ticket.get("language")
                        ))
                        is_duplicate = True
                
                # Sort matches by similarity score (highest first)
                matches.sort(key=lambda x: x.similarity_score, reverse=True)
            
            return DuplicateResult(
                ticket_id=ticket.ticket_id,
                is_duplicate=is_duplicate,
                matches=matches,
                processed_text=processed_text,
                detected_language=detected_language,
                cluster_id=cluster_id
            )
            
        except Exception as e:
            logger.error(f"Error checking duplicate for ticket {ticket.ticket_id}: {e}")
            raise
    
    def check_batch_duplicates(self, tickets: List[TicketInput]) -> List[DuplicateResult]:
        """Check multiple tickets for duplicates with batch processing optimization"""
        results = []
        language_stats = {}
        
        for ticket in tickets:
            try:
                result = self.check_single_duplicate(ticket)
                results.append(result)
                
                # Track language statistics
                if result.detected_language:
                    lang = result.detected_language
                    language_stats[lang] = language_stats.get(lang, 0) + 1
                    
            except Exception as e:
                logger.error(f"Error processing ticket {ticket.ticket_id}: {e}")
                # Add error result
                results.append(DuplicateResult(
                    ticket_id=ticket.ticket_id,
                    is_duplicate=False,
                    matches=[],
                    processed_text=None,
                    detected_language=None,
                    cluster_id=None
                ))
        
        logger.info(f"Batch processing completed: {len(results)} tickets, languages detected: {language_stats}")
        return results
    
    def add_ticket(self, ticket: TicketInput) -> bool:
        """Add a new ticket with enhanced features"""
        try:
            # Check if ticket ID already exists
            existing_ids = [t["ticket_id"] for t in self.existing_tickets]
            if ticket.ticket_id in existing_ids:
                logger.warning(f"Ticket {ticket.ticket_id} already exists")
                return False
            
            # Language detection
            detected_language, language_confidence = self.language_detector.detect_with_fallback(
                ticket.text, default_language='en'
            )
            
            # Create new ticket with enhanced metadata
            new_ticket = {
                "ticket_id": ticket.ticket_id,
                "text": ticket.text,
                "language": ticket.language or detected_language,
                "language_confidence": language_confidence,
                "cluster_id": None
            }
            
            # Generate embedding
            processed_text = preprocess_text(ticket.text)
            new_embedding = self.embedding_model.embed_text(processed_text)
            
            # Ensure new_embedding has the right shape
            if len(new_embedding.shape) == 1:
                new_embedding = new_embedding.unsqueeze(0)
            elif len(new_embedding.shape) == 3:
                new_embedding = new_embedding.squeeze(0)
            
            # Add to existing tickets and embeddings
            self.existing_tickets.append(new_ticket)
            
            if self.existing_embeddings.size(0) == 0:
                self.existing_embeddings = new_embedding
            else:
                self.existing_embeddings = torch.cat([
                    self.existing_embeddings,
                    new_embedding
                ], dim=0)
            
            # Update clustering if enabled
            if settings.clustering_enabled and self.cluster_results:
                cluster_id = self.clustering_service.get_cluster_for_new_ticket(new_embedding)
                new_ticket['cluster_id'] = cluster_id
                
                # Re-cluster if we have enough new tickets (optional optimization)
                if len(self.existing_tickets) % 10 == 0:  # Re-cluster every 10 new tickets
                    self._initialize_clustering()
            
            logger.info(f"Successfully added ticket {ticket.ticket_id} (language: {detected_language}). Total: {len(self.existing_tickets)}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding ticket {ticket.ticket_id}: {e}")
            # Rollback
            try:
                if len(self.existing_tickets) > 0 and self.existing_tickets[-1]["ticket_id"] == ticket.ticket_id:
                    self.existing_tickets.pop()
            except:
                pass
            return False
    
    def submit_feedback(self, ticket_id: str, matched_ticket_id: Optional[str], 
                       is_correct_duplicate: bool, user_comment: Optional[str] = None,
                       confidence_score: Optional[float] = None) -> str:
        """Submit feedback on duplicate detection results"""
        if not settings.feedback_enabled:
            raise ValueError("Feedback collection is disabled")
        
        return self.feedback_service.submit_feedback(
            ticket_id, matched_ticket_id, is_correct_duplicate, user_comment, confidence_score
        )
    
    def get_language_statistics(self) -> List[LanguageStats]:
        """Get language distribution statistics"""
        language_counts = {}
        language_confidences = {}
        
        for ticket in self.existing_tickets:
            lang = ticket.get('language', 'unknown')
            confidence = ticket.get('language_confidence', 0.0)
            
            language_counts[lang] = language_counts.get(lang, 0) + 1
            if lang not in language_confidences:
                language_confidences[lang] = []
            language_confidences[lang].append(confidence)
        
        stats = []
        for lang_code, count in language_counts.items():
            avg_confidence = sum(language_confidences[lang_code]) / len(language_confidences[lang_code])
            stats.append(LanguageStats(
                language_code=lang_code,
                language_name=self.language_detector.get_language_name(lang_code),
                ticket_count=count,
                avg_confidence=round(avg_confidence, 3)
            ))
        
        return sorted(stats, key=lambda x: x.ticket_count, reverse=True)
    
    def get_clustering_results(self) -> Optional[Dict]:
        """Get current clustering results"""
        return self.cluster_results
    
    def recluster_tickets(self, method: str = 'kmeans') -> Dict:
        """Force re-clustering of all tickets"""
        if not settings.clustering_enabled:
            raise ValueError("Clustering is disabled")
        
        if len(self.existing_tickets) < settings.min_cluster_size:
            raise ValueError(f"Insufficient tickets for clustering (minimum: {settings.min_cluster_size})")
        
        ticket_ids = [t['ticket_id'] for t in self.existing_tickets]
        ticket_texts = [t['text'] for t in self.existing_tickets]
        
        self.cluster_results = self.clustering_service.cluster_tickets(
            self.existing_embeddings, ticket_ids, ticket_texts, method
        )
        
        # Update tickets with new cluster assignments
        cluster_assignments = self.cluster_results.get('cluster_assignments', {})
        for ticket in self.existing_tickets:
            ticket['cluster_id'] = cluster_assignments.get(ticket['ticket_id'])
        
        return self.cluster_results
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        base_stats = {
            "total_tickets": len(self.existing_tickets),
            "model_name": settings.embedding_model_name,
            "similarity_threshold": settings.similarity_threshold,
            "embedding_dimension": self.existing_embeddings.size(1) if self.existing_embeddings.size(0) > 0 else 0
        }
        
        # Language statistics
        base_stats["language_distribution"] = self.get_language_statistics()
        
        # Clustering statistics
        if settings.clustering_enabled and self.cluster_results:
            base_stats["clustering_stats"] = self.cluster_results
        
        # Feedback statistics
        if settings.feedback_enabled:
            base_stats["feedback_stats"] = self.feedback_service.get_feedback_stats()
        
        # Feature status
        base_stats["features_enabled"] = {
            "multilingual_support": settings.auto_detect_language,
            "clustering": settings.clustering_enabled,
            "feedback_collection": settings.feedback_enabled
        }
        
        return base_stats