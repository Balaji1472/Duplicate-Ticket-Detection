"""
Feedback service for active learning and model improvement.
Collects user feedback on duplicate detection accuracy.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FeedbackService:
    """Service for collecting and managing user feedback on duplicate detection"""
    
    def __init__(self, feedback_storage_path: str = "data/feedback.json"):
        self.feedback_storage_path = feedback_storage_path
        self.feedback_data = []
        self._ensure_storage_directory()
        self._load_existing_feedback()
    
    def _ensure_storage_directory(self):
        """Ensure the feedback storage directory exists"""
        storage_dir = Path(self.feedback_storage_path).parent
        storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_existing_feedback(self):
        """Load existing feedback from storage"""
        try:
            if os.path.exists(self.feedback_storage_path):
                with open(self.feedback_storage_path, 'r', encoding='utf-8') as f:
                    self.feedback_data = json.load(f)
                logger.info(f"Loaded {len(self.feedback_data)} existing feedback entries")
            else:
                self.feedback_data = []
                logger.info("No existing feedback file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            self.feedback_data = []
    
    def _save_feedback(self):
        """Save feedback data to storage"""
        try:
            with open(self.feedback_storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved feedback data to {self.feedback_storage_path}")
        except Exception as e:
            logger.error(f"Error saving feedback data: {e}")
            raise
    
    def submit_feedback(self, ticket_id: str, matched_ticket_id: Optional[str], 
                       is_correct_duplicate: bool, user_comment: Optional[str] = None,
                       confidence_score: Optional[float] = None) -> str:
        """
        Submit user feedback on duplicate detection results.
        
        Args:
            ticket_id: ID of the ticket that was checked
            matched_ticket_id: ID of the ticket it was matched with (if any)
            is_correct_duplicate: Whether the duplicate detection was correct
            user_comment: Optional user comment
            confidence_score: User's confidence in their feedback (0.0-1.0)
            
        Returns:
            Unique feedback ID
        """
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        feedback_entry = {
            'feedback_id': feedback_id,
            'timestamp': timestamp,
            'ticket_id': ticket_id,
            'matched_ticket_id': matched_ticket_id,
            'is_correct_duplicate': is_correct_duplicate,
            'user_comment': user_comment,
            'confidence_score': confidence_score
        }
        
        self.feedback_data.append(feedback_entry)
        self._save_feedback()
        
        logger.info(f"Received feedback for ticket {ticket_id}: {'correct' if is_correct_duplicate else 'incorrect'}")
        return feedback_id
    
    def get_feedback_stats(self) -> Dict:
        """Get statistics about collected feedback"""
        if not self.feedback_data:
            return {
                'total_feedback': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'accuracy': 0.0,
                'avg_user_confidence': 0.0
            }
        
        total = len(self.feedback_data)
        correct = sum(1 for f in self.feedback_data if f['is_correct_duplicate'])
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate average user confidence (only for entries with confidence scores)
        confidence_scores = [f['confidence_score'] for f in self.feedback_data 
                           if f.get('confidence_score') is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_feedback': total,
            'correct_predictions': correct,
            'incorrect_predictions': incorrect,
            'accuracy': round(accuracy, 3),
            'avg_user_confidence': round(avg_confidence, 3)
        }
    
    def get_problematic_cases(self, limit: int = 10) -> List[Dict]:
        """
        Get feedback entries where the model made incorrect predictions.
        Useful for identifying patterns in model failures.
        
        Args:
            limit: Maximum number of cases to return
            
        Returns:
            List of problematic feedback entries
        """
        incorrect_cases = [
            f for f in self.feedback_data 
            if not f['is_correct_duplicate']
        ]
        
        # Sort by timestamp (most recent first)
        incorrect_cases.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return incorrect_cases[:limit]
    
    def get_feedback_trends(self, days: int = 30) -> Dict:
        """
        Analyze feedback trends over the specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_feedback = [
            f for f in self.feedback_data
            if datetime.fromisoformat(f['timestamp']) >= cutoff_date
        ]
        
        if not recent_feedback:
            return {'message': f'No feedback received in the last {days} days'}
        
        # Group by day
        daily_stats = {}
        for feedback in recent_feedback:
            date_str = feedback['timestamp'][:10]  # YYYY-MM-DD
            if date_str not in daily_stats:
                daily_stats[date_str] = {'total': 0, 'correct': 0}
            
            daily_stats[date_str]['total'] += 1
            if feedback['is_correct_duplicate']:
                daily_stats[date_str]['correct'] += 1
        
        # Calculate daily accuracy
        for date_str in daily_stats:
            stats = daily_stats[date_str]
            stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        return {
            'period_days': days,
            'total_feedback': len(recent_feedback),
            'daily_breakdown': daily_stats,
            'overall_accuracy': sum(f['is_correct_duplicate'] for f in recent_feedback) / len(recent_feedback)
        }
    
    def export_feedback(self, format: str = 'json') -> str:
        """
        Export feedback data in the specified format.
        
        Args:
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported data as string
        """
        if format.lower() == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = ['feedback_id', 'timestamp', 'ticket_id', 'matched_ticket_id', 
                         'is_correct_duplicate', 'user_comment', 'confidence_score']
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.feedback_data)
            
            return output.getvalue()
        else:
            return json.dumps(self.feedback_data, indent=2, ensure_ascii=False)
    
    def get_threshold_recommendations(self) -> Dict:
        """
        Analyze feedback to suggest optimal similarity thresholds.
        
        Returns:
            Dictionary with threshold recommendations
        """
        if len(self.feedback_data) < 10:
            return {'message': 'Insufficient feedback data for threshold analysis'}
        
        # This is a simplified analysis - in practice, you'd need the similarity scores
        # from the original predictions to make better recommendations
        correct_duplicate_feedback = [f for f in self.feedback_data if f['is_correct_duplicate']]
        incorrect_feedback = [f for f in self.feedback_data if not f['is_correct_duplicate']]
        
        accuracy = len(correct_duplicate_feedback) / len(self.feedback_data)
        
        recommendation = "maintain current threshold"
        if accuracy < 0.7:
            recommendation = "consider increasing threshold (reduce false positives)"
        elif accuracy > 0.95:
            recommendation = "consider decreasing threshold (catch more duplicates)"
        
        return {
            'current_accuracy': round(accuracy, 3),
            'total_samples': len(self.feedback_data),
            'recommendation': recommendation,
            'confidence': 'low' if len(self.feedback_data) < 50 else 'medium' if len(self.feedback_data) < 200 else 'high'
        }
    
    def clear_old_feedback(self, days: int = 365):
        """
        Remove feedback older than specified days to manage storage.
        
        Args:
            days: Age threshold in days
        """
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        original_count = len(self.feedback_data)
        
        self.feedback_data = [
            f for f in self.feedback_data
            if datetime.fromisoformat(f['timestamp']) >= cutoff_date
        ]
        
        removed_count = original_count - len(self.feedback_data)
        if removed_count > 0:
            self._save_feedback()
            logger.info(f"Removed {removed_count} old feedback entries (older than {days} days)")
        
        return removed_count