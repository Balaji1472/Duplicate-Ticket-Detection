"""
Language detection module for multilingual ticket processing.
Uses langdetect library for automatic language identification.
"""

from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import logging
from typing import Tuple, Optional, Dict

# Set seed for reproducible results
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Language detection service for ticket text"""
    
    # Language code to name mapping
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'ta': 'Tamil'
    }
    
    def __init__(self):
        self.confidence_threshold = 0.7
        
    def detect_language(self, text: str) -> Tuple[Optional[str], float]:
        """
        Detect the language of the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        if not text or len(text.strip()) < 3:
            return None, 0.0
            
        try:
            # Get language probabilities
            language_probs = detect_langs(text)
            
            if language_probs:
                best_lang = language_probs[0]
                language_code = best_lang.lang
                confidence = best_lang.prob
                
                # Map zh-cn/zh-tw to zh for consistency
                if language_code in ['zh-cn', 'zh-tw']:
                    language_code = 'zh'
                
                logger.debug(f"Detected language: {language_code} (confidence: {confidence:.3f})")
                return language_code, confidence
            else:
                return None, 0.0
                
        except LangDetectException as e:
            logger.warning(f"Language detection failed for text: {text[:50]}... Error: {e}")
            return None, 0.0
        except Exception as e:
            logger.error(f"Unexpected error in language detection: {e}")
            return None, 0.0
    
    def is_supported_language(self, language_code: str, supported_languages: list) -> bool:
        """Check if the detected language is in the supported languages list"""
        return language_code in supported_languages if language_code else False
    
    def get_language_name(self, language_code: str) -> str:
        """Get the human-readable name for a language code"""
        return self.LANGUAGE_NAMES.get(language_code, f"Unknown ({language_code})")
    
    def get_language_stats(self, tickets: list) -> Dict[str, Dict]:
        """
        Analyze language distribution across a collection of tickets.
        
        Args:
            tickets: List of ticket dictionaries with 'text' field
            
        Returns:
            Dictionary with language statistics
        """
        language_counts = {}
        language_confidences = {}
        total_tickets = len(tickets)
        
        for ticket in tickets:
            text = ticket.get('text', '')
            lang_code, confidence = self.detect_language(text)
            
            if lang_code:
                # Count occurrences
                language_counts[lang_code] = language_counts.get(lang_code, 0) + 1
                
                # Track confidence scores
                if lang_code not in language_confidences:
                    language_confidences[lang_code] = []
                language_confidences[lang_code].append(confidence)
        
        # Calculate statistics
        stats = {}
        for lang_code, count in language_counts.items():
            avg_confidence = sum(language_confidences[lang_code]) / len(language_confidences[lang_code])
            stats[lang_code] = {
                'language_name': self.get_language_name(lang_code),
                'count': count,
                'percentage': (count / total_tickets) * 100,
                'avg_confidence': round(avg_confidence, 3)
            }
        
        return stats
    
    def detect_with_fallback(self, text: str, default_language: str = 'en') -> Tuple[str, float]:
        """
        Detect language with fallback to default if detection fails or confidence is low.
        
        Args:
            text: Input text
            default_language: Fallback language code
            
        Returns:
            Tuple of (language_code, confidence_score)
        """
        lang_code, confidence = self.detect_language(text)
        
        if not lang_code or confidence < self.confidence_threshold:
            logger.info(f"Using fallback language '{default_language}' for low-confidence detection")
            return default_language, 0.5  # Moderate confidence for fallback
        
        return lang_code, confidence
