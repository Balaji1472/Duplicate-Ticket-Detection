
"""
Configuration settings for the duplicate ticket detection system.
Enhanced with multilingual support and new bonus features.
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Model configuration
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    similarity_threshold: float = 0.8
    
    # Multilingual settings
    supported_languages: list = ["en", "es", "fr", "de", "it", "pt", "hi", "zh"]
    auto_detect_language: bool = True
    
    # Clustering settings
    clustering_enabled: bool = True
    min_cluster_size: int = 3
    max_clusters: int = 10
    
    # Active learning settings
    feedback_enabled: bool = True
    feedback_storage_path: str = "data/feedback.json"
    
    # Data paths
    data_path: str = "data/sample_tickets.json"
    
    # API configuration
    api_title: str = "Advanced Duplicate Ticket Detection API"
    api_description: str = "Multilingual API for detecting duplicate support tickets with clustering and active learning"
    api_version: str = "2.0.0"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Performance settings
    batch_size: int = 32
    max_batch_size: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()