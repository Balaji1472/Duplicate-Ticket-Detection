"""
Clustering service for grouping similar tickets.
Implements KMeans and AgglomerativeClustering for ticket analysis.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import logging
from typing import List, Dict, Tuple, Optional
import json

logger = logging.getLogger(__name__)

class TicketClusteringService:
    """Service for clustering tickets based on semantic similarity"""
    
    def __init__(self, min_cluster_size: int = 3, max_clusters: int = 10):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.scaler = StandardScaler()
        self.current_clusters = None
        self.cluster_centers = None
        self.cluster_model = None
        
    def find_optimal_clusters(self, embeddings: torch.Tensor) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            embeddings: Tensor of ticket embeddings
            
        Returns:
            Optimal number of clusters
        """
        if len(embeddings) < self.min_cluster_size:
            return 1
            
        embeddings_np = embeddings.numpy()
        max_k = min(self.max_clusters, len(embeddings) // 2)
        
        if max_k < 2:
            return 1
        
        silhouette_scores = []
        inertias = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings_np)
                
                # Calculate silhouette score
                silhouette_avg = silhouette_score(embeddings_np, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                inertias.append(kmeans.inertia_)
                
                logger.debug(f"K={k}: Silhouette={silhouette_avg:.3f}, Inertia={kmeans.inertia_:.2f}")
                
            except Exception as e:
                logger.warning(f"Clustering failed for k={k}: {e}")
                continue
        
        if not silhouette_scores:
            return 1
        
        # Choose k with highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        return optimal_k
    
    def cluster_tickets(self, embeddings: torch.Tensor, ticket_ids: List[str], 
                       ticket_texts: List[str], method: str = 'kmeans') -> Dict:
        """
        Cluster tickets using the specified method.
        
        Args:
            embeddings: Tensor of ticket embeddings
            ticket_ids: List of ticket IDs
            ticket_texts: List of ticket texts
            method: Clustering method ('kmeans' or 'agglomerative')
            
        Returns:
            Dictionary with clustering results
        """
        if len(embeddings) < self.min_cluster_size:
            logger.info(f"Too few tickets ({len(embeddings)}) for clustering. Minimum required: {self.min_cluster_size}")
            return self._create_single_cluster_result(ticket_ids, ticket_texts)
        
        try:
            embeddings_np = embeddings.numpy()
            
            # Find optimal number of clusters
            optimal_k = self.find_optimal_clusters(embeddings)
            
            if optimal_k == 1:
                return self._create_single_cluster_result(ticket_ids, ticket_texts)
            
            # Perform clustering
            if method == 'kmeans':
                self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            else:  # agglomerative
                self.cluster_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
            
            cluster_labels = self.cluster_model.fit_predict(embeddings_np)
            
            # Store cluster information
            self.current_clusters = cluster_labels
            if hasattr(self.cluster_model, 'cluster_centers_'):
                self.cluster_centers = self.cluster_model.cluster_centers_
            
            # Generate cluster results
            results = self._generate_cluster_results(
                cluster_labels, ticket_ids, ticket_texts, embeddings_np
            )
            
            logger.info(f"Successfully clustered {len(ticket_ids)} tickets into {optimal_k} clusters")
            return results
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self._create_single_cluster_result(ticket_ids, ticket_texts)
    
    def _create_single_cluster_result(self, ticket_ids: List[str], ticket_texts: List[str]) -> Dict:
        """Create result for single cluster (fallback)"""
        return {
            'total_clusters': 1,
            'clusters': [{
                'cluster_id': 0,
                'ticket_count': len(ticket_ids),
                'representative_text': ticket_texts[0] if ticket_texts else "No tickets",
                'avg_similarity': 1.0,
                'ticket_ids': ticket_ids
            }],
            'unclustered_tickets': 0,
            'cluster_assignments': {tid: 0 for tid in ticket_ids}
        }
    
    def _generate_cluster_results(self, cluster_labels: np.ndarray, ticket_ids: List[str], 
                                 ticket_texts: List[str], embeddings: np.ndarray) -> Dict:
        """Generate detailed clustering results"""
        unique_labels = np.unique(cluster_labels)
        clusters = []
        cluster_assignments = {}
        
        for cluster_id in unique_labels:
            # Get tickets in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_ticket_ids = [ticket_ids[i] for i in range(len(ticket_ids)) if cluster_mask[i]]
            cluster_texts = [ticket_texts[i] for i in range(len(ticket_texts)) if cluster_mask[i]]
            cluster_embeddings = embeddings[cluster_mask]
            
            # Calculate average similarity within cluster
            if len(cluster_embeddings) > 1:
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(cluster_embeddings)
                # Get upper triangle (excluding diagonal)
                upper_tri = np.triu(similarities, k=1)
                avg_similarity = np.mean(upper_tri[upper_tri > 0])
            else:
                avg_similarity = 1.0
            
            # Find representative text (closest to centroid)
            if len(cluster_embeddings) > 1:
                centroid = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                representative_idx = np.argmin(distances)
                representative_text = cluster_texts[representative_idx]
            else:
                representative_text = cluster_texts[0] if cluster_texts else "Empty cluster"
            
            clusters.append({
                'cluster_id': int(cluster_id),
                'ticket_count': len(cluster_ticket_ids),
                'representative_text': representative_text,
                'avg_similarity': round(float(avg_similarity), 4),
                'ticket_ids': cluster_ticket_ids
            })
            
            # Store assignments
            for tid in cluster_ticket_ids:
                cluster_assignments[tid] = int(cluster_id)
        
        return {
            'total_clusters': len(clusters),
            'clusters': clusters,
            'unclustered_tickets': 0,  # All tickets are assigned to clusters
            'cluster_assignments': cluster_assignments
        }
    
    def get_cluster_for_new_ticket(self, new_embedding: torch.Tensor) -> Optional[int]:
        """
        Assign a new ticket to the most appropriate existing cluster.
        
        Args:
            new_embedding: Embedding of the new ticket
            
        Returns:
            Cluster ID or None if no good match
        """
        if self.cluster_centers is None or self.cluster_model is None:
            return None
        
        try:
            new_embedding_np = new_embedding.numpy().reshape(1, -1)
            
            if hasattr(self.cluster_model, 'predict'):
                # For KMeans
                cluster_id = self.cluster_model.predict(new_embedding_np)[0]
                return int(cluster_id)
            else:
                # For AgglomerativeClustering, find closest centroid
                distances = np.linalg.norm(self.cluster_centers - new_embedding_np, axis=1)
                closest_cluster = np.argmin(distances)
                return int(closest_cluster)
                
        except Exception as e:
            logger.error(f"Error assigning ticket to cluster: {e}")
            return None
    
    def get_cluster_summary(self) -> Dict:
        """Get summary of current clustering state"""
        if self.current_clusters is None:
            return {'status': 'No clustering performed yet'}
        
        unique_clusters = np.unique(self.current_clusters)
        cluster_sizes = [np.sum(self.current_clusters == c) for c in unique_clusters]
        
        return {
            'total_clusters': len(unique_clusters),
            'cluster_sizes': cluster_sizes,
            'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0
        }