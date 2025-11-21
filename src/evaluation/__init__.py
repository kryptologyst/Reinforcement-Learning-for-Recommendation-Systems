"""Evaluation metrics for recommendation systems."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import precision_score, recall_score


class RecommendationMetrics:
    """Collection of recommendation evaluation metrics."""
    
    def __init__(self, k_values: List[int] = None):
        """Initialize metrics calculator.
        
        Args:
            k_values: List of k values for top-k metrics.
        """
        if k_values is None:
            k_values = [1, 5, 10, 20]
        self.k_values = k_values
    
    def precision_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate Precision@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K value.
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / k
    
    def recall_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate Recall@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K value.
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        relevant_in_top_k = len(set(top_k_recs) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
    
    def ndcg_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K value.
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def map_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate MAP@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K value.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        
        # Calculate average precision
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recs):
            if item in relevant_items:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(relevant_items)
    
    def hit_rate_at_k(
        self, 
        recommendations: List[int], 
        relevant_items: List[int], 
        k: int
    ) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant items.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K value (0 or 1).
        """
        if k == 0:
            return 0.0
        
        top_k_recs = recommendations[:k]
        return 1.0 if len(set(top_k_recs) & set(relevant_items)) > 0 else 0.0
    
    def coverage(
        self, 
        all_recommendations: List[List[int]], 
        total_items: int
    ) -> float:
        """Calculate catalog coverage.
        
        Args:
            all_recommendations: List of recommendation lists for all users.
            total_items: Total number of items in catalog.
            
        Returns:
            Coverage value.
        """
        if total_items == 0:
            return 0.0
        
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items
    
    def diversity(
        self, 
        recommendations: List[int], 
        item_features: Optional[np.ndarray] = None
    ) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommendations: List of recommended item IDs.
            item_features: Item feature matrix (optional).
            
        Returns:
            Diversity value.
        """
        if len(recommendations) <= 1:
            return 0.0
        
        if item_features is not None:
            # Use feature-based diversity
            rec_features = item_features[recommendations]
            similarities = np.dot(rec_features, rec_features.T)
            np.fill_diagonal(similarities, 0)
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity
        else:
            # Use simple diversity (inverse of average pairwise similarity)
            # For simplicity, assume items are diverse if they're different
            return 1.0 - (len(set(recommendations)) / len(recommendations))
    
    def novelty(
        self, 
        recommendations: List[int], 
        item_popularity: np.ndarray
    ) -> float:
        """Calculate novelty of recommendations.
        
        Args:
            recommendations: List of recommended item IDs.
            item_popularity: Array of item popularity scores.
            
        Returns:
            Novelty value.
        """
        if len(recommendations) == 0:
            return 0.0
        
        rec_popularity = item_popularity[recommendations]
        # Novelty is inverse of popularity
        return 1.0 - np.mean(rec_popularity)
    
    def evaluate_user(
        self, 
        recommendations: List[int], 
        relevant_items: List[int],
        item_features: Optional[np.ndarray] = None,
        item_popularity: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate recommendations for a single user.
        
        Args:
            recommendations: List of recommended item IDs.
            relevant_items: List of relevant items.
            item_features: Item feature matrix (optional).
            item_popularity: Item popularity scores (optional).
            
        Returns:
            Dictionary of metric values.
        """
        metrics = {}
        
        for k in self.k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(recommendations, relevant_items, k)
            metrics[f"recall@{k}"] = self.recall_at_k(recommendations, relevant_items, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(recommendations, relevant_items, k)
            metrics[f"map@{k}"] = self.map_at_k(recommendations, relevant_items, k)
            metrics[f"hit_rate@{k}"] = self.hit_rate_at_k(recommendations, relevant_items, k)
        
        # Additional metrics
        metrics["diversity"] = self.diversity(recommendations, item_features)
        
        if item_popularity is not None:
            metrics["novelty"] = self.novelty(recommendations, item_popularity)
        
        return metrics
    
    def evaluate_system(
        self, 
        all_recommendations: List[List[int]], 
        all_relevant_items: List[List[int]],
        total_items: int,
        item_features: Optional[np.ndarray] = None,
        item_popularity: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate recommendations for the entire system.
        
        Args:
            all_recommendations: List of recommendation lists for all users.
            all_relevant_items: List of relevant item lists for all users.
            total_items: Total number of items in catalog.
            item_features: Item feature matrix (optional).
            item_popularity: Item popularity scores (optional).
            
        Returns:
            Dictionary of average metric values.
        """
        # Calculate per-user metrics
        user_metrics = []
        for recs, relevant in zip(all_recommendations, all_relevant_items):
            user_metric = self.evaluate_user(recs, relevant, item_features, item_popularity)
            user_metrics.append(user_metric)
        
        # Average across users
        avg_metrics = {}
        for metric_name in user_metrics[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in user_metrics])
        
        # System-level metrics
        avg_metrics["coverage"] = self.coverage(all_recommendations, total_items)
        
        return avg_metrics


class RLRecommenderEvaluator:
    """Evaluator for RL-based recommendation systems."""
    
    def __init__(self, metrics: RecommendationMetrics):
        """Initialize evaluator.
        
        Args:
            metrics: Metrics calculator instance.
        """
        self.metrics = metrics
    
    def evaluate_online(
        self, 
        recommender, 
        test_interactions: np.ndarray,
        n_episodes: int = 100,
        k: int = 10
    ) -> Dict[str, float]:
        """Evaluate RL recommender in online setting.
        
        Args:
            recommender: RL recommender instance.
            test_interactions: Test interaction matrix.
            n_episodes: Number of evaluation episodes.
            k: Number of recommendations per episode.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        n_users, n_items = test_interactions.shape
        
        episode_rewards = []
        episode_precisions = []
        episode_recalls = []
        
        for episode in range(n_episodes):
            episode_reward = 0
            episode_precision = 0
            episode_recall = 0
            
            for user_id in range(n_users):
                # Get recommendations
                recommendations = recommender.recommend(user_id, k)
                
                # Simulate user feedback
                relevant_items = np.where(test_interactions[user_id] == 1)[0].tolist()
                
                # Calculate metrics
                precision = self.metrics.precision_at_k(recommendations, relevant_items, k)
                recall = self.metrics.recall_at_k(recommendations, relevant_items, k)
                
                episode_precision += precision
                episode_recall += recall
                
                # Simulate reward (could be based on clicks, purchases, etc.)
                reward = len(set(recommendations) & set(relevant_items)) / k
                episode_reward += reward
                
                # Update recommender
                if recommendations:
                    item_id = recommendations[0]  # Assume user interacts with first item
                    recommender.update(user_id, item_id, reward)
            
            episode_rewards.append(episode_reward / n_users)
            episode_precisions.append(episode_precision / n_users)
            episode_recalls.append(episode_recall / n_users)
        
        return {
            "avg_reward": np.mean(episode_rewards),
            "avg_precision": np.mean(episode_precisions),
            "avg_recall": np.mean(episode_recalls),
            "reward_std": np.std(episode_rewards),
            "precision_std": np.std(episode_precisions),
            "recall_std": np.std(episode_recalls)
        }
    
    def evaluate_offline(
        self, 
        recommender, 
        test_interactions: np.ndarray,
        k: int = 10
    ) -> Dict[str, float]:
        """Evaluate RL recommender in offline setting.
        
        Args:
            recommender: RL recommender instance.
            test_interactions: Test interaction matrix.
            k: Number of recommendations.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        n_users, n_items = test_interactions.shape
        
        all_recommendations = []
        all_relevant_items = []
        
        for user_id in range(n_users):
            recommendations = recommender.recommend(user_id, k)
            relevant_items = np.where(test_interactions[user_id] == 1)[0].tolist()
            
            all_recommendations.append(recommendations)
            all_relevant_items.append(relevant_items)
        
        return self.metrics.evaluate_system(
            all_recommendations, 
            all_relevant_items, 
            n_items
        )
