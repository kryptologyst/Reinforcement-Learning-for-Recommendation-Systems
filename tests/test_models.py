"""Tests for RL recommendation systems."""

import numpy as np
import pytest
import torch

from src.models import (
    QLearningRecommender,
    SARSARecommender,
    DQNRecommender,
    ContextualBanditRecommender
)
from src.evaluation import RecommendationMetrics
from src.utils import set_seed, create_user_item_matrix, split_interactions


class TestRLRecommenders:
    """Test cases for RL recommenders."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.n_users = 10
        self.n_items = 20
        self.k = 5
    
    def test_q_learning_recommender(self):
        """Test Q-Learning recommender."""
        recommender = QLearningRecommender(
            self.n_users, 
            self.n_items,
            learning_rate=0.1,
            epsilon=0.1
        )
        
        # Test recommendation
        recommendations = recommender.recommend(0, self.k)
        assert len(recommendations) == self.k
        assert all(0 <= rec < self.n_items for rec in recommendations)
        
        # Test update
        initial_q = recommender.q_table[0, 0]
        recommender.update(0, 0, 1.0)
        assert recommender.q_table[0, 0] != initial_q
    
    def test_sarsa_recommender(self):
        """Test SARSA recommender."""
        recommender = SARSARecommender(
            self.n_users,
            self.n_items,
            learning_rate=0.1,
            epsilon=0.1
        )
        
        # Test recommendation
        recommendations = recommender.recommend(0, self.k)
        assert len(recommendations) == self.k
        assert all(0 <= rec < self.n_items for rec in recommendations)
        
        # Test update
        initial_q = recommender.q_table[0, 0]
        recommender.update(0, 0, 1.0)
        assert recommender.q_table[0, 0] != initial_q
    
    def test_dqn_recommender(self):
        """Test DQN recommender."""
        recommender = DQNRecommender(
            self.n_users,
            self.n_items,
            embedding_dim=32,
            hidden_dim=64
        )
        
        # Test recommendation
        recommendations = recommender.recommend(0, self.k)
        assert len(recommendations) == self.k
        assert all(0 <= rec < self.n_items for rec in recommendations)
        
        # Test update
        recommender.update(0, 0, 1.0)
        assert len(recommender.replay_buffer) == 1
    
    def test_contextual_bandit_recommender(self):
        """Test Contextual Bandit recommender."""
        recommender = ContextualBanditRecommender(
            self.n_users,
            self.n_items,
            context_dim=5
        )
        
        # Test recommendation
        recommendations = recommender.recommend(0, self.k)
        assert len(recommendations) == self.k
        assert all(0 <= rec < self.n_items for rec in recommendations)
        
        # Test update
        recommender.update(0, 0, 1.0)
        assert recommender.A.shape == (5, 5)  # context_dim x context_dim


class TestRecommendationMetrics:
    """Test cases for recommendation metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = RecommendationMetrics(k_values=[1, 5, 10])
        self.recommendations = [0, 1, 2, 3, 4]
        self.relevant_items = [0, 2, 4, 6, 8]
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        precision = self.metrics.precision_at_k(self.recommendations, self.relevant_items, 5)
        expected = 3 / 5  # 3 relevant items in top 5
        assert abs(precision - expected) < 1e-6
    
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recall = self.metrics.recall_at_k(self.recommendations, self.relevant_items, 5)
        expected = 3 / 5  # 3 relevant items found out of 5 total
        assert abs(recall - expected) < 1e-6
    
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        ndcg = self.metrics.ndcg_at_k(self.recommendations, self.relevant_items, 5)
        assert 0 <= ndcg <= 1
    
    def test_map_at_k(self):
        """Test MAP@k calculation."""
        map_score = self.metrics.map_at_k(self.recommendations, self.relevant_items, 5)
        assert 0 <= map_score <= 1
    
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        hit_rate = self.metrics.hit_rate_at_k(self.recommendations, self.relevant_items, 5)
        assert hit_rate == 1.0  # At least one relevant item found
    
    def test_coverage(self):
        """Test coverage calculation."""
        all_recommendations = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4]
        ]
        total_items = 10
        coverage = self.metrics.coverage(all_recommendations, total_items)
        expected = 5 / 10  # 5 unique items out of 10 total
        assert abs(coverage - expected) < 1e-6
    
    def test_diversity(self):
        """Test diversity calculation."""
        diversity = self.metrics.diversity(self.recommendations)
        assert 0 <= diversity <= 1
    
    def test_novelty(self):
        """Test novelty calculation."""
        item_popularity = np.random.random(10)
        novelty = self.metrics.novelty(self.recommendations, item_popularity)
        assert 0 <= novelty <= 1


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        random1 = np.random.random()
        
        set_seed(42)
        random2 = np.random.random()
        
        assert abs(random1 - random2) < 1e-10
    
    def test_create_user_item_matrix(self):
        """Test user-item matrix creation."""
        matrix = create_user_item_matrix(10, 20, sparsity=0.5, seed=42)
        
        assert matrix.shape == (10, 20)
        assert matrix.dtype == np.int64
        assert np.all((matrix == 0) | (matrix == 1))
    
    def test_split_interactions(self):
        """Test interaction splitting."""
        interactions = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0]
        ])
        
        train, val, test = split_interactions(interactions, test_ratio=0.3, val_ratio=0.2, seed=42)
        
        # Check shapes
        assert train.shape == interactions.shape
        assert val.shape == interactions.shape
        assert test.shape == interactions.shape
        
        # Check that all interactions are preserved
        total_interactions = np.sum(interactions)
        split_interactions_total = np.sum(train) + np.sum(val) + np.sum(test)
        assert total_interactions == split_interactions_total


if __name__ == "__main__":
    pytest.main([__file__])
