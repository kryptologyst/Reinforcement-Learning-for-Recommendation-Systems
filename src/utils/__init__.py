"""Core utilities for the RL recommendation system."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_user_item_matrix(
    n_users: int, 
    n_items: int, 
    sparsity: float = 0.8,
    seed: Optional[int] = None
) -> np.ndarray:
    """Create a synthetic user-item interaction matrix.
    
    Args:
        n_users: Number of users.
        n_items: Number of items.
        sparsity: Fraction of missing interactions (0.0 = dense, 1.0 = sparse).
        seed: Random seed for reproducibility.
        
    Returns:
        Binary interaction matrix of shape (n_users, n_items).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create random interactions
    interactions = np.random.random((n_users, n_items))
    
    # Apply sparsity by masking interactions
    mask = np.random.random((n_users, n_items)) > sparsity
    
    # Convert to binary (1 = interaction, 0 = no interaction)
    matrix = (interactions > 0.5).astype(int)
    matrix = matrix * mask
    
    return matrix


def split_interactions(
    interactions: np.ndarray,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split interaction matrix into train/validation/test sets.
    
    Args:
        interactions: Binary interaction matrix.
        test_ratio: Fraction of interactions for test set.
        val_ratio: Fraction of interactions for validation set.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_matrix, val_matrix, test_matrix).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Find all positive interactions
    user_indices, item_indices = np.where(interactions == 1)
    n_interactions = len(user_indices)
    
    # Shuffle indices
    indices = np.random.permutation(n_interactions)
    
    # Calculate split sizes
    n_test = int(n_interactions * test_ratio)
    n_val = int(n_interactions * val_ratio)
    n_train = n_interactions - n_test - n_val
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create matrices
    train_matrix = np.zeros_like(interactions)
    val_matrix = np.zeros_like(interactions)
    test_matrix = np.zeros_like(interactions)
    
    # Fill matrices
    train_matrix[user_indices[train_indices], item_indices[train_indices]] = 1
    val_matrix[user_indices[val_indices], item_indices[val_indices]] = 1
    test_matrix[user_indices[test_indices], item_indices[test_indices]] = 1
    
    return train_matrix, val_matrix, test_matrix


def calculate_popularity_bias(interactions: np.ndarray) -> Dict[str, float]:
    """Calculate popularity bias metrics.
    
    Args:
        interactions: Binary interaction matrix.
        
    Returns:
        Dictionary containing popularity bias metrics.
    """
    item_popularity = np.sum(interactions, axis=0)
    user_activity = np.sum(interactions, axis=1)
    
    return {
        "item_popularity_gini": _calculate_gini(item_popularity),
        "user_activity_gini": _calculate_gini(user_activity),
        "avg_item_popularity": np.mean(item_popularity),
        "avg_user_activity": np.mean(user_activity),
    }


def _calculate_gini(values: np.ndarray) -> float:
    """Calculate Gini coefficient for measuring inequality."""
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)
    
    if cumsum[-1] == 0:
        return 0.0
    
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
