"""Data loading and preprocessing utilities."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetLoader:
    """Load and preprocess recommendation datasets."""
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        """Initialize dataset loader.
        
        Args:
            data_dir: Directory containing dataset files.
        """
        self.data_dir = Path(data_dir)
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
    def load_interactions(
        self, 
        filename: str = "interactions.csv",
        columns: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Load interaction data from CSV file.
        
        Args:
            filename: Name of the CSV file.
            columns: Column mapping for user_id, item_id, timestamp, weight.
            
        Returns:
            DataFrame with interaction data.
        """
        if columns is None:
            columns = {
                "user_id": "user_id",
                "item_id": "item_id", 
                "timestamp": "timestamp",
                "weight": "weight"
            }
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Generate synthetic data if file doesn't exist
            return self._generate_synthetic_interactions()
        
        df = pd.read_csv(filepath)
        
        # Rename columns if needed
        df = df.rename(columns={v: k for k, v in columns.items()})
        
        return df
    
    def load_items(
        self, 
        filename: str = "items.csv",
        columns: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Load item metadata from CSV file.
        
        Args:
            filename: Name of the CSV file.
            columns: Column mapping for item_id, title, tags, etc.
            
        Returns:
            DataFrame with item metadata.
        """
        if columns is None:
            columns = {
                "item_id": "item_id",
                "title": "title",
                "tags": "tags",
                "category": "category"
            }
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Generate synthetic data if file doesn't exist
            return self._generate_synthetic_items()
        
        df = pd.read_csv(filepath)
        
        # Rename columns if needed
        df = df.rename(columns={v: k for k, v in columns.items()})
        
        return df
    
    def load_users(
        self, 
        filename: str = "users.csv",
        columns: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Load user metadata from CSV file.
        
        Args:
            filename: Name of the CSV file.
            columns: Column mapping for user_id, age, gender, etc.
            
        Returns:
            DataFrame with user metadata.
        """
        if columns is None:
            columns = {
                "user_id": "user_id",
                "age": "age",
                "gender": "gender",
                "location": "location"
            }
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Generate synthetic data if file doesn't exist
            return self._generate_synthetic_users()
        
        df = pd.read_csv(filepath)
        
        # Rename columns if needed
        df = df.rename(columns={v: k for k, v in columns.items()})
        
        return df
    
    def create_interaction_matrix(
        self, 
        interactions: pd.DataFrame,
        normalize: bool = False
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Create user-item interaction matrix.
        
        Args:
            interactions: DataFrame with user_id, item_id, weight columns.
            normalize: Whether to normalize weights to [0, 1].
            
        Returns:
            Tuple of (interaction_matrix, user_ids, item_ids).
        """
        # Encode user and item IDs
        user_ids = self.user_encoder.fit_transform(interactions["user_id"])
        item_ids = self.item_encoder.fit_transform(interactions["item_id"])
        
        # Get unique IDs
        unique_users = self.user_encoder.classes_
        unique_items = self.item_encoder.classes_
        
        # Create matrix
        matrix = np.zeros((len(unique_users), len(unique_items)))
        
        # Fill matrix with weights
        weights = interactions["weight"].values
        if normalize:
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        matrix[user_ids, item_ids] = weights
        
        return matrix, unique_users.tolist(), unique_items.tolist()
    
    def _generate_synthetic_interactions(self) -> pd.DataFrame:
        """Generate synthetic interaction data."""
        np.random.seed(42)
        
        n_users = 1000
        n_items = 500
        n_interactions = 10000
        
        # Generate random interactions
        user_ids = np.random.randint(0, n_users, n_interactions)
        item_ids = np.random.randint(0, n_items, n_interactions)
        
        # Add some temporal patterns
        timestamps = np.random.randint(0, 365 * 24 * 3600, n_interactions)  # 1 year in seconds
        
        # Add popularity bias
        item_popularity = np.random.power(0.5, n_items)
        weights = np.random.beta(2, 5, n_interactions) * item_popularity[item_ids]
        
        df = pd.DataFrame({
            "user_id": [f"user_{uid}" for uid in user_ids],
            "item_id": [f"item_{iid}" for iid in item_ids],
            "timestamp": timestamps,
            "weight": weights
        })
        
        # Save synthetic data
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "interactions.csv", index=False)
        
        return df
    
    def _generate_synthetic_items(self) -> pd.DataFrame:
        """Generate synthetic item metadata."""
        np.random.seed(42)
        
        n_items = 500
        categories = ["electronics", "books", "clothing", "home", "sports"]
        
        items = []
        for i in range(n_items):
            category = np.random.choice(categories)
            items.append({
                "item_id": f"item_{i}",
                "title": f"Item {i}",
                "category": category,
                "price": np.random.uniform(10, 1000),
                "rating": np.random.uniform(1, 5)
            })
        
        df = pd.DataFrame(items)
        
        # Save synthetic data
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "items.csv", index=False)
        
        return df
    
    def _generate_synthetic_users(self) -> pd.DataFrame:
        """Generate synthetic user metadata."""
        np.random.seed(42)
        
        n_users = 1000
        genders = ["M", "F", "Other"]
        locations = ["US", "EU", "Asia", "Other"]
        
        users = []
        for i in range(n_users):
            users.append({
                "user_id": f"user_{i}",
                "age": np.random.randint(18, 80),
                "gender": np.random.choice(genders),
                "location": np.random.choice(locations),
                "signup_date": np.random.randint(0, 365 * 24 * 3600)
            })
        
        df = pd.DataFrame(users)
        
        # Save synthetic data
        self.data_dir.mkdir(exist_ok=True)
        df.to_csv(self.data_dir / "users.csv", index=False)
        
        return df
