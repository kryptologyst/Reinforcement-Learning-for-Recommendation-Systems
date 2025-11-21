"""Evaluation script for RL recommendation systems."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from src.data import DatasetLoader
from src.evaluation import RecommendationMetrics, RLRecommenderEvaluator
from src.models import (
    QLearningRecommender,
    SARSARecommender,
    DQNRecommender,
    ContextualBanditRecommender
)
from src.utils import set_seed


def load_model(model_path: Path, model_type: str, n_users: int, n_items: int):
    """Load a trained model from file.
    
    Args:
        model_path: Path to model file.
        model_type: Type of model.
        n_users: Number of users.
        n_items: Number of items.
        
    Returns:
        Loaded model instance.
    """
    if model_type == "q_learning":
        model = QLearningRecommender(n_users, n_items)
        if model_path.exists():
            model.q_table = np.load(model_path)
        return model
    elif model_type == "sarsa":
        model = SARSARecommender(n_users, n_items)
        if model_path.exists():
            model.q_table = np.load(model_path)
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate RL recommendation models")
    parser.add_argument("--models_dir", type=str, default="outputs",
                       help="Directory containing trained models")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing test data")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                       help="Output file for results")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of recommendations")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Load test data
    loader = DatasetLoader(args.data_dir)
    interactions_df = loader.load_interactions()
    interactions_matrix, user_ids, item_ids = loader.create_interaction_matrix(interactions_df)
    
    # Load models
    models_dir = Path(args.models_dir)
    models = {}
    
    # Look for Q-table files
    q_table_files = list(models_dir.glob("*_q_table.npy"))
    
    for q_table_file in q_table_files:
        model_name = q_table_file.stem.replace("_q_table", "")
        
        # Determine model type from name
        if "q_learning" in model_name:
            model_type = "q_learning"
        elif "sarsa" in model_name:
            model_type = "sarsa"
        else:
            continue
        
        model = load_model(q_table_file, model_type, len(user_ids), len(item_ids))
        models[model_name] = model
    
    if not models:
        print("No trained models found!")
        return
    
    # Evaluate models
    metrics = RecommendationMetrics(k_values=[args.k])
    evaluator = RLRecommenderEvaluator(metrics)
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        
        # Offline evaluation
        offline_metrics = evaluator.evaluate_offline(model, interactions_matrix, args.k)
        
        # Online evaluation
        online_metrics = evaluator.evaluate_online(
            model, interactions_matrix, 
            n_episodes=50, k=args.k
        )
        
        results[model_name] = {
            **offline_metrics,
            **online_metrics
        }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\nEvaluation Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
