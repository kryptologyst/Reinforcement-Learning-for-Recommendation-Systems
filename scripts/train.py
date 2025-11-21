"""Main training script for RL recommendation systems."""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.data import DatasetLoader
from src.evaluation import RecommendationMetrics, RLRecommenderEvaluator
from src.models import (
    QLearningRecommender,
    SARSARecommender,
    DQNRecommender,
    ContextualBanditRecommender
)
from src.utils import set_seed, split_interactions


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_recommender(model_type: str, config: Dict) -> object:
    """Create recommender instance based on model type.
    
    Args:
        model_type: Type of recommender model.
        config: Model configuration.
        
    Returns:
        Recommender instance.
    """
    if model_type == "q_learning":
        return QLearningRecommender(**config)
    elif model_type == "sarsa":
        return SARSARecommender(**config)
    elif model_type == "dqn":
        return DQNRecommender(**config)
    elif model_type == "contextual_bandit":
        return ContextualBanditRecommender(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_recommender(
    recommender: object,
    train_interactions: np.ndarray,
    val_interactions: np.ndarray,
    config: Dict
) -> Dict:
    """Train RL recommender.
    
    Args:
        recommender: Recommender instance.
        train_interactions: Training interaction matrix.
        val_interactions: Validation interaction matrix.
        config: Training configuration.
        
    Returns:
        Training history.
    """
    n_users, n_items = train_interactions.shape
    n_episodes = config.get("n_episodes", 1000)
    k = config.get("k", 10)
    eval_freq = config.get("eval_freq", 100)
    
    history = {
        "episode": [],
        "train_reward": [],
        "val_precision": [],
        "val_recall": [],
        "val_ndcg": []
    }
    
    metrics = RecommendationMetrics(k_values=[k])
    evaluator = RLRecommenderEvaluator(metrics)
    
    for episode in tqdm(range(n_episodes), desc="Training"):
        episode_reward = 0
        
        # Training episode
        for user_id in range(n_users):
            # Get recommendations
            recommendations = recommender.recommend(user_id, k)
            
            # Simulate user feedback from training data
            relevant_items = np.where(train_interactions[user_id] == 1)[0].tolist()
            
            # Calculate reward
            reward = len(set(recommendations) & set(relevant_items)) / k
            episode_reward += reward
            
            # Update recommender
            if recommendations:
                item_id = recommendations[0]
                recommender.update(user_id, item_id, reward)
        
        episode_reward /= n_users
        
        # Evaluation
        if episode % eval_freq == 0:
            val_metrics = evaluator.evaluate_offline(recommender, val_interactions, k)
            
            history["episode"].append(episode)
            history["train_reward"].append(episode_reward)
            history["val_precision"].append(val_metrics[f"precision@{k}"])
            history["val_recall"].append(val_metrics[f"recall@{k}"])
            history["val_ndcg"].append(val_metrics[f"ndcg@{k}"])
    
    return history


def evaluate_models(
    models: Dict[str, object],
    test_interactions: np.ndarray,
    config: Dict
) -> pd.DataFrame:
    """Evaluate multiple models and create comparison table.
    
    Args:
        models: Dictionary of model name -> recommender instance.
        test_interactions: Test interaction matrix.
        config: Evaluation configuration.
        
    Returns:
        DataFrame with evaluation results.
    """
    k = config.get("k", 10)
    metrics = RecommendationMetrics(k_values=[k])
    evaluator = RLRecommenderEvaluator(metrics)
    
    results = []
    
    for model_name, recommender in models.items():
        print(f"Evaluating {model_name}...")
        
        # Offline evaluation
        offline_metrics = evaluator.evaluate_offline(recommender, test_interactions, k)
        
        # Online evaluation
        online_metrics = evaluator.evaluate_online(
            recommender, test_interactions, 
            n_episodes=config.get("n_eval_episodes", 100), k=k
        )
        
        # Combine metrics
        combined_metrics = {**offline_metrics, **online_metrics}
        combined_metrics["model"] = model_name
        
        results.append(combined_metrics)
    
    return pd.DataFrame(results)


def main():
    """Main training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Train RL recommendation systems")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing dataset files")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    loader = DatasetLoader(args.data_dir)
    interactions_df = loader.load_interactions()
    items_df = loader.load_items()
    
    # Create interaction matrix
    interactions_matrix, user_ids, item_ids = loader.create_interaction_matrix(interactions_df)
    
    # Split data
    train_matrix, val_matrix, test_matrix = split_interactions(
        interactions_matrix, 
        test_ratio=config["data"]["test_ratio"],
        val_ratio=config["data"]["val_ratio"],
        seed=args.seed
    )
    
    print(f"Data shapes - Train: {train_matrix.shape}, Val: {val_matrix.shape}, Test: {test_matrix.shape}")
    
    # Train models
    models = {}
    histories = {}
    
    for model_config in config["models"]:
        model_type = model_config["type"]
        model_name = model_config["name"]
        
        print(f"\nTraining {model_name} ({model_type})...")
        
        # Create recommender
        recommender_config = {
            "n_users": len(user_ids),
            "n_items": len(item_ids),
            "seed": args.seed,
            **model_config["params"]
        }
        
        recommender = create_recommender(model_type, recommender_config)
        
        # Train
        start_time = time.time()
        history = train_recommender(recommender, train_matrix, val_matrix, config["training"])
        training_time = time.time() - start_time
        
        models[model_name] = recommender
        histories[model_name] = history
        
        print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate models
    print("\nEvaluating models...")
    results_df = evaluate_models(models, test_matrix, config["evaluation"])
    
    # Save results
    results_df.to_csv(output_dir / "evaluation_results.csv", index=False)
    
    # Save training histories
    for model_name, history in histories.items():
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / f"{model_name}_history.csv", index=False)
    
    # Save configuration
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Print results
    print("\nEvaluation Results:")
    print(results_df.round(4))
    
    # Save model checkpoints
    for model_name, recommender in models.items():
        if hasattr(recommender, 'q_table'):
            np.save(output_dir / f"{model_name}_q_table.npy", recommender.q_table)
        elif hasattr(recommender, 'q_network'):
            torch.save(recommender.q_network.state_dict(), 
                      output_dir / f"{model_name}_network.pth")


if __name__ == "__main__":
    main()
