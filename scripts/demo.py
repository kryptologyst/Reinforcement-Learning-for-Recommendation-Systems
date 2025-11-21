"""Streamlit demo for RL recommendation systems."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml

from src.data import DatasetLoader
from src.models import (
    QLearningRecommender,
    SARSARecommender,
    DQNRecommender,
    ContextualBanditRecommender
)
from src.evaluation import RecommendationMetrics
from src.utils import set_seed


def load_trained_models(models_dir: Path):
    """Load trained model checkpoints."""
    models = {}
    
    # Load Q-Learning model
    q_table_path = models_dir / "q_learning_q_table.npy"
    if q_table_path.exists():
        q_table = np.load(q_table_path)
        n_users, n_items = q_table.shape
        models["Q-Learning"] = QLearningRecommender(n_users, n_items)
        models["Q-Learning"].q_table = q_table
    
    # Load SARSA model
    sarsa_table_path = models_dir / "sarsa_q_table.npy"
    if sarsa_table_path.exists():
        sarsa_table = np.load(sarsa_table_path)
        n_users, n_items = sarsa_table.shape
        models["SARSA"] = SARSARecommender(n_users, n_items)
        models["SARSA"].q_table = sarsa_table
    
    return models


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Recommendation Systems Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Reinforcement Learning for Recommendation Systems")
    st.markdown("""
    This demo showcases different RL-based recommendation algorithms including Q-Learning, 
    SARSA, Deep Q-Networks, and Contextual Bandits.
    """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load data
    data_dir = st.sidebar.text_input("Data Directory", value="data")
    models_dir = st.sidebar.text_input("Models Directory", value="outputs")
    
    # Load configuration
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        st.error("Configuration file not found!")
        return
    
    # Load data
    try:
        loader = DatasetLoader(data_dir)
        interactions_df = loader.load_interactions()
        items_df = loader.load_items()
        
        # Create interaction matrix
        interactions_matrix, user_ids, item_ids = loader.create_interaction_matrix(interactions_df)
        
        st.success(f"Loaded data: {len(user_ids)} users, {len(item_ids)} items")
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Load trained models
    try:
        models = load_trained_models(Path(models_dir))
        if not models:
            st.warning("No trained models found. Training new models...")
            # Train simple models for demo
            set_seed(42)
            models["Q-Learning"] = QLearningRecommender(len(user_ids), len(item_ids))
            models["SARSA"] = SARSARecommender(len(user_ids), len(item_ids))
            
            # Quick training
            for model_name, model in models.items():
                for _ in range(100):
                    for user_id in range(len(user_ids)):
                        recommendations = model.recommend(user_id, 5)
                        if recommendations:
                            relevant_items = np.where(interactions_matrix[user_id] == 1)[0].tolist()
                            reward = len(set(recommendations) & set(relevant_items)) / 5
                            model.update(user_id, recommendations[0], reward)
        
        st.success(f"Loaded {len(models)} trained models")
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Recommendations", "Model Comparison", "Data Analysis", "Training History"])
    
    with tab1:
        st.header("Get Recommendations")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # User selection
            user_idx = st.selectbox(
                "Select User",
                range(len(user_ids)),
                format_func=lambda x: f"User {x} ({user_ids[x]})"
            )
            
            # Model selection
            model_name = st.selectbox("Select Model", list(models.keys()))
            
            # Number of recommendations
            k = st.slider("Number of Recommendations", 1, 20, 10)
            
            # Get recommendations
            if st.button("Get Recommendations"):
                model = models[model_name]
                recommendations = model.recommend(user_idx, k)
                
                # Display recommendations
                st.subheader(f"Top {k} Recommendations for User {user_ids[user_idx]}")
                
                rec_items = []
                for i, item_idx in enumerate(recommendations):
                    item_id = item_ids[item_idx]
                    item_info = items_df[items_df["item_id"] == item_id].iloc[0] if len(items_df) > 0 else None
                    
                    rec_items.append({
                        "Rank": i + 1,
                        "Item ID": item_id,
                        "Title": item_info["title"] if item_info is not None else f"Item {item_idx}",
                        "Category": item_info["category"] if item_info is not None else "Unknown",
                        "Q-Value": model.q_table[user_idx, item_idx] if hasattr(model, 'q_table') else "N/A"
                    })
                
                rec_df = pd.DataFrame(rec_items)
                st.dataframe(rec_df, use_container_width=True)
        
        with col2:
            # Show user's interaction history
            st.subheader("User Interaction History")
            user_interactions = np.where(interactions_matrix[user_idx] == 1)[0]
            
            if len(user_interactions) > 0:
                interaction_items = []
                for item_idx in user_interactions:
                    item_id = item_ids[item_idx]
                    item_info = items_df[items_df["item_id"] == item_id].iloc[0] if len(items_df) > 0 else None
                    
                    interaction_items.append({
                        "Item ID": item_id,
                        "Title": item_info["title"] if item_info is not None else f"Item {item_idx}",
                        "Category": item_info["category"] if item_info is not None else "Unknown"
                    })
                
                interaction_df = pd.DataFrame(interaction_items)
                st.dataframe(interaction_df, use_container_width=True)
            else:
                st.info("No interaction history for this user.")
    
    with tab2:
        st.header("Model Comparison")
        
        # Evaluate all models
        metrics = RecommendationMetrics(k_values=[5, 10])
        
        comparison_data = []
        for model_name, model in models.items():
            all_recommendations = []
            all_relevant_items = []
            
            for user_idx in range(len(user_ids)):
                recommendations = model.recommend(user_idx, 10)
                relevant_items = np.where(interactions_matrix[user_idx] == 1)[0].tolist()
                
                all_recommendations.append(recommendations)
                all_relevant_items.append(relevant_items)
            
            # Calculate metrics
            system_metrics = metrics.evaluate_system(
                all_recommendations, all_relevant_items, len(item_ids)
            )
            
            comparison_data.append({
                "Model": model_name,
                **system_metrics
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display metrics table
        st.subheader("Performance Metrics")
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # Visualization
        st.subheader("Metrics Visualization")
        
        metric_cols = [col for col in comparison_df.columns if col != "Model"]
        
        for metric in metric_cols[:4]:  # Show first 4 metrics
            fig = px.bar(
                comparison_df, 
                x="Model", 
                y=metric,
                title=f"{metric.replace('_', ' ').title()}",
                color="Model"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User activity distribution
            st.subheader("User Activity Distribution")
            user_activity = np.sum(interactions_matrix, axis=1)
            
            fig = px.histogram(
                x=user_activity,
                title="Number of Interactions per User",
                labels={"x": "Number of Interactions", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Item popularity distribution
            st.subheader("Item Popularity Distribution")
            item_popularity = np.sum(interactions_matrix, axis=0)
            
            fig = px.histogram(
                x=item_popularity,
                title="Number of Interactions per Item",
                labels={"x": "Number of Interactions", "y": "Count"}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Data statistics
        st.subheader("Dataset Statistics")
        
        stats_data = {
            "Metric": [
                "Total Users",
                "Total Items", 
                "Total Interactions",
                "Sparsity",
                "Avg Interactions per User",
                "Avg Interactions per Item"
            ],
            "Value": [
                len(user_ids),
                len(item_ids),
                np.sum(interactions_matrix),
                f"{1 - np.sum(interactions_matrix) / (len(user_ids) * len(item_ids)):.3f}",
                f"{np.mean(user_activity):.2f}",
                f"{np.mean(item_popularity):.2f}"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
    with tab4:
        st.header("Training History")
        
        # Load training histories
        history_files = list(Path(models_dir).glob("*_history.csv"))
        
        if history_files:
            for history_file in history_files:
                model_name = history_file.stem.replace("_history", "")
                
                st.subheader(f"{model_name} Training History")
                
                history_df = pd.read_csv(history_file)
                
                # Plot training curves
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=history_df["episode"],
                    y=history_df["train_reward"],
                    mode="lines",
                    name="Training Reward",
                    line=dict(color="blue")
                ))
                
                fig.add_trace(go.Scatter(
                    x=history_df["episode"],
                    y=history_df["val_precision"],
                    mode="lines",
                    name="Validation Precision",
                    line=dict(color="red")
                ))
                
                fig.add_trace(go.Scatter(
                    x=history_df["episode"],
                    y=history_df["val_recall"],
                    mode="lines",
                    name="Validation Recall",
                    line=dict(color="green")
                ))
                
                fig.update_layout(
                    title=f"{model_name} Training Progress",
                    xaxis_title="Episode",
                    yaxis_title="Metric Value",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No training history files found.")


if __name__ == "__main__":
    main()
