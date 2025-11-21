# Reinforcement Learning for Recommendation Systems

A comprehensive implementation of reinforcement learning algorithms for recommendation systems, featuring Q-Learning, SARSA, Deep Q-Networks (DQN), and Contextual Bandits.

## Overview

This project demonstrates how reinforcement learning can be applied to recommendation systems, where the system learns to recommend items based on user interactions and feedback. The system treats each recommendation as an action and user feedback (clicks, purchases, ratings) as rewards, aiming to maximize long-term user engagement.

## Features

- **Multiple RL Algorithms**: Q-Learning, SARSA, DQN, and Contextual Bandits
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate, Coverage, Diversity, Novelty
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Production Ready**: Clean code with type hints, docstrings, and proper testing
- **Configurable**: YAML-based configuration for easy experimentation
- **Reproducible**: Deterministic seeding and proper data splitting

## Installation

### Using pip

```bash
pip install -r requirements.txt
```

### Using conda

```bash
conda create -n rl-recs python=3.10
conda activate rl-recs
pip install -r requirements.txt
```

### Development setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick Start

### 1. Generate Synthetic Data

The system will automatically generate synthetic data if no dataset is provided:

```bash
python scripts/train.py --config configs/default.yaml
```

### 2. Train Models

Train all RL models with default configuration:

```bash
python scripts/train.py --config configs/default.yaml --output_dir outputs
```

### 3. Run Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run scripts/demo.py
```

### 4. Evaluate Models

Compare model performance:

```bash
python scripts/evaluate.py --models_dir outputs --test_data data/test.csv
```

## Project Structure

```
├── src/                    # Source code
│   ├── models/            # RL recommendation models
│   ├── data/              # Data loading and preprocessing
│   ├── evaluation/        # Evaluation metrics and tools
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── tests/                 # Unit tests
├── notebooks/             # Jupyter notebooks for analysis
├── data/                  # Dataset files (auto-generated)
├── outputs/               # Model outputs and results
└── assets/                # Static assets for documentation
```

## Models

### Q-Learning Recommender

Classical Q-Learning algorithm adapted for recommendation systems:

- **State**: User ID
- **Action**: Item to recommend
- **Reward**: User feedback (click, purchase, rating)
- **Policy**: Epsilon-greedy exploration

```python
from src.models import QLearningRecommender

recommender = QLearningRecommender(
    n_users=1000,
    n_items=500,
    learning_rate=0.1,
    discount_factor=0.9,
    epsilon=0.1
)
```

### SARSA Recommender

State-Action-Reward-State-Action algorithm for on-policy learning:

- Similar to Q-Learning but uses actual next action instead of optimal action
- Better for online learning scenarios

### Deep Q-Network (DQN) Recommender

Neural network-based Q-Learning with experience replay:

- **Architecture**: Embedding layers + fully connected layers
- **Experience Replay**: Stores and samples past experiences
- **Target Network**: Stabilizes training with separate target network

### Contextual Bandit Recommender

Upper Confidence Bound (UCB) algorithm with context features:

- **Context**: User features, item features, temporal information
- **Exploration**: UCB-based exploration strategy
- **Update**: Ridge regression for parameter updates

## Data Format

### Interactions CSV

```csv
user_id,item_id,timestamp,weight
user_0,item_0,1640995200,0.8
user_0,item_1,1640995800,0.6
...
```

### Items CSV

```csv
item_id,title,category,price,rating
item_0,Product A,electronics,99.99,4.5
item_1,Product B,books,19.99,4.2
...
```

### Users CSV (Optional)

```csv
user_id,age,gender,location,signup_date
user_0,25,M,US,1640995200
user_1,30,F,EU,1640995800
...
```

## Configuration

Models and training parameters can be configured via YAML files:

```yaml
# configs/default.yaml
data:
  test_ratio: 0.2
  val_ratio: 0.1
  sparsity: 0.8

training:
  n_episodes: 1000
  k: 10
  eval_freq: 100

models:
  - name: "q_learning"
    type: "q_learning"
    params:
      learning_rate: 0.1
      discount_factor: 0.9
      epsilon: 0.1
```

## Evaluation Metrics

### Accuracy Metrics

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users with at least one relevant recommendation

### Diversity and Coverage Metrics

- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Intra-list diversity of recommendations
- **Novelty**: Average novelty of recommended items

### RL-Specific Metrics

- **Cumulative Reward**: Total reward accumulated over episodes
- **Exploration Rate**: Fraction of exploratory actions
- **Convergence**: Rate of Q-value convergence

## Usage Examples

### Basic Training

```python
from src.models import QLearningRecommender
from src.data import DatasetLoader
from src.evaluation import RecommendationMetrics

# Load data
loader = DatasetLoader("data")
interactions_df = loader.load_interactions()
interactions_matrix, user_ids, item_ids = loader.create_interaction_matrix(interactions_df)

# Create recommender
recommender = QLearningRecommender(len(user_ids), len(item_ids))

# Train
for episode in range(1000):
    for user_id in range(len(user_ids)):
        recommendations = recommender.recommend(user_id, 10)
        # Simulate user feedback
        reward = calculate_reward(recommendations, user_id)
        recommender.update(user_id, recommendations[0], reward)
```

### Model Comparison

```python
from src.evaluation import RLRecommenderEvaluator

models = {
    "Q-Learning": q_learning_model,
    "SARSA": sarsa_model,
    "DQN": dqn_model
}

evaluator = RLRecommenderEvaluator(RecommendationMetrics())

for name, model in models.items():
    metrics = evaluator.evaluate_offline(model, test_interactions)
    print(f"{name}: {metrics}")
```

## Advanced Features

### Hyperparameter Tuning

Use Optuna or similar tools for hyperparameter optimization:

```python
import optuna

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    epsilon = trial.suggest_float("epsilon", 0.05, 0.3)
    
    recommender = QLearningRecommender(
        n_users, n_items,
        learning_rate=learning_rate,
        epsilon=epsilon
    )
    
    # Train and evaluate
    metrics = train_and_evaluate(recommender)
    return metrics["precision@10"]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

### Multi-Armed Bandit Integration

Combine with multi-armed bandit algorithms for exploration:

```python
from src.models import ContextualBanditRecommender

bandit = ContextualBanditRecommender(
    n_users, n_items,
    context_dim=10,
    learning_rate=0.01
)
```

### Real-time Serving

Deploy models for real-time recommendations:

```python
from fastapi import FastAPI
from src.models import QLearningRecommender

app = FastAPI()
model = QLearningRecommender.load("models/q_learning.pkl")

@app.post("/recommend")
async def recommend(user_id: int, k: int = 10):
    recommendations = model.recommend(user_id, k)
    return {"recommendations": recommendations}
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific test categories:

```bash
pytest tests/test_models.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_utils.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{rl_recommendations,
  title={Reinforcement Learning for Recommendation Systems},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Reinforcement-Learning-for-Recommendation-Systems}
}
```

## Acknowledgments

- OpenAI Gym for RL environment framework
- PyTorch for deep learning components
- Streamlit for interactive demos
- Scikit-learn for evaluation metrics
# Reinforcement-Learning-for-Recommendation-Systems
