# Q-Learning Solver for OpenAI Gym Environments

## Overview
This project implements a Q-Learning-based reinforcement learning agent for solving OpenAI Gym environments. The agent interacts with the environment, learns an optimal policy through trial and error, and evaluates the impact of hyperparameters like learning rate, discount factor, and exploration rate.

## Features
- Implements Q-learning with adjustable hyperparameters
- Trains an agent on OpenAI Gym's `Taxi-v3` environment
- Evaluates the impact of:
  - Learning rate (alpha)
  - Discount factor (gamma)
  - Exploration rate (epsilon)
  - Number of training episodes
- Visualizes the effects of hyperparameters using Matplotlib

## Installation
To run this project, ensure you have Python installed and install the required dependencies:

```bash
pip install numpy matplotlib gymnasium[all]
```

## Usage
### Running the Q-learning Agent
To train the agent and analyze hyperparameter effects, run:

```python
import gym
from qlearning import test_hyperparameters_impact

env = gym.make("Taxi-v3")
test_hyperparameters_impact(env)
```

### Understanding the Q-Learning Solver
The `QLearningSolver` class initializes and trains the Q-learning agent. It supports:
- Choosing an action using an epsilon-greedy policy
- Updating Q-values based on observed rewards
- Reducing exploration (`epsilon`) over time

### Hyperparameter Testing
The function `test_hyperparameters_impact(env, steps=100)`
- Varies learning rate, gamma, epsilon, and episode count
- Trains multiple agents with different hyperparameter settings
- Plots performance comparisons

## Results and Visualization
- The script plots the effects of different hyperparameters on learning efficiency.
- Results help in selecting the best values for faster convergence and optimal policy learning.

----------------
Made by Anna Wierzbik

