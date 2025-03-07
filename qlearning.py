import numpy as np
import matplotlib.pyplot as plt
import random
import gym

class QLearningSolver:
    def __init__(
        self,
        observation_space: int,
        action_space: int,
        learning_rate: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = np.zeros((observation_space, action_space))

    def __call__(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.qtable[state, action]

    def update(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray) -> None:
        max_next_q = np.max(self.qtable[next_state, :])  
        current_q = self.qtable[state, action]
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.qtable[state, action] = new_q

    def get_best_action(self, state: np.ndarray) -> np.ndarray:
        return np.argmax(self.qtable[state, :])

    def choose_action(self, state: np.ndarray) -> np.ndarray:
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.get_best_action(state)
        return action

    def train(self, env, episodes: int, steps: int):
        rewards_per_episode = []
        steps_per_episode = []
        
        for episode in range(episodes):
            state = env.reset()[0]
            total_reward = 0
            total_steps = 0
            done = False
            while not done and total_steps < steps:
                action = self.choose_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                total_steps += 1
            rewards_per_episode.append(total_reward)
            steps_per_episode.append(total_steps)
            self.epsilon = max(0.01, self.epsilon * 0.995)
        return rewards_per_episode, steps_per_episode

def test_hyperparameters_impact(env, steps=100):
    def run_experiment(agent, episodes, steps):
        avg_reward, avg_steps = [], []
        rewards, steps = agent.train(env, episodes, steps)
        avg_reward.append(np.mean(rewards))
        avg_steps.append(np.mean(steps))
        return avg_reward[0], avg_steps[0]

    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    avg_rewards_lr, avg_steps_lr = [], []
    
    for alpha in learning_rates:
        agent = QLearningSolver(observation_space=env.observation_space.n, 
                                action_space=env.action_space.n,
                                learning_rate=alpha, 
                                gamma=0.8, 
                                epsilon=0.1)
        avg_reward, avg_steps = run_experiment(agent, episodes=5000, steps=steps)
        avg_rewards_lr.append(avg_reward)
        avg_steps_lr.append(avg_steps)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(learning_rates, avg_rewards_lr, marker='o')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Reward')
    plt.title('Effect of Learning Rate on Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates, avg_steps_lr, marker='o', color='r')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Steps')
    plt.title('Effect of Learning Rate on Average Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    gamma_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    avg_rewards_gamma, avg_steps_gamma = [], []
    
    for gamma in gamma_values:
        agent = QLearningSolver(observation_space=env.observation_space.n, 
                                action_space=env.action_space.n,
                                learning_rate=0.8, 
                                gamma=gamma, 
                                epsilon=0.1)
        avg_reward, avg_steps = run_experiment(agent, episodes=5000, steps=steps)
        avg_rewards_gamma.append(avg_reward)
        avg_steps_gamma.append(avg_steps)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gamma_values, avg_rewards_gamma, marker='o')
    plt.xlabel('Gamma')
    plt.ylabel('Average Reward')
    plt.title('Effect of Gamma on Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(gamma_values, avg_steps_gamma, marker='o', color='r')
    plt.xlabel('Gamma')
    plt.ylabel('Average Steps')
    plt.title('Effect of Gamma on Average Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    epsilon_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    avg_rewards_epsilon, avg_steps_epsilon = [], []
    
    for epsilon in epsilon_values:
        agent = QLearningSolver(observation_space=env.observation_space.n, 
                                action_space=env.action_space.n,
                                learning_rate=0.8, 
                                gamma=0.8, 
                                epsilon=epsilon)
        avg_reward, avg_steps = run_experiment(agent, episodes=5000, steps=steps)
        avg_rewards_epsilon.append(avg_reward)
        avg_steps_epsilon.append(avg_steps)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epsilon_values, avg_rewards_epsilon, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Average Reward')
    plt.title('Effect of Epsilon on Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_values, avg_steps_epsilon, marker='o', color='r')
    plt.xlabel('Epsilon')
    plt.ylabel('Average Steps')
    plt.title('Effect of Epsilon on Average Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    episode_counts = [100, 500, 1000, 5000, 8000, 10000]
    avg_rewards_episodes, avg_steps_episodes = [], []
    
    for episodes in episode_counts:
        agent = QLearningSolver(observation_space=env.observation_space.n, 
                                action_space=env.action_space.n,
                                learning_rate=0.8, 
                                gamma=0.8, 
                                epsilon=0.1)
        avg_reward, avg_steps = run_experiment(agent, episodes=episodes, steps=steps)
        avg_rewards_episodes.append(avg_reward)
        avg_steps_episodes.append(avg_steps)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_counts, avg_rewards_episodes, marker='o')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Reward')
    plt.title('Effect of Number of Episodes on Average Reward')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_counts, avg_steps_episodes, marker='o', color='r')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Average Steps')
    plt.title('Effect of Number of Episodes on Average Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

env = gym.make("Taxi-v3")
test_hyperparameters_impact(env)
