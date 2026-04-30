# main.py

import os
import pickle

from env.traffic_env import TrafficEnv
from agent.q_agent import QAgent
from config import EPISODES, STEPS_PER_EPISODE
from visualization.plot import plot_comparison
from utils.baseline import run_baseline


def train():
    env = TrafficEnv()
    agent = QAgent()

    episode_rewards = []

    print("Training RL agent...\n")

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode+1} | RL Reward: {total_reward}")

        # decay exploration
        agent.decay_epsilon()

    print("\nTraining completed ✅")

    return agent, episode_rewards


def save_q_table(agent):
    with open("results/q_table.pkl", "wb") as f:
        pickle.dump(agent.q_table, f)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Train
    agent, rl_rewards = train()

    # Save model
    save_q_table(agent)
    print("Q-table saved ✅")

    # Run baseline
    print("\nRunning baseline...")
    baseline_rewards = run_baseline()

    # Plot comparison
    plot_comparison(rl_rewards, baseline_rewards)