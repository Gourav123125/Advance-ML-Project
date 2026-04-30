# utils/baseline.py

from env.traffic_env import TrafficEnv
from config import EPISODES, STEPS_PER_EPISODE


def run_baseline():
    env = TrafficEnv()
    episode_rewards = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(STEPS_PER_EPISODE):
            action = step % 2  # alternate signals

            next_state, reward = env.step(action)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

    return episode_rewards