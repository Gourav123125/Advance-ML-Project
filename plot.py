# visualization/plot.py

import matplotlib.pyplot as plt


def plot_comparison(rl_rewards, baseline_rewards):
    plt.figure()

    plt.plot(rl_rewards, label="Q-Learning (RL)")
    plt.plot(baseline_rewards, label="Fixed Signal (Baseline)")

    plt.title("RL vs Baseline: Traffic Signal Performance")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    plt.legend()
    plt.grid()

    plt.savefig("results/comparison_plot.png")
    plt.show()