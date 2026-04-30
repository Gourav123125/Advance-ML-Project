# config.py

# Training
EPISODES = 500
STEPS_PER_EPISODE = 100

# Q-learning parameters
ALPHA = 0.1
GAMMA = 0.9

# Exploration (VERY IMPORTANT)
EPSILON = 1.0          # start with full exploration
EPSILON_DECAY = 0.995  # decay rate per episode
MIN_EPSILON = 0.05     # minimum exploration

# Traffic settings
MAX_CARS_PASS = 2
MAX_CAR_ARRIVAL = 3