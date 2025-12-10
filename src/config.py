# config.py

from tiny_game import GameNames

BATCH_SIZE = 64
EPISODES_TEST = 100
EPISODES_TRAIN = 10_000

RESULTS_DIR = "Results/"

# RL Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 1.0 # Finite horizon, usually 1.0
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995