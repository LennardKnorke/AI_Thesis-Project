# test_baselines.py
import numpy as np
import os
import pandas as pd

from agents import (
    AgentList, BaseAgent, 
    RandomAgent
)

from tiny_game import (
    GAMES, get_game_Rework, GameNames, Settings,
    DecPOMP_Rework, PAYOFFS, normalize_payoffs
)
from runner import run_episode

from config import RESULTS_DIR

def test_baselines():
    test_random_agents()
    test_heuristic_agents()
    test_decPOMDP_agents()
    print("Successfully tested all types of Agents!\n")
    return



def test_random_agents():
    """
    Test the random agents on all 6 Games.
    """
    all_results_dict = {}
    file_name = "random_agents.csv"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_path = os.path.join(RESULTS_DIR, file_name)


    for game_id in range(len(GAMES)):
        # Init GameType
        game_enum = GameNames(GAMES[game_id])

        # Ugly but works
        game : DecPOMP_Rework = get_game_Rework(gamename=game_enum, setting=Settings.decpomdp)

        # Set up Agents
        num_actions = game.num_actions
        agent_list = AgentList([
            RandomAgent(0, num_actions), 
            RandomAgent(1, num_actions)
        ])

        # Starting States
        s_states = game.start_states()

        # Loop Over Possible Starting State
        for start_state in s_states:
            col_name = f"{game_enum.name}_{str(start_state)}"
            current_reward = np.zeros(1_000)

            # Run each for a number of episodes
            for i in range(1_000):
                episode_reward = run_episode(game, agent_list, test_episode=True, start_state=start_state)
                current_reward[i] = episode_reward
            all_results_dict[col_name] = current_reward
    # Convert Results to Dataframe
    df = pd.DataFrame(all_results_dict)
    # Save Results
    df.to_csv(full_path, index_label="Episode")
    print(f"Saving results to {full_path}...\n")
    return



def test_heuristic_agents():
    print("Testing Heuristic Agents not Implemented Yet.\n")

def test_decPOMDP_agents():
    print("Testing decPOMDP Agents not Implemented Yet.\n")


###################################