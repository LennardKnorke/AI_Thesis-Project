# train_pbdp.py

import numpy as np
import pandas as pd
from tqdm import tqdm

from tiny_game import *
from runner import *
from agents import *
from config import *

_name = "PBDP"

# Setting Folders
_agent_sub_dir = _name.replace(" ", "_")
_searchresults_dir = os.path.join(HYPERSEARCH_RESULTS_DIR, _agent_sub_dir)
os.makedirs(_searchresults_dir, exist_ok=True)
_results_dir = os.path.join(RESULTS_DIR, _agent_sub_dir)
os.makedirs(_results_dir, exist_ok=True)

def train_pbdp():
    final_agents = {}
    final_results = {}

    pbar = tqdm(GAMES, desc="Iter Games", leave=False)
    for g in pbar:
        _postfix = {
            "G" : g
        }
        pbar.set_postfix(_postfix)
        env : Game = ENVIRONMENTS[g]

        agents_ = PBDP_Central_Planner(env, env.num_cards, env.num_actions, g)
        rewards, _, agents_ = run_model_based_planning(env, g, agents_, max_iterations=50, attempts=10)
        final_agents[g] = agents_
        final_results[f"{g}_reward"] = rewards
        #final_results[f"{g}_loss"] = loss


        agents_dir = os.path.join(_results_dir, f"{g}_agents.pkl")
        agents_.save(agents_dir)

    results_df = pd.DataFrame(final_results)
    csv_path = os.path.join(_results_dir, "final_results.csv")
    results_df.to_csv(csv_path)
    return


if __name__ == "__main__":
    train_pbdp()