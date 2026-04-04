# train_test_baselines.py
import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from tiny_game import GAMES, Settings, GameNames, DecPOMDP, MyHanabi, get_game_Rework
from runner import run_training
from agents import *
from config import *

def load_best_params(agent_name: str) -> list[dict[str, Any]]:
    """
    Reads from: Results/{Agent_Name}/best_params.json
    """
    folder_name = agent_name.replace(" ", "_")
    path = os.path.join(RESULTS_DIR, folder_name, "best_params.json")
    
    if not os.path.exists(path):
        return []
        
    with open(path, 'r') as f:
        return [json.load(f)]


AGENT_REGISTRY = [
    Experiment(
        name="IQ Learning",
        agent_class=IQ_Learning_Agent,
        param_list=load_best_params("IQ Learning"),
        list_class=AgentList
    ),
    Experiment(
        name="VDN",
        agent_class=VDN_Agent,
        param_list=load_best_params("VDN"),
        list_class=VDN_CentralPlanner
    ),
    Experiment(
        name="PBVI",
        agent_class=PBVI_Agent,
        param_list=load_best_params("PBVI"),
        list_class=PBVI_List
    ),
    Experiment(
        name="MA Belief DP",
        agent_class=DP_Agent,
        param_list=load_best_params("MA Belief DP"),
        list_class=DP_List
    ),
]


def save_final_model(
    agents: AgentList, 
    agent_name: str, 
    game_name: str,
):
    """
    Saves the trained models into Results/{Agent_Name}/
    Prefixes files with G_{GameName}_agent_{agent id}.pkl...
    """
    folder_name = agent_name.replace(" ", "_")
    agent_save_dir = os.path.join(RESULTS_DIR, folder_name)
    os.makedirs(agent_save_dir, exist_ok=True)

    # Save Agents
    if agents.centralized_planning:
        # Centralized list: single shared model file
        path = os.path.join(agent_save_dir, f"G_{game_name}_shared_model.pkl")
        agents.save(path)
    else:
        # Decentralized: one file per agent
        for i, agent in enumerate(agents):
            path = os.path.join(agent_save_dir, f"G_{game_name}_agent_{i}.pkl")
            agent.save(path)
            
    return agent_save_dir


def train_test_baselines(): 
    agent_experiments = tqdm(AGENT_REGISTRY, desc="Agent Types")
    for exp in agent_experiments:
        if len(exp.param_list) != 1:
            print(f"[WARNING] Skipping {exp.name}: no best_params.json found")
            continue

        params = exp.param_list[0]
        agent_experiments.set_description(f"Processing {exp.name}")
        folder_name = exp.name.replace(" ", "_")
        results_cache = {}

        # --- SKIP LOGIC: Check if results already exist ---
        final_csv_path = os.path.join(RESULTS_DIR, folder_name, "final_results.csv")
        if os.path.exists(final_csv_path):
            continue
        
        # Iterate over All Games
        for game_name in GAMES:
            # Setup Environment
            game = get_game_Rework(GameNames(game_name))
            
            # Setup Agents
            agents = exp.make_agents(game, params)

            # Configure Run Arguments
            run_kwargs = params.copy()
            run_kwargs['game_name'] = game_name
            run_kwargs['pbar'] = agent_experiments

            # 3. Run Training
            rewards, losses, trained_agents = run_training(
                env=game,
                agents=agents,
                **run_kwargs
            )

            # 4. Save Final Model (Pickles)
            save_dir = save_final_model(trained_agents, exp.name, game_name)
            
            # 5. Accumulate Results
            results_cache[f"reward_{game_name}"] = rewards
            results_cache[f"loss_{game_name}"] = losses

        # 6. Save Aggregated Results
        results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_cache.items()]))
        results_df.to_csv(final_csv_path, index=False)
    return
