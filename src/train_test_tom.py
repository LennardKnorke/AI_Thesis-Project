# train_test_tom.py
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Own Imports from your project
from tiny_game import *
from runner import *
from agents import *
from agents.model_based.dtde_ToM import OBS_DIM
from config import (HYPERSEARCH_RESULTS_DIR, RESULTS_DIR)
from hypersearch_tom import load_partner_models, load_worldmodels


def train_test_tom():
    os.makedirs(os.path.join(RESULTS_DIR, "DTDE_ToMBI"), exist_ok=True)

    print("Making Exhaustive Testing and reporting for ToM Agent")
    world_models = load_worldmodels()
    all_partners = load_partner_models()

    agent_name = "DTDE ToMBI"
    folder_name = agent_name.replace(" ", "_") 

    best_params_path = os.path.join(HYPERSEARCH_RESULTS_DIR, folder_name, "best_params.json")

    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    tom_agents = train_tom_agent(world_models, best_params)

    for game in GAMES:
        ENV = get_game(GameNames(game), Settings.decpomdp, normalize=False)
        game_results = exhaustive_testing_tom(tom_agents[game], all_partners[game], ENV)
        save_results(game_results, game)
    return


def train_tom_agent(world_models : Dict[str, ToM_WorldModel], best_params:Dict[str,Any])->Dict[str,DTDE_ToMBI_Agent]:
    TOM_AGENTS = {}
    for game in GAMES:
        ENV = get_game(GameNames(game), Settings.decpomdp, normalize=True)
        agent = DTDE_ToMBI_Agent(
            num_actions=ENV.num_actions,
            num_cards=ENV.num_cards,
            env=ENV,
            world_model=world_models[game]
        )

        for _ in range(best_params['max_iterations']):
            agent.train()
        TOM_AGENTS[game] = agent
    return TOM_AGENTS


def exhaustive_testing_tom(tom_agent : DTDE_ToMBI_Agent, all_partners : Dict[str, BaseAgent], env : DecPOMDP)->Dict[str, float]:
    results = {}
    for partner_key, partner in all_partners.items():
        if partner_key[-1] == "0":
            agents = AgentList([partner, tom_agent])
        elif partner_key[-1] == "1":
            agents = AgentList([tom_agent, partner])
        else:
            raise ValueError("Error Check")
        
        test_reward : float = test_on_all_start_states(env, agents)
        results[partner_key] = test_reward
    return results


def save_results(game_results : Dict[str, float], game_name : str):
    csv_filename = f"ToM_Agent_Evaluation_Results_{game_name}.csv"
    csv_dir = os.path.join(RESULTS_DIR, "DTDE_ToMBI")
    csv_filepath = os.path.join(csv_dir,csv_filename)
    
    # Convert dictionary to DataFrame
    df_data = []
    for partner_id, reward in game_results.items():        
        df_data.append({
            "game": game_name,
            "partner_agent_type": partner_id,
            "reward": reward
        })
    
    df = pd.DataFrame(df_data)
    
    df.to_csv(csv_filepath, index=False)
    return csv_filepath