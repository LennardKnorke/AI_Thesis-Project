# hypersearch_tom.py
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Own Imports
from tiny_game import *
from runner import run_episode, test_on_all_start_states
from config import (
    HYPERSEARCH_RESULTS_DIR, WORLD_MODELS_DIR, RESULTS_DIR,
    BASELINE_EXPERIMENTS, DTDE_ToMBI_params
)
from agents import *
from agents.model_based.dtde_ToM import OBS_DIM # Import constants for model initialization
from hypersearch_baselines import results_do_exists # Reuse helper function



def load_worldmodels()->Dict[str, ToM_WorldModel]:
    print("Loading World Models into Games")

    num_agent_types = len(BASELINE_EXPERIMENTS)
    models = {}
    for game in GAMES:
        ENV = get_game(GameNames(game), Settings.decpomdp, normalize=False)

        world_model_path = os.path.join(WORLD_MODELS_DIR, f"ToM_WorldModel_{game}.pth")
        world_model_params_path = os.path.join(WORLD_MODELS_DIR, f"best_params.json")

        if not os.path.exists(world_model_path) or not os.path.exists(world_model_params_path):
            raise FileNotFoundError(f"World Models not found for game - {game}")
        
        with open(world_model_params_path, 'r') as f:
            wm_params = json.load(f)

        # Instantiate ToM_WorldModel with the CORRECT parameters
        world_model = ToM_WorldModel(
            obs_dim=OBS_DIM, 
            action_dim=ENV.num_actions, # Use ENV's num_actions
            num_agent_types=num_agent_types,
            char_embed_dim=wm_params['char_dim'],
            mental_embed_dim=wm_params['mental_dim']
        )
        world_model.load_state_dict(torch.load(world_model_path))
        world_model.eval()
        models[game] = world_model
    
    return models


def load_partner_models():
    # {Game_name : {Agent Name, loaded Agent instances}} Only DTDE ones.
    all_partners: Dict[str, Dict[str, BaseAgent]] = defaultdict(dict)
    for game_name in GAMES:
        ENV = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)
        game_partners = {}
        for exp in BASELINE_EXPERIMENTS:
            # Skip Centralized Trained Agents
            if exp.name[:4] == "CTDE": continue
            

            folder_name = exp.name.replace(' ', '_')
            MODEL_DIR = os.path.join(RESULTS_DIR, folder_name)
            best_params_path = os.path.join(HYPERSEARCH_RESULTS_DIR, folder_name, "best_params.json")

            if not os.path.exists(best_params_path):
                raise FileNotFoundError(f"Couldnt find file - {best_params_path}")
            with open(best_params_path, 'r') as f:
                params = json.load(f)

            baseline_agents_list = exp.make_agents(ENV, params)
            for i in range(2):
                p_path = os.path.join(MODEL_DIR, f"G_{game_name}_agent_{i}.pkl")
                if not os.path.exists(p_path):
                    raise FileNotFoundError(f"Not Found File - {p_path}")
                baseline_agents_list[i].load(p_path)
                game_partners[f"{exp.name}_{i}"] = baseline_agents_list[i]
        all_partners[game_name] = game_partners
    return all_partners            

def hypersearch_tom() -> None:
    """
    Performs hyperparameter search for the Theory of Mind (ToM) agent.
    
    This function iterates through defined ToM agent parameters, and for each set,
    it trains the ToM agent against all specified games, leveraging the pre-trained
    ToM World Model for that specific game.
    """
    agent_name = "DTDE ToMBI"
    folder_name = agent_name.replace(" ", "_")
    
    # 1. Create Directory Structure
    results_dir = os.path.join(HYPERSEARCH_RESULTS_DIR, folder_name)
    os.makedirs(results_dir, exist_ok=True)

    # Determine num_agent_types (needed for ToM_WorldModel initialization)
    world_models : Dict[str, ToM_WorldModel] = load_worldmodels()
    all_partners : Dict[str, Dict[str, BaseAgent]] = load_partner_models()

    # START - LOOP OVER PARAM SETS FOR THE TOM AGENT
    print(f"\nHyperparameter Search for {agent_name}")
    pbar = tqdm(range(len(DTDE_ToMBI_params)))

    best_idx = None
    best_reward = None

    for idx in pbar:
        params = DTDE_ToMBI_params[idx]
        results_file = f"{idx}_results.csv"
        results_path = os.path.join(results_dir, results_file)
        param_results = {}

        tmp_final_reward = 0.0
        for game_name in GAMES:
            ENV = get_game(GameNames(game_name), Settings.decpomdp, normalize=True)
            tom_agent = DTDE_ToMBI_Agent(
                num_actions=ENV.num_actions,
                num_cards=ENV.num_cards,
                env=ENV,
                world_model=world_models[game_name]
            )
            game_partners = all_partners[game_name]

            # Training + Testing
            losses = []
            rewards = []
            for _ in range(params['max_iterations']):
                train_delta = tom_agent.train()
                losses.append(train_delta)

                tmp_reward = 0
                for _partner_id, _partner in game_partners.items():
                    if _partner_id[-1] == '1':
                        agents = AgentList([tom_agent, _partner])
                    elif _partner_id[-1] == '0':
                        agents = AgentList([_partner, tom_agent])
                    else:
                        raise ValueError(f"You messed something up lenny")

                    tmp_reward += test_on_all_start_states(ENV, agents)
                tmp_reward /= len(all_partners)
                rewards.append(tmp_reward)
            param_results[f"reward_{game_name}"] = np.array(rewards)
            param_results[f"loss_{game_name}"] = np.array(losses)
            tmp_final_reward += param_results[f"reward_{game_name}"][-1]
        save_results(results_path, param_results)
        if best_reward is None or tmp_final_reward > best_reward:
            best_reward = tmp_final_reward
            best_idx = idx

    best_json_path = os.path.join(results_dir, "best_params.json")
    if not os.path.exists(best_json_path):
        print(f"Creating preliminary best params")
        with open(best_json_path, 'w') as f:
            json.dump(params[best_idx], f, indent=4)
    return


def save_results(path : str, data : Dict[str, np.array]):
    results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in data.items()]))
    results_df.index.name = "episode"
    results_df.reset_index(inplace=True)
    results_df.to_csv(path, index=False)
    return