# train_baselines.py

import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Tuple, Dict, Union, Any, List, Optional

from tiny_game import GAMES, Settings, GameNames, DecPOMP_Rework, get_game_Rework
from runner import run_training
from config import (
    BASELINE_RESULTS_DIR, TRAINING_EPISODES_HYPERSEARCH,
    modelfree_IQ_Learner_paramList, 
    modelbased_IQ_Learner_paramList, 
    vdn_rl_paramList,
    vdn_vi_paramList
)

from agents import (
    AgentList, BaseAgent,
    ModelFreeAgent, ModelBasedAgent,

    Independent_RL_Agent, Independent_VI_Agent,
    VDN_RL_Agent, VDN_AgentList,
    VDN_VI_Agent, VDN_VI_AgentList
)

# Baseline Experiments Setup
baselines_experiments = {
    "Independent_RL_Agent": {
        "learning_agents": Independent_RL_Agent,
        "parameters": modelfree_IQ_Learner_paramList,
        "list_type": AgentList # Standard decentralized list
    },
    "Independent_VI_Agent": {
        "learning_agents": Independent_VI_Agent,
        "parameters": modelbased_IQ_Learner_paramList,
        "list_type": AgentList
    },
    "VDN_RL_Agents" : {
        "learning_agents" : Independent_VI_Agent,
        "parameters" : vdn_rl_paramList,
        "list_type" : VDN_AgentList
    },
    "VDN_VI_Agents" : {
        "learning_agents" :VDN_VI_Agent,
        "parameters" : vdn_vi_paramList,
        "list_type" : VDN_VI_AgentList
    }
}


def train_baselines()-> None:
    """
    Loop over all baseline algorithms and perform hyperparameter search training.
    """
    for algo_name, algo_data in baselines_experiments.items():
        if algo_data is None:
            print(f"Skipping unimplemented Baseline Algorithm: {algo_name}")
            continue
        else:
            hypersearch_algorithm(algo_name, **algo_data)
    return


def setup_agents(
        algorithm: str,
        learning_agents : Any, # Class reference
        parameters: Dict[str, Any],
        num_cards: int,
        num_actions: int,
        env: DecPOMP_Rework, # Env is crucial for Model-Based
        list_type : Any, # Class reference for the List
    ) -> AgentList:
    """
    Set up agents for a specific baseline algorithm with given parameters.
    Args:
        algorithm (str): The name of the baseline algorithm.
        learning_agents (BaseAgent): The agent class to instantiate.
        parameters (Dict[str, Any]): The hyperparameters for the agents.
        num_cards (int): Number of cards in the environment.
        num_actions (int): Number of actions available to the agents.
    """
    if list_type != AgentList:
        if "iterations" in parameters or "convergence_threshold" in parameters:
             agents_instances = list_type(
                num_cards=num_cards,
                num_actions=num_actions,
                env=env,
                **parameters
            )
        else:
            agents_instances = list_type(
                num_cards=num_cards,
                num_actions=num_actions,
                **parameters
            )
        agents = agents_instances
    else:
        if issubclass(learning_agents, ModelBasedAgent):
            # Model-Based: Must pass ENV
            agents_instances = [
                learning_agents(num_cards, num_actions, env=env, **parameters),
                learning_agents(num_cards, num_actions, env=env, **parameters)
            ]
        else:
            # Model-Free: Do NOT pass ENV
            agents_instances = [
                learning_agents(num_cards, num_actions, **parameters),
                learning_agents(num_cards, num_actions, **parameters)
            ]
        agents = AgentList(agents_instances)
    return agents

def results_do_exists(results_path: str, **params) -> bool:
    """
    Checks if results exist and if the parameters match the saved metadata.
    
    Returns:
        True: If files exist and parameters match.
        False: If files do not exist (safe to train).
        Raises ValueError: If files exist but parameters differ (collision/consistency error).
    """
    json_path = results_path.replace("_results.csv", "_params.json")
    if not os.path.exists(results_path) or not os.path.exists(json_path):
        return False

    # Load the existing JSON parameters
    try:
        with open(json_path, 'r') as f:
            saved_params = json.load(f)
    except json.JSONDecodeError:
        # If JSON is corrupted, treat as not existing (or you could raise error)
        return False

    # 4. Compare provided params with saved params
    # We iterate over the provided input 'params' to ensure they match the saved ones.
    mismatches = []
    
    for key, value in params.items():
        # Check if key exists in saved data
        if key not in saved_params:
            mismatches.append(f"Key '{key}' missing in saved file.")
            continue
        
        # Check if values match
        # Note: Be careful with float precision if comparing floats
        if saved_params[key] != value:
            mismatches.append(
                f"Key '{key}' mismatch. Saved: {saved_params[key]} | Request: {value}"
            )

    if mismatches:
        error_msg = "\n".join(mismatches)
        raise ValueError(
            f"Result files exist at '{results_path}' but parameters do not match!\n{error_msg}"
        )

    return True

def hypersearch_algorithm(
        algorithm: str, 
        learning_agents : BaseAgent, 
        parameters: List[Dict[str, Any]],
        list_type: Any,
        *args, **kwargs)-> None:
    """
    Train a specific baseline algorithm and do hyperparameter search on the given parameters list.
    """
    # Create main Results Folder for Algorithm
    results_baseline_dir = os.path.join(BASELINE_RESULTS_DIR, algorithm.replace(" ", "_") + "/")
    os.makedirs(results_baseline_dir, exist_ok=True)

    # Data results folder
    data_results_dir = os.path.join(results_baseline_dir, "Search Results/")
    os.makedirs(data_results_dir, exist_ok=True)

    # Status Bar + START - LOOP OVER PARAM SETS
    pbar = tqdm(range(len(parameters)), desc=f"{algorithm} Hyperparameter Search")
    for idx in pbar:
        # Set Parameters
        params : Dict[str, Any] = parameters[idx]
        pbar.set_description(f"{algorithm} - {idx+1}/{len(parameters)}")

        # Results filepaths + Check if exist
        results_filepath = os.path.join(data_results_dir, f"{idx}_results.csv")

        # Update Status Bar
        if results_do_exists(results_path=results_filepath, **params):
            continue

        # Prepare Parameter Results Cache
        results_cache = {}

        # START - TRAIN PARAMS ON ALL GAMES
        for game_name in GAMES:
            # Set up Game Instance
            game : DecPOMP_Rework = get_game_Rework(GameNames(game_name), Settings.decpomdp)
            num_cards = game.num_cards
            num_actions = game.num_actions

            # Set up Agents
            agents : AgentList = setup_agents(
                algorithm,
                learning_agents,
                params,
                num_cards,
                num_actions,
                env=game,
                list_type=list_type,
                **kwargs
            )

            # Update kwargs for search
            run_kwargs = params.copy()
            if agents[0].MODEL_BASED:
                if 'iterations' in run_kwargs:
                    run_kwargs['max_iterations'] = run_kwargs.pop('iterations')
            else:
                # Model-Free: Ensure train_episodes is set
                run_kwargs['train_episodes'] = TRAINING_EPISODES_HYPERSEARCH
            run_kwargs['pbar'] = pbar
            run_kwargs['game_name'] = game_name
            
            # Run Training
            game_rewards, game_losses, _ = run_training(
                env=game,
                agents=agents,
                **run_kwargs
            )

            # Update Results in cache
            results_cache[f"reward_{game_name}"] = game_rewards
            results_cache[f"loss_{game_name}"] = game_losses
        # END - TRAIN PARAMS ON ALL GAMES
        results_df = pd.DataFrame(dict([ (k, pd.Series(v)) for k, v in results_cache.items() ]))
        results_df.to_csv(results_filepath, index=False)

        # Save params used
        params_filepath = os.path.join(data_results_dir, f"{idx}_params.json")
        with open(params_filepath, 'w') as f:
            json.dump(params, f, indent=4)
    # END - LOOP OVER PARAM SETS
    return