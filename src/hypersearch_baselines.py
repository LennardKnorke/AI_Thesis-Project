import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Tuple, Dict, Union, Any, List, Optional

# Own Imports
from tiny_game import GAMES, Settings, GameNames, DecPOMDP, get_game
from runner import run_training
from config import (
    TRAINING_EPISODES_HYPERSEARCH,
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

RESULTS_DIR = "HyperSearchResults/"

# Baseline Experiments Setup
baselines_experiments = {
    "Independent_RL_Agent": {
        "learning_agents": Independent_RL_Agent,
        "parameters": modelfree_IQ_Learner_paramList,
        "list_type": AgentList 
    },
    "Independent_VI_Agent": {
        "learning_agents": Independent_VI_Agent,
        "parameters": modelbased_IQ_Learner_paramList,
        "list_type": AgentList
    },
    "VDN_RL_Agents" : {
        "learning_agents" : VDN_RL_Agent,
        "parameters" : vdn_rl_paramList,
        "list_type" : VDN_AgentList
    },
    "VDN_VI_Agents" : {
        "learning_agents" : VDN_VI_Agent,
        "parameters" : vdn_vi_paramList,
        "list_type" : VDN_VI_AgentList
    }
}


def hypersearch_baselines() -> None:
    """
    Loop over all baseline algorithms and perform hyperparameter search training.
    """
    for algo_name, algo_data in baselines_experiments.items():
        if algo_data is None:
            continue
        hypersearch_algorithm(algo_name, **algo_data)
    return


def hypersearch_algorithm(
        algorithm: str, 
        learning_agents : BaseAgent, 
        parameters: List[Dict[str, Any]],
        list_type: Any,
        *args, **kwargs) -> None:
    """
    Train a specific baseline algorithm and do hyperparameter search.
    """
    # 1. Create Directory Structure
    # Structure: HyperSearchResults / {Agent_Name} / Search_Results /
    results_dir = os.path.join(RESULTS_DIR, algorithm.replace(" ", "_"))
    os.makedirs(results_dir, exist_ok=True)

    # START - LOOP OVER PARAM SETS
    pbar = tqdm(range(len(parameters)), desc=f"{algorithm} Search")
    
    for idx in pbar:
        params : Dict[str, Any] = parameters[idx]

        results_filepath = os.path.join(results_dir, f"{idx}_results.csv")
        
        # Check if completed
        if results_do_exists(results_path=results_filepath, **params):
            continue

        results_cache = {}

        # START - TRAIN PARAMS ON ALL GAMES
        for game_name in GAMES:
            # Set up Game Instance
            game = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)
            num_cards = game.num_cards
            num_actions = game.num_actions

            # Set up Agents
            agents = setup_agents(
                learning_agents,
                params,
                num_cards,
                num_actions,
                env=game,
                list_type=list_type
            )

            # Update kwargs for search
            run_kwargs = params.copy()
            
            # Handle Model-Based vs Model-Free arguments
            if agents[0].MODEL_BASED:
                if 'iterations' in run_kwargs:
                    run_kwargs['max_iterations'] = run_kwargs.pop('iterations')
            else:
                run_kwargs['train_episodes'] = TRAINING_EPISODES_HYPERSEARCH
            
            # Pass metadata to runner
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
        
        # Save Combined CSV
        # Use pd.Series to handle different convergence lengths
        results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_cache.items()]))
        results_df.to_csv(results_filepath, index=False)

        # Save params used
        params_filepath = os.path.join(results_dir, f"{idx}_params.json")
        with open(params_filepath, 'w') as f:
            json.dump(params, f, indent=4)
            
    return


def results_do_exists(results_path: str, **params) -> bool:
    """
    Checks if results exist and if the parameters match.
    """
    json_path = results_path.replace("_results.csv", "_params.json")
    if not os.path.exists(results_path) or not os.path.exists(json_path):
        return False

    try:
        with open(json_path, 'r') as f:
            saved_params = json.load(f)
    except json.JSONDecodeError:
        return False

    # Check for mismatches
    for key, value in params.items():
        if key not in saved_params or saved_params[key] != value:
            return False
    return True


def setup_agents(
        learning_agents : Any, 
        parameters: Dict[str, Any],
        num_cards: int,
        num_actions: int,
        env: DecPOMDP,
        list_type : Any, 
    ) -> AgentList:
    """
    Set up agents for a specific baseline algorithm.
    """
    # 1. Centralized Lists (VDN)
    if list_type is not AgentList:
        # Check params to guess if Model-Based (heuristic check)
        if "iterations" in parameters or "convergence_threshold" in parameters:
             # Model-Based Centralized
             agents_instances = list_type(
                num_cards=num_cards,
                num_actions=num_actions,
                env=env,
                **parameters
            )
        else:
            # Model-Free Centralized
            agents_instances = list_type(
                num_cards=num_cards,
                num_actions=num_actions,
                **parameters
            )
        agents = agents_instances
        
    # 2. Decentralized (Standard List)
    else:
        if issubclass(learning_agents, ModelBasedAgent):
            # Model-Based Independent
            # FIX: Removed 'agent_id' as they are now role-agnostic/unified
            agents_instances = [
                learning_agents(num_cards, num_actions, env=env, **parameters),
                learning_agents(num_cards, num_actions, env=env, **parameters)
            ]
        else:
            # Model-Free Independent
            agents_instances = [
                learning_agents(num_cards, num_actions, **parameters),
                learning_agents(num_cards, num_actions, **parameters)
            ]
        agents = AgentList(agents_instances)
        
    return agents