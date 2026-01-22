import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Union

# Environment
from tiny_game import GAMES, Settings, GameNames, get_game, DecPOMDP

# Runner
from runner import run_training

# Agents
from agents import (
    AgentList, BaseAgent, ModelBasedAgent,
    Independent_RL_Agent, 
    Independent_VI_Agent,
    VDN_RL_Agent,
    VDN_AgentList,
    VDN_VI_AgentList,
    VDN_VI_Agent
)

from config import TRAINING_EPISODES_FINAL

# --- DIRECTORY CONFIGURATION ---
HYPERSEARCH_DIR = "HyperSearchResults"

# Registry to map string names to Classes
AGENT_REGISTRY = {
    "Independent_RL_Agent": {
        "agent_cls": Independent_RL_Agent,
        "list_cls": AgentList,
        "is_model_based": False
    },
    "Independent_VI_Agent": {
        "agent_cls": Independent_VI_Agent,
        "list_cls": AgentList,
        "is_model_based": True
    },
    "VDN_RL_Agents": {
        "agent_cls": VDN_RL_Agent,
        "list_cls": VDN_AgentList,
        "is_model_based": False
    },
    "VDN_VI_Agents": {
        "agent_cls": VDN_VI_Agent,
        "list_cls": VDN_VI_AgentList,
        "is_model_based": True
    }
}

def load_best_params(agent_name: str) -> Dict[str, Any]:
    """
    Reads from: HyperSearchResults/{Agent_Name}/best_params.json
    """
    folder_name = agent_name.replace(" ", "_")
    path = os.path.join(HYPERSEARCH_DIR, folder_name, "best_params.json")
    
    if not os.path.exists(path):
        return None
        
    with open(path, 'r') as f:
        return json.load(f)

def setup_final_agents(
    registry_entry: Dict, 
    params: Dict, 
    num_cards: int, 
    num_actions: int, 
    env: DecPOMDP
) -> AgentList:
    """
    Instantiates the agents using the loaded parameters.
    """
    agent_cls = registry_entry["agent_cls"]
    list_cls = registry_entry["list_cls"]
    is_model_based = registry_entry["is_model_based"]

    # 1. Centralized / Custom List (VDN)
    if list_cls != AgentList:
        if is_model_based:
            agents = list_cls(num_cards, num_actions, env=env, **params)
        else:
            agents = list_cls(num_cards, num_actions, **params)
    
    # 2. Decentralized Standard List (Independent)
    else:
        if is_model_based:
            # Model-Based Independent (needs env)
            agents_list = [
                agent_cls(num_cards, num_actions, env=env, agent_id=0, **params),
                agent_cls(num_cards, num_actions, env=env, agent_id=1, **params)
            ]
        else:
            # Model-Free Independent
            agents_list = [
                agent_cls(num_cards, num_actions, **params),
                agent_cls(num_cards, num_actions, **params)
            ]
        agents = AgentList(agents_list)

    return agents

def save_final_model(
    agents: AgentList, 
    agent_name: str, 
    game_name: str,
):
    """
    Saves the trained models into Baselines/{Agent_Name}/
    Prefixes files with G_{GameName}_...
    """
    folder_name = agent_name.replace(" ", "_")
    agent_save_dir = os.path.join(HYPERSEARCH_DIR, folder_name)
    os.makedirs(agent_save_dir, exist_ok=True)

    # Save Agents
    if agents.__class__.__name__.startswith("VDN"):
        # Centralized List (Shared Policy)
        path = os.path.join(agent_save_dir, f"G_{game_name}_shared_model.pkl")
        agents.save(path)
    else:
        # Independent Agents (Separate Files)
        for i, agent in enumerate(agents):
            path = os.path.join(agent_save_dir, f"G_{game_name}_agent_{i}.pkl")
            agent.save(path)
            
    return agent_save_dir

def train_test_baselines(): 
    pbar_agents = tqdm(AGENT_REGISTRY.items(), desc="Agent Types")
    
    for agent_name, registry_info in pbar_agents:
        # 1. Load Params
        params = load_best_params(agent_name)
        if params is None:
            continue
            
        pbar_agents.set_description(f"Processing {agent_name}")

        folder_name = agent_name.replace(" ", "_")
        
        # --- SKIP LOGIC: Check if results already exist ---
        final_csv_path = os.path.join(HYPERSEARCH_DIR, folder_name, "final_results.csv")
        if os.path.exists(final_csv_path):
            continue
        
        # Cache for aggregating results across games
        results_cache = {}
        
        # 2. Iterate over All Games
        for game_name in GAMES:
            # Setup Environment
            game = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)
            
            # Setup Agents
            agents = setup_final_agents(
                registry_info, 
                params, 
                game.num_cards, 
                game.num_actions, 
                game
            )

            # Configure Run Arguments
            run_kwargs = params.copy()
            run_kwargs['game_name'] = game_name
            run_kwargs['pbar'] = pbar_agents
            
            if registry_info["is_model_based"]:
                # --- Model Based ---
                # Use iterations from best_params, default to 50
                run_kwargs['max_iterations'] = run_kwargs.get('iterations', 50)
                # Remove config keys not needed by runner
                run_kwargs.pop('iterations', None)
                run_kwargs.pop('convergence_threshold', None) # Passed explicitly if needed, or runner default
            else:
                # --- Model Free ---
                run_kwargs['train_episodes'] = TRAINING_EPISODES_FINAL
                run_kwargs['train_test_freq'] = 1000

            # 3. Run Training
            rewards, losses, trained_agents = run_training(
                env=game,
                agents=agents,
                **run_kwargs
            )

            # 4. Save Final Model (Pickles)
            save_dir = save_final_model(trained_agents, agent_name, game_name)
            
            # 5. Accumulate Results
            # Keys: "reward_A", "loss_A", "reward_B", etc.
            results_cache[f"reward_{game_name}"] = rewards
            results_cache[f"loss_{game_name}"] = losses

        # 6. Save Aggregated Results
        results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results_cache.items()]))
        
        folder_name = agent_name.replace(" ", "_")
        csv_path = os.path.join(HYPERSEARCH_DIR, folder_name, "final_results.csv")
        results_df.to_csv(csv_path, index=False)