import json
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from typing import Dict, Any, List, Optional

# Own Imports
from tiny_game import GAMES, Settings, GameNames, DecPOMDP, get_game
from runner import run_training
from config import (
    TRAINING_EPISODES_HYPERSEARCH,
    TOM_AGENT_EXPERIMENT, Experiment, BASELINE_EXPERIMENTS
)
from agents import DTDE_ToMBI_Agent, ToM_WorldModel # Your ToM agent and world model
from agents.model_based.dtde_ToM import OBS_DIM # Import constants for model initialization
from hypersearch_baselines import results_do_exists # Reuse helper function


RESULTS_DIR = "HyperSearchResults/"
WORLD_MODELS_DIR = "Results/WorldModels/" # Directory where best ToM World Models are saved


def hypersearch_tom() -> None:
    """
    Performs hyperparameter search for the Theory of Mind (ToM) agent.
    
    This function iterates through defined ToM agent parameters, and for each set,
    it trains the ToM agent against all specified games, leveraging the pre-trained
    ToM World Model for that specific game.
    """
    exp = TOM_AGENT_EXPERIMENT # Get the defined ToM agent experiment
    
    # 1. Create Directory Structure
    # Structure: HyperSearchResults / {Agent_Name}
    results_dir = os.path.join(RESULTS_DIR, exp.name.replace(" ", "_"))
    os.makedirs(results_dir, exist_ok=True)

    # Determine num_agent_types (needed for ToM_WorldModel initialization)
    num_agent_types = len(BASELINE_EXPERIMENTS)

    # START - LOOP OVER PARAM SETS FOR THE TOM AGENT
    print(f"\nHyperparameter Search for {exp.name}")
    pbar = tqdm(range(len(exp.param_list)))
    
    for idx in pbar:
        params: Dict[str, Any] = exp.param_list[idx]

        results_file = os.path.join(results_dir, f"{idx}_results.csv")
        
        # Check if completed already
        if results_do_exists(results_path=results_file, **params):
            continue
        results_cache = {}

        # START - TRAIN CURRENT PARAM SET ON ALL GAMES
        for game_name in GAMES:
            # 1. Set up Game Environment
            ENV = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)

            # 2. Load the best pre-trained ToM World Model parameters for THIS game
            world_model_path = os.path.join(WORLD_MODELS_DIR, f"ToM_WorldModel_{game_name}.pth")
            world_model_params_path = os.path.join(WORLD_MODELS_DIR, f"best_params.json")
            
            if not os.path.exists(world_model_path) or not os.path.exists(world_model_params_path):
                print(f"Warning: World Model or its parameters not found for {game_name}. Skipping game.")
                # Fill with NaNs for this game's results
                results_cache[f"reward_{game_name}"] = [np.nan] * 1 
                results_cache[f"loss_{game_name}"] = [np.nan] * 1
                continue
            
            # Load the parameters used to train this specific world model instance
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
            # Move to appropriate device (e.g., CPU, or GPU if specified in config)
            world_model.to("cpu") # For now, assume CPU for ToMBI agent's inference
            world_model.eval() # Set to eval mode for inference

            # 3. Prepare run arguments for the ToM agent
            run_kwargs = params.copy()
            run_kwargs['world_model'] = world_model # Inject the loaded world model
            run_kwargs['device'] = "cpu" # Agent will use this device for its tensors

            # 4. Instantiate ToM Agents (both P0 and P1 are ToMBI agents in this setup)
            AGENTS = exp.make_agents(ENV, run_kwargs)

            # 5. Pass metadata to runner
            run_kwargs['pbar'] = pbar
            run_kwargs['game_name'] = game_name
            
            # 6. Run Training (Model-Based Planning)
            game_rewards, game_losses, _ = run_training(
                env=ENV,
                agents=AGENTS,
                **run_kwargs
            )

            # 7. Update Results Cache
            results_cache[f"reward_{game_name}"] = game_rewards
            results_cache[f"loss_{game_name}"] = game_losses
        
        # END - TRAIN CURRENT PARAM SET ON ALL GAMES
        
        # 8. Save Combined CSV
        max_len = 0
        for k in results_cache:
            if isinstance(results_cache[k], np.ndarray):
                max_len = max(max_len, len(results_cache[k]))
            elif isinstance(results_cache[k], list): 
                max_len = max(max_len, len(results_cache[k]))

        for k in results_cache:
            if isinstance(results_cache[k], np.ndarray):
                current_len = len(results_cache[k])
                if current_len < max_len:
                    padding = np.full(max_len - current_len, np.nan)
                    results_cache[k] = np.concatenate([results_cache[k], padding])
            elif isinstance(results_cache[k], list):
                 current_len = len(results_cache[k])
                 if current_len < max_len:
                     results_cache[k].extend([np.nan] * (max_len - current_len))

        results_df = pd.DataFrame(results_cache)
        results_df.index.name = "iteration" 
        results_df.reset_index(inplace=True)
        results_df.to_csv(results_file, index=False)

        # 9. Save parameters used
        params_filepath = os.path.join(results_dir, f"{idx}_params.json")
        with open(params_filepath, 'w') as f:
            params_to_save = {k: v for k, v in params.items() if k not in ['world_model', 'device']}
            json.dump(params_to_save, f, indent=4)
            
    print(f"\nHyperparameter search for {exp.name} completed.")
    return