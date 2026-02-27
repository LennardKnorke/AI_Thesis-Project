# train_test_tom.py
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Project-specific imports
from tiny_game import GAMES, Settings, GameNames, get_game_Rework, DecPOMDP, MyHanabi
from runner import run_training # Make sure this is the MODIFIED runner.py for player_id
from runner import run_episode # Also ensure this is the MODIFIED one
from agents import *
from config import *
from train_worldmodel import setup_baseline_agents # Function to load pre-trained baseline agents

# --- Helper functions ---
def load_world_model_and_config(game_name: str, device: str) -> tuple[ToM_WorldModel, dict[str, Any]]:
    """
    Loads a specific pre-trained ToM_WorldModel and its configuration for a given game.
    The world model's architecture parameters are derived from the environment structure
    and the 'best_params.json' used during WM training.
    """
    wm_best_params_path = os.path.join(WORLD_MODELS_DIR, "best_params.json")
    if not os.path.exists(wm_best_params_path):
        raise FileNotFoundError(f"World model best_params.json not found at {wm_best_params_path}. "
                                f"Please ensure the world model has been trained.")
    
    with open(wm_best_params_path, 'r') as f:
        wm_training_params = json.load(f)

    dummy_env = get_game_Rework(GameNames(game_name), Settings.decpomdp, normalize=False)
    dummy_action_output_dim = dummy_env.num_actions

    if isinstance(dummy_env, DecPOMDP):
        num_cards_in_focal_hand = 1
        env_max_card_value = dummy_env.num_cards
        dummy_max_seq_length = dummy_env.horizon - 1 
    elif isinstance(dummy_env, MyHanabi):
        num_cards_in_focal_hand = 2
        env_max_card_value = dummy_env.max_card_value
        dummy_max_seq_length = dummy_env.horizon - 3
    else:
        raise ValueError("Unsupported environment type encountered when deriving WM config.")

    dummy_feat_dim = dummy_action_output_dim
    if isinstance(dummy_env, DecPOMDP):
        dummy_feat_dim += env_max_card_value
    else:
        dummy_feat_dim += (num_cards_in_focal_hand * env_max_card_value)

    wm_config = {
        'feat_dim': dummy_feat_dim,
        'action_output_dim': dummy_action_output_dim,
        'max_seq_len': dummy_max_seq_length,
        'num_agent_types': len(BASELINE_EXPERIMENTS),
        'char_embed_dim': wm_training_params['char_dim'],
        'mental_embed_dim': wm_training_params['mental_dim'],
        'trunk_dim': wm_training_params['trunk_dim']
    }

    # CRITICAL FIX: Use the correct parameter names for ToM_WorldModel __init__
    world_model = ToM_WorldModel(
        obs_dim=wm_config['feat_dim'], # Corrected name
        action_dim=wm_config['action_output_dim'], # Corrected name
        num_agent_types=wm_config['num_agent_types'],
        max_seq_len=wm_config['max_seq_len'],
        char_embed_dim=wm_config['char_embed_dim'],
        mental_embed_dim=wm_config['mental_embed_dim'],
        trunk_dim=wm_config['trunk_dim']
    )
    
    wm_path = os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth")
    if not os.path.exists(wm_path):
        raise FileNotFoundError(f"World model .pth file not found for game {game_name} at {wm_path}. "
                                f"Please ensure world models are trained via `train_worldmodel.py`.")

    world_model.load_state_dict(torch.load(wm_path, map_location=device))
    world_model.to(device)
    world_model.eval()

    return world_model, wm_config

def load_all_world_models(device: str) -> dict[str, tuple[ToM_WorldModel, dict[str, Any]]]:
    """
    Loads all pre-trained ToM_WorldModel instances and their configurations,
    keyed by game name. Robustly handles missing or erroneous models.
    """
    all_world_models: dict[str, tuple[ToM_WorldModel, dict[str, Any]]] = {}
    print("\nLoading all World Models...")
    for game_name in GAMES:
        wm, wm_config = load_world_model_and_config(game_name, device)
        all_world_models[game_name] = (wm, wm_config)
    print("Finished loading World Models.")
    return all_world_models
# --- End of helper functions ---


# Define the ToM Experiment (for the Experiment registry)
TOM_EXPERIMENT = Experiment(
    name="DTDE ToMBI",
    agent_class=DTDE_ToMBI_Agent, # Refers to the individual agent class
    # max_iterations for model-based planning (BFS + BI) and attempts for random restarts (not applicable for BI usually)
    param_list=[{"gamma": 0.99, "max_iterations": 1, "attempts": 1}], 
    list_class=ToMBI_AgentList # This is the wrapper for the single ToMBI agent
)

def train_test_tom():
    """
    Main function to train and test the DTDE ToMBI agent across all games.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tom_folder_name = TOM_EXPERIMENT.name.replace(" ", "_")
    tom_results_dir = os.path.join(RESULTS_DIR, tom_folder_name)
    os.makedirs(tom_results_dir, exist_ok=True)
    
    # This CSV will store the results of the ToMBI agent *against each baseline*
    final_csv_path = os.path.join(tom_results_dir, "final_results.csv")

    world_models_by_game = load_all_world_models(device)
    baseline_agents_by_game = setup_baseline_agents() # This is dict[game_name, dict[baseline_name, AgentList]]

    if not world_models_by_game:
        print("No world models loaded. Exiting ToMBI training.")
        return
    if not baseline_agents_by_game:
        print("No baseline agents loaded. Cannot evaluate ToMBI. Exiting.")
        return

    print(f"\nStarting training and evaluation for {TOM_EXPERIMENT.name} agents...")
    
    all_evaluation_results = {}

    pbar_games = tqdm(GAMES, desc=f"Processing Games for {TOM_EXPERIMENT.name}")
    for game_name in pbar_games:
        if game_name not in world_models_by_game:
            pbar_games.write(f"[SKIPPING] Game {game_name}: No World Model loaded.")
            continue
        if game_name not in baseline_agents_by_game:
            pbar_games.write(f"[SKIPPING] Game {game_name}: No Baseline Agents loaded.")
            continue
        
        # Select Environment
        env = get_game_Rework(GameNames(game_name), Settings.decpomdp, normalize=False)
        world_model, wm_config = world_models_by_game[game_name]
        baseline_agents = baseline_agents_by_game[game_name]

        # Set up ToM Agent
        tom_params = TOM_EXPERIMENT.param_list[0].copy()
        tom_agent = DTDE_ToMBI_Agent(
            env=env,
            num_cards = 5,
            num_actions=4,
            world_model=world_model,
            world_model_config=wm_config,
            device=device
        )
        agent_list : ToMBI_AgentList = ToMBI_AgentList(tom_agent, baseline_agents)

        # Prep Training + testing
        test_results = []
        train_results = []
        start_states = env.start_states()

        # Planning and Testing Step
        for it in range(tom_params['max_iterations']):
            # Planning
            train_loss = tom_agent.train()
            train_results.append(train_loss)

            # Testing
            it_reward = 0.0
            n_rewards = len(start_states) * len(agent_list.baseline_agents.keys()) * 2
            for s0 in start_states:
                for b_agent in agent_list.baseline_agents.keys():
                    for side in [0,1]:
                        agent_list.set_current_partner(b_agent, side)
                        episode = run_episode(env, agent_list, s0, True)

                        it_reward += episode[-1]
                    #
            it_reward = it_reward // n_rewards
            test_results.append(it_reward)
        # Save Game specific policies
        agent_statedict_path = os.path.join(tom_results_dir, f"G_{game_name}_agent.pkl")
        agent_list.save(agent_statedict_path)

        all_evaluation_results[f"reward_{game_name}"] = test_results
        all_evaluation_results[f"loss_{game_name}"] = train_loss

    # Save Results
    results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_evaluation_results.items()]))
    results_df.to_csv(final_csv_path, index=False)
