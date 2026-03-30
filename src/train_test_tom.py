# train_test_tom.py
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Project-specific imports
from tiny_game import GAMES, GameNames, get_game_Rework, DecPOMDP, MyHanabi, Game
from runner import run_training # Make sure this is the MODIFIED runner.py for player_id
from runner import run_episode # Also ensure this is the MODIFIED one
from agents import *
from config import *
from train_worldmodel import setup_baseline_agents, setup_hanabi_environments


# --- Helper functions ---
def load_world_model_and_config(game_name: str, device: str, env : Game) -> tuple[ToM_WorldModel, dict[str, Any]]:
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
    
    ACT_DIM = env.num_actions + 1   # Num of possible actions + Null action
    if isinstance(env, DecPOMDP):
        start_len = 2
        MAX_SEQ_LEN = env.horizon - 1
        obs_act_dim = env.num_actions
        obs_card_dim = env.num_cards * 2

    elif isinstance(env, MyHanabi):
        start_len = 4
        MAX_SEQ_LEN = env.horizon - 3
        obs_act_dim = env.num_actions + env.num_cards + 1
        obs_card_dim = env.num_cards * start_len
    OBS_DIM = obs_act_dim + obs_card_dim
    JOINT_OBS_DIM = OBS_DIM


    wm_config = {
        'obs_dim': OBS_DIM,
        "joint_obs_dim" : JOINT_OBS_DIM,
        'action_dim': ACT_DIM,
        'max_seq_len': MAX_SEQ_LEN,
        'num_agent_types': len(BASELINE_EXPERIMENTS),
        'char_embed_dim': wm_training_params['char_dim'],
        'mental_embed_dim': wm_training_params['mental_dim'],
        'trunk_dim': wm_training_params['trunk_dim'],
        'action_output_dim' : ACT_DIM
    }

    # CRITICAL FIX: Use the correct parameter names for ToM_WorldModel __init__
    world_model = ToM_WorldModel(**wm_config)
    
    wm_path = os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth")
    if not os.path.exists(wm_path):
        raise FileNotFoundError(f"World model .pth file not found for game {game_name} at {wm_path}. "
                                f"Please ensure world models are trained via `train_worldmodel.py`.")

    world_model.load_state_dict(torch.load(wm_path, map_location=device))
    world_model.to(device)
    world_model.eval()

    return world_model, wm_config

def load_all_world_models(device: str, environments) -> dict[str, tuple[ToM_WorldModel, dict[str, Any]]]:
    """
    Loads all pre-trained ToM_WorldModel instances and their configurations,
    keyed by game name. Robustly handles missing or erroneous models.
    """
    all_world_models: dict[str, tuple[ToM_WorldModel, dict[str, Any]]] = {}
    print("\nLoading all World Models...")
    for game_name in GAMES:
        wm, wm_config = load_world_model_and_config(game_name, device, environments[game_name])
        all_world_models[game_name] = (wm, wm_config)
    print("Finished loading World Models.")
    return all_world_models
# --- End of helper functions ---



def train_test_tom():
    """
    Main function to train and test the DTDE ToMBI agent across all games.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Path Preliminaries
    tom_results_dir = "DTDE ToMBI".replace(" ", "_")
    tom_results_dir = os.path.join(RESULTS_DIR, tom_results_dir)
    os.makedirs(tom_results_dir, exist_ok=True)
    
    final_csv_path = os.path.join(tom_results_dir, "final_results.csv")

    environments = setup_hanabi_environments()

    world_models_by_game = load_all_world_models(device, environments)

    baseline_agents_by_game = setup_baseline_agents(environments)

    ensembles : dict[str, np.ndarray] = load_ensembles()

    if not world_models_by_game or not baseline_agents_by_game or not ensembles:
        print("Missing Components. Exiting ToMBI training.")
        return

    # Train and evaluate    
    all_evaluation_results = {}
    pbar = tqdm(GAMES, desc=f"DTDE_ToMBI")
    for game_name in pbar:
        postfix = {
            "Game" : game_name
        }
        pbar.set_postfix(postfix)

        if game_name not in world_models_by_game:
            pbar.write(f"[SKIPPING] Game {game_name}: No World Model loaded.")
            continue
        if game_name not in baseline_agents_by_game:
            pbar.write(f"[SKIPPING] Game {game_name}: No Baseline Agents loaded.")
            continue
        if game_name not in ensembles:
            pbar.write(f"[SKIPPING] Game {game_name}: No Ensemble")
            continue

        # Select Environment
        baseline_agents = baseline_agents_by_game[game_name]
        world_model, wm_config = world_models_by_game[game_name]
        ensemble = ensembles[game_name]
        env = environments[game_name]

        # Set up ToM Agent
        tom_agent_p0 = DTDE_ToMBI_Agent(
            env=env,
            num_cards = env.num_cards,
            ensemble=ensemble,
            num_actions=env.num_actions,
            world_model=world_model,
            world_model_config=wm_config,
            device=device,
            gamma=0.99,
            agent_id = 0
        )
        tqdm.write(f"{game_name} | Num. s0 : {len(env.start_states())} | Num. S {len(tom_agent_p0.all_joint_histories)} | Num priv h {len(tom_agent_p0.all_private_histories)}")
        tom_agent_p1 = DTDE_ToMBI_Agent(
            env=env,
            num_cards = env.num_cards,
            ensemble=ensemble,
            num_actions=env.num_actions,
            world_model=world_model,
            world_model_config=wm_config,
            device=device,
            gamma=0.99,
            agent_id = 1
        )
        agent_list : AgentList = AgentList([tom_agent_p0, tom_agent_p1])

        # Prep Training + testing
        p0_loss_results = []
        p1_loss_results = []

        reward_results = []
        p0_reward_results = []
        p1_reward_results = []
        start_states = env.start_states()

        # Planning and Testing Step
        pbar_2 = tqdm(range(5), desc="Iterations", leave=False)

        p0_reward = None
        p1_reward = None
        reward = None
        for it in pbar_2:
            postfix_2 = {
                "Iter" : it,
                "Rew" : reward if reward else "None",
                "Rew_p0" : p0_reward if p0_reward else "None",
                "Rew_p1" : p1_reward if p1_reward else "None",
            }
            pbar_2.set_postfix(postfix_2)

            # Planning
            p0_delta = tom_agent_p0.train()
            p0_loss_results.append(p0_delta)
            p1_delta = tom_agent_p1.train()
            p1_loss_results.append(p1_delta)

            # Testing
            p0_reward = 0.0
            p1_reward = 0.0
            reward = 0.0

            
            for s0 in start_states:
                ep = run_episode(env, agent_list, s0, True)
                reward += ep[-1]

                for base_agent_type in baseline_agents.keys():
                    p0_agent_list = AgentList([tom_agent_p0, baseline_agents[base_agent_type][1]])
                    p1_agent_list = AgentList([baseline_agents[base_agent_type][0], tom_agent_p1])

                    p0_ep = run_episode(env, p0_agent_list, s0, True)
                    p0_reward += p0_ep[-1]
                    p1_ep = run_episode(env, p1_agent_list, s0, True)
                    p1_reward += p1_ep[-1]

            # Average reward
            reward = reward / len(start_states)
            p0_reward = p0_reward / (len(start_states) * len(baseline_agents.keys()))
            p1_reward = p1_reward / (len(start_states) * len(baseline_agents.keys()))
            reward_results.append(reward)
            p0_reward_results.append(p0_reward)
            p1_reward_results.append(p1_reward)

        # Save Game specific policies
        tom_agent_p0.save(os.path.join(tom_results_dir, f"G_{game_name}_agent_0.pkl"))
        tom_agent_p1.save(os.path.join(tom_results_dir, f"G_{game_name}_agent_1.pkl"))

        # Cache Results
        all_evaluation_results[f"reward_{game_name}"] = reward_results
        all_evaluation_results[f"reward_{game_name}_0"] = p0_reward_results
        all_evaluation_results[f"reward_{game_name}_1"] = p1_reward_results
        all_evaluation_results[f"loss_{game_name}_0"] = p0_loss_results
        all_evaluation_results[f"loss_{game_name}_1"] = p1_loss_results

    # Save Results
    results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_evaluation_results.items()]))
    results_df.to_csv(final_csv_path, index=False)
    return




def load_ensembles():
    try:
        ensembles = {}
        for gname in GAMES:
            filepath = os.path.join(WORLD_MODELS_DIR, f"G_{gname}_ensemble.npy")
            e = np.load(filepath)
            ensembles[gname] = e
        return ensembles
    except:
        return None
    

if __name__ == "__main__":
    train_test_tom()