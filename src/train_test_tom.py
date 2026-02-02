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
from config import *


# --- DIRECTORY CONFIGURATION ---
HYPERSEARCH_TOM_DIR = "HyperSearchResults/DTDE_ToMBI"
WORLD_MODELS_DIR = "Results/WorldModels"
BASELINE_MODELS_DIR = "Results"
FINAL_RESULTS_TOM_DIR = "Results/ToM_Agent"


N_EVAL_EPISODES = 100
NUM_AGENT_TYPES = len(BASELINE_EXPERIMENTS)


def load_world_model_for_game(game_name: str, device: str = "cpu") -> Optional[ToM_WorldModel]:
    """Loads a pre-trained ToM_WorldModel for a given game."""
    world_model_path = os.path.join(WORLD_MODELS_DIR, f"ToM_WorldModel_{game_name}.pth")
    # Global best_params.json for the World Model, not specific to game
    world_model_global_params_path = os.path.join(WORLD_MODELS_DIR, f"best_params.json") 

    if not os.path.exists(world_model_path) or not os.path.exists(world_model_global_params_path):
        print(f"Warning: World Model ({world_model_path}) or its global best parameters ({world_model_global_params_path}) not found for {game_name}.")
        return None
    
    # Load the global best parameters that define the world model's architecture
    with open(world_model_global_params_path, 'r') as f:
        wm_params = json.load(f)

    # Get num_actions for the specific game
    dummy_game = get_game(GameNames(game_name), Settings.decpomdp)
    num_actions = dummy_game.num_actions

    # Instantiate ToM_WorldModel with the CORRECT parameters
    world_model = ToM_WorldModel(
        obs_dim=OBS_DIM, 
        action_dim=num_actions,
        num_agent_types=NUM_AGENT_TYPES,
        char_embed_dim=wm_params['char_dim'],
        mental_embed_dim=wm_params['mental_dim']
    )
    world_model.load_state_dict(torch.load(world_model_path, map_location=device))
    world_model.to(device)
    world_model.eval() # Set to eval mode for inference
    return world_model


def save_tom_agent_policy(agents: AgentList, agent_name: str, game_name: str):
    """Saves the trained ToM agent's policy for a specific game.
       Assumes both agents in the AgentList share the same underlying policy for DTDE_ToMBI."""
    os.makedirs(FINAL_RESULTS_TOM_DIR, exist_ok=True)
    # Since DTDE_ToMBI_Agent instances share the policy during training,
    # saving one is sufficient.
    if agents and isinstance(agents[0], DTDE_ToMBI_Agent):
        path = os.path.join(FINAL_RESULTS_TOM_DIR, f"G_{game_name}_ToMBI_policy.pkl")
        agents[0].save(path) 
        return path
    else:
        raise TypeError("Attempted to save non-ToMBI agent or empty AgentList with ToMBI save function.")


def load_baseline_agents_for_game(exp: Experiment, game_name: str, env: DecPOMDP) -> Optional[AgentList]:
    """Loads a trained baseline agent (or AgentList) for a given game."""
    
    # 1. Load best params for this baseline agent from hypersearch
    # This comes from HyperSearchResults/{AGENT_NAME}/best_params.json
    hypersearch_baseline_dir = os.path.join("HyperSearchResults", exp.name.replace(" ", "_"))
    baseline_params_path = os.path.join(hypersearch_baseline_dir, "best_params.json")

    if not os.path.exists(baseline_params_path):
        print(f"Warning: Best params not found for baseline {exp.name}.")
        return None
    
    with open(baseline_params_path, 'r') as f:
        baseline_params = json.load(f)
    
    # 2. Instantiate agents (unloaded) using the Experiment factory
    agents = exp.make_agents(env, baseline_params)

    # 3. Load weights from final results directory
    final_baseline_load_dir = os.path.join(BASELINE_MODELS_DIR, exp.name.replace(" ", "_"))

    try:
        if exp.list_class.__name__.startswith("CTDE"): # Centralized agents (e.g., VDN, CTDE BI)
            path = os.path.join(final_baseline_load_dir, f"G_{game_name}_shared_model.pkl")
            if os.path.exists(path):
                agents.load(path)
                return agents
        else: # Decentralized agents (e.g., QSarsa, DTDE BI)
            path_p0 = os.path.join(final_baseline_load_dir, f"G_{game_name}_agent_0.pkl")
            path_p1 = os.path.join(final_baseline_load_dir, f"G_{game_name}_agent_1.pkl")
            if os.path.exists(path_p0) and os.path.exists(path_p1):
                agents[0].load(path_p0)
                agents[1].load(path_p1)
                return agents
        
    except Exception as e:
        print(f"Error loading {exp.name} for {game_name}: {e}")
        return None
    
    print(f"Could not find saved model files for baseline {exp.name} for game {game_name}.")
    return None


# --- MAIN FUNCTION ---
def train_test_tom():
    os.makedirs(FINAL_RESULTS_TOM_DIR, exist_ok=True)
    
    print("\n--- Training and Testing the Theory of Mind Agent ---")
    
    # 1. Load Best Hyperparameters for the ToM Agent
    # This comes from HyperSearchResults/DTDE_ToMBI/best_params.json
    tom_best_params_path = os.path.join(HYPERSEARCH_TOM_DIR, "best_params.json")
    if not os.path.exists(tom_best_params_path):
        print(f"❌ No best_params.json found for {TOM_AGENT_EXPERIMENT.name} at {tom_best_params_path}. Run hypersearch_tom first.")
        return
    
    with open(tom_best_params_path, 'r') as f:
        tom_best_params = json.load(f)
    print(f"Loaded Best ToM Agent Params: {tom_best_params}")

    comparison_results = [] # To store all evaluation metrics

    # 2. Loop through each Game
    for game_name in tqdm(GAMES, desc="Processing Games"):
        print(f"\n--- Game: {game_name} ---")
        env = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)

        # 2a. Load the pre-trained ToM World Model for this game
        world_model = load_world_model_for_game(game_name, device="cpu") # ToM Agent runs on CPU
        if world_model is None:
            print(f"   Skipping {game_name} due to missing World Model.")
            continue
        
        # 2b. Train the ToM Agent for this specific game
        print(f"   Training ToM Agent for {game_name}...")
        tom_run_args = tom_best_params.copy()
        tom_run_args['world_model'] = world_model
        tom_run_args['device'] = "cpu" # Device for ToM agent's internal tensors (e.g., for World Model inference)
        
        # Instantiate ToM agents (both P0 and P1 will be DTDE_ToMBI_Agent instances)
        tom_agents_for_training = TOM_AGENT_EXPERIMENT.make_agents(env, tom_run_args)
        
        # Run training (Backward Induction is a "planning" phase)
        # The rewards/losses here are from the planning convergence, not direct gameplay
        _, _, trained_tom_agents = run_training(
            env=env,
            agents=tom_agents_for_training,
            game_name=game_name,
            **tom_run_args # Pass max_iterations, convergence_threshold, attempts
        )
        
        # Save the trained ToM agent's policy
        tom_policy_path = save_tom_agent_policy(trained_tom_agents, TOM_AGENT_EXPERIMENT.name, game_name)
        print(f"   Saved trained ToM Agent policy to {tom_policy_path}")

        # 2c. Evaluation: ToM Agent vs. All Baseline Agents
        print(f"   Evaluating ToM Agent against baselines for {game_name}...")

        # Reload the ToM agent to ensure it's using the saved policy (good practice)
        # Or, just use 'trained_tom_agents' returned from run_training.
        # For simplicity, let's use 'trained_tom_agents' directly for this run.

        for baseline_exp in BASELINE_EXPERIMENTS:
            baseline_name = baseline_exp.name
            
            # Load the trained baseline agent(s) for this specific game
            baseline_agents = load_baseline_agents_for_game(baseline_exp, game_name, env)
            if baseline_agents is None:
                print(f"      Skipping comparison with {baseline_name} (could not load baseline).")
                continue

            # --- Evaluation Scenario 1: ToM (P0) vs. Baseline (P1) ---
            # Create a temporary AgentList for this specific pair
            current_eval_agents_p0_tom = AgentList([trained_tom_agents[0], baseline_agents[1]]) # ToM is P0, Baseline is P1
            total_reward_p0_tom = 0.0
            for _ in range(N_EVAL_EPISODES):
                env.reset()
                total_reward_p0_tom += run_episode(env, current_eval_agents_p0_tom, test_episode=True)
            avg_reward_p0_tom = total_reward_p0_tom / N_EVAL_EPISODES

            # --- Evaluation Scenario 2: Baseline (P0) vs. ToM (P1) ---
            # Create a temporary AgentList for this specific pair
            current_eval_agents_p1_tom = AgentList([baseline_agents[0], trained_tom_agents[1]]) # Baseline is P0, ToM is P1
            total_reward_p1_tom = 0.0
            for _ in range(N_EVAL_EPISODES):
                env.reset()
                total_reward_p1_tom += run_episode(env, current_eval_agents_p1_tom, test_episode=True)
            avg_reward_p1_tom = total_reward_p1_tom / N_EVAL_EPISODES

            comparison_results.append({
                "game": game_name,
                "tom_agent_type": TOM_AGENT_EXPERIMENT.name,
                "partner_agent_type": baseline_name,
                "tom_role": "P0",
                "partner_role": "P1",
                "avg_reward": avg_reward_p0_tom
            })
            comparison_results.append({
                "game": game_name,
                "tom_agent_type": TOM_AGENT_EXPERIMENT.name,
                "partner_agent_type": baseline_name,
                "tom_role": "P1",
                "partner_role": "P0",
                "avg_reward": avg_reward_p1_tom
            })
            print(f"      vs. {baseline_name} | ToM P0: {avg_reward_p0_tom:.2f} | ToM P1: {avg_reward_p1_tom:.2f}")

    # 3. Save All Comparison Results to CSV
    if comparison_results:
        results_df = pd.DataFrame(comparison_results)
        results_csv_path = os.path.join(FINAL_RESULTS_TOM_DIR, "ToM_Agent_Comparison_Results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"\n✅ All ToM Agent comparison results saved to {results_csv_path}")
    else:
        print("\n⚠️ No comparison results generated.")

    print("\n--- ToM Agent Training and Testing Complete ---")