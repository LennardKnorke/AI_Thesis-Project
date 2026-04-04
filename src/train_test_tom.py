# train_test_tom.py
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Project-specific imports
from tiny_game import GAMES, GameNames, get_game_Rework, DecPOMDP, MyHanabi, Game, OPTIMAL_RETURNS
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
    tom_results_dir = "ToM_PBVI"
    tom_results_dir = os.path.join(RESULTS_DIR, tom_results_dir)
    os.makedirs(tom_results_dir, exist_ok=True)
    
    final_csv_path = os.path.join(tom_results_dir, "final_results.csv")

    environments = setup_hanabi_environments()

    world_models_by_game = load_all_world_models(device, environments)

    baseline_agents_by_game = setup_baseline_agents(environments)

    ensembles : dict[str, np.ndarray] = load_ensembles()
    special_ensembles : dict[str, dict[str, np.ndarray]] = load_specialized_ensembles()
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
        special_ensemble = special_ensembles[game_name]
        env = environments[game_name]

        # Set up ToM Agent
        tom_agent_p0 = ToM_PBVI_Agent(
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
        tom_agent_p1 = ToM_PBVI_Agent(
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
        start_states = env.start_states()
        # Prep Training + testing
        p0_loss_results = []
        p1_loss_results = []

        reward_results = []

        spc_ens_p0_reward_results = []
        spc_ens_p1_reward_results = []

        mix_ens_p0_reward_results = []
        mix_ens_p1_reward_results = []

        # Planning and Testing Step
        pbar_2 = tqdm(range(5), desc="Iterations", leave=False)

        self_play_reward, spc_ens_p0_reward, spc_ens_p1_reward, mix_ens_p0_reward, mix_ens_p1_reward = None, None, None, None, None
        for it in pbar_2:
            postfix_2 = {
                "Iter" : it,
                "Rew" : self_play_reward if self_play_reward else "None",
                "Rew_p0" : spc_ens_p0_reward if spc_ens_p0_reward else "None",
                "Rew_p1" : spc_ens_p1_reward if spc_ens_p1_reward else "None",
            }
            pbar_2.set_postfix(postfix_2)

            # Planning
            p0_delta = tom_agent_p0.train()
            p0_loss_results.append(p0_delta)
            p1_delta = tom_agent_p1.train()
            p1_loss_results.append(p1_delta)

            rewards = _evaluate_agents(env, 
                                       tom_agent_p0, 
                                       tom_agent_p1, 
                                       agent_list,
                                       start_states,
                                       baseline_agents,
                                       game_name,
                                       ensemble,
                                       special_ensemble)
            
            self_play_reward, spc_ens_p0_reward, spc_ens_p1_reward, mix_ens_p0_reward, mix_ens_p1_reward = rewards
            reward_results.append(self_play_reward)
            spc_ens_p0_reward_results.append(spc_ens_p0_reward)
            spc_ens_p1_reward_results.append(spc_ens_p1_reward)
            mix_ens_p0_reward_results.append(mix_ens_p0_reward)
            mix_ens_p1_reward_results.append(mix_ens_p1_reward)


        # Save Game specific policies
        tom_agent_p0.save(os.path.join(tom_results_dir, f"G_{game_name}_agent_0.pkl"))
        tom_agent_p1.save(os.path.join(tom_results_dir, f"G_{game_name}_agent_1.pkl"))

        # Cache Results
        all_evaluation_results[f"reward_{game_name}"] = reward_results
        all_evaluation_results[f"reward_{game_name}_0_mix"] = mix_ens_p0_reward_results
        all_evaluation_results[f"reward_{game_name}_1_mix"] = mix_ens_p1_reward_results
        all_evaluation_results[f"reward_{game_name}_0_spc"] = spc_ens_p0_reward_results
        all_evaluation_results[f"reward_{game_name}_1_spc"] = spc_ens_p0_reward_results

        all_evaluation_results[f"loss_{game_name}_0"] = p0_loss_results
        all_evaluation_results[f"loss_{game_name}_1"] = p1_loss_results

    # Save Results
    results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_evaluation_results.items()]))
    results_df.to_csv(final_csv_path, index=False)
    return




def load_ensembles():
    """Load the mixed (general) ensemble per game: G_{game}_ensemble.npy"""
    try:
        ensembles = {}
        for gname in GAMES:
            filepath = os.path.join(WORLD_MODELS_DIR, f"G_{gname}_ensemble.npy")
            ensembles[gname] = np.load(filepath)
        return ensembles
    except:
        return None


def load_specialized_ensembles() -> dict[str, dict[str, np.ndarray]]:
    """
    Load per-agent-type ensembles keyed by [game_name][agent_type_name].
    Files: G_{game}_{agent_name.replace(' ','_')}_ensemble.npy
    Missing files are silently omitted; callers fall back to the mixed ensemble.
    """
    specialized: dict[str, dict[str, np.ndarray]] = {}
    for gname in GAMES:
        specialized[gname] = {}
        for exp in BASELINE_EXPERIMENTS:
            safe_name = exp.name.replace(" ", "_")
            fpath = os.path.join(WORLD_MODELS_DIR, f"G_{gname}_{safe_name}_ensemble.npy")
            if os.path.exists(fpath):
                specialized[gname][exp.name] = np.load(fpath)
    return specialized


def _evaluate_agents(
    env: Game,
    tom_agent_p0: ToM_PBVI_Agent,
    tom_agent_p1: ToM_PBVI_Agent,
    agent_list: AgentList,
    start_states: list,
    baseline_agents: dict,
    game_name: str,
    mixed_ensemble: np.ndarray,
    specialized_ensembles: dict[str, np.ndarray] | None,
) -> tuple[float, float, float]:
    optimal = OPTIMAL_RETURNS[game_name]
    n_starts = len(start_states)   

    # Self play - Mixed ensemble
    tom_agent_p0.set_ensemble(mixed_ensemble)
    tom_agent_p1.set_ensemble(mixed_ensemble)
    self_play_total = 0.0
    for s0 in start_states:
        episode = run_episode(env, agent_list, s0, True)
        self_play_total += episode[-1]
    self_play_total = self_play_total / n_starts / optimal


    p0_spc_total = 0.0
    p1_spc_total = 0.0
    p0_mix_total = 0.0
    p1_mix_total = 0.0

    for base_type, base_agent_list in baseline_agents.items():
        ens = specialized_ensembles[base_type]

        for s0 in start_states:
            # Run with mixed ensemble
            tom_agent_p0.set_ensemble(mixed_ensemble)
            tom_agent_p1.set_ensemble(mixed_ensemble)
        
            p0_mix_total += run_episode(env, AgentList([tom_agent_p0, base_agent_list[1]]), s0, True)[-1]
            p1_mix_total += run_episode(env, AgentList([base_agent_list[0], tom_agent_p1]), s0, True)[-1]
            
            # Run with specialized ensemble
            tom_agent_p0.set_ensemble(ens)
            tom_agent_p1.set_ensemble(ens)

            p0_spc_total += run_episode(env, AgentList([tom_agent_p0, base_agent_list[1]]), s0, True)[-1]
            p1_spc_total += run_episode(env, AgentList([base_agent_list[0], tom_agent_p1]), s0, True)[-1]

    # Restore mixed for the next planning sweep / cooperative eval
    tom_agent_p0.set_ensemble(mixed_ensemble)
    tom_agent_p1.set_ensemble(mixed_ensemble)

    n_types = len(baseline_agents)
    p0_spc_total = p0_spc_total / (n_starts * n_types) / optimal
    p1_spc_total = p1_spc_total / (n_starts * n_types) / optimal
    return self_play_total, p0_spc_total, p1_spc_total, p0_mix_total, p1_mix_total
    

if __name__ == "__main__":
    train_test_tom()