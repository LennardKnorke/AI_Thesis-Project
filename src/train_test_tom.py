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

# DEVELOPMENTAL PLACEHOLDER #REMOVE LATER
BASELINES_TO_SKIP = ['OSarsa', 'MAA', 'MAA*', 'JESP', 'JEPS']

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
        'lr' : wm_training_params['lr'],
        
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
    pbar = tqdm(GAMES, desc=f"ToM-PBVI")
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
        for key in BASELINES_TO_SKIP:
            if key in baseline_agents.keys():
                baseline_agents.pop(key)
        world_model, wm_config = world_models_by_game[game_name]
        ensemble = ensembles[game_name]
        special_ensemble = special_ensembles[game_name]
        env = environments[game_name]

        # Set up ToM Agent — single instance covers both player positions.
        # All ensembles (mixed + per-type specialized) are passed so the agent
        # reasons jointly over type uncertainty during planning and execution.
        tom_agent = ToM_PBVI_Agent(
            env=env,
            num_cards=env.num_cards,
            ensemble=ensemble,
            spec_ensembles=special_ensemble,
            num_actions=env.num_actions,
            world_model=world_model,
            world_model_config=wm_config,
            device=device,
            gamma=0.99,
        )
        tqdm.write(f"{game_name} | Num. s0 : {len(env.start_states())} | Num. S {len(tom_agent.all_joint_histories)} | Num priv h {len(tom_agent.all_private_histories)}")
        agent_list : AgentList = AgentList([tom_agent, tom_agent])
        start_states = env.start_states()

        # Result accumulators
        loss_results      = []
        reward_results    = []
        p0_reward_results = []
        p1_reward_results = []

        # Planning and Testing Step
        pbar_2 = tqdm(range(50), desc="Iterations", leave=False)

        self_play_reward, p0_reward, p1_reward = None, None, None
        for it in pbar_2:
            pbar_2.set_postfix({
                "Iter" : it,
                "Rew"  : f"{self_play_reward:.3f}" if self_play_reward is not None else "—",
                "P0"   : f"{p0_reward:.3f}"        if p0_reward        is not None else "—",
                "P1"   : f"{p1_reward:.3f}"        if p1_reward        is not None else "—",
            })

            # Planning — joint (type × world) PBVI sweep
            delta = tom_agent.train()
            loss_results.append(delta)

            self_play_reward, p0_reward, p1_reward = _evaluate_agents(
                env, tom_agent, agent_list, start_states, baseline_agents, game_name
            )
            reward_results.append(self_play_reward)
            p0_reward_results.append(p0_reward)
            p1_reward_results.append(p1_reward)

            if delta == 0.0 and it != 49:
                tqdm.write(f"Game {game_name} - Finished early at iter {it + 1}")
                break

        # Save Game specific policy
        tom_agent.save(os.path.join(tom_results_dir, f"G_{game_name}_agent.pkl"))

        # Cache Results
        all_evaluation_results[f"reward_{game_name}"]    = reward_results
        all_evaluation_results[f"reward_{game_name}_p0"] = p0_reward_results
        all_evaluation_results[f"reward_{game_name}_p1"] = p1_reward_results
        all_evaluation_results[f"loss_{game_name}"]      = loss_results

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
    tom_agent: ToM_PBVI_Agent,
    agent_list: AgentList,
    start_states: list,
    baseline_agents: dict,
    game_name: str,
) -> tuple[float, float, float]:
    """Evaluate the ToM agent using its joint (type × world) belief.

    The agent no longer needs an explicit ensemble swap — it infers the
    partner type online via its pre-computed type posteriors.

    Returns
    -------
    (self_play, cross_play_p0, cross_play_p1)
        Each normalised by the optimal return for the game.
    """
    optimal  = OPTIMAL_RETURNS[game_name]
    n_starts = len(start_states)
    n_types  = len(baseline_agents)

    # Self-play: both seats are the same ToM agent
    self_play_total = 0.0
    for s0 in start_states:
        self_play_total += run_episode(env, agent_list, s0, True)[-1]
    self_play_total /= n_starts * optimal

    # Cross-play: ToM vs each baseline type, averaged over types and starts
    p0_total = 0.0
    p1_total = 0.0
    for s0 in start_states:
        for base_agent_list in baseline_agents.values():
            p0_total += run_episode(env, AgentList([tom_agent, base_agent_list[1]]), s0, True)[-1]
            p1_total += run_episode(env, AgentList([base_agent_list[0], tom_agent]), s0, True)[-1]

    denom    = n_starts * n_types * optimal
    p0_total /= denom
    p1_total /= denom
    return self_play_total, p0_total, p1_total
    

if __name__ == "__main__":
    train_test_tom()