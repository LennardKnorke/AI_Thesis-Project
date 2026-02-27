# train_worldmodel.py
import os
import json
import itertools
import copy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from tiny_game import GAMES, Settings, GameNames, get_game_Rework, MyHanabi, DecPOMDP, Game
from agents import ToM_WorldModel
from agents.model_based.dtde_ToM import _encode_observation, _encode_action, _encode_joint_observation
from runner import run_episode
from config import * 
from train_test_baselines import load_best_baselinesagents

# --- CONSTANTS ---
TRAIN_SIZE_RATIO = 0.8

# --- HYPERPARAMETER GRID ---
TOM_PARAM_GRID = {
    "char_dim": [4, 8, 16],
    "mental_dim": [16, 32],
    "trunk_dim": [16, 32, 64,],
    "lr": [0.001, 0.01],
    "epochs": [1000],
    "batch_size" : [32, 64]
}

# --- UTILITY FUNCTIONS ---
def generate_param_combinations(grid: dict) -> list[dict]:
    """Generates all permutations of hyperparameters."""
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


# --- Dataset Class ---
class ToMDataset(torch.utils.data.Dataset):
    """
    Dataset for ToM World Model.
    Each item returned:
        past:      (N_PAST, SEQ, FEAT)
        history:   (SEQ, FEAT)  h_t
        obs:       (OBS_DIM,)   z_t
        tgt_act:   (ACT_DIM,)
        tgt_obs:   (OBS_DIM,)
        tgt_type:  (ACT_DIM,)
    """
    def __init__(
            self,
            x_past_traj,  # (N, N_PAST, SEQ, OBS_DIM)
            x_history,    # (N, SEQ, FEAT)
            x_obs,        # (N, OBS_DIM) -> Own Next Obs
            x_act,        # (N, ACT_DIM) -> Own Action (One-Hot)
            tgt_obs,      # (N, OBS_DIM) -> Partner Next Obs
            tgt_act,      # (N,)         -> Partner Action (Indices)
            tgt_type      # (N,)         -> Agent Identity (Indices)
    ):
        self.x_past_traj = x_past_traj
        self.x_history = x_history
        self.x_obs = x_obs
        self.x_act = x_act
        self.y_obs = tgt_obs
        self.y_act = tgt_act
        self.y_type = tgt_type 

    def __len__(self):
        return len(self.x_obs)

    def __getitem__(self, idx):
        return {
            # Inputs
            "past": torch.tensor(self.x_past_traj[idx], dtype=torch.float32),
            "history": torch.tensor(self.x_history[idx], dtype=torch.float32),
            "obs": torch.tensor(self.x_obs[idx], dtype=torch.float32),
            "act": torch.tensor(self.x_act[idx], dtype=torch.float32), # Own action is float input
            
            # Targets
            "tgt_obs": torch.tensor(self.y_obs[idx], dtype=torch.float32), # Regression target
            "tgt_act": torch.tensor(self.y_act[idx], dtype=torch.float32),    # Classification target (Long)
            "tgt_type": torch.tensor(self.y_type[idx], dtype=torch.long)   # Classification target (Long)
        }
        

Dataframe = dict[str, Dataset|DataLoader|int]
dataframe_template : Dataframe = {
    "train" : ToMDataset,
    "val" : ToMDataset,
    "obs_dim" : int,
    "action_dim" : int,
    "max_seq_length" : int
}


# --- MAIN FUNCTION ---
def train_worldmodel()->None:
    os.makedirs(WORLD_MODELS_DIR, exist_ok=True)

    # 1. Set up ToM param configurations and Baseline Agents
    param_configs = generate_param_combinations(TOM_PARAM_GRID)
    baseline_agents_per_game = setup_baseline_agents()

    # 2. Collect Data per Game
    full_dataset: dict[str, Dataframe] = setup_dataset(baseline_agents_per_game)
    if full_dataset is None:
        print("No data collected. Check trained baselines.")
        return

    best_params = None
    best_loss_per_game = {}   # {game: loss}
    best_models_per_game = {} # {game: state_dict}

    # 3. Grid Search Loop
    pbar = tqdm(param_configs, desc="ToM WM Parameter Search")
    for params in pbar:

        current_loss_per_game = {}
        current_models_per_game = {}

        # Train for every game with these params
        pbar2 = tqdm(full_dataset.items(), desc="Iterate over Games", leave=False)
        for game_name, dataset in pbar2:
            val_loss, state_dict = train_evaluate_world_model(params, dataset, NUM_AGENT_TYPES)
            current_loss_per_game[game_name] = val_loss
            current_models_per_game[game_name] = state_dict

        # Compare vs Global Best
        if best_params is None or is_better_results(best_loss_per_game, current_loss_per_game):
            best_params = params
            best_loss_per_game = current_loss_per_game
            best_models_per_game = current_models_per_game
            avg_last_loss = np.mean([l[-1] for _, l in best_loss_per_game.items()])
            tqdm.write(f"New Best! Avg Loss: {avg_last_loss} - {best_params}")

            # Save Params
            with open(os.path.join(WORLD_MODELS_DIR, "best_params.json"), 'w') as f:
                json.dump(best_params, f, indent=4)

            # Save Results CSV
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in best_loss_per_game.items()]))
            df.to_csv(os.path.join(WORLD_MODELS_DIR, "final_results.csv"), index=False)

            # Save Models
            for game_name, state_dict in best_models_per_game.items():
                save_path = os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth")
                torch.save(state_dict, save_path)
    return


def setup_baseline_agents():
    base_experiments = load_best_baselinesagents()
    agents_per_game : dict[str, dict[str,AgentList]]= {}

    # Loop over game environments
    print("Loading Baseline Agents")
    for game_name in GAMES:

        tmp_env = get_game_Rework(GameNames(game_name), Settings.decpomdp, normalize=False)

        game_agents : dict[str, AgentList]= {}      # To be filled

        # Loop over agenttype experiments
        for exp in base_experiments:
            new_agents : AgentList = exp.make_agents(tmp_env, exp.param_list[0])

            # Load Policies (centralized or decentralized)
            folder_name = os.path.join(RESULTS_DIR, exp.name.replace(" ", "_"))
            if isinstance(new_agents, CTDE_BI_MB_List) or isinstance(new_agents, CTDE_VDN_MF_List):
                shared_path = os.path.join(folder_name, f"G_{game_name}_shared_model.pkl")
                new_agents.load(shared_path)
            else:
                p0_path = os.path.join(folder_name, f"G_{game_name}_agent_0.pkl")
                p1_path = os.path.join(folder_name, f"G_{game_name}_agent_1.pkl")
                new_agents[0].load(p0_path)
                new_agents[1].load(p1_path)

            game_agents[exp.name] = new_agents
        agents_per_game[game_name] = game_agents
    return agents_per_game


def setup_dataset(baseline_agents_per_game : dict[str, dict[str, AgentList]], *args, **kwargs)->dict[str, Dataframe]:
    full_dataset = {}
    ensembles_per_game = {}

    # Loop over potential game environments
    pbar = tqdm(baseline_agents_per_game.items(), desc="Set up Games Datasets")
    for game_name, agents in pbar:

        # and collect a dataset/planning ensemble for each.
        env = get_game_Rework(GameNames(game_name), Settings.decpomdp, normalize=False)
        game_dataset, ensemble = collect_game_datasets(env, agents, game_name, *args, **kwargs)
        tqdm.write(f"{game_name} - Data Created")
        ensembles_per_game[game_name] = ensemble

        full_dataset[game_name] = game_dataset

    return full_dataset#, ensembles_per_game


def collect_game_datasets(env : Game, baseline_agents : dict[str, AgentList], game_name : str, *args, **kwargs) -> Dataframe:
    """
    Main Data Collection Loop.
    Returns: 
        1. Datasets dict
    """

    # Action Dimension
    ACT_DIM = env.num_actions + 1   # Num of possible actions + Null action
    null_action = env.num_actions

    # ENV SPECIFIC SPECS
    max_card_value = env.num_cards  # Num of possible cards
    num_players = 2
    if isinstance(env, DecPOMDP):
        start_len = 2
        MAX_SEQ_LEN = env.horizon - 1  #(Converge 2 cards into 1 obs)
        cards_per_player  = 1
    else:
        start_len = 4
        MAX_SEQ_LEN = env.horizon - 3     #(converge 4 cards into 1 obs)
        cards_per_player = 2
    OBS_DIM = env.num_actions + (max_card_value * cards_per_player) # Private observation dimension
    JOINT_OBS_DIM = env.num_actions + ((max_card_value * cards_per_player) * num_players)

    # PAST CONTEXT
    start_states = env.start_states()
    PAST_EPISODES_CONTEXT = NUM_AGENT_TYPES * len(start_states)    
    past_episodes_ensemble = _setup_ensemble(
        env, baseline_agents, 
        PAST_EPISODES_CONTEXT, 
        MAX_SEQ_LEN,
        JOINT_OBS_DIM,
        game_name
    )

    # Dataset
    DATASET_EPISODES = 15 # Per agent type and per start state (PAST_EPISODES_CONTEXT)
    # DATASET_EPISODES *= PAST_EPISODES_CONTEXT
    train_storage = []
    val_storage = []

    # Indexing tool
    p0_card_cutoff_idx = start_len // 2
    
    # Loop over each agent partner duo
    for type_idx, (type_name, agents_list) in enumerate(baseline_agents.items()):
        # Datapoints storage per agent type
        memory_bank_train = []
        memory_bank_val = []

        # Encode the current agents
        type_id_enc = np.zeros(NUM_AGENT_TYPES, dtype=np.float32)
        type_id_enc[type_idx] = 1.0
        
        episodes_collected = 0
        while episodes_collected < DATASET_EPISODES:
            for s0 in start_states:
                if episodes_collected >= DATASET_EPISODES: break

                # Play an episode
                full_episode_log = run_episode(env, agents_list, list(s0), True) # Returns history + payoff in list
                full_history = full_episode_log[:-1] # Fully observable history (no masking)
                
                # Convert (C0, C1, A0, A1) -> ((C0,C1), A0, A1) or (P0C1, P0C2, P1C0, P1C1, P0A0, P1A1...) -> ((P0C1, P0C2, P1C0, P1C1), P0A0, P1A0,...)
                cards = full_history[:start_len]
                actions = full_history[start_len:]

                # Private Hisotires
                p0_cards = cards[p0_card_cutoff_idx:] # THEY OBSERVE THE CARDS OF THEIR PARTNER NOT THEIR OWN!
                p1_cards = cards[:p0_card_cutoff_idx]
                p0_history_seq = [p0_cards] + actions
                p1_history_seq = [p1_cards] + actions

                full_history = [cards] + actions

                h_enc = np.zeros((MAX_SEQ_LEN, JOINT_OBS_DIM), dtype=np.float32) # Encoded step-t joint history

                assert len(full_history) == len(p0_history_seq) and len(full_history) == len(p1_history_seq)
                is_p0_turn : bool = True

                # Loop Over Timesteps to create inputs and outputs for each step
                for i, (joint_obs, p0_obs, p1_obs) in enumerate(zip(full_history, p0_history_seq, p1_history_seq)):
                    # 1. Encode the current private observations
                    p0_z_enc = _encode_observation(p0_history_seq[i], OBS_DIM, env, s0, is_p0_turn)
                    p1_z_enc = _encode_observation(p1_history_seq[i], OBS_DIM, env, s0, is_p0_turn)

                    # Encode step t joint observation/history 
                    z_enc = _encode_joint_observation(full_history[i], JOINT_OBS_DIM, env, s0, is_p0_turn)
                    h_enc[i] = z_enc

                    # 2. Encode the action taken for both parties
                    current_act_idx = i + 1
                    if current_act_idx < len(full_history):
                        current_act_int = full_history[current_act_idx]
                        if is_p0_turn:
                            p0_act_enc = _encode_action(current_act_int, ACT_DIM)
                            p1_act_enc = _encode_action(null_action, ACT_DIM)
                        else:
                            p0_act_enc = _encode_action(null_action, ACT_DIM)
                            p1_act_enc = _encode_action(current_act_int, ACT_DIM)
                    else:
                        break # Safety break
                    
                    # 3. Encode the next_obs
                    next_obs_idx = current_act_idx # Action is seen.
                    tgt_obs_int = full_history[next_obs_idx]
                    next_obs_enc = _encode_observation(tgt_obs_int, OBS_DIM, env, s0, is_p0_turn)


                    p0_datapoint = {
                        "obs" : p0_z_enc,
                        "act" : p0_act_enc,
                        "tgt_act": p1_act_enc,       # Integer
                        "tgt_obs": next_obs_enc,
                        "tgt_type": type_idx,          # Integer
                        # CRITICAL: Copy the current state of past episodes buffer
                        "past_episodes": past_episodes_ensemble.copy(),
                        "current_history": h_enc.copy()
                    }
                    p1_datapoint = {
                        "obs": p1_z_enc,
                        "act": p1_act_enc,
                        "tgt_act": p0_act_enc,       # Integer
                        "tgt_obs": next_obs_enc,
                        "tgt_type": type_idx,          # Integer
                        # CRITICAL: Copy the current state of past episodes buffer
                        "past_episodes": past_episodes_ensemble.copy(),
                        "current_history": h_enc.copy()
                    }
                  
                    # 5. Append relevant perspective to dataset
                    if episodes_collected < (DATASET_EPISODES * TRAIN_SIZE_RATIO):
                        memory_bank_train.append(p0_datapoint)
                        memory_bank_train.append(p1_datapoint)
                    else:
                        memory_bank_val.append(p0_datapoint)
                        memory_bank_val.append(p1_datapoint)
                    # Change flag
                    is_p0_turn = not is_p0_turn
                episodes_collected += 1

        # Update total database with experiences of the current agent
        train_storage.extend(memory_bank_train)
        val_storage.extend(memory_bank_val)
    return convert_game_dataset(train_storage, val_storage, OBS_DIM, JOINT_OBS_DIM, ACT_DIM, MAX_SEQ_LEN), past_episodes_ensemble


def _setup_ensemble(env : Game, baseline_agent : dict[AgentList], past_episodes_context, max_seq_len, obs_dim, game_name):
    vec = np.zeros((past_episodes_context, max_seq_len, obs_dim))

    ensemble_path = os.path.join(WORLD_MODELS_DIR, f"G_{game_name}_ensemble.npy")

    # Already_exists? load it
    if os.path.exists(ensemble_path):
        tqdm.write(f"Loaded Game Ensemble: {game_name}")
        vec = np.load(ensemble_path)
        return vec
    
    start_states = env.start_states()
    cuttoff_idx = 2 if isinstance(env, DecPOMDP) else 4
    p_idx = 0
    while p_idx < past_episodes_context:
        for keys, agents in baseline_agent.items():
            for s0 in start_states:
                full_history = run_episode(env, agents, list(s0), test_episode=True)
                full_history = full_history[:-1]
                cards = full_history[:cuttoff_idx]
                actions = full_history[cuttoff_idx:]
                full_history = [cards] + actions

                h_enc = np.zeros((max_seq_len, obs_dim))

                # Encode current joint observation sequence
                is_p0_turn = True
                for i, obs in enumerate(full_history):
                    z_enc = _encode_joint_observation(obs, obs_dim, env, s0, is_p0_turn)
                    is_p0_turn = not is_p0_turn
                    h_enc[i] = z_enc

                # Copy in ensemble
                vec[p_idx] = h_enc.copy()
                p_idx += 1

    np.save(ensemble_path, vec)
    tqdm.write(f"Saved Game Ensemble: {game_name}")
    return vec


def convert_game_dataset(train_data, val_data, obs_dim, joint_obs_dim, act_dim, seq_len):
    def _unpack_data(data_list):            
        return ToMDataset(
            x_past_traj=[d['past_episodes'] for d in data_list],
            x_history=[d['current_history'] for d in data_list],
            x_obs=[d['obs'] for d in data_list],
            x_act=[d['act'] for d in data_list],
            tgt_obs=[d['tgt_obs'] for d in data_list],
            tgt_act=[d['tgt_act'] for d in data_list],
            tgt_type=[d['tgt_type'] for d in data_list]
        )
    df : Dataframe = dataframe_template.copy()
    
    df["train"] = _unpack_data(train_data)
    df["val"] = _unpack_data(val_data)

    df["joint_obs_dim"] = joint_obs_dim
    df["obs_dim"] = obs_dim
    df["act_dim"] = act_dim
    df["max_seq_length"] = seq_len
    
    return df


# --- TRAINING FUNCTION --- 
def train_evaluate_world_model(params: dict, dataset_info: Dataframe, num_agents : int) -> tuple[float, dict]:
    """
    Trains one model on one game for X epochs. Returns Validation Loss and State Dict.
    """
    # 1. Setup Model
    model = ToM_WorldModel(
        joint_obs_dim=dataset_info["joint_obs_dim"],
        obs_dim=dataset_info["obs_dim"],
        action_dim=dataset_info["act_dim"],
        num_agent_types=num_agents,
        max_seq_len=dataset_info['max_seq_length'],
        char_embed_dim=params['char_dim'],
        mental_embed_dim=params['mental_dim'],
        trunk_dim=params['trunk_dim']
    )
    
    train_loader = DataLoader(dataset_info["train"], batch_size=params['batch_size'], shuffle=True, drop_last=False)
    val_loader = DataLoader(dataset_info["val"], batch_size=params['batch_size'], shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    epoch_val_losses = []
    
    # 2. Training Loop
    pbar = tqdm(range(params['epochs']), leave= False)
    for _ in pbar:
        model.train()

        for batch in train_loader:
            # Unpack Dictionary from DataLoader
            past = batch['past']
            hist = batch['history']
            obs = batch['obs']
            act = batch['act']

            tgt_act = batch['tgt_act']
            tgt_obs = batch['tgt_obs']
            tgt_type = batch['tgt_type']

            optimizer.zero_grad()
            
            act_logits, obs_pred, id_logits = model(past, hist, obs, act)
            
            l_act = ce_loss(act_logits, tgt_act)
            l_obs = mse_loss(obs_pred, tgt_obs)
            l_id = ce_loss(id_logits, tgt_type)
            
            loss = l_act + l_obs + (0.2 * l_id)

            loss.backward()
            optimizer.step()
            
        # 3. Final Validation
        model.eval()
        val_loss_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                past = batch['past']
                hist = batch['history']
                obs = batch['obs']
                act = batch['act']

                tgt_act = batch['tgt_act']
                tgt_obs = batch['tgt_obs']

                act_logits, obs_pred, _ = model(past, hist, obs, act)

                # Metric: Action Loss + Obs Loss
                l_act = ce_loss(act_logits, tgt_act)
                l_obs = mse_loss(obs_pred, tgt_obs)
                
                val_loss_sum += (l_act + l_obs).item()
                num_batches += 1

        avg_val_loss  = val_loss_sum / max(1, num_batches)
        epoch_val_losses.append(avg_val_loss)
            
    return epoch_val_losses, model.state_dict()


def is_better_results(old_loss_dict: None|dict[str, list[float]], new_loss_dict: dict[str, list[float]]) -> bool:
    """
    Determines if the new parameter set is better globally.
    Metric: Lower Average Validation Loss across available games.
    """
    # Calculate avg loss for old vs new
    final_old_losses = [l[-1] for _, l in old_loss_dict.items()]
    final_new_losses = [l[-1] for _, l in new_loss_dict.items()]

    old_loss = np.mean(final_old_losses)
    new_loss = np.mean(final_new_losses)

    return new_loss < old_loss