# train_worldmodel.py
import os
import json
import itertools
import copy
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from tiny_game import GAMES, Settings, GameNames, get_game
from agents import ToM_WorldModel
from config import * 
from train_test_baselines import load_best_params

# Data Collection Settings
PAST_EPISODES_CONTEXT = 5  
DATASET_EPISODES = 250     
MAX_SEQ_LEN = 4            
AGENT_IDS = {exp.name: i for i, exp in enumerate(BASELINE_EXPERIMENTS)}


# --- HYPERPARAMETER GRID ---
TOM_PARAM_GRID = {
    "char_dim": [16, 32, 64],
    "mental_dim": [16, 32, 64],
    "lr": [0.001, 0.0005, 0.01],
    "batch_size": [32, 64],
    "epochs": [10, 20]
}


# --- HELPER FUNCTIONS ---
def generate_param_combinations(grid: Dict) -> List[Dict]:
    keys, values = zip(*grid.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def one_hot(idx, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def pad_sequence(seq_vectors, max_len, feat_dim):
    arr = np.array(seq_vectors, dtype=np.float32)
    if len(arr) == 0: return np.zeros((max_len, feat_dim), dtype=np.float32)
    if len(arr) < max_len:
        padding = np.zeros((max_len - len(arr), feat_dim), dtype=np.float32)
        arr = np.vstack([arr, padding])
    return arr[:max_len] 


# --- DATASET ---
class ToMDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list 
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


# --- DATA COLLECTION ---
def _collect_data_for_game(game_name: str) -> Optional[ToMDataset]:
    """
    Simulates games using trained baselines to create a dataset for a specific game.
    """
    game_data = []
    
    # Get Env specs
    env = get_game(GameNames(game_name), Settings.decpomdp, normalize=False)
    obs_dim = 4 
    act_dim = env.num_actions
    step_feat_dim = obs_dim + act_dim

    for exp in BASELINE_EXPERIMENTS:
        agent_type_id = AGENT_IDS[exp.name]
        
        # Load Best Params
        params_list = load_best_params(exp.name)
        if not params_list: continue
        
        # Instantiate Agents
        try:
            agents = exp.make_agents(env, params_list[0])
        except: continue

        # Load Weights
        folder_name = exp.name.replace(" ", "_")
        load_dir = os.path.join(RESULTS_DIR, folder_name)
        shared_path = os.path.join(load_dir, f"G_{game_name}_shared_model.pkl")
        p0_path = os.path.join(load_dir, f"G_{game_name}_agent_0.pkl")
        p1_path = os.path.join(load_dir, f"G_{game_name}_agent_1.pkl")
        
        try:
            if os.path.exists(shared_path): 
                agents.load(shared_path)
            elif os.path.exists(p0_path): 
                agents[0].load(p0_path); agents[1].load(p1_path)
            else: 
                continue
        except: continue

        # Simulation Loop
        memory_bank = [] 
        for _ in range(DATASET_EPISODES):
            # Player Episode
            env.reset()
            done = False
            trajectory_raw = []
            
            while not done:
                ctx = env.context()
                pid = 0 if len(ctx) == 2 else 1
                obs = list(ctx); obs[pid] = -1 
                obs_padded = obs + [-1] * (obs_dim - len(obs))
                obs_vec = np.array(obs_padded, dtype=np.float32)
                
                action = agents[pid].act(tuple(obs), exploit=True)
                env.step(action)
                done = env.is_terminal()
                trajectory_raw.append((pid, obs_vec, action))

            # Add to Memory Bank
            formatted_traj_steps = []
            for _, o_v, a_i in trajectory_raw:
                a_oh = one_hot(a_i, act_dim)
                formatted_traj_steps.append(np.concatenate([o_v, a_oh]))
            memory_bank.append(formatted_traj_steps)
            
            # Generate Samples
            if len(memory_bank) <= PAST_EPISODES_CONTEXT: continue

            # Create Context
            past_indices = np.random.choice(len(memory_bank)-1, PAST_EPISODES_CONTEXT, replace=False)
            past_tensor = np.zeros((PAST_EPISODES_CONTEXT, MAX_SEQ_LEN, step_feat_dim), dtype=np.float32)
            for i, past_idx in enumerate(past_indices):
                past_tensor[i] = pad_sequence(memory_bank[past_idx], MAX_SEQ_LEN, step_feat_dim)
            
            current_steps_accumulated = []
            
            for i in range(len(trajectory_raw)):
                pid, obs_vec, action_int = trajectory_raw[i]
                
                tgt_act = torch.tensor(action_int, dtype=torch.long)
                tgt_type = torch.tensor(agent_type_id, dtype=torch.long)
                
                # Determine Next Observation (Target)
                if i + 1 < len(trajectory_raw):
                    next_obs_vec = np.zeros(obs_dim, dtype=np.float32)
                    for j in range(i+1, len(trajectory_raw)):
                        if trajectory_raw[j][0] == pid:
                            next_obs_vec = trajectory_raw[j][1]
                            break
                else:
                    next_obs_vec = np.zeros(obs_dim, dtype=np.float32)
                    
                tgt_next_obs = torch.tensor(next_obs_vec, dtype=torch.float32)
                curr_hist = pad_sequence(current_steps_accumulated, MAX_SEQ_LEN, step_feat_dim)
                
                game_data.append((
                    torch.tensor(past_tensor),      
                    torch.tensor(curr_hist),        
                    torch.tensor(obs_vec),          
                    tgt_act,
                    tgt_next_obs,
                    tgt_type
                ))
                
                a_oh = one_hot(action_int, act_dim)
                current_steps_accumulated.append(np.concatenate([obs_vec, a_oh]))

    if not game_data: return None
    return ToMDataset(game_data)


def prepare_all_datasets():
    """
    Pre-loads all datasets into memory to avoid regeneration during search.
    Returns Dict[game_name: (train_loader, val_loader, act_dim)]
    """
    print("Preparing Datasets for all games...")
    datasets_cache = {}
    
    for game_name in GAMES:
        ds = _collect_data_for_game(game_name)
        if ds is None:
            print(f"Warning: No data for {game_name}")
            continue
            
        train_size = int(0.8 * len(ds))
        val_size = len(ds) - train_size
        train_set, val_set = random_split(ds, [train_size, val_size])
        
        # Store raw sets, batching depends on params
        game = get_game(GameNames(game_name), Settings.decpomdp)
        act_dim = game.num_actions
        
        datasets_cache[game_name] = {
            "train": train_set,
            "val": val_set,
            "act_dim": act_dim
        }
        
    print(f"Data ready for {len(datasets_cache)} games.")
    return datasets_cache


def train_evaluate_single_model(train_set, val_set, act_dim, params):
    """
    Trains one model on one game for X epochs. Returns Validation Loss.
    """
    obs_dim = 4 # Fixed for TinyHanabi
    num_agent_types = len(BASELINE_EXPERIMENTS)

    # 1. Setup Model & Loader
    model = ToM_WorldModel(
        obs_dim=obs_dim,
        action_dim=act_dim,
        num_agent_types=num_agent_types,
        char_embed_dim=params['char_dim'],
        mental_embed_dim=params['mental_dim']
    )
    
    train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=params['batch_size'], shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    
    train_loss_history = []
    val_loss_history = []
    val_act_acc_history = []
    val_id_acc_history = []

    # 2. Training Loop
    for _ in range(params['epochs']):
        model.train()
        epoch_train_loss = 0.0

        for batch in train_loader:
            past, hist, obs, tgt_act, tgt_obs, tgt_type = batch
            optimizer.zero_grad()
            
            act_logits, obs_pred, id_logits = model(past, hist, obs)
            
            l_act = ce_loss(act_logits, tgt_act)
            l_obs = mse_loss(obs_pred, tgt_obs)
            l_id = ce_loss(id_logits, tgt_type)
            
            loss = l_act + l_obs + (0.5 * l_id) 
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_loss_history.append(epoch_train_loss / len(train_loader))
            
        # 3. Validation Loop
        model.eval()
        val_loss_sum = 0.0
        val_act_correct = 0
        val_act_total = 0
        val_id_correct = 0
        val_id_total = 0

        with torch.no_grad():
            for batch in val_loader:
                past, hist, obs, tgt_act, tgt_obs, tgt_type = batch

                act_logits, obs_pred, id_logits = model(past, hist, obs)

                l_act = ce_loss(act_logits, tgt_act)
                l_obs = mse_loss(obs_pred, tgt_obs)
                l_id = ce_loss(id_logits, tgt_type)

                total_loss = l_act + l_obs + (0.5 * l_id)
                val_loss_sum += total_loss.item()

                # Action Accuracy
                preds_act = torch.argmax(act_logits, dim=1)
                val_act_correct += (preds_act == tgt_act).sum().item()
                val_act_total += tgt_act.size(0)

                # Identity Accuracy (Auxiliary)
                preds_id = torch.argmax(id_logits, dim=1)
                val_id_correct += (preds_id == tgt_type).sum().item()
                val_id_total += tgt_type.size(0)
            
        val_loss_history.append(val_loss_sum / max(1, len(val_loader)))
        val_act_acc_history.append(val_act_correct / max(1, val_act_total))
        val_id_acc_history.append(val_id_correct / max(1, val_id_total))
            
    # Return histories and the final model state
    return train_loss_history, val_loss_history, val_act_acc_history, val_id_acc_history, model.state_dict()


# --- MAIN SEARCH ROUTINE ---
def train_worldmodel():
    os.makedirs(WORLD_MODELS_DIR, exist_ok=True)
    
    # 1. Pre-Collect Data (Once)
    datasets_cache = prepare_all_datasets()
    if not datasets_cache:
        print("No data available. Exiting.")
        return

    # 2. Generate Grid
    param_configs = generate_param_combinations(TOM_PARAM_GRID)
    print(f"\nStarting Grid Search over {len(param_configs)} configurations...")
    print("Optimization Target: Average Validation Loss across ALL games.\n")
    
    best_global_loss = float('inf')
    best_params = None
    
    best_global_histories_per_game = {} # {game_name: (train_loss_hist, val_loss_hist, val_act_acc_hist, val_id_acc_hist)}
    best_global_model_states = {}       # {game_name: state_dict}

    # 3. Loop: Parameters
    for params in tqdm(param_configs, desc="World Model Hyperparam Sets"):
        current_global_final_val_loss_sum = 0.0
        valid_games_count = 0
        
        temp_histories_this_param_set = {}
        temp_model_states_this_param_set = {}

        # 4. Loop: Games
        for game_name, data in datasets_cache.items():
            train_loss_hist, val_loss_hist, val_act_acc_hist, val_id_acc_hist, state_dict = train_evaluate_single_model(
                data["train"], 
                data["val"], 
                data["act_dim"], 
                params
            )
            
            final_val_loss = val_loss_hist[-1]
            current_global_final_val_loss_sum += final_val_loss
            valid_games_count += 1

            temp_histories_this_param_set[game_name] = (
                train_loss_hist, val_loss_hist, val_act_acc_hist, val_id_acc_hist
            )
            temp_model_states_this_param_set[game_name] = state_dict
        
        # Average final validation loss across games for this parameter set
        if valid_games_count > 0:
            avg_final_val_loss_this_param_set = current_global_final_val_loss_sum / valid_games_count
            
            # Check if this parameter set is the best overall so far
            if avg_final_val_loss_this_param_set < best_global_loss:
                best_global_loss = avg_final_val_loss_this_param_set
                best_params = params.copy()
                
                # Update the globally best histories and model states
                best_global_histories_per_game = copy.deepcopy(temp_histories_this_param_set)
                best_global_model_states = copy.deepcopy(temp_model_states_this_param_set)
                
                tqdm.write(f"New Best Global Avg Loss: {best_global_loss:.4f} | Params: {best_params}")


    # --- FINAL SAVING ---
    print(f"\nBest Avg Validation Loss: {best_global_loss:.4f}")
    print(f"Best Parameters: {best_params}")
    print("="*60)
    
    # Save Params
    with open(os.path.join(WORLD_MODELS_DIR, "best_params.json"), 'w') as f:
        json.dump(best_params, f, indent=4)
        
    if best_params:
        history_records = []
        max_epochs = 0
        for game_name, (train_l, val_l, val_aa, val_ia) in best_global_histories_per_game.items():
            max_epochs = max(max_epochs, len(train_l))

            # Store histories as lists for each game, padding if epoch counts differ (unlikely with fixed epochs)
            game_data = {
                f'train_loss_{game_name}': train_l,
                f'val_loss_{game_name}': val_l,
                f'val_act_acc_{game_name}': val_aa,
                f'val_id_acc_{game_name}': val_ia,
            }
            history_records.append(game_data)
        
        # Reshape the history_records for easier DataFrame creation with separate columns per game/metric
        df_data = defaultdict(list)
        for epoch in range(max_epochs):
            row = {'epoch': epoch + 1}
            for game_name, (train_l, val_l, val_aa, val_ia) in best_global_histories_per_game.items():
                row[f'train_loss_{game_name}'] = train_l[epoch] if epoch < len(train_l) else np.nan
                row[f'val_loss_{game_name}'] = val_l[epoch] if epoch < len(val_l) else np.nan
                row[f'val_act_acc_{game_name}'] = val_aa[epoch] if epoch < len(val_aa) else np.nan
                row[f'val_id_acc_{game_name}'] = val_ia[epoch] if epoch < len(val_ia) else np.nan
            history_records.append(row)
        
        # Create DataFrame from the records
        history_df = pd.DataFrame(history_records)
        history_df.to_csv(os.path.join(WORLD_MODELS_DIR, "best_world_model_training_history.csv"), index=False)
        print(f"Saved best model's training history to {os.path.join(WORLD_MODELS_DIR, 'best_world_model_training_history.csv')}")


    # 3. Save the individual model state_dicts for each game, using the GLOBAL best parameters
    if best_global_model_states:
        print("\nSaving individual models for the global best parameters...")
        for game_name, state_dict in best_global_model_states.items():
            save_path = os.path.join(WORLD_MODELS_DIR, f"ToM_WorldModel_{game_name}.pth")
            torch.save(state_dict, save_path)
            # print(f"Saved {save_path}") # Uncomment for verbose output
    else:
        print("No models to save (best_global_model_states is empty).")


    return best_params, best_global_loss