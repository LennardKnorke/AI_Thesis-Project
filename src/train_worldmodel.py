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
from agents.model_based.ToM_pbvi import _encode_observation, _encode_action, _encode_joint_observation
from runner import run_episode
from config import * 
from train_test_baselines import load_best_baselinesagents

# --- CONSTANTS ---
TRAIN_SIZE_RATIO = 0.8
TOM_PARAM_LOG_FILE = "tested_params.json"
TOM_PARAM_LOG_FILE = os.path.join(WORLD_MODELS_DIR, TOM_PARAM_LOG_FILE)

# --- HYPERPARAMETER GRID ---
TOM_PARAM_GRID = {
    "char_dim": [8, 16],
    "mental_dim": [16, 32],
    "trunk_dim": [16, 32, 64,],
    "lr": [0.001, 0.01],
    "epochs": [250],
    "batch_size" : [32]
}
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
        past = torch.tensor(self.x_past_traj[idx], dtype=torch.float32)
        history = torch.tensor(self.x_history[idx], dtype=torch.float32)
        obs = torch.tensor(self.x_obs[idx], dtype=torch.float32)
        act = torch.tensor(self.x_act[idx], dtype=torch.float32)
        tgt_obs = torch.tensor(self.y_obs[idx], dtype=torch.float32)
        tgt_act = torch.tensor(self.y_act[idx], dtype=torch.long)
        tgt_type = torch.tensor(self.y_type[idx], dtype=torch.long)
        return {
            # Inputs
            "past": past,
            "history": history,
            "obs": obs,
            "act": act,
            # Targets
            "tgt_obs": tgt_obs,
            "tgt_act": tgt_act,
            "tgt_type": tgt_type
        }
    

Dataframe = dict[str, Dataset|DataLoader|int]
dataframe_template : Dataframe = {
    "train" : ToMDataset,
    "val" : ToMDataset,
    # Dimensions
    "obs_dim" : int,
    "action_dim" : int,
    "max_seq_length" : int,
    "joint_obs_dim" : int,
}


# --- MAIN FUNCTION ---
def train_worldmodel()->None:
    os.makedirs(WORLD_MODELS_DIR, exist_ok=True)

    environments = setup_hanabi_environments()

    param_configs = generate_param_combinations(TOM_PARAM_GRID)

    baseline_agents_per_game = setup_baseline_agents(environments)

    full_dataset: dict[str, Dataframe] = setup_dataset(baseline_agents_per_game, environments)
    if full_dataset is None:
        print("No data collected. Check trained baselines.")
        return

    # Load previous best results
    best_params, best_results = load_tmp_best_results()
    processed_params = load_processed_params(best_params) if best_params is not None else []

    # Grid Search Loop
    pbar = tqdm(param_configs, desc="ToM WM Parameter Search")
    for params in pbar:
        # Check for processed params
        if are_params_processed(processed_params, params):
            continue

        # Results Cachce
        current_loss_per_game = {}
        current_obs_action_acc_per_game = {}
        current_obs_card_acc_per_game = {}
        current_act_acc_per_game = {}
        current_models_per_game = {}

        # Train for every game with these params
        pbar2 = tqdm(full_dataset.items(), desc="Iterate over Games", leave=False)
        for game_name, dataset in pbar2:
            #pbar2.set_postfix()

            env = environments[game_name]

            val_loss_list, obs_act_acc_list, obs_card_acc_list, act_acc_list, state_dict = train_evaluate_world_model(params, dataset, NUM_AGENT_TYPES, env)
            
            current_loss_per_game[game_name] = val_loss_list
            current_obs_action_acc_per_game[game_name] = obs_act_acc_list
            current_obs_card_acc_per_game[game_name] = obs_card_acc_list
            current_act_acc_per_game[game_name] = act_acc_list
            current_models_per_game[game_name] = state_dict

        # Compute averages across games
        avg_last_loss = np.mean([l[-1] for l in current_loss_per_game.values()])
        avg_last_obs_action_acc = np.mean(list(current_obs_action_acc_per_game.values()))
        avg_last_obs_card_acc = np.mean(list(current_obs_card_acc_per_game.values()))
        avg_last_act_acc = np.mean(list(current_act_acc_per_game.values()))

        # Write summary for this parameter set
        is_better = (best_params is None) or is_better_results(best_results, current_loss_per_game)
        _text =  f"New Best {params}!! | " if is_better else  f"Tested {params} | "
        tqdm.write(
            f"{_text}"
            f"Avg Val Loss: {avg_last_loss:.4f} | "
            f"Obs Action Acc: {avg_last_obs_action_acc:.4f} | "
            f"Obs Card Acc: {avg_last_obs_card_acc:.4f} | "
            f"Action Pred Acc: {avg_last_act_acc:.4f}"
        )

        # Identify model improvement
        if is_better:
            best_params = params

            # Save Params
            with open(os.path.join(WORLD_MODELS_DIR, "best_params.json"), 'w') as f:
                json.dump(best_params, f, indent=4)

            # Save Results CSV
            results_data = {}
            for game_name in current_loss_per_game.keys():
                results_data[f"loss_{game_name}"] = current_loss_per_game[game_name]
                results_data[f"obs_card_acc_{game_name}"] = current_obs_card_acc_per_game[game_name]
                results_data[f"obs_act_acc_{game_name}"] = current_obs_action_acc_per_game[game_name]
                results_data[f"act_acc_{game_name}"] = current_act_acc_per_game[game_name]
            best_results = results_data
            
            df = pd.DataFrame(best_results)
            df.to_csv(os.path.join(WORLD_MODELS_DIR, "final_results.csv"), index=False)

            # Save Models
            #for game_name, state_dict in current_models_per_game.items():
            #    save_path = os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth")
            #    torch.save(state_dict, save_path)

        # Process Params tested
        processed_params.append(params)
        log_processed_params(params)




    # Train best params for slightly longer
    current_models_per_game = {}
    pbar2 = tqdm(full_dataset.items(), desc="Iterate over Games", leave=False)
    best_params['epochs'] = 1000
    results_data = {}
    for game_name, dataset in pbar2:
        #pbar2.set_postfix()
        env = environments[game_name]

        val_loss_list, obs_act_acc_list, obs_card_acc_list, act_acc_list, state_dict = train_evaluate_world_model(best_params, dataset, NUM_AGENT_TYPES, env)
        
        results_data[f"loss_{game_name}"] = val_loss_list
        results_data[f"obs_act_acc_{game_name}"] = obs_act_acc_list
        results_data[f"obs_card_acc_{game_name}"] = obs_card_acc_list
        results_data[f"act_acc_{game_name}"] = act_acc_list
        current_models_per_game[game_name] = state_dict
    
    df = pd.DataFrame(results_data)
    df.to_csv(os.path.join(WORLD_MODELS_DIR, "final_results.csv"), index=False)
    # Save Models
    for game_name, state_dict in current_models_per_game.items():
        save_path = os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth")
        torch.save(state_dict, save_path)

    return


def setup_hanabi_environments()->dict[str, Game]:
    environments = {}
    for game_name in GAMES:
        tmp_env = get_game_Rework(GameNames(game_name))
        environments[game_name] = tmp_env
    return environments


def setup_baseline_agents(environments : dict[str, Game]):
    base_experiments = load_best_baselinesagents()
    agents_per_game : dict[str, dict[str,AgentList]]= {}

    # Loop over game environments
    print("Loading Baseline Agents")
    for game_name in GAMES:
        tmp_env = environments[game_name]        

        game_agents : dict[str, AgentList]= {}      # To be filled

        # Loop over agenttype experiments
        for exp in base_experiments:
            new_agents : AgentList = exp.make_agents(tmp_env, exp.param_list[0])

            # Load Policies (centralized or decentralized)
            folder_name = os.path.join(RESULTS_DIR, exp.name.replace(" ", "_"))
            if new_agents.centralized_planning:
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


def setup_dataset(baseline_agents_per_game : dict[str, dict[str, AgentList]], environments : dict[str, Game], *args, **kwargs)->dict[str, Dataframe]:
    full_dataset = {}
    ensembles_per_game = {}

    # Loop over potential game environments
    pbar = tqdm(baseline_agents_per_game.keys(), desc="Set up Games Datasets")
    for game_name in pbar:
        env = environments[game_name]
        agents = baseline_agents_per_game[game_name]

        game_dataset, ensemble = collect_game_datasets(env, agents, game_name, *args, **kwargs)

        ensembles_per_game[game_name] = ensemble
        full_dataset[game_name] = game_dataset

    return full_dataset#, ensembles_per_game


def collect_game_datasets(env : Game, baseline_agents : dict[str, AgentList], game_name : str, *args, **kwargs) -> Dataframe:
    # Action Dimension
    ACT_DIM = env.num_actions + 1   # Num of possible actions + Null action
    null_action = env.num_actions
    # ENV SPECIFIC SPECS
    if isinstance(env, DecPOMDP):
        start_len = 2
        MAX_SEQ_LEN = env.horizon - 1   #(Converge 2 cards into 1 obs)
        obs_act_dim = env.num_actions       # Action observed only
        obs_card_dim = env.num_cards * 2
    elif isinstance(env, MyHanabi):
        start_len = 4
        MAX_SEQ_LEN = env.horizon - 3     #(converge 4 cards into 1 obs)
        obs_act_dim = env.num_actions + env.num_cards + 1   # Action observed includes a potential card revealed
        obs_card_dim = env.num_cards * start_len
    else:
        raise ValueError("Upsi?")
    OBS_DIM = obs_act_dim + obs_card_dim
    JOINT_OBS_DIM = OBS_DIM

    # PAST CONTEXT
    start_states = env.start_states()
    PAST_EPISODES_CONTEXT = NUM_AGENT_TYPES * len(start_states)

    # Mixed ensemble — one shared pool across all agent types (backup / testing)
    mixed_ensemble = _setup_ensemble(
        env, baseline_agents,
        PAST_EPISODES_CONTEXT, MAX_SEQ_LEN, JOINT_OBS_DIM, game_name
    )

    # Per-agent-type ensembles — CharNet sees only episodes from the target agent type
    per_type_ensembles: dict[str, np.ndarray] = {}
    for _type_name, _agents in baseline_agents.items():
        per_type_ensembles[_type_name] = _setup_ensemble(
            env, {_type_name: _agents},
            PAST_EPISODES_CONTEXT, MAX_SEQ_LEN, JOINT_OBS_DIM, game_name,
            type_name=_type_name
        )

    # Dataset
    DATASET_EPISODES = 5
    train_storage = []
    val_storage = []

    # Indexing tool
    p0_card_cutoff_idx = start_len // 2
    
    # Loop over each agent partner duo
    pbar2 = tqdm(enumerate(baseline_agents.keys()), leave=False)
    for type_idx, type_name in pbar2:
        agents_list = baseline_agents[type_name]

        # Datapoints storage per agent type
        memory_bank_train = []
        memory_bank_val = []

        pbar3 = tqdm(range(DATASET_EPISODES), leave=False)
        for ep in pbar3:
            for s0 in start_states:
                # Play an episode
                full_episode_log = run_episode(env, agents_list, list(s0), True)
                full_history = full_episode_log[:-1]
                
                # Split into cards and actions
                cards = full_history[:start_len]
                actions = full_history[start_len:]
                full_history = [cards] + actions

                #
                h_enc = np.zeros((MAX_SEQ_LEN, JOINT_OBS_DIM), dtype=np.float32)
                # Loop Over Timesteps to create inputs and outputs for each step
                for i, joint_obs in enumerate(full_history):
                    # t+1 index.
                    next_idx = i + 1
                    if next_idx >= len(full_history):
                        break
                    next_obs = full_history[next_idx]
                    
                    # Encode step t joint history: (h^i, h^k)_t = h_t
                    z_enc = _encode_joint_observation(joint_obs, JOINT_OBS_DIM, env)
                    h_enc[i] = z_enc

                    # Encode own action (always null), a_t
                    priv_a_enc = _encode_action(null_action, ACT_DIM)

                    # Encode own next obs z_{t+1}
                    priv_z_enc = _encode_observation(next_obs, OBS_DIM, env)

                    # Encode partner next obs z_{t+1}. In this game z^k_{t+1}==z^i_{t+1}
                    partner_z_enc = _encode_observation(next_obs, OBS_DIM, env)

                    # Encode partner action, a^k_t
                    partner_a_enc = next_obs[0] if isinstance(next_obs, (tuple, list)) else next_obs

                    
                    datapoint = {
                        "obs" : priv_z_enc,
                        "act" : priv_a_enc,
                        "tgt_act": partner_a_enc,
                        "tgt_obs": partner_z_enc,
                        "tgt_type": type_idx,
                        # CRITICAL: Copy the current state of past episodes buffer
                        "past_episodes": per_type_ensembles[type_name].copy(),
                        "current_history": h_enc.copy()
                    }
                  
                    # 5. Append relevant perspective to dataset
                    if ep < (DATASET_EPISODES * TRAIN_SIZE_RATIO):
                        memory_bank_train.append(datapoint)
                    else:
                        memory_bank_val.append(datapoint)

        # Update total database with experiences of the current agent
        train_storage.extend(memory_bank_train)
        val_storage.extend(memory_bank_val)
    return convert_game_dataset(train_storage, val_storage, OBS_DIM, JOINT_OBS_DIM, ACT_DIM, MAX_SEQ_LEN), mixed_ensemble


def _setup_ensemble(env: Game, baseline_agent: dict[AgentList], past_episodes_context, max_seq_len, obs_dim, game_name, type_name: str | None = None):
    vec = np.zeros((past_episodes_context, max_seq_len, obs_dim))

    # Per-agent-type ensemble OR mixed ensemble (type_name=None)
    if type_name is not None:
        path_name = type_name.replace(" ", "_")
        ensemble_path = os.path.join(WORLD_MODELS_DIR, f"G_{game_name}_{path_name}_ensemble.npy")
    else:
        ensemble_path = os.path.join(WORLD_MODELS_DIR, f"G_{game_name}_ensemble.npy")

    if os.path.exists(ensemble_path):
        tqdm.write(f"Loaded Ensemble: {os.path.basename(ensemble_path)}")
        vec = np.load(ensemble_path)
        return vec

    start_states = env.start_states()
    cuttoff_idx = 2 if isinstance(env, DecPOMDP) else 4
    p_idx = 0
    while p_idx < past_episodes_context:
        for keys, agents in baseline_agent.items():
            for s0 in start_states:
                if p_idx >= past_episodes_context:
                    break
                full_history = run_episode(env, agents, list(s0), test_episode=True)
                full_history = full_history[:-1]
                cards = full_history[:cuttoff_idx]
                actions = full_history[cuttoff_idx:]
                full_history = [cards] + actions

                h_enc = np.zeros((max_seq_len, obs_dim))

                # Encode current joint observation sequence
                is_p0_turn = True
                for i, obs in enumerate(full_history):
                    z_enc = _encode_joint_observation(obs, obs_dim, env)
                    is_p0_turn = not is_p0_turn
                    h_enc[i] = z_enc

                vec[p_idx] = h_enc.copy()
                p_idx += 1

    np.save(ensemble_path, vec)
    tqdm.write(f"Created Ensemble: {os.path.basename(ensemble_path)}")
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


def load_tmp_best_results():
    #
    params_path = os.path.join(WORLD_MODELS_DIR, "best_params.json")
    results_path = os.path.join(WORLD_MODELS_DIR, "final_results.csv")
    if not os.path.exists(params_path) or not os.path.exists(results_path):
        return None, {}
    
    # Load params
    with open(params_path, 'r') as f:
        best_params = json.load(f)

    # Load results
    df = pd.read_csv(results_path)
    best_results_per_game = {
        col: df[col].dropna().tolist()
        for col in df.columns
    }

    return best_params, best_results_per_game


# --- TRAINING FUNCTION --- 
def train_evaluate_world_model(params: dict, dataset_info: Dataframe, num_agents : int, env : Game) -> tuple[list[float], list[float], list[float], dict]:
    """
    Trains one model on one game for X epochs. Returns Validation Loss and State Dict.
    """
    action_part_dim = env.num_actions + 1  

    if isinstance(env, MyHanabi):
        card_part_dim = env.num_cards * 4

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
    bce_loss = nn.BCEWithLogitsLoss()


    epoch_train_losses = []
    epoch_obs_action_acc = []
    epoch_obs_card_acc = []
    epoch_action_acc = []
    
    # 2. Training Loop
    pbar = tqdm(range(params['epochs']), leave= False)
    for epoch in pbar:
        # Training Step
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for batch in train_loader:
            past = batch['past']
            hist = batch['history']
            obs = batch['obs']
            act = batch['act']
            tgt_act = batch['tgt_act']
            tgt_obs = batch['tgt_obs']
            tgt_type = batch['tgt_type']

            optimizer.zero_grad()
            
            act_logits, obs_pred, id_logits = model(past, hist, obs, act)

            # Loss
            loss_obs = bce_loss(obs_pred, tgt_obs)      # Obs Prediction (complex)
            loss_act = ce_loss(act_logits, tgt_act)     # Action Prediction (onehot)
            loss_id  = ce_loss(id_logits, tgt_type)
            loss = loss_act + loss_obs + (0.2 * loss_id)
            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train_loss = train_loss_sum / train_batches
        epoch_train_losses.append(avg_train_loss) 

        # Final Validation
        model.eval()
        correct_obs_action = 0
        correct_obs_cards = 0
        correct_act = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                past = batch['past']
                hist = batch['history']
                obs = batch['obs']
                act = batch['act']

                tgt_act = batch['tgt_act']
                tgt_obs = batch['tgt_obs']

                act_logits, obs_pred, _ = model(past, hist, obs, act)

                if isinstance(env, DecPOMDP):
                    pred_obs = obs_pred.argmax(dim=1)
                    true_obs = tgt_obs.argmax(dim=1)
                    correct_obs = (pred_obs == true_obs).sum().item()
                    # For consistency with return values, set both action and card acc to same
                    correct_obs_action += correct_obs
                    correct_obs_cards += correct_obs   

                elif isinstance(env, MyHanabi):
                    # Split into action part and card part
                    obs_pred_action = obs_pred[:, :action_part_dim]
                    obs_pred_cards = obs_pred[:, action_part_dim:]

                    # Convert target one-hot to class indices
                    tgt_obs_action = tgt_obs[:, :action_part_dim].argmax(dim=1)
                    tgt_obs_cards = tgt_obs[:, action_part_dim:].argmax(dim=1)

                    # Calculate accuracies
                    pred_obs_action = obs_pred_action.argmax(dim=1)
                    pred_obs_cards = obs_pred_cards.argmax(dim=1)

                    correct_obs_action += (pred_obs_action == tgt_obs_action).sum().item()
                    correct_obs_cards += (pred_obs_cards == tgt_obs_cards).sum().item()

                pred_act = act_logits.argmax(dim=1)
                correct_act += (pred_act == tgt_act).sum().item()
                total += tgt_act.size(0)

        # Calculate validation metrics
        avg_obs_action_acc = correct_obs_action / total if total > 0 else 0.0
        avg_obs_cards_acc = correct_obs_cards / total if total > 0 else 0.0
        avg_act_acc = correct_act / total if total > 0 else 0.0

        #epoch_val_losses.append(avg_val_loss)
        epoch_obs_action_acc.append(avg_obs_action_acc)
        epoch_obs_card_acc.append(avg_obs_cards_acc)
        epoch_action_acc.append(avg_act_acc)

        postfix_dict = {
            'loss': f'{avg_train_loss:.4f}',
            'act_acc': f'{avg_act_acc:.4f}'
        }
        if isinstance(env, MyHanabi):
            postfix_dict['obsA_acc'] = f'{avg_obs_action_acc:.4f}'
            postfix_dict['obsC_acc'] = f'{avg_obs_cards_acc:.4f}'
        else:  # DecPOMDP
            postfix_dict['obs_acc'] = f'{avg_obs_action_acc:.4f}'
            
        pbar.set_postfix(postfix_dict)
    # Finalize Training
    return epoch_train_losses, epoch_obs_action_acc, epoch_obs_card_acc, epoch_action_acc, model.state_dict()


def is_better_results(best_results: None|dict[str, list[float]], new_loss_dict: dict[str, list[float]]) -> bool:
    """
    Determines if the new parameter set is better globally.
    Metric: Lower Average Validation Loss across available games.
    """
    if best_results is None:
        return True
    
    final_old_losses = []
    for g in GAMES:
        key = f"loss_{g}"
        final_old_losses.append(
            best_results[key][-1]
        )
    # Calculate avg loss for old vs new
    final_new_losses = [l[-1] for _, l in new_loss_dict.items()]

    old_loss = np.mean(final_old_losses)
    new_loss = np.mean(final_new_losses)

    return new_loss < old_loss


def log_processed_params(params : dict):
    with open(TOM_PARAM_LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(params, f)
        f.write("\n")
    return


def load_processed_params(best_params):
    if best_params is None:
        return []
    if not os.path.exists(TOM_PARAM_LOG_FILE):
        return []

    # Load parameters processed.
    dictionaries = []
    with open(TOM_PARAM_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line in dictionaries:
                dictionaries.append(json.loads(line))
    return dictionaries


def are_params_processed(processed_params : list[dict[str, float|int]], new_params : dict[str, float|int])->bool:
    def dicts_equal_strict(d1: dict[str, float | int], d2: dict[str, float | int]) -> bool:
        """Compare two dictionaries with strict type checking."""
        if d1.keys() != d2.keys():
            return False
        for key in d1:
            v1, v2 = d1[key], d2[key]
            if type(v1) is not type(v2) or v1 != v2:
                return False
        return True

    # Compare each param dict
    for processed in processed_params:
        if dicts_equal_strict(processed, new_params):
            return True
    return False



if __name__ == "__main__":
    train_worldmodel()
