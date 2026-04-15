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
from agents.model_based.ToM_pbvi import _encode_observation, _encode_joint_observation
from runner import run_episode
from config import * 
from train_test_baselines import load_best_baselinesagents

# --- CONSTANTS ---
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
            #x_act,        # (N, ACT_DIM) -> Own Action (One-Hot)
            tgt_obs,      # (N, OBS_DIM) -> Partner Next Obs
            tgt_act,      # (N,)         -> Partner Action (Indices)
            tgt_type      # (N,)         -> Agent Identity (Indices)
    ):
        self.x_past_traj = x_past_traj
        self.x_history = x_history
        self.x_obs = x_obs
        #self.x_act = x_act
        self.y_obs = tgt_obs
        self.y_act = tgt_act
        self.y_type = tgt_type 

    def __len__(self):
        return len(self.x_obs)

    def __getitem__(self, idx):
        past = torch.tensor(self.x_past_traj[idx], dtype=torch.float32)
        history = torch.tensor(self.x_history[idx], dtype=torch.float32)
        obs = torch.tensor(self.x_obs[idx], dtype=torch.float32)
        #act = torch.tensor(self.x_act[idx], dtype=torch.float32)
        tgt_obs = torch.tensor(self.y_obs[idx], dtype=torch.float32)
        tgt_act = torch.tensor(self.y_act[idx], dtype=torch.long)
        tgt_type = torch.tensor(self.y_type[idx], dtype=torch.long)
        return {
            # Inputs
            "past": past,
            "history": history,
            "obs": obs,
            #"act": act,
            # Targets
            "tgt_obs": tgt_obs,
            "tgt_act": tgt_act,
            "tgt_type": tgt_type
        }
    

Dataframe = dict[str, Dataset|DataLoader|int]
dataframe_template : Dataframe = {
    "data" : ToMDataset,   # full dataset — no train/val split (deterministic policies)
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

        # Results cache
        current_loss_per_game = {}
        #current_obs_action_acc_per_game = {}
        #current_obs_card_acc_per_game = {}
        current_act_acc_per_game = {}
        current_type_acc_per_game = {}
        current_models_per_game = {}

        # Train for every game with these params
        pbar2 = tqdm(full_dataset.items(), desc="Iterate over Games", leave=False)
        for game_name, dataset in pbar2:
            env = environments[game_name]

            val_loss_list, obs_act_acc_list, obs_card_acc_list, act_acc_list, type_acc_list, state_dict = train_evaluate_world_model(params, dataset, NUM_AGENT_TYPES, env)

            current_loss_per_game[game_name] = val_loss_list
            #current_obs_action_acc_per_game[game_name] = obs_act_acc_list
            #current_obs_card_acc_per_game[game_name] = obs_card_acc_list
            current_act_acc_per_game[game_name] = act_acc_list
            current_type_acc_per_game[game_name] = type_acc_list
            current_models_per_game[game_name] = state_dict

        # Compute averages across games
        avg_last_loss          = np.mean([l[-1] for l in current_loss_per_game.values()])
        #avg_last_obs_action_acc = np.mean([v[-1] for v in current_obs_action_acc_per_game.values()])
        #avg_last_obs_card_acc  = np.mean([v[-1] for v in current_obs_card_acc_per_game.values()])
        avg_last_act_acc       = np.mean([v[-1] for v in current_act_acc_per_game.values()])
        avg_last_type_acc      = np.mean([v[-1] for v in current_type_acc_per_game.values()])

        # Write summary for this parameter set
        is_better = (best_params is None) or is_better_results(best_results, current_loss_per_game)
        _text = f"New Best {params}!! | " if is_better else f"Tested {params} | "
        tqdm.write(
            f"{_text}"
            f"Avg Val Loss: {avg_last_loss:.4f} | "
            #f"Obs Action Acc: {avg_last_obs_action_acc:.4f} | "
            #f"Obs Card Acc: {avg_last_obs_card_acc:.4f} | "
            f"Action Pred Acc: {avg_last_act_acc:.4f} | "
            f"Type ID Acc: {avg_last_type_acc:.4f}"
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
                results_data[f"loss_{game_name}"]        = current_loss_per_game[game_name]
                #results_data[f"obs_card_acc_{game_name}"] = current_obs_card_acc_per_game[game_name]
                #results_data[f"obs_act_acc_{game_name}"] = current_obs_action_acc_per_game[game_name]
                results_data[f"act_acc_{game_name}"]     = current_act_acc_per_game[game_name]
                results_data[f"type_acc_{game_name}"]    = current_type_acc_per_game[game_name]
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

        val_loss_list, obs_act_acc_list, obs_card_acc_list, act_acc_list, type_acc_list, state_dict = train_evaluate_world_model(best_params, dataset, NUM_AGENT_TYPES, env)

        results_data[f"loss_{game_name}"]         = val_loss_list
        #results_data[f"obs_act_acc_{game_name}"]  = obs_act_acc_list
        #results_data[f"obs_card_acc_{game_name}"] = obs_card_acc_list
        results_data[f"act_acc_{game_name}"]      = act_acc_list
        results_data[f"type_acc_{game_name}"]     = type_acc_list
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
    # null_action = env.num_actions  # only used in _encode_action which is commented out
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

    # Dataset — one pass only.
    # Baseline policies are deterministic, so repeating episodes yields identical trajectories.
    # The full dataset is num_agent_types × num_start_states × steps_per_episode unique points.
    # We run each (agent_type, start_state) pair exactly once and train on the full dataset.
    storage = []

    # Alternating-turn datapoint collection.
    # The game is strictly 2-player alternating: P0 acts at next_idx=1,3,5,...
    # and P1 acts at next_idx=2,4,6,...  At every step exactly one player takes
    # a real action while the other takes the null action.
    #
    # We collect from BOTH perspectives in a single pass:
    #   next_idx odd  → P0 acts; from P1's view P1 holds null, P1 predicts P0.
    #   next_idx even → P1 acts; from P0's view P0 holds null, P0 predicts P1.
    #
    # Every transition is a valid partner-prediction datapoint from exactly one
    # perspective, so no step is skipped and no step is double-counted.
    pbar2 = tqdm(enumerate(baseline_agents.keys()), leave=False)
    for type_idx, type_name in pbar2:
        agents_list = baseline_agents[type_name]

        for s0 in start_states:
            full_episode_log = run_episode(env, agents_list, list(s0), True)
            full_history = full_episode_log[:-1]

            cards = full_history[:start_len]
            actions = full_history[start_len:]
            full_history = [cards] + actions

            h_enc = np.zeros((MAX_SEQ_LEN, JOINT_OBS_DIM), dtype=np.float32)
            for i, joint_obs in enumerate(full_history):
                next_idx = i + 1
                if next_idx >= len(full_history):
                    break
                next_obs = full_history[next_idx]

                z_enc = _encode_joint_observation(joint_obs, JOINT_OBS_DIM, env)
                h_enc[i] = z_enc

                # Encode own action (always null), a_t
                #priv_a_enc = _encode_action(null_action, ACT_DIM)

                # Encode own next obs z_{t+1} (equals partner obs — z^i == z^{-i})
                priv_z_enc = _encode_observation(next_obs, OBS_DIM, env)

                # tgt_obs REMOVED — obs head disabled (z^i == z^{-i}, trivial identity)
                # partner_z_enc = _encode_observation(next_obs, OBS_DIM, env)

                # Encode partner action a^{-i}_{t+1}
                partner_a_enc = next_obs[0] if isinstance(next_obs, (tuple, list)) else next_obs

                storage.append({
                    "obs":             priv_z_enc,
                    #"act":            priv_a_enc,
                    "tgt_act":         partner_a_enc,
                    "tgt_obs":         priv_z_enc,   # API compat only; not used in loss
                    "tgt_type":        type_idx,
                    "past_episodes":   per_type_ensembles[type_name].copy(),
                    "current_history": h_enc.copy()
                })

    return convert_game_dataset(storage, OBS_DIM, JOINT_OBS_DIM, ACT_DIM, MAX_SEQ_LEN), mixed_ensemble


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


def convert_game_dataset(data, obs_dim, joint_obs_dim, act_dim, seq_len):
    df : Dataframe = dataframe_template.copy()
    df["data"] = ToMDataset(
        x_past_traj=[d['past_episodes'] for d in data],
        x_history=[d['current_history'] for d in data],
        x_obs=[d['obs'] for d in data],
        #x_act=[d['act'] for d in data],
        tgt_obs=[d['tgt_obs'] for d in data],
        tgt_act=[d['tgt_act'] for d in data],
        tgt_type=[d['tgt_type'] for d in data]
    )
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
def train_evaluate_world_model(params: dict, dataset_info: Dataframe, num_agents : int, env : Game = None) -> tuple[list[float], list[float], list[float], dict]:
    """
    Trains one model on one game for X epochs using full-batch gradient descent.
    env retained in signature for API compatibility (all game-specific branches removed with obs head).
    """
    # action_part_dim and card_part_dim were used for obs accuracy — removed with obs head
    # action_part_dim = env.num_actions + 1
    # if isinstance(env, MyHanabi):
    #     card_part_dim = env.num_cards * 4

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
    
    # Full-batch: load the entire (small, deterministic) dataset as one tensor batch.
    # Baseline policies are deterministic → no new information from repeated sampling.
    full_dataset = dataset_info["data"]
    full_loader  = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False)
    full_batch   = next(iter(full_loader))   # pre-load once; data never changes
    past     = full_batch['past']
    hist     = full_batch['history']
    obs      = full_batch['obs']
    tgt_act  = full_batch['tgt_act']
    tgt_type = full_batch['tgt_type']

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    ce_loss   = nn.CrossEntropyLoss()
    # bce_loss = nn.BCEWithLogitsLoss()  # REMOVED — obs head disabled

    epoch_train_losses = []
    # epoch_obs_action_acc = []  # REMOVED — obs head disabled
    # epoch_obs_card_acc = []    # REMOVED — obs head disabled
    epoch_action_acc = []
    epoch_type_acc   = []

    # 2. Training Loop — one full-batch gradient step per epoch
    pbar = tqdm(range(params['epochs']), leave=False)
    for _ in pbar:
        model.train()
        optimizer.zero_grad()

        act_logits, _, id_logits = model(past, hist, obs)

        # Loss: action prediction + identity regularisation
        # loss_obs removed — obs head disabled (z^i == z^{-i})
        loss_act = ce_loss(act_logits, tgt_act)
        loss_id  = ce_loss(id_logits,  tgt_type)
        loss     = loss_act + (0.2 * loss_id)
        loss.backward()
        optimizer.step()

        # Accuracy from the same logits (full batch, no separate eval pass needed)
        act_acc  = (act_logits.detach().argmax(dim=1) == tgt_act).float().mean().item()
        type_acc = (id_logits.detach().argmax(dim=1)  == tgt_type).float().mean().item()

        epoch_train_losses.append(loss.item())
        epoch_action_acc.append(act_acc)
        epoch_type_acc.append(type_acc)

        pbar.set_postfix({
            'loss':     f'{loss.item():.4f}',
            'act_acc':  f'{act_acc:.4f}',
            'type_acc': f'{type_acc:.4f}',
        })
    # Finalize Training
    # obs_act and obs_card lists returned as [0.0] sentinels (obs head removed)
    return epoch_train_losses, [0.0], [0.0], epoch_action_acc, epoch_type_acc, model.state_dict()


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
