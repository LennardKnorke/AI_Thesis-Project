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

from agents import *
from agents.model_based.worldmodel import _encode_observation, _encode_joint_observation
from config import *
from tiny_game import *
from runner import run_episode

# --- Constants ---
TOM_PARAM_LOG_FILE = os.path.join(WORLD_MODELS_DIR, "tested_params.json")

VAL_RATIO        = 0.3
IDENTITY_WEIGHT  = 0.1
DELTA_WEIGHT     = 0.6

# --- Hyperparameter grid ---
tom_net_grid = {
    "char_dim":    [8],
    "mental_dim":  [16, 32, 8],
    "trunk_dim":   [16, 32, 64],
    "lr":          [0.01, 0.005, 0.001],
    "epochs":      [100],
    "batch_size":  [64],
    "use_obs":     [True],
}
tom_net_grid = generate_param_grid(tom_net_grid)


class Random_Agent(ModelBasedAgent):
    """Thin executor whose policy is maintained by Random_Planner."""

    def __init__(self, env: Game, num_cards: int, num_actions: int, agent_id: int, policy: dict):
        super().__init__(env, num_cards, num_actions)
        self.agent_id = agent_id
        self.policy   = policy

    def act(self, input_state: tuple, exploit: bool = False) -> int:
        return self.policy[input_state]

    def train(self):                return 0.0
    def save_transition(self, *_):  pass
    def save(self, *_):             pass
    def load(self, *_):             pass
    def reset(self):                pass


class Random_Planner(AgentList):
    """
    Assigns a fixed, random valid action to every private history.
    Actions are weighted inversely to how often existing baselines choose them,
    so the random agent fills underexplored parts of the action space.
    """

    def __init__(
        self,
        env: Game,
        num_cards: int,
        num_actions: int,
        game_name: str,
        baselines,
        *args, **kwargs,
    ):
        self.env         = env
        self.game_name   = game_name
        self.num_cards   = num_cards
        self.num_actions = num_actions
        self.baselines   = baselines
        self.policy: dict[tuple, int] = {}
        self._init_tables()

        agent_0 = Random_Agent(env, num_cards, num_actions, 0, self.policy)
        agent_1 = Random_Agent(env, num_cards, num_actions, 1, self.policy)
        super().__init__([agent_0, agent_1])

    @property
    def centralized_planning(self) -> bool:
        return True

    def _init_tables(self):
        for priv_h, actions, done, turn_id, _ in PRIV_HISTORIES[self.game_name]:
            if not done:
                legal  = list(actions)
                counts = {a: 0 for a in legal}
                for _, b_agents in self.baselines.items():
                    chosen_a = b_agents[int(turn_id)].act(priv_h, exploit=True)
                    counts[chosen_a] += 1
                weights = [1.0 / (1.0 + counts[a]) for a in legal]
                self.policy[priv_h] = random.choices(legal, weights=weights, k=1)[0]

    def train(self) -> float:   return 0.0
    def reset(self):            pass


# --- Dataset ---

class ToMDataset(torch.utils.data.Dataset):
    """
    Dataset for the ToM World Model. Each item:
        past:      (N_PAST, SEQ, FEAT)   — past episode encodings
        past_mask: (N_PAST,)             — 1 = real episode, 0 = padded
        history:   (SEQ, FEAT)           — current joint history h_t
        obs:       (OBS_DIM,)            — own current observation z_t
        tgt_obs:   (OBS_DIM,)            — partner next obs (API compat; unused in loss)
        tgt_act:   scalar long           — partner action index
        tgt_type:  scalar long           — agent profile index
    """
    def __init__(self, x_past_traj, x_past_mask, x_history, x_obs, tgt_obs, tgt_act, tgt_type):
        self.x_past_traj = torch.as_tensor(np.asarray(x_past_traj), dtype=torch.float32, device=DEVICE)
        self.x_past_mask = torch.as_tensor(np.asarray(x_past_mask), dtype=torch.float32, device=DEVICE)
        self.x_history   = torch.as_tensor(np.asarray(x_history),   dtype=torch.float32, device=DEVICE)
        self.x_obs       = torch.as_tensor(np.asarray(x_obs),       dtype=torch.float32, device=DEVICE)
        self.y_obs       = torch.as_tensor(np.asarray(tgt_obs),     dtype=torch.float32, device=DEVICE)
        self.y_act       = torch.as_tensor(np.asarray(tgt_act),     dtype=torch.long,    device=DEVICE)
        self.y_type      = torch.as_tensor(np.asarray(tgt_type),    dtype=torch.long,    device=DEVICE)

    def __len__(self):
        return len(self.x_obs)

    def __getitem__(self, idx):
        return {
            "past":      self.x_past_traj[idx],
            "past_mask": self.x_past_mask[idx],
            "history":   self.x_history[idx],
            "obs":       self.x_obs[idx],
            "tgt_obs":   self.y_obs[idx],
            "tgt_act":   self.y_act[idx],
            "tgt_type":  self.y_type[idx],
        }


Dataframe = dict[str, Dataset | DataLoader | int]
dataframe_template: Dataframe = {
    "data":           ToMDataset,
    "obs_dim":        int,
    "action_dim":     int,
    "max_seq_length": int,
    "joint_obs_dim":  int,
}


# --- Main ---

def train_worldmodel() -> None:
    os.makedirs(WORLD_MODELS_DIR, exist_ok=True)

    baseline_agents_per_game = setup_baseline_agents()
    full_dataset = setup_dataset(baseline_agents_per_game)
    if full_dataset is None:
        print("No data collected. Check trained baselines.")
        return

    best_params, best_results = load_tmp_best_results()
    processed_params = load_processed_params(best_params) if best_params is not None else []

    # Precompute the number of unique policy profiles per game
    num_profiles_per_game = {
        game_name: len(identify_unique_profiles(agents))
        for game_name, agents in baseline_agents_per_game.items()
    }

    pbar = tqdm(tom_net_grid, desc="ToM WM Parameter Search")
    for params in pbar:
        if are_params_processed(processed_params, params):
            continue

        current_loss_per_game            = {}
        current_train_act_acc_per_game   = {}
        current_train_type_acc_per_game  = {}
        current_val_act_acc_per_game     = {}
        current_val_type_acc_per_game    = {}
        current_val_act_delta_per_game   = {}
        current_models_per_game          = {}

        pbar2 = tqdm(full_dataset.items(), desc="Iterate over Games", leave=False)
        for game_name, dataset in pbar2:
            env = ENVIRONMENTS[game_name]
            train_data, val_data = dataset
            train_losses, train_act_acc, val_act_acc, train_type_acc, val_type_acc, val_act_deltas, state_dict = \
                train_evaluate_world_model(params, train_data, val_data, num_profiles_per_game[game_name], env)

            current_loss_per_game[game_name]           = train_losses
            current_train_act_acc_per_game[game_name]  = train_act_acc
            current_train_type_acc_per_game[game_name] = train_type_acc
            current_val_act_acc_per_game[game_name]    = val_act_acc
            current_val_type_acc_per_game[game_name]   = val_type_acc
            current_val_act_delta_per_game[game_name]  = val_act_deltas
            current_models_per_game[game_name]         = state_dict

        avg_val_act_acc   = np.mean([v[-1] for v in current_val_act_acc_per_game.values()])
        avg_val_type_acc  = np.mean([v[-1] for v in current_val_type_acc_per_game.values()])
        avg_val_delta     = current_val_act_delta_per_game.get("G", [0.0])[-1]
        avg_val_delta_all = np.mean([v[-1] for v in current_val_act_delta_per_game.values()])

        is_better, composite = is_better_results(best_results, avg_val_act_acc, avg_val_type_acc, avg_val_delta_all)

        _text = "NEW BEST!" if is_better else "Tested:"
        tqdm.write(
            f"{_text} Params: {params} | "
            f"ID Acc: {avg_val_type_acc:.3f} | "
            f"Mean Act: {avg_val_act_acc:.3f} | "
            f"Delta G: {avg_val_delta:+.3f} | "
            f"Delta All: {avg_val_delta_all:+.3f}"
        )

        if is_better:
            best_params  = params
            best_results = {"best_composite": [composite]}
            with open(os.path.join(WORLD_MODELS_DIR, "best_params.json"), 'w') as f:
                json.dump(best_params, f, indent=4)
            for game_name, state_dict in current_models_per_game.items():
                torch.save(state_dict, os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth"))
            for game_name in current_loss_per_game.keys():
                best_results[f"loss_{game_name}"]           = current_loss_per_game[game_name]
                best_results[f"train_act_acc_{game_name}"]  = current_train_act_acc_per_game[game_name]
                best_results[f"val_act_acc_{game_name}"]    = current_val_act_acc_per_game[game_name]
                best_results[f"train_type_acc_{game_name}"] = current_train_type_acc_per_game[game_name]
                best_results[f"val_type_acc_{game_name}"]   = current_val_type_acc_per_game[game_name]
                best_results[f"val_act_delta_{game_name}"]  = current_val_act_delta_per_game[game_name]
            df = pd.DataFrame({k: pd.Series(v) for k, v in best_results.items()})
            df.to_csv(os.path.join(WORLD_MODELS_DIR, "final_results.csv"), index=False)

        processed_params.append(params)
        log_processed_params(params)


# --- Agent setup ---

def setup_baseline_agents():
    agents_per_game: dict[str, dict[str, AgentList]] = {}

    def setup_iql_agent(game_name: str):
        path       = os.path.join(RESULTS_DIR, "IQL")
        params     = load_best_params(os.path.join(path, "best_params.json"))
        env        = ENVIRONMENTS[game_name]
        agent_0    = IQ_Learning_Agent(env, game_name, env.num_cards, env.num_actions, **params)
        agent_1    = IQ_Learning_Agent(env, game_name, env.num_cards, env.num_actions, **params)
        agent_0.load(os.path.join(path, f"{game_name}_agent0.pkl"))
        agent_1.load(os.path.join(path, f"{game_name}_agent1.pkl"))
        return AgentList([agent_0, agent_1])

    def setup_vdn_agent(game_name: str):
        path    = os.path.join(RESULTS_DIR, "VDN")
        params  = load_best_params(os.path.join(path, "best_params.json"))
        env     = ENVIRONMENTS[game_name]
        agents  = VDN_CentralPlanner(env, game_name, env.num_cards, env.num_actions, **params)
        agents.load(os.path.join(path, f"{game_name}_agents.pkl"))
        return agents

    def setup_osarsa_agent(game_name: str):
        path    = os.path.join(RESULTS_DIR, "Osarsa")
        params  = load_best_params(os.path.join(path, "best_params.json"))
        env     = ENVIRONMENTS[game_name]
        agents  = OSarsa_Planner(env, game_name, env.num_cards, env.num_actions, **params)
        agents.load(os.path.join(path, f"G_{game_name}_shared_model.pkl"))
        return agents

    def setup_pbdp_agent(game_name: str):
        path    = os.path.join(RESULTS_DIR, "PBDP")
        params  = load_best_params(os.path.join(path, "best_params.json"))
        env     = ENVIRONMENTS[game_name]
        agents  = PBDP_Central_Planner(env, env.num_cards, env.num_actions, game_name, **params)
        agents.load(os.path.join(path, f"{game_name}_agents.pkl"))
        return agents

    def setup_random_agent(game_name: str, baseline_agents: dict[str, AgentList]):
        state = random.getstate()
        random.seed(42)
        env    = ENVIRONMENTS[game_name]
        agents = Random_Planner(env, env.num_cards, env.num_actions, game_name, baseline_agents)
        random.setstate(state)
        return agents

    print("Loading Baseline Agents")
    for game_name in GAMES:
        game_agents: dict[str, AgentList] = {
            "OSarsa": setup_osarsa_agent(game_name),
            "IQL":    setup_iql_agent(game_name),
            "VDN":    setup_vdn_agent(game_name),
            "PBDP":   setup_pbdp_agent(game_name),
        }
        game_agents["Random"] = setup_random_agent(game_name, game_agents)
        agents_per_game[game_name] = game_agents
    return agents_per_game


def identify_unique_profiles(baseline_agents: dict[str, AgentList]) -> dict[int, list[str]]:
    """
    Groups algorithms by unique policy profile.
    Returns {profile_id: [algo_name, ...]} with consecutive integer keys.
    """
    profiles: dict[int, list[str]] = defaultdict(list)
    seen: list[dict] = []

    for algo, agent_list in baseline_agents.items():
        if algo != "IQL":
            policy = agent_list.policy
        else:
            # Merge P0 and P1 then restrict to the state keys used by other agents
            # so the comparison is over the same observation space.
            combined      = {**agent_list[0].policy, **agent_list[1].policy}
            template_keys = seen[0].keys()   # IQL is never first, so seen[0] is guaranteed
            policy = {k: combined[k] for k in template_keys if k in combined}

        for i, known_policy in enumerate(seen):
            if policy == known_policy:
                profiles[i].append(algo)
                break
        else:
            profiles[len(seen)].append(algo)
            seen.append(policy)

    return dict(profiles)


# --- Dataset construction ---

def setup_dataset(baseline_agents_per_game: dict[str, dict[str, AgentList]], *args, **kwargs) -> dict[str, Dataframe]:
    full_datasets = {}
    random.seed(MAIN_SEED)
    np.random.seed(MAIN_SEED)
    split_generator = torch.Generator().manual_seed(MAIN_SEED)

    pbar = tqdm(baseline_agents_per_game.keys(), desc="Set up Games Datasets")
    for game_name in pbar:
        env          = ENVIRONMENTS[game_name]
        agents       = baseline_agents_per_game[game_name]
        game_dataset = collect_game_datasets(env, agents, game_name, *args, **kwargs)

        total_size = len(game_dataset["data"])
        val_size   = int(VAL_RATIO * total_size)
        train_size = total_size - val_size
        train_subset, val_subset = random_split(
            game_dataset["data"], [train_size, val_size], generator=split_generator
        )

        # Save val tensors for post-hoc context-accuracy analysis
        val_path = os.path.join(WORLD_MODELS_DIR, f"val_{game_name}.pt")
        d   = game_dataset["data"]
        idx = torch.tensor(val_subset.indices)
        torch.save({
            'past':           d.x_past_traj[idx],
            'past_mask':      d.x_past_mask[idx],
            'history':        d.x_history[idx],
            'obs':            d.x_obs[idx],
            'tgt_act':        d.y_act[idx],
            'tgt_type':       d.y_type[idx],
            'obs_dim':        game_dataset["obs_dim"],
            'joint_obs_dim':  game_dataset["joint_obs_dim"],
            'act_dim':        game_dataset["act_dim"],
            'max_seq_length': game_dataset["max_seq_length"],
        }, val_path)

        train_df = game_dataset.copy()
        val_df   = game_dataset.copy()
        train_df["data"] = train_subset
        val_df["data"]   = val_subset
        full_datasets[game_name] = (train_df, val_df)
    return full_datasets


def collect_game_datasets(env: Game, baseline_agents: dict[str, AgentList], game_name: str, *args, **kwargs) -> Dataframe:
    ACT_DIM = env.num_actions

    if isinstance(env, DecPOMDP):
        start_len     = 2
        MAX_SEQ_LEN   = env.horizon - 1
        obs_act_dim   = env.num_actions
        obs_card_dim  = env.num_cards * 2
    elif isinstance(env, MyHanabi):
        start_len     = 4
        MAX_SEQ_LEN   = env.horizon - 3
        obs_act_dim   = env.num_actions + env.num_cards + 1
        obs_card_dim  = env.num_cards * start_len
    else:
        raise ValueError("Upsi?")

    OBS_DIM       = obs_act_dim + obs_card_dim
    JOINT_OBS_DIM = OBS_DIM

    start_states          = list(env.start_states())
    PAST_EPISODES_CONTEXT = len(start_states)

    # Mixed ensemble (single shared pool) — disabled in favour of per-type ensembles below.
    # mixed_ensemble = _setup_ensemble(
    #     env, baseline_agents, PAST_EPISODES_CONTEXT, MAX_SEQ_LEN, JOINT_OBS_DIM, game_name
    # )

    # Per-agent-type ensembles — CharNet sees only episodes from the target agent type
    per_type_ensembles: dict[str, np.ndarray] = {
        _type_name: _setup_ensemble(
            env, {_type_name: _agents},
            PAST_EPISODES_CONTEXT, MAX_SEQ_LEN, JOINT_OBS_DIM, game_name,
            type_name=_type_name,
        )
        for _type_name, _agents in baseline_agents.items()
    }

    unique_profiles = identify_unique_profiles(baseline_agents)
    type_to_profile = {
        name: profile_id
        for profile_id, names in unique_profiles.items()
        for name in names
    }

    storage = []
    pbar2   = tqdm(enumerate(baseline_agents.keys()), leave=False)
    for _, type_name in pbar2:
        agents_list    = baseline_agents[type_name]
        type_ensemble  = per_type_ensembles[type_name]
        type_idx       = type_to_profile[type_name]

        random.shuffle(start_states)
        for s0 in start_states:
            full_episode_log = run_episode(env, agents_list, list(s0), True)
            full_history     = full_episode_log[:-1]

            cards        = full_history[:start_len]
            actions      = full_history[start_len:]
            full_history = [cards] + actions

            for i, joint_obs in enumerate(full_history):
                next_idx = i + 1
                if next_idx >= len(full_history):
                    break

                next_obs      = full_history[next_idx]
                current_seq   = full_history[:i + 1]
                seq_length    = min(len(current_seq), MAX_SEQ_LEN)
                truncated_seq = current_seq[-seq_length:]

                h_enc = np.zeros((MAX_SEQ_LEN, JOINT_OBS_DIM), dtype=np.float32)
                for t_idx, obs_t in enumerate(truncated_seq):
                    h_enc[MAX_SEQ_LEN - seq_length + t_idx] = _encode_joint_observation(obs_t, JOINT_OBS_DIM, env)

                z_enc        = _encode_joint_observation(joint_obs, JOINT_OBS_DIM, env)
                h_enc[MAX_SEQ_LEN - seq_length + i] = z_enc

                priv_z_enc   = _encode_observation(next_obs, OBS_DIM, env)
                partner_a_enc = next_obs[0] if isinstance(next_obs, (tuple, list)) else next_obs

                for k in range(0, PAST_EPISODES_CONTEXT + 1):
                    current_ensemble = np.zeros_like(type_ensemble)
                    mask             = np.zeros(PAST_EPISODES_CONTEXT, dtype=np.float32)
                    if k > 0:
                        indices      = np.random.choice(len(type_ensemble), k, replace=False)
                        mask[:k]     = 1.0
                        current_ensemble[:k] = type_ensemble[indices]
                        np.random.shuffle(current_ensemble[:k])

                    storage.append({
                        "obs":             priv_z_enc,
                        "tgt_act":         partner_a_enc,
                        "tgt_obs":         priv_z_enc,
                        "tgt_type":        type_idx,
                        "past_episodes":   current_ensemble.copy(),
                        "past_mask":       mask.copy(),
                        "current_history": h_enc.copy(),
                    })

    return convert_game_dataset(storage, OBS_DIM, JOINT_OBS_DIM, ACT_DIM, MAX_SEQ_LEN)


def _setup_ensemble(env: Game, baseline_agent: dict, past_episodes_context, max_seq_len, obs_dim, game_name, type_name: str | None = None):
    # Determine cache path — per-type or mixed
    if type_name is not None:
        path_name     = type_name.replace(" ", "_")
        ensemble_path = os.path.join(WORLD_MODELS_DIR, f"G_{game_name}_{path_name}_ensemble.npy")
    else:
        ensemble_path = os.path.join(WORLD_MODELS_DIR, f"G_{game_name}_ensemble.npy")

    if os.path.exists(ensemble_path):
        tqdm.write(f"Loaded Ensemble: {os.path.basename(ensemble_path)}")
        return np.load(ensemble_path)

    vec          = np.zeros((past_episodes_context, max_seq_len, obs_dim))
    start_states = env.start_states()
    cuttoff_idx  = 2 if isinstance(env, DecPOMDP) else 4
    p_idx        = 0

    while p_idx < past_episodes_context:
        for _, agents in baseline_agent.items():
            for s0 in start_states:
                if p_idx >= past_episodes_context:
                    break
                full_history = run_episode(env, agents, list(s0), test_episode=True)[:-1]
                cards        = full_history[:cuttoff_idx]
                actions      = full_history[cuttoff_idx:]
                full_history = [cards] + actions

                seq_length    = min(len(full_history), max_seq_len)
                truncated_seq = full_history[-seq_length:]
                h_enc         = np.zeros((max_seq_len, obs_dim))
                for i, obs in enumerate(truncated_seq):
                    h_enc[max_seq_len - seq_length + i] = _encode_joint_observation(obs, obs_dim, env)
                vec[p_idx] = h_enc.copy()
                p_idx += 1

    np.save(ensemble_path, vec)
    tqdm.write(f"Created Ensemble: {os.path.basename(ensemble_path)}")
    return vec


def convert_game_dataset(data, obs_dim, joint_obs_dim, act_dim, seq_len):
    df = dataframe_template.copy()
    df["data"] = ToMDataset(
        x_past_traj=[d['past_episodes']   for d in data],
        x_past_mask=[d['past_mask']        for d in data],
        x_history  =[d['current_history']  for d in data],
        x_obs      =[d['obs']              for d in data],
        tgt_obs    =[d['tgt_obs']          for d in data],
        tgt_act    =[d['tgt_act']          for d in data],
        tgt_type   =[d['tgt_type']         for d in data],
    )
    df["joint_obs_dim"]  = joint_obs_dim
    df["obs_dim"]        = obs_dim
    df["act_dim"]        = act_dim
    df["max_seq_length"] = seq_len
    return df


# --- Checkpoint helpers ---

def load_tmp_best_results():
    params_path  = os.path.join(WORLD_MODELS_DIR, "best_params.json")
    results_path = os.path.join(WORLD_MODELS_DIR, "final_results.csv")
    if not os.path.exists(params_path) or not os.path.exists(results_path):
        return None, {}
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    df = pd.read_csv(results_path)
    best_results = {col: df[col].dropna().tolist() for col in df.columns}
    return best_params, best_results


# --- Training ---

def train_evaluate_world_model(params: dict, train_info: Dataframe, val_info: Dataframe, num_profiles: int, env: Game = None):
    """Trains and evaluates one world model on one game with the given hyperparameters."""
    model = ToM_WorldModel(
        joint_obs_dim        = train_info["joint_obs_dim"],
        obs_dim              = train_info["obs_dim"],
        action_dim           = train_info["act_dim"],
        num_agent_types      = num_profiles,
        max_seq_len          = train_info['max_seq_length'],
        char_embed_dim       = params['char_dim'],
        mental_embed_dim     = params['mental_dim'],
        trunk_dim            = params['trunk_dim'],
        use_obs              = params.get('use_obs', False),
    ).to(DEVICE)

    batch_size   = params['batch_size']
    train_loader = DataLoader(train_info["data"], batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_info["data"],   batch_size=batch_size, shuffle=False)
    optimizer    = optim.Adam(model.parameters(), lr=params['lr'])
    ce_loss      = nn.CrossEntropyLoss()

    train_losses    = []
    train_act_acc   = []
    train_type_acc  = []
    val_act_acc     = []
    val_type_acc    = []
    val_act_deltas  = []

    pbar = tqdm(range(params['epochs']), leave=False)
    for epoch in pbar:
        # Training pass
        model.train()
        epoch_loss = epoch_train_act = epoch_train_type = 0.0
        num_train_batches = 0

        for batch in train_loader:
            past, mask    = batch['past'], batch['past_mask']
            history, obs  = batch['history'], batch['obs']
            tgt_act       = batch['tgt_act']
            tgt_type      = batch['tgt_type']

            optimizer.zero_grad()
            act_logits, _, id_logits, _ = model(past, mask, history, obs)
            loss = ce_loss(act_logits, tgt_act) + IDENTITY_WEIGHT * ce_loss(id_logits, tgt_type)
            loss.backward()
            optimizer.step()

            epoch_loss       += loss.item()
            epoch_train_act  += (act_logits.argmax(dim=1) == tgt_act).float().mean().item()
            epoch_train_type += (id_logits.argmax(dim=1)  == tgt_type).float().mean().item()
            num_train_batches += 1

        train_losses.append(epoch_loss       / num_train_batches)
        train_act_acc.append(epoch_train_act  / num_train_batches)
        train_type_acc.append(epoch_train_type / num_train_batches)

        # Validation pass
        model.eval()
        epoch_val_act = epoch_val_type = 0.0
        num_val_batches = 0
        val_k_correct   = defaultdict(float)
        val_k_total     = defaultdict(float)

        with torch.no_grad():
            for batch in val_loader:
                past, mask    = batch['past'], batch['past_mask']
                history, obs  = batch['history'], batch['obs']
                tgt_act       = batch['tgt_act']
                tgt_type      = batch['tgt_type']

                act_logits, _, id_logits, _ = model(past, mask, history, obs)
                preds = act_logits.argmax(dim=1)

                epoch_val_act  += (preds == tgt_act).float().mean().item()
                epoch_val_type += (id_logits.argmax(dim=1) == tgt_type).float().mean().item()
                num_val_batches += 1

                k_vals   = mask.sum(dim=1).long()
                corrects = (preds == tgt_act).float()
                for c, k in zip(corrects, k_vals):
                    val_k_correct[k.item()] += c.item()
                    val_k_total[k.item()]   += 1.0

        val_act_acc.append(epoch_val_act  / num_val_batches)
        val_type_acc.append(epoch_val_type / num_val_batches)

        # Delta: accuracy at high context vs. zero context
        k_accs = {k: val_k_correct[k] / max(1.0, val_k_total[k]) for k in val_k_total}
        max_k             = max(val_k_total.keys()) if val_k_total else 0
        informed_threshold = int(max_k * 0.7)
        informed_values   = [k_accs[k] for k in k_accs if k >= informed_threshold]
        acc_informed      = np.mean(informed_values) if informed_values else 0.0
        acc_k0            = k_accs.get(0, 0.0)
        epoch_delta       = acc_informed - acc_k0
        val_act_deltas.append(epoch_delta)

        pbar.set_postfix({
            'loss':  f'{train_losses[-1]:.4f}',
            'v_act': f'{val_act_acc[-1]:.3f}',
            'v_id':  f'{val_type_acc[-1]:.3f}',
            'delta': f'{epoch_delta:+.3f}',
        })

    return train_losses, train_act_acc, val_act_acc, train_type_acc, val_type_acc, val_act_deltas, model.state_dict()


def is_better_results(best_results: dict | None, current_val_act_acc: float, current_val_type_acc: float, current_val_act_delta: float) -> tuple[bool, float]:
    """Returns (is_better, composite_score). Higher composite is better."""
    ACT_WEIGHT = 1.0 - IDENTITY_WEIGHT - DELTA_WEIGHT
    current_composite = (
        IDENTITY_WEIGHT * current_val_type_acc +
        ACT_WEIGHT      * current_val_act_acc  +
        DELTA_WEIGHT    * current_val_act_delta
    )
    if best_results is None:
        return True, current_composite
    prev_composite = best_results.get("best_composite", [0.0])[0]
    return current_composite > prev_composite, current_composite


# --- Param logging helpers ---

def log_processed_params(params: dict):
    with open(TOM_PARAM_LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(params, f)
        f.write("\n")


def load_processed_params(best_params):
    if best_params is None or not os.path.exists(TOM_PARAM_LOG_FILE):
        return []
    dictionaries = []
    with open(TOM_PARAM_LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and line not in dictionaries:
                dictionaries.append(json.loads(line))
    return dictionaries


def are_params_processed(processed_params: list[dict], new_params: dict) -> bool:
    def dicts_equal_strict(d1: dict, d2: dict) -> bool:
        if d1.keys() != d2.keys():
            return False
        return all(type(d1[k]) is type(d2[k]) and d1[k] == d2[k] for k in d1)
    return any(dicts_equal_strict(p, new_params) for p in processed_params)


if __name__ == "__main__":
    train_worldmodel()
