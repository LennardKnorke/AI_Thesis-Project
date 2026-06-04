# train_pomcp_tom.py — Hyperparameter search and evaluation for POMCP-ToM agent.
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from tiny_game import *
from runner import *
from agents import *
from config import *
from train_worldmodel import setup_baseline_agents

NUM_RUNS = 5

_name = "ToM-POMCP"
# Setting Folders
_agent_sub_dir = _name.replace(" ", "_")
_results_dir = os.path.join(RESULTS_DIR, _agent_sub_dir)
os.makedirs(_results_dir, exist_ok=True)


pomcp_param_grid = {
    'n_simulations' : [2_500],
    'exploration_constant' : [1.41, 2.0],
    'gamma' : [0.99],
    'selection_rule' : ["ucb1", "ucb1_tuned", "puct"],
}
pomcp_param_grid = generate_param_grid(pomcp_param_grid)


def load_world_model_and_config(game_name: str, device: str, env : Game) -> tuple[ToM_WorldModel, dict[str, Any]]:
    wm_best_params_path = os.path.join(WORLD_MODELS_DIR, "best_params.json")
    if not os.path.exists(wm_best_params_path):
        raise FileNotFoundError(f"World model best_params.json not found at {wm_best_params_path}. "
                                f"Please ensure the world model has been trained.")
    
    with open(wm_best_params_path, 'r') as f:
        wm_training_params = json.load(f)
    
    # Must match train_worldmodel.py: ACT_DIM = env.num_actions (no null slot).
    # The saved checkpoint was trained with this value.
    ACT_DIM = env.num_actions
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


    wm_path = os.path.join(WORLD_MODELS_DIR, f"WM_{game_name}.pth")
    if not os.path.exists(wm_path):
        raise FileNotFoundError(f"World model .pth file not found for game {game_name} at {wm_path}. "
                                f"Please ensure world models are trained via `train_worldmodel.py`.")

    state_dict = torch.load(wm_path, map_location=device)
    ckpt_num_agent_types = state_dict['char_net.identity_classifier.weight'].shape[0]

    wm_config = {
        'obs_dim': OBS_DIM,
        "joint_obs_dim" : JOINT_OBS_DIM,
        'action_dim': ACT_DIM,
        'max_seq_len': MAX_SEQ_LEN,
        'num_agent_types': ckpt_num_agent_types,
        'char_embed_dim': wm_training_params['char_dim'],
        'mental_embed_dim': wm_training_params['mental_dim'],
        'trunk_dim': wm_training_params['trunk_dim'],
        'use_obs': wm_training_params.get('use_obs', True),
        'lr' : wm_training_params['lr'],

        'action_output_dim' : ACT_DIM
    }

    world_model = ToM_WorldModel(**wm_config)
    world_model.load_state_dict(state_dict)
    world_model.to(device)
    world_model.eval()

    return world_model, wm_config


def load_all_world_models() -> dict[str, tuple[ToM_WorldModel, dict[str, Any]]]:
    all_world_models: dict[str, tuple[ToM_WorldModel, dict[str, Any]]] = {}
    print("\nLoading all World Models...")
    for game_name in GAMES:
        wm, wm_config = load_world_model_and_config(game_name, 'cpu', ENVIRONMENTS[game_name])
        all_world_models[game_name] = (wm, wm_config)
    print("Finished loading World Models.")
    return all_world_models

def load_ensembles():
    """Load the mixed (general) ensemble per game: G_{game}_ensemble.npy."""
    try:
        ensembles = {}
        for gname in GAMES:
            filepath = os.path.join(WORLD_MODELS_DIR, f"G_{gname}_ensemble.npy")
            ensembles[gname] = np.load(filepath)
        return ensembles
    except:
        return None


def load_specialized_ensembles() -> dict[str, dict[str, np.ndarray]]:
    """Load per-agent-type ensembles keyed by [game_name][agent_type]. Missing files are silently omitted."""
    specialized: dict[str, dict[str, np.ndarray]] = {}
    for gname in GAMES:
        specialized[gname] = {}
        for exp in ["IQL", "VDN", "PBDP", "OSarsa"]:
            fpath = os.path.join(WORLD_MODELS_DIR, f"G_{gname}_{exp}_ensemble.npy")
            if os.path.exists(fpath):
                specialized[gname][exp] = np.load(fpath)
    return specialized


def _evaluate_agents(
        env: Game,
        agent_params: dict,
        baseline_agents: dict[str, AgentList],
        game_name: str,
        special_ensembles: dict[str, np.ndarray],
        cheat_toggle: bool = False,
    ) -> tuple[dict, "POMCP_ToM_Agent"]:
    initial_ensemble = np.zeros_like(special_ensembles['IQL'])
    tom_agent = POMCP_ToM_Agent(
        env=env, game_name=game_name,
        base_ensemble=initial_ensemble,
        **agent_params,
    )
    start_states = list(env.start_states())
    num_s0       = len(start_states)
    extra_parts  = list(env.start_states())
    results      = {}

    # --- Evaluate as P0 ---
    tom_agent.agent_id = 0
    pbar = tqdm(baseline_agents.items(), desc="Evaluating as P0", leave=False)
    for base_name, base_agents in pbar:
        agents = AgentList([tom_agent, base_agents[1]])
        if cheat_toggle:
            tom_agent.cheat_partner = base_agents[1]

        sum_rewards = sum_moving_avg = None
        for n in tqdm(range(num_s0), desc="Runs", leave=False):
            tom_agent.reset()
            run_rewards, run_moving_avg = [], []

            remaining = [s for i, s in enumerate(start_states) if i != n]
            random.shuffle(remaining)
            run_start_states = [start_states[n]] + remaining

            if not cheat_toggle:
                random.shuffle(extra_parts)
                run_start_states.extend(extra_parts)

            for s0 in tqdm(run_start_states, desc="s0", leave=False):
                tom_agent.reuse_tree()
                episode_log = run_episode(env, agents, s0, test_episode=True)
                reward = normalize_payoffs(episode_log[-1],
                                          PAYOFFS[game_name].max(),
                                          PAYOFFS[game_name].min())
                run_rewards.append(np.float32(reward))
                avg = np.mean(run_rewards[-min(len(start_states), len(run_rewards)):])
                run_moving_avg.append(avg)
                tom_agent.update_ensemble(episode_log[:-1])

            run_rewards    = np.array(run_rewards)
            run_moving_avg = np.array(run_moving_avg)
            results[f"p0_reward_{n}_{base_name}"]     = run_rewards
            results[f"p0_moving_avg_{n}_{base_name}"] = run_moving_avg

            if sum_rewards is None:
                sum_rewards    = np.zeros_like(run_rewards)
                sum_moving_avg = np.zeros_like(run_moving_avg)
            sum_rewards    += run_rewards
            sum_moving_avg += run_moving_avg

        results[f"p0_reward_{base_name}"]     = sum_rewards    / num_s0
        results[f"p0_moving_avg_{base_name}"] = sum_moving_avg / num_s0

    # --- Evaluate as P1 ---
    tom_agent.agent_id = 1
    pbar = tqdm(baseline_agents.items(), desc="Evaluating as P1", leave=False)
    for base_name, base_agents in pbar:
        agents = AgentList([base_agents[0], tom_agent])
        if cheat_toggle:
            tom_agent.cheat_partner = base_agents[0]

        sum_rewards = sum_moving_avg = None
        for n in tqdm(range(num_s0), desc="Runs", leave=False):
            tom_agent.reset()
            run_rewards, run_moving_avg = [], []

            remaining = [s for i, s in enumerate(start_states) if i != n]
            random.shuffle(remaining)
            run_start_states = [start_states[n]] + remaining
            if not cheat_toggle:
                random.shuffle(extra_parts)
                run_start_states.extend(extra_parts)

            for s0 in tqdm(run_start_states, desc="s0", leave=False):
                tom_agent.reuse_tree()
                episode_log = run_episode(env, agents, s0, test_episode=True)
                reward = normalize_payoffs(episode_log[-1],
                                          PAYOFFS[game_name].max(),
                                          PAYOFFS[game_name].min())
                run_rewards.append(np.float32(reward))
                window = min(len(start_states), len(run_moving_avg))
                avg = np.mean(run_rewards[-window:]) if run_rewards else reward
                run_moving_avg.append(avg)
                tom_agent.update_ensemble(episode_log[:-1])

            run_rewards    = np.array(run_rewards)
            run_moving_avg = np.array(run_moving_avg)
            results[f"p1_reward_{n}_{base_name}"]     = run_rewards
            results[f"p1_moving_avg_{n}_{base_name}"] = run_moving_avg

            if sum_rewards is None:
                sum_rewards    = np.zeros_like(run_rewards)
                sum_moving_avg = np.zeros_like(run_moving_avg)
            sum_rewards    += run_rewards
            sum_moving_avg += run_moving_avg

        results[f"p1_reward_{base_name}"]     = sum_rewards    / num_s0
        results[f"p1_moving_avg_{base_name}"] = sum_moving_avg / num_s0

    
    for base_name in baseline_agents.keys():
        p0_r  = results[f"p0_reward_{base_name}"]
        p1_r  = results[f"p1_reward_{base_name}"]
        p0_ma = results[f"p0_moving_avg_{base_name}"]
        p1_ma = results[f"p1_moving_avg_{base_name}"]
        n = min(len(p0_r), len(p1_r))
        results[f"reward_{base_name}"]      = (p0_r[:n]  + p1_r[:n])  / 2
        results[f"moving_avg_{base_name}"]  = (p0_ma[:n] + p1_ma[:n]) / 2

    n = min(len(results[f"reward_{b}"]) for b in baseline_agents)
    results["reward"]      = np.mean([results[f"reward_{b}"][:n]     for b in baseline_agents], axis=0)
    results["moving_avg"]  = np.mean([results[f"moving_avg_{b}"][:n] for b in baseline_agents], axis=0)

    results = {k: v[:n] if isinstance(v, np.ndarray) else v for k, v in results.items()}

    return results, tom_agent



def train_pomcp_tom():
    all_wms = load_all_world_models()
    all_baseline_agents = setup_baseline_agents()
    spec_ensemble = load_specialized_ensembles()

    final_params = []  # [0]=uniform, [1]=update

    pbar0 = tqdm(['uniform', 'update'], leave=True)
    for u_rule in pbar0:
        _postfix0 = {
            'Rule' : u_rule
        }
        pbar0.set_postfix(_postfix0)
        best_avg = -float("inf")
        best_params = None
        best_per_game_results = {}
        best_planners = {}   

        best_params_path = os.path.join(_results_dir, f"{u_rule}_cheat_best_params.json")
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                data = json.load(f)
                best_params = data['best_params']
            cheat_sweep_done = True
        else:
            cheat_sweep_done = False


        pbar1 = tqdm(pomcp_param_grid, desc="HyperSearch", disable=cheat_sweep_done, leave=False)
        
        for params in pbar1:
            if cheat_sweep_done:
                break

            avg, results, planners = train_on_params(
                all_baseline_agents,
                all_wms,
                spec_ensemble,
                cheat_toggle=True,
                update_rule=u_rule,
                **params
            )

            if avg > best_avg:
                best_avg = avg
                best_params = params
                best_per_game_results = results
                best_planners = planners
                pbar1.write(f"  ↑ new best")

        if not cheat_sweep_done:
            for game_name, results in best_per_game_results.items():
                csv_path = os.path.join(_results_dir, f"{game_name}_{u_rule}_final_results_cheat.csv")
                pd.DataFrame(results).to_csv(csv_path, index=False)

                planner = best_planners[game_name]
                planner.save(os.path.join(_results_dir, f"G_{game_name}_{u_rule}_agent_cheat.pkl"))

            with open(best_params_path, "w") as f:
                json.dump({"best_params": best_params}, f, indent=4)
        
        final_params.append(best_params)

    # --- World-model evaluation with best params ---
    pbar2 = tqdm(['uniform', 'update'], desc="World-model eval", leave=True)
    for u_rule in pbar2:
        tqdm.write(f"\nRunning {u_rule}")
        csv_path = os.path.join(_results_dir, f"A_{u_rule}_final_results.csv")
        if os.path.exists(csv_path):
            continue

        if u_rule=='uniform':
            params = final_params[0]
        else:
            params = final_params[1]

        if params is None:
            raise RuntimeError("Something went wrong")

        avg, results, planners = train_on_params(
            all_baseline_agents,
            all_wms,
            spec_ensemble,
            cheat_toggle=False,
            update_rule=u_rule,
            **params
        )

        for game_name, game_results in results.items():
            csv_path = os.path.join(_results_dir, f"{game_name}_{u_rule}_final_results.csv")
            pd.DataFrame(game_results).to_csv(csv_path, index=False)

            planner = planners[game_name]
            planner.save(os.path.join(_results_dir, f"G_{game_name}_{u_rule}_agent.pkl"))


    return



def train_on_params(
        all_baseline_agents: dict[str, AgentList],
        all_wms: dict[str, tuple[ToM_WorldModel, dict]],
        all_ensembles: dict[str, np.ndarray],
        cheat_toggle: bool,
        update_rule: str = 'uniform',
        **params,
    ):
    per_game_eval = {}
    results  = {}
    planners = {}

    pbar = tqdm(GAMES, desc="ToM-POMCP", leave=False)
    for game_name in pbar:
        pbar.set_postfix({"Game": game_name})

        # Exclude Random from training partners
        baseline_agents = {k: v for k, v in all_baseline_agents[game_name].items()
                           if "random" not in k.lower()}
        world_model, wm_config = all_wms[game_name]
        special_ensemble = all_ensembles[game_name]
        wm_config['past_episodes_context'] = special_ensemble['IQL'].shape[0]
        env = ENVIRONMENTS[game_name]

        agent_params = {
            'num_cards':          env.num_cards,
            'num_actions':        env.num_actions,
            'world_model':        world_model,
            'world_model_config': wm_config,
            'device':             'cpu',
            'update_rule':        update_rule,
            **params,
        }
        game_results, final_planner = _evaluate_agents(
            env, agent_params, baseline_agents, game_name, special_ensemble, cheat_toggle
        )
        eval_reward = float(game_results['moving_avg'][-1])
        per_game_eval[game_name] = eval_reward
        results[game_name]  = game_results
        planners[game_name] = final_planner
    
    avg = float(np.mean(list(per_game_eval.values())))
    tqdm.write(f"Config {params}")
    per_game_str = "  ".join(f"{g}={r:.3f}" for g, r in per_game_eval.items())
    tqdm.write(f"  per-game: {per_game_str}")
    tqdm.write(f"  avg:      {avg:.3f}")
    return avg, results, planners

if __name__ == "__main__":
    train_pomcp_tom()