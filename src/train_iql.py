# train_iql.py

import numpy as np
import pandas as pd
from tqdm import tqdm

from tiny_game import *
from runner import *
from agents import *
from config import *

_name = "IQL"

# Setting Folders
_agent_sub_dir = _name.replace(" ", "_")
_searchresults_dir = os.path.join(HYPERSEARCH_RESULTS_DIR, _agent_sub_dir)
os.makedirs(_searchresults_dir, exist_ok=True)
_results_dir = os.path.join(RESULTS_DIR, _agent_sub_dir)
os.makedirs(_results_dir, exist_ok=True)


iql_grid = {
    "lr": [0.1, 0.05, 0.01],
    'gamma': [0.99],
    "batch_size": [32, 64],
    'updates_per_train' : [1, 3],
    'buffer_size': [128, 256],
    "epsilon_start": [1.0],
    'epsilon_min': [0.05, 0.1],
    "epsilon_decay": [0.99, 0.995, 0.999]
}
iql_grid = generate_param_grid(iql_grid)


def run_iql_training(
        params : dict, 
        save_path : str, 
        episodes : int,
        save_policy : bool = True
    )->tuple[pd.DataFrame, AgentList]:
    path = os.path.join(save_path, "final_results.csv")
    final_agents : dict[str, AgentList] = {}
    results : dict[str, np.ndarray] = {}

    pbar = tqdm(GAMES, desc="Iter Games", leave=False)
    for g in pbar:
        _postfix = {
            "G" : g
        }
        pbar.set_postfix(_postfix)

        best_results = None
        best_agents = None

        pbar2 = tqdm(range(NUM_RUNS), desc="Runs", leave=False)
        for n in pbar2:
            _postfix2 = {
                "Run" : n
            }

            pbar2.set_postfix(_postfix2)
            env : Game = ENVIRONMENTS[g]

            agent_0 = IQ_Learning_Agent(
                env,
                g,
                env.num_cards,
                env.num_actions,
                **params
            )
            agent_1 = IQ_Learning_Agent(
                env,
                g,
                env.num_cards,
                env.num_actions,
                **params
            )
            _agents : AgentList = AgentList([agent_0, agent_1])

            reward, loss, _agents = run_model_free_training(
                env,
                g,
                _agents,
                train_episodes = episodes
            )
            # Save all results
            results[f"{g}_{n}_reward"] = reward
            results[f"{g}_{n}_loss"] = loss

            # Remember best agents
            if best_results is None or np.mean(reward[-100:]) > best_results:
                best_agents = _agents
                best_results = np.mean(reward[-100:])

        # Remember best agents
        final_agents[g] = best_agents

    # Final Save
    if save_policy:
        for key, _agents in final_agents.items():
            _path0 = os.path.join(save_path, f"{key}_agent0.pkl")
            _agents[0].save(_path0)

            _path1 = os.path.join(save_path, f"{key}_agent1.pkl")
            _agents[1].save(_path1)

    
    results_df = pd.DataFrame(results)
    
    results_df.to_csv(path, index=False)
    return results_df, final_agents



def hyper_search_iql():
    best_params = None
    best_results = None
    best_param_path = os.path.join(_results_dir, "best_params.json")

    pbar = tqdm(enumerate(iql_grid), desc="Hypersearch_IQL", leave=False)
    for i, param_set in pbar:
        # Param Folder Dir
        path = os.path.join(_searchresults_dir, f"{i}")
        os.makedirs(path, exist_ok=True)
        if os.path.exists(os.path.join(path, "final_results.csv")):
            continue

        results, _ = run_iql_training(param_set, path, TRAINING_EPISODES_HYPERSEARCH, False)

        # Cache best params
        is_better_result : bool = compare_results(results, best_results)
        if is_better_result:
            best_params = param_set
            best_results = results
            with open(best_param_path, 'w') as f:
                json.dump(best_params, f, indent=4)
    return

def train_iql():
    tqdm.write("IQL Training Pipeline")
    hyper_search_iql()

    best_params = load_best_params(os.path.join(_results_dir, "best_params.json"))
    run_iql_training(
        params=best_params, 
        save_path=_results_dir, 
        episodes=TRAINING_EPISODES_FINAL, 
        save_policy=True
    )
    return

def compare_results(new_results : pd.DataFrame, prev_best_results : pd.DataFrame|None)->bool:
    """
    Returns True if new resuls are better
    """

    if prev_best_results is None:
        return True
    
    new_score = compute_avg_final_reward(new_results)
    prev_score = compute_avg_final_reward(prev_best_results)

    is_better = bool(new_score > prev_score)
    if is_better:
        tqdm.write(f"New Best avg: {new_score}")
    return is_better
    

def compute_avg_final_reward(df: pd.DataFrame) -> float:
    game_scores = []
    for game in GAMES:
        reward_cols = [col for col in df.columns if col.startswith(f"{game}_") and col.endswith("_reward")]
        if not reward_cols:
            continue
        run_means = []
        for col in reward_cols:
            rewards = df[col].dropna().values  
            last_100 = rewards[-100:] if len(rewards) >= 100 else rewards
            run_means.append(np.mean(last_100))

        game_scores.append(np.mean(run_means))

    return np.mean(game_scores) if game_scores else -np.inf

if __name__ == "__main__":
    train_iql()