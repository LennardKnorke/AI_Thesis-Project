"""
Load C++ oSarsa-seq policies into OSarsa_Planner and save in the standard
baseline format used by train_test_baselines.py.

Pipeline:
  1. export_hanabi_dpomdp.py / hanabi_g.cpp  → .dpomdp / compiled generator
  2. run_hanabi_osarsa.py                    → Results/OSarsa/<game>/policy_s<seed>.pkl
  3. import_oSarsa.py (this)                → Results/OSarsa/G_<game>_shared_model.pkl

OSarsa_Planner pre-fills every reachable history with a legal-random fallback;
the C++ policy is merged on top, so the saved agent is always fully defined.
The best seed is selected by Python evaluation, not the solver's reported value.
"""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agents import OSarsa_Planner
from tiny_game import *
from runner.engine import test_on_all_start_states


def load_policy(filepath: str) -> dict[tuple, int]:
    """Load the policy dict from a pickle produced by run_hanabi_osarsa.py."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Policy file not found: {filepath}")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data.get("policy", {})


def load_best_value(filepath: str) -> float | None:
    """Return the best_value stored in a pickle, or None."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data.get("best_value")


def _evaluate_seed(env: Game, num_cards: int, num_actions: int,
                   pkl_path: Path, game_letter: str,
                   ) -> tuple[float, "OSarsa_Planner", dict]:
    """Build an OSarsa_Planner, merge the seed's C++ policy, and evaluate it."""
    planner = OSarsa_Planner(env, game_letter, num_cards, num_actions)
    cpp_policy = load_policy(str(pkl_path))
    planner.best_value = load_best_value(str(pkl_path))

    try:
        legal_map = {
            priv_h: legal_as
            for priv_h, legal_as, done, _, _ in PRIV_HISTORIES[game_letter]
            if not done
        }
        valid_cpp_policy = {}
        illegal_count = 0
        for priv_h, action in cpp_policy.items():
            if priv_h in legal_map:
                if action in legal_map[priv_h]:
                    valid_cpp_policy[priv_h] = action
                else:
                    illegal_count += 1

        planner.policy.update(valid_cpp_policy)

        if illegal_count > 0:
            print(f"    Filtered out {illegal_count} illegal C++ fallback actions.")

        norm_reward = test_on_all_start_states(env, planner, game_name=game_letter)

    except Exception as e:
        print(f"    Evaluation failed: {e}")
        norm_reward = None

    return norm_reward, planner, cpp_policy


def main():
    """
    For each game: evaluate every seed's policy, select the best by Python
    reward, save as G_<game>_shared_model.pkl, and write final_results.csv.
    """
    project_root = Path(__file__).parent.resolve()
    results_root = project_root / "Results" / "OSarsa"
    output_dir   = project_root / "Results" / "OSarsa"

    output_dir.mkdir(parents=True, exist_ok=True)
    results_cache: dict[str, list] = {}

    for game_letter in GAMES:
        game_dir  = results_root / game_letter
        pkl_files = sorted(
            game_dir.glob("policy_s*.pkl"),
            key=lambda p: int(p.stem.split("_s")[-1]),
        )
        seeds = [int(p.stem.split("_s")[-1]) for p in pkl_files]
        print(f"\n[{game_letter}] evaluating {len(seeds)} seeds: {seeds}")

        env         = get_game_Rework(GameNames(game_letter))
        num_cards   = env.num_cards
        num_actions = env.num_actions

        best_reward  = -float("inf")
        best_planner: OSarsa_Planner | None = None
        best_seed:   int | None = None
        best_cpp_n:  int        = 0
        seed_rewards: list[tuple[int, float]] = []

        for seed, pkl_path in zip(seeds, pkl_files):
            norm_reward, planner, cpp_policy = _evaluate_seed(
                env, num_cards, num_actions, pkl_path, game_letter,
            )
            if norm_reward is not None:
                seed_rewards.append((seed, norm_reward))
                print(f"  seed {seed}: norm_reward={norm_reward:.4f}  "
                      f"cpp_entries={len(cpp_policy)}  "
                      f"reported_best_value={planner.best_value}")
                if norm_reward > best_reward:
                    best_reward  = norm_reward
                    best_planner = planner
                    best_seed    = seed
                    best_cpp_n   = len(cpp_policy)
            else:
                print(f"  seed {seed}: norm_reward=FaultyPolicy  "
                      f"cpp_entries={len(cpp_policy)}  "
                      f"reported_best_value={planner.best_value}")

        if seed_rewards:
            rewards_arr = [r for _, r in seed_rewards]
            print(f"  seed reward distribution: "
                  f"min={min(rewards_arr):.4f}  "
                  f"max={max(rewards_arr):.4f}  "
                  f"mean={sum(rewards_arr)/len(rewards_arr):.4f}  "
                  f"n={len(rewards_arr)}")

        if best_planner:
            optimal = OPTIMAL_RETURNS.get(game_letter)
            print(f"  → selected seed {best_seed}: normalized reward={best_reward:.4f}  "
                  f"(optimal={optimal}, cpp_entries={best_cpp_n})")

            model_path = output_dir / f"G_{game_letter}_shared_model.pkl"
            best_planner.save(str(model_path))
            print(f"  saved → {model_path}")

            results_cache[f"reward_{game_letter}"] = [best_reward]

    df = pd.DataFrame(results_cache)
    csv_path = output_dir / "final_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved → {csv_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
