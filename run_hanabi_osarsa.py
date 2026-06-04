"""
Run the C++ oSarsa-seq solver on Hanabi games (A–F and G), translate each
resulting policy to Python-native format, and save per-game pickles.

Usage:
    python run_hanabi_osarsa.py [--games A B C G] [--seeds 0 1 2] [--timeout 300] [--force]

Outputs:
    Results/OSarsa/<game>/final_results.csv   – per-seed reward summary
    Results/OSarsa/<game>/policy_s<seed>.pkl  – Python-native policy pickle

Binary: osarsa-aaai-25/code/build/sdms.exe  (cwd set to osarsa-aaai-25/code/).
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tiny_game import MyHanabi, get_game_Rework, GameNames, get_all_possible_histories

CPP_ROOT   = PROJECT_ROOT / "osarsa-aaai-25" / "code"
BINARY     = CPP_ROOT / "build" / "sdms.exe"
MINGW_BIN  = r"C:\MyPrograms\mingw64\bin"

RESULTS_ROOT = PROJECT_ROOT / "Results" / "OSarsa"

# G: truncation=5 works in practice (crash occurs post-output during cleanup).
GAME_CONFIGS = {
    "G": dict(bench="Hanabi_G", horizon=13, truncation=5, game_type="myhanabi"),
    "A": dict(bench="Hanabi_A", horizon=9,  truncation=3, game_type="tiny"),
    "B": dict(bench="Hanabi_B", horizon=9,  truncation=3, game_type="tiny"),
    "C": dict(bench="Hanabi_C", horizon=9,  truncation=3, game_type="tiny"),
    "D": dict(bench="Hanabi_D", horizon=9,  truncation=3, game_type="tiny"),
    "E": dict(bench="Hanabi_E", horizon=9,  truncation=3, game_type="tiny"),
    "F": dict(bench="Hanabi_F", horizon=9,  truncation=3, game_type="tiny"),
}

if sys.platform == "win32" and os.path.isdir(MINGW_BIN):
    os.environ["PATH"] = MINGW_BIN + os.pathsep + os.environ.get("PATH", "")


def run_cpp_solver(
        game_letter: str, seed: int, timeout: float,
        log_csv: Path,
        epsilon: float | None = None,
        iter_max: int | None = None,
    ) -> tuple[float | None, Path | None]:
    """Run the C++ oSarsa-seq binary for one game/seed. Returns (best_value, policy_csv_path)."""
    cfg = GAME_CONFIGS[game_letter]

    cmd = [
        str(BINARY),
        "-f", cfg["bench"],
        "-N", "2",
        "-n", "oSarsa-seq",
        "-p", str(cfg["horizon"]),
        "-m", str(cfg["truncation"]),
        "-t", str(int(timeout)),
        "-s", str(seed),
        "-v", "0",
        "-l", str(log_csv),
    ]
    if epsilon is not None:
        cmd += ["-e", str(epsilon)]
    if iter_max is not None:
        cmd += ["-i", str(int(iter_max))]

    print(f"  [{game_letter} s={seed}] {' '.join(cmd)}")
    subprocess.run(cmd, capture_output=False, text=True, cwd=str(CPP_ROOT))

    best_value = _parse_best_value(log_csv)
    policy_csv = Path(str(log_csv).replace(".csv", "_policy.csv"))
    if not policy_csv.exists():
        print(f"  WARNING: policy CSV not found at {policy_csv}")
        policy_csv = None

    return best_value, policy_csv


def _parse_best_value(log_csv: Path) -> float | None:
    """Return the final 'best' value from an oSarsa log CSV."""
    if not log_csv.exists():
        return None
    try:
        with open(log_csv) as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
        if last and "best" in last:
            return float(last["best"])
    except Exception as e:
        print(f"  WARNING: could not parse best value from {log_csv}: {e}")
    return None


def _decode_obs_tiny(obs_seq: list[int], agent: int,
                     n_cards: int, n_real: int) -> tuple:
    """
    Translate a C++ obs sequence to a Python TinyHanabi private history.

    agent 0: obs_seq[-1] = c1                              → (-1, c1)
    agent 1: obs_seq[-2] = c0, [-1] = n_cards+c0*n_real+a0 → (c0, -1, a0)
    """
    if agent == 0:
        c1 = obs_seq[-1]
        return (-1, c1)
    else:
        c0       = obs_seq[-2]
        z_phase2 = obs_seq[-1]
        a0       = (z_phase2 - n_cards) % n_real
        return (c0, -1, a0)


def _read_policy_csv(
    policy_csv: Path,
) -> tuple[list[tuple[int, int, tuple, int]], dict[tuple[int, int], int]]:
    """
    Read C++ policy CSV. Returns (rows, defaults).

    rows     – list of (step, agent, obs_seq, action)
    defaults – (step, agent) → fallback action from DEFAULT sentinel rows
    """
    rows: list[tuple[int, int, tuple, int]] = []
    defaults: dict[tuple[int, int], int] = {}
    with open(policy_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("step"):
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            step    = int(parts[0])
            agent   = int(parts[1])
            obs_raw = parts[2].strip()
            action  = int(parts[3])
            if obs_raw == "DEFAULT":
                defaults[(step, agent)] = action
                continue
            obs_seq = tuple(int(x) for x in obs_raw.split()) if obs_raw else ()
            rows.append((step, agent, obs_seq, action))
    return rows, defaults


def _encode_full_obs_seq_myhanabi(priv_h: tuple, agent: int) -> tuple[int, ...]:
    """
    Build the full C++ observation sequence an agent would have received for
    this private history, mirroring hanabi_g.cpp's compressed 5-bit encoding.

        z0                    = partner_c0 * 4 + partner_c1        (0..15)
        z_k (play action)     = 16 + action * 4 + card             (16..23, a ∈ {0,1})
        z_k (declare action)  = 24 + (action - HAND_SIZE)          (24..25, a ∈ {2,3})
    """
    HAND_SIZE = 2
    if agent == 0:
        p_c0, p_c1 = priv_h[2], priv_h[3]
    else:
        p_c0, p_c1 = priv_h[0], priv_h[1]
    z0 = p_c0 * 4 + p_c1

    z_rest = []
    for a, c in priv_h[4:]:
        if a < HAND_SIZE:
            z_rest.append(16 + a * 4 + c)
        else:
            z_rest.append(24 + (a - HAND_SIZE))
    return (z0, *z_rest)


def _build_myhanabi_policy(rows, defaults, env, truncation: int) -> dict[tuple, int]:
    """
    Enumerate reachable Python private histories, compute each one's truncated
    C++ obs-seq, and look up the action: exact match → default → skip.
    Illegal actions are dropped; legal-random fallbacks fill the gaps.
    """
    cpp_entries: dict[tuple, int] = {}
    for step, agent, obs_seq, action in rows:
        cpp_entries[(step, agent, obs_seq)] = action

    priv_obs, _ = get_all_possible_histories(env)

    policy: dict[tuple, int] = {}
    matched_exact   = 0
    matched_default = 0
    illegal_dropped = 0
    no_default      = 0

    for priv_h, _, done, turn_id, _ in priv_obs:
        if done:
            continue
        agent = int(turn_id)
        step = (len(priv_h) - 4) + 1
        if step < 1:
            continue

        full_obs  = _encode_full_obs_seq_myhanabi(priv_h, agent)
        truncated = full_obs[-truncation:] if truncation > 0 else full_obs
        key       = (step, agent, tuple(truncated))

        if key in cpp_entries:
            action = cpp_entries[key]
            source = "exact"
        elif (step, agent) in defaults:
            action = defaults[(step, agent)]
            source = "default"
        else:
            no_default += 1
            continue

        _, legal = env.num_legal_actions(priv_h)
        if action in legal:
            policy[priv_h] = action
            if source == "exact":
                matched_exact += 1
            else:
                matched_default += 1
        else:
            illegal_dropped += 1

    print(f"    matched exact={matched_exact}  default={matched_default}  "
          f"illegal_dropped={illegal_dropped}  no_default={no_default}")
    return policy


def parse_policy_csv(policy_csv: Path, env) -> dict[tuple, int]:
    """Parse the C++ policy CSV and return a Python-native policy dict."""
    rows, defaults = _read_policy_csv(policy_csv)

    if isinstance(env, MyHanabi):
        truncation = GAME_CONFIGS["G"]["truncation"]
        return _build_myhanabi_policy(rows, defaults, env, truncation)

    n_cards = env.num_cards
    n_real  = env.num_actions

    policy: dict[tuple, int] = {}
    for step, agent, obs_seq, action in rows:
        # 3-phase cycle: phase-1 (step%3==1) → agent 0; phase-2 → agent 1
        if not ((step % 3 == 1 and agent == 0) or
                (step % 3 == 2 and agent == 1)):
            continue
        if not obs_seq:
            continue
        priv_h = _decode_obs_tiny(list(obs_seq), agent, n_cards, n_real)
        policy[priv_h] = action

    return policy


def run_game(game_letter: str, seeds: list[int], timeout: float,
             epsilon: float | None = None,
             iter_max: int | None = None):
    """Run oSarsa-seq for all seeds of one game and save results + policy pickles."""
    cfg  = GAME_CONFIGS[game_letter]
    gdir = RESULTS_ROOT / game_letter
    gdir.mkdir(parents=True, exist_ok=True)

    if cfg["game_type"] == "myhanabi":
        from tiny_game import MyHanabi as MH
        env = MH()
    else:
        env = get_game_Rework(GameNames(game_letter))

    rows = []
    for seed in seeds:
        log_csv = gdir / f"log_s{seed}.csv"
        best_value, policy_csv = run_cpp_solver(
            game_letter, seed, timeout, log_csv,
            epsilon=epsilon, iter_max=iter_max,
        )

        policy: dict[tuple, int] = {}
        if policy_csv is not None:
            try:
                policy = parse_policy_csv(policy_csv, env)
                print(f"  [{game_letter} s={seed}] policy entries: {len(policy)}")
            except Exception as e:
                print(f"  WARNING: policy parsing failed: {e}")

        pkl_path = gdir / f"policy_s{seed}.pkl"
        _save_policy_pickle(pkl_path, policy, best_value, game_letter, seed)
        print(f"  [{game_letter} s={seed}] best_value={best_value}  -> {pkl_path}")

        rows.append({
            "game":       game_letter,
            "seed":       seed,
            "best_value": best_value if best_value is not None else "",
            "policy_pkl": str(pkl_path),
        })

    results_csv = gdir / "final_results.csv"
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "seed", "best_value", "policy_pkl"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [{game_letter}] results -> {results_csv}")

    return rows


def _save_policy_pickle(path: Path, policy: dict, best_value: float | None,
                        game_letter: str, seed: int):
    data = {
        "policy":      policy,
        "best_value":  best_value,
        "game_letter": game_letter,
        "seed":        seed,
    }
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games",     nargs="+", default=list("G"),
                        help="Games to run (default: A B C D E F G)")
    parser.add_argument("--seeds",     nargs="+", type=int, default=None,
                        help="Explicit seed list, e.g. --seeds 0 1 2. "
                             "If omitted, uses range(num_seeds).")
    parser.add_argument("--num-seeds", type=int, default=100,
                        help="Shortcut: run seeds 0..num_seeds-1 (default: 3). "
                             "Ignored if --seeds is given.")
    parser.add_argument("--timeout",   type=float, default=90.0,
                        help="Per-run timeout in seconds (default: 300)")
    parser.add_argument("--epsilon",   type=float, default=None,
                        help="Solver exploration rate (-e). Default: solver's built-in.")
    parser.add_argument("--iter-max",  type=int, default=None,
                        help="Solver hard iteration cap (-i). Default: solver's built-in.")
    parser.add_argument("--force",     action="store_true",
                        help="Re-run even if policy pickles already exist")
    args = parser.parse_args()

    if args.seeds is None:
        args.seeds = list(range(args.num_seeds))

    games   = [g.upper() for g in args.games]
    unknown = [g for g in games if g not in GAME_CONFIGS]
    if unknown:
        parser.error(f"Unknown games: {unknown}.  Valid: {list(GAME_CONFIGS)}")

    game_seeds: dict[str, list[int]] = {}
    for g in games:
        gdir    = RESULTS_ROOT / g
        pending = [s for s in args.seeds
                   if args.force or not (gdir / f"policy_s{s}.pkl").exists()]
        if not pending:
            print(f"[{g}] all seed pickles found — skipping (use --force to re-run)")
        else:
            skipped = len(args.seeds) - len(pending)
            if skipped:
                print(f"[{g}] {skipped} seeds already done, running {len(pending)} remaining")
            game_seeds[g] = pending

    if not game_seeds:
        print("All requested games already solved. Done.")
        return

    all_rows = []
    for g, seeds in game_seeds.items():
        print(f"\n=== Game {g} ===")
        rows = run_game(g, seeds, args.timeout,
                        epsilon=args.epsilon, iter_max=args.iter_max)
        all_rows.extend(rows)

    summary = RESULTS_ROOT / "final_results.csv"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game", "seed", "best_value", "policy_pkl"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nAll results -> {summary}")


if __name__ == "__main__":
    main()
