
import seaborn as sns

from runner import *
from config import *
from tiny_game import *
from results_wm import _finalize_plot


def read_bagents_results() -> dict[str, pd.DataFrame]:
    def _read(name: str) -> pd.DataFrame:
        path = os.path.join(RESULTS_DIR, name, "final_results.csv")
        if not os.path.exists(path):
            raise ValueError("")
        return pd.read_csv(path)

    pbdp = _read("PBDP")
    if not pbdp.empty and pbdp.columns[0] == "Unnamed: 0":
        pbdp = pbdp.drop(columns=pbdp.columns[0])

    return {
        "IQL":    _read("IQL"),
        "VDN":    _read("VDN"),
        "PBDP":   pbdp,
        "OSarsa": _read("OSarsa"),
    }


def read_tom_results(cheat_results: bool, u_rule: str) -> dict[str, pd.DataFrame]:
    """Load per-game ToM result CSVs. Key columns: reward, reward_{base}, p0/p1_reward_{base}."""
    tom_dir = os.path.join(RESULTS_DIR, "ToM-POMCP")
    results = {}
    for game in GAMES:
        path = os.path.join(tom_dir, f"{game}_{u_rule}_final_results")
        if cheat_results:
            path += "_cheat.csv"
        else:
            path += ".csv"
        df = pd.read_csv(path)
        results[game] = df
    return results


def baselines_crossplay(bagents: dict[str, dict[str, AgentList]]) -> pd.DataFrame:
    """Every baseline pair plays from both perspectives across all start states."""
    results = {}
    for game_name in GAMES:
        env = ENVIRONMENTS[game_name]
        baseline_agents = bagents[game_name]

        game_results = {}

        for type0, p0_agents in baseline_agents.items():
            for type1, p1_agents in baseline_agents.items():
                key = f"{type0}_{type1}"
                if type0 == type1:
                    continue

                agent_list = AgentList([p0_agents[0], p1_agents[1]])
                r = test_on_all_start_states(env, agent_list, game_name)
                game_results[key] = r
        results[game_name] = pd.DataFrame([game_results])
    return results


def _tom_avg(df: pd.DataFrame, row_idx: int) -> float:
    """Average reward over all available baseline partners at the given row index."""
    cols = [f"reward_{b}" for b in BASELINES if f"reward_{b}" in df.columns]
    return float(df[cols].iloc[row_idx].mean()) if cols else np.nan


def _tom_partner(df: pd.DataFrame, row_idx: int, partner: str) -> float:
    """Reward against one specific partner at the given row index."""
    col = f"reward_{partner}"
    return float(df[col].iloc[row_idx]) if col in df.columns else np.nan


def _tom_partner_std(df: pd.DataFrame, partner: str, row_idx: int = -1) -> float:
    """Std across evaluation runs of the (P0+P1)/2 reward for one partner at the given row index."""
    vals = []
    for role in ("p0", "p1"):
        col = f"{role}_reward_{partner}_std"
        if col in df.columns:
            vals.append(float(df[col].iloc[row_idx]))
    return float(np.mean(vals)) if vals else np.nan


def _tom_avg_std(df: pd.DataFrame, row_idx: int = -1) -> float:
    """Mean of per-partner stds across all available baselines."""
    stds = [_tom_partner_std(df, b, row_idx) for b in BASELINES
            if f"p0_reward_{b}_std" in df.columns]
    return float(np.mean(stds)) if stds else np.nan


def _fmt(val: float, std: float) -> str:
    if np.isnan(val):
        return ""
    if np.isnan(std):
        return f"{val:.2f}"
    return f"{val:.2f}±{std:.2f}"


def _render_perf_heatmap(df_heat: pd.DataFrame, filename: str, annot_df: pd.DataFrame | None = None) -> None:
    _, ax = plt.subplots(figsize=(16, 5))
    annot = annot_df if annot_df is not None else True
    fmt   = "" if annot_df is not None else ".2f"
    sns.heatmap(
        df_heat.astype(float),
        annot=annot, fmt=fmt,
        vmin=0, vmax=1, cmap="RdYlGn", linewidths=0.5,
        annot_kws={"size": 11}, ax=ax,
    )
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Scenario")
    _finalize_plot(filename)


def plot_reward_heatmap(
        tom_results: dict[str, pd.DataFrame],
        cheat_results: dict[str, pd.DataFrame],
        bagents_results: dict[str, pd.DataFrame],
        u_rule: str,
    ) -> None:
    """
    Performance heatmap: scenarios (A–G) on Y-axis, algorithms on X-axis.
    ToM (base) = first episode (no prior adaptation); ToM (full) = last episode (after full adaptation).
    Generates one averaged heatmap plus one per ToM partner (IQL/VDN/PBDP/OSarsa).
    """
    iql_df    = bagents_results["IQL"]
    vdn_df    = bagents_results["VDN"]
    pbdp_df   = bagents_results["PBDP"]
    osarsa_df = bagents_results["OSarsa"]
    columns   = ["IQL", "VDN", "PBDP", "OSarsa", "ToM (Oracle)", "ToM (base)", "ToM (full)"]

    def _baseline_row(game: str) -> dict:
        row = {}
        for name, df in [("IQL", iql_df), ("VDN", vdn_df)]:
            reward_cols = [f"{game}_{r}_reward" for r in range(NUM_RUNS)
                           if f"{game}_{r}_reward" in df.columns]
            row[name] = float(np.mean([df[c].dropna().iloc[-1] for c in reward_cols])) if reward_cols else np.nan
        pbdp_col   = f"{game}_reward"
        osarsa_col = f"reward_{game}"
        row["PBDP"]   = float(pbdp_df[pbdp_col].iloc[0])     if pbdp_col   in pbdp_df.columns   else np.nan
        row["OSarsa"] = float(osarsa_df[osarsa_col].iloc[0]) if osarsa_col in osarsa_df.columns else np.nan
        return row

    def _build_data(partner: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        tom_fn     = (lambda df, i: _tom_avg(df, i))         if partner is None \
                     else (lambda df, i: _tom_partner(df, i, partner))
        tom_std_fn = (lambda df, i: _tom_avg_std(df, i))     if partner is None \
                     else (lambda df, i: _tom_partner_std(df, partner, i))
        data, annot = {}, {}
        for game in GAMES:
            row = _baseline_row(game)
            annot_row = {k: f"{v:.2f}" if not np.isnan(v) else "" for k, v in row.items()}

            oracle_val = tom_fn(cheat_results[game], -1)     if game in cheat_results else np.nan
            oracle_std = tom_std_fn(cheat_results[game], -1) if game in cheat_results else np.nan
            row["ToM (Oracle)"]       = oracle_val
            annot_row["ToM (Oracle)"] = _fmt(oracle_val, oracle_std)

            if game in tom_results:
                # row 0 = first episode with no prior adaptation; row -1 = after full adaptation sequence
                base_val = tom_fn(tom_results[game],  0);  base_std = tom_std_fn(tom_results[game],  0)
                full_val = tom_fn(tom_results[game], -1);  full_std = tom_std_fn(tom_results[game], -1)
                row["ToM (base)"]       = base_val;  annot_row["ToM (base)"] = _fmt(base_val, base_std)
                row["ToM (full)"]       = full_val;  annot_row["ToM (full)"] = _fmt(full_val, full_std)
            else:
                row["ToM (base)"] = row["ToM (full)"] = np.nan
                annot_row["ToM (base)"] = annot_row["ToM (full)"] = ""

            data[game]  = row
            annot[game] = annot_row

        df_values = pd.DataFrame(data,  index=columns).T.reindex(GAMES)
        df_annot  = pd.DataFrame(annot, index=columns).T.reindex(GAMES)
        return df_values, df_annot

    # Averaged over all partners
    df, annot = _build_data()
    _render_perf_heatmap(df, f"performance_heatmap_{u_rule}.png", annot)

    # One heatmap per ToM partner
    for partner in BASELINES:
        df, annot = _build_data(partner)
        _render_perf_heatmap(df, f"performance_heatmap_{u_rule}_vs_{partner}.png", annot)


def _model_free_last(name: str, df: pd.DataFrame, games: list[str]) -> float:
    """Last-episode reward for a baseline, averaged across the given games and runs."""
    if name in ("IQL", "VDN"):
        vals = []
        for game in games:
            cols = [
                f"{game}_{r}_reward"
                for r in range(NUM_RUNS)
                if f"{game}_{r}_reward" in df.columns
            ]
            if cols:
                vals.append(np.mean([df[c].dropna().iloc[-1] for c in cols]))
        return float(np.mean(vals)) if vals else np.nan
    if name == "PBDP":
        vals = [df[f"{g}_reward"].iloc[0] for g in games if f"{g}_reward" in df.columns]
        return float(np.mean(vals)) if vals else np.nan
    if name == "OSarsa":
        vals = [df[f"reward_{g}"].iloc[0] for g in games if f"reward_{g}" in df.columns]
        return float(np.mean(vals)) if vals else np.nan
    return np.nan


def _crossplay_score(crossplay_results: dict[str, pd.DataFrame], p0: str, p1: str) -> float:
    """Average reward for p0 vs p1 across all games, reading column '{p0}_{p1}'."""
    key = f"{p0}_{p1}"
    vals = [
        float(df[key].iloc[0])
        for df in crossplay_results.values()
        if key in df.columns
    ]
    return float(np.mean(vals)) if vals else np.nan


def _tom_crossplay_std(dfs: dict[str, pd.DataFrame], role: str, partner: str) -> float:
    """Mean per-game std across evaluation runs for ToM as role ('p0'/'p1') against partner."""
    col = f"{role}_reward_{partner}_std"
    stds = [float(df[col].iloc[-1]) for df in dfs.values() if col in df.columns]
    return float(np.mean(stds)) if stds else np.nan


def _build_crossplay_matrix(
        tom_results: dict[str, pd.DataFrame],
        cheat_results: dict[str, pd.DataFrame],
        bagents_results: dict[str, pd.DataFrame],
        crossplay_results: dict[str, pd.DataFrame],
        games: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the crossplay matrix averaged over `games`. Returns (values, annotations)."""
    agents = BASELINES + ["ToM (Oracle)", "ToM (full)"]
    matrix = pd.DataFrame(np.nan,  index=agents, columns=agents)
    annot  = pd.DataFrame("",      index=agents, columns=agents)

    # --- Diagonal: self-play ---
    for name in BASELINES:
        val = _model_free_last(name, bagents_results[name], games)
        matrix.loc[name, name] = val
        annot.loc[name, name]  = f"{val:.2f}" if not np.isnan(val) else ""

    # --- Baseline vs Baseline crossplay ---
    filtered_crossplay = {g: df for g, df in crossplay_results.items() if g in games}
    for p0 in BASELINES:
        for p1 in BASELINES:
            if p0 == p1:
                continue
            val = _crossplay_score(filtered_crossplay, p0, p1)
            if not np.isnan(val):
                matrix.loc[p0, p1] = val
                annot.loc[p0, p1]  = f"{val:.2f}"

    # --- ToM (full) vs baselines: last episode after full adaptation ---
    filtered_tom = {g: df for g, df in tom_results.items() if g in games}
    for base in BASELINES:
        p0_vals, p1_vals = [], []
        for df in filtered_tom.values():
            if f"p0_reward_{base}" in df.columns:
                p0_vals.append(float(df[f"p0_reward_{base}"].iloc[-1]))
            if f"p1_reward_{base}" in df.columns:
                p1_vals.append(float(df[f"p1_reward_{base}"].iloc[-1]))
        if p0_vals:
            val = float(np.mean(p0_vals))
            std = _tom_crossplay_std(filtered_tom, "p0", base)
            matrix.loc["ToM (full)", base] = val
            annot.loc["ToM (full)", base]  = _fmt(val, std)
        if p1_vals:
            val = float(np.mean(p1_vals))
            std = _tom_crossplay_std(filtered_tom, "p1", base)
            matrix.loc[base, "ToM (full)"] = val
            annot.loc[base, "ToM (full)"]  = _fmt(val, std)

    # --- ToM (Oracle) vs baselines: single episode (no adaptation sequence) ---
    filtered_cheat = {g: df for g, df in cheat_results.items() if g in games}
    for base in BASELINES:
        p0_vals, p1_vals = [], []
        for df in filtered_cheat.values():
            if f"p0_reward_{base}" in df.columns:
                p0_vals.append(float(df[f"p0_reward_{base}"].iloc[-1]))
            if f"p1_reward_{base}" in df.columns:
                p1_vals.append(float(df[f"p1_reward_{base}"].iloc[-1]))
        if p0_vals:
            val = float(np.mean(p0_vals))
            std = _tom_crossplay_std(filtered_cheat, "p0", base)
            matrix.loc["ToM (Oracle)", base] = val
            annot.loc["ToM (Oracle)", base]  = _fmt(val, std)
        if p1_vals:
            val = float(np.mean(p1_vals))
            std = _tom_crossplay_std(filtered_cheat, "p1", base)
            matrix.loc[base, "ToM (Oracle)"] = val
            annot.loc[base, "ToM (Oracle)"]  = _fmt(val, std)

    return matrix, annot


def _render_crossplay(matrix: pd.DataFrame, filename: str, annot_df: pd.DataFrame | None = None) -> None:
    _, ax = plt.subplots(figsize=(12, 7))
    annot = annot_df if annot_df is not None else True
    fmt   = "" if annot_df is not None else ".2f"
    sns.heatmap(
        matrix.astype(float),
        annot=annot, fmt=fmt,
        vmin=0, vmax=1, cmap="RdYlGn", linewidths=0.5,
        mask=matrix.isna(), annot_kws={"size": 11}, ax=ax,
    )
    ax.set_xlabel("Player 1")
    ax.set_ylabel("Player 0")
    _finalize_plot(filename)


def plot_crossplay(
        tom_results: dict[str, pd.DataFrame],
        cheat_results: dict[str, pd.DataFrame],
        bagents_results: dict[str, pd.DataFrame],
        crossplay_results: dict[str, pd.DataFrame],
        u_rule: str,
) -> None:
    shared = dict(
        tom_results=tom_results,
        cheat_results=cheat_results,
        bagents_results=bagents_results,
        crossplay_results=crossplay_results,
    )

    # Heatmap averaged across all games
    matrix_all, annot_all = _build_crossplay_matrix(**shared, games=GAMES)
    _render_crossplay(matrix_all, f"crossplay_heatmap_{u_rule}.png", annot_all)

    # One heatmap per individual game
    for game in GAMES:
        matrix, annot = _build_crossplay_matrix(**shared, games=[game])
        _render_crossplay(matrix, f"crossplay_heatmap_{game}_{u_rule}.png", annot)


def run_results_tom(b_agents):
    cross_play_results = baselines_crossplay(b_agents)
    baselines_results  = read_bagents_results()
    for u_rule in ['update']:
        cheat_results = read_tom_results(True,  u_rule)
        tom_results   = read_tom_results(False, u_rule)
        plot_reward_heatmap(tom_results, cheat_results, baselines_results, u_rule)
        plot_crossplay(tom_results, cheat_results, baselines_results, cross_play_results, u_rule)
