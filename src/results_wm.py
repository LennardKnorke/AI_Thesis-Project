

from tiny_game import GAMES
from config import *

from train_worldmodel import setup_dataset


def report_wm_dataset_size(b_agents) -> None:
    full_dataset = setup_dataset(b_agents)
    print(f"\n{'Game':<6} {'Total':>8} {'Train':>8} {'Val':>8}")
    print("-" * 32)
    for game_name, (train_df, val_df) in full_dataset.items():
        train_size = len(train_df["data"])
        val_size   = len(val_df["data"])
        print(f"{game_name:<6} {train_size + val_size:>8} {train_size:>8} {val_size:>8}")


def _stack_per_game(curves: dict[str, np.ndarray]) -> np.ndarray:
    """Stack per-game curves to a common length (min across games)."""
    min_len = min(len(c) for c in curves.values())
    return np.stack([c[:min_len] for c in curves.values()], axis=0)

def analyze_wm(df: pd.DataFrame) -> None:
    """Train/val action and type accuracy per epoch, averaged across games."""
    valid_games = [g for g in GAMES if f"train_act_acc_{g}" in df.columns]
    if not valid_games:
        print("No world-model training columns found.")
        return

    train_act  = {g: df[f"train_act_acc_{g}"].dropna().values  for g in valid_games}
    val_act    = {g: df[f"val_act_acc_{g}"].dropna().values    for g in valid_games}
    train_type = {g: df[f"train_type_acc_{g}"].dropna().values for g in valid_games}
    val_type   = {g: df[f"val_type_acc_{g}"].dropna().values   for g in valid_games}

    avg_train_act  = _stack_per_game(train_act).mean(axis=0)
    avg_val_act    = _stack_per_game(val_act).mean(axis=0)
    avg_train_type = _stack_per_game(train_type).mean(axis=0)
    avg_val_type   = _stack_per_game(val_type).mean(axis=0)
    epochs_act     = np.arange(1, len(avg_train_act) + 1)
    epochs_type    = np.arange(1, len(avg_train_type) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_act, avg_train_act, label="Train",      color="#1f77b4")
    ax.plot(epochs_act, avg_val_act,   label="Validation", color="#ff7f0e", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Action Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize_plot("wm_action_accuracy_avg.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_type, avg_train_type, label="Train",      color="#1f77b4")
    ax.plot(epochs_type, avg_val_type,   label="Validation", color="#ff7f0e", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Type Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _finalize_plot("wm_type_accuracy_avg.png")


def plot_wm_context_accuracy() -> None:
    params_path = os.path.join(WORLD_MODELS_DIR, "best_params.json")
    if not os.path.exists(params_path):
        print("No best_params.json found for world model.")
        return
    params = load_best_params(params_path)

    act_per_game:  dict[str, tuple[list, list]] = {}
    type_per_game: dict[str, tuple[list, list]] = {}

    for game in GAMES:
        val_path = os.path.join(WORLD_MODELS_DIR, f"val_{game}.pt")
        wm_path  = os.path.join(WORLD_MODELS_DIR, f"WM_{game}.pth")
        if not os.path.exists(val_path) or not os.path.exists(wm_path):
            continue

        val_data = torch.load(val_path, map_location=DEVICE, weights_only=False)
        state_dict = torch.load(wm_path, map_location=DEVICE, weights_only=True)
        actual_num_agent_types = state_dict['char_net.identity_classifier.weight'].shape[0]

        model = ToM_WorldModel(
            joint_obs_dim    = val_data["joint_obs_dim"],
            obs_dim          = val_data["obs_dim"],
            action_dim       = val_data["act_dim"],
            num_agent_types  = actual_num_agent_types,
            max_seq_len      = val_data["max_seq_length"],
            char_embed_dim   = params["char_dim"],
            use_obs          = True,
            mental_embed_dim = params["mental_dim"],
            trunk_dim        = params["trunk_dim"],
        ).to(DEVICE)
        model.load_state_dict(torch.load(wm_path, map_location=DEVICE, weights_only=True))
        model.eval()

        with torch.no_grad():
            act_logits, _, id_logits, _ = model(
                val_data["past"], val_data["past_mask"],
                val_data["history"], val_data["obs"],
            )
        act_correct  = (act_logits.argmax(dim=1) == val_data["tgt_act"])
        type_correct = (id_logits.argmax(dim=1)  == val_data["tgt_type"])
        k_values     = val_data["past_mask"].sum(dim=1).long()

        act_per_k, type_per_k = {}, {}
        for k in range(int(k_values.max().item()) + 1):
            sel = (k_values == k)
            if sel.sum() == 0:
                continue
            act_per_k[k]  = act_correct[sel].float().mean().item()
            type_per_k[k] = type_correct[sel].float().mean().item()

        ks = sorted(act_per_k)
        act_per_game[game]  = (ks, [act_per_k[k]  for k in ks])
        type_per_game[game] = (ks, [type_per_k[k] for k in ks])

    if not act_per_game:
        print("No validation data found for context accuracy plot.")
        return

    all_ks = sorted({k for ks, _ in act_per_game.values() for k in ks})

    def _avg(per_game: dict) -> np.ndarray:
        return np.mean(
            [np.interp(all_ks, ks, accs) for ks, accs in per_game.values()], axis=0
        )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_ks, _avg(act_per_game), color="#1f77b4", marker="o")
    ax.set_xlabel("(k) Past Episodes in Ensemble")
    ax.set_ylabel("Validation Action Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _finalize_plot("wm_context_action_accuracy_avg.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_ks, _avg(type_per_game), color="#1f77b4", marker="o")
    ax.set_xlabel("(k) Past Episodes in Ensemble")
    ax.set_ylabel("Validation Type Accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    _finalize_plot("wm_context_type_accuracy_avg.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for game, (ks, accs) in act_per_game.items():
        ax.plot(ks, accs, color=GAME_COLORS[game], marker="o", label=f"Game {game}")
    ax.set_xlabel("Past Episodes in Ensemble (k)")
    ax.set_ylabel("Validation Action Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    _finalize_plot("wm_context_action_accuracy_per_game.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for game, (ks, accs) in type_per_game.items():
        ax.plot(ks, accs, color=GAME_COLORS[game], marker="o", label=f"Game {game}")
    ax.set_xlabel("Past Episodes in Ensemble (k)")
    ax.set_ylabel("Validation Type Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    _finalize_plot("wm_context_type_accuracy_per_game.png")


def _finalize_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def run_results_wm(b_agents):
    report_wm_dataset_size(b_agents)
    wm_path = os.path.join(WORLD_MODELS_DIR, "final_results.csv")
    df = pd.read_csv(wm_path)
    analyze_wm(df)
    plot_wm_context_accuracy()
    return