"""
Export Tiny-Hanabi games A–F to .dpomdp format for the oSarsa-seq C++ solver.

Usage:
    python export_hanabi_dpomdp.py               # exports all games A–F
    python export_hanabi_dpomdp.py A C

Output:
    osarsa-aaai-25/code/src/problem_examples/dpomdp/Hanabi_<X>_light.dpomdp

Encoding: 3-phase simultaneous Dec-POMDP with noop actions
-----------------------------------------------------------
Tiny-Hanabi A–F is sequential: deal → agent 0 acts → agent 1 acts → reward.
The .dpomdp format has no initial observations, so a Phase-0 "reveal" step
delivers cards via a transition before any real decision is made.

    Phase-0 (c0,c1): (noop,noop) → Phase-1; others self-loop with dummy obs.
        obs: agent 0 ← c1,  agent 1 ← c0

    Phase-1 (c0,c1): agent 0 plays a0; agent 1 noops → Phase-2.
        obs: agent 0 ← c1,  agent 1 ← n_cards + c0*n_real + a0

    Phase-2 (c0,c1,a0): agent 1 plays a1; agent 0 noops → reward + Phase-0 reset.
        obs: dummy on entering Phase-0

Observation counts:
    agent 0: n_obs_0 = n_cards + 1              (c1 values + dummy)
    agent 1: n_obs_1 = n_cards*(1+n_real) + 1   (c0, c0+a0 combos + dummy)

Solver: horizon=9, truncation=2  (agent 0 needs 1 obs, agent 1 needs 2)
"""

import sys
import os
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from tiny_game import PAYOFFS, OPTIMAL_RETURNS, get_game_Rework, GameNames


OUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "osarsa-aaai-25", "code", "src", "problem_examples", "dpomdp",
)
DISCOUNT = 0.99


def export_game(game_letter: str):
    """Write the 3-phase .dpomdp encoding for one TinyHanabi game."""
    gname   = GameNames(game_letter)
    env     = get_game_Rework(gname)
    payoffs = PAYOFFS[game_letter]          # shape [c0][c1][a0][a1]

    # ---- game dimensions ---------------------------------------------------
    n_real   = env.num_actions              # real actions per agent
    n_cards  = env.num_cards                # distinct card values
    noop     = n_real                       # noop action index
    n_act    = n_real + 1                   # total actions per agent (real + noop)

    deals      = sorted(env.start_states(), key=lambda s: (s[0], s[1]))
    n_deals    = len(deals)
    deal_idx   = {(s[0], s[1]): i for i, s in enumerate(deals)}

    # ---- state indices -----------------------------------------------------
    # Phase-0: states 0 .. n_deals-1                (observation reveal)
    # Phase-1: states n_deals .. 2*n_deals-1         (agent 0 acts)
    # Phase-2: states 2*n_deals .. 2*n_deals+n_deals*n_real-1  (agent 1 acts)
    n_states_p0 = n_deals
    n_states_p1 = n_deals
    n_states_p2 = n_deals * n_real
    n_states    = n_states_p0 + n_states_p1 + n_states_p2

    def p0_state(c0, c1):
        return deal_idx[(c0, c1)]

    def p1_state(c0, c1):
        return n_deals + deal_idx[(c0, c1)]

    def p2_state(c0, c1, a0):
        return 2 * n_deals + deal_idx[(c0, c1)] * n_real + a0

    # ---- observation counts ------------------------------------------------
    # Agent 0: sees c1 when entering Phase-1 or Phase-2; dummy when entering Phase-0
    #   real obs: 0 … n_cards-1
    #   dummy:    n_cards
    #   n_obs_0 = n_cards + 1
    #
    # Agent 1: sees c0 when entering Phase-1;
    #          sees n_cards + c0*n_real + a0 when entering Phase-2;
    #          dummy when entering Phase-0
    #   real obs: 0 … n_cards*(1+n_real)-1
    #   dummy:    n_cards * (1 + n_real)
    #   n_obs_1 = n_cards * (1 + n_real) + 1
    n_obs_0 = n_cards + 1
    n_obs_1 = n_cards * (1 + n_real) + 1

    dummy_obs_0 = n_cards                   # agent 0 dummy (out-of-range)
    dummy_obs_1 = n_cards * (1 + n_real)    # agent 1 dummy (out-of-range)

    def obs0_phase1(_, c1):      return c1
    def obs1_phase1(c0, _):     return c0
    def obs0_phase2(_, c1):     return c1
    def obs1_phase2(c0, _, a0): return n_cards + c0 * n_real + a0

    # ---- all joint actions -------------------------------------------------
    joint_actions = list(itertools.product(range(n_act), repeat=2))

    lines = []

    # ---- header ------------------------------------------------------------
    lines.append(f"agents: 2")
    lines.append(f"discount: {DISCOUNT}")
    lines.append(f"states: {' '.join(str(i) for i in range(n_states))}")
    lines.append(f"actions:")
    lines.append(f"{' '.join(str(a) for a in range(n_act))}")   # agent 0
    lines.append(f"{' '.join(str(a) for a in range(n_act))}")   # agent 1
    lines.append(f"observations:")
    lines.append(f"{' '.join(str(o) for o in range(n_obs_0))}")  # agent 0
    lines.append(f"{' '.join(str(o) for o in range(n_obs_1))}")  # agent 1

    # Start: uniform over Phase-0 states
    p0_prob = 1.0 / n_deals
    start_probs = (
        [f"{p0_prob:.6f}"] * n_deals +
        ["0.000000"] * n_states_p1 +
        ["0.000000"] * n_states_p2
    )
    lines.append(f"start: {' '.join(start_probs)}")

    # ---- transitions -------------------------------------------------------
    for a0, a1 in joint_actions:
        # Phase-0: only (noop, noop) is valid → Phase-1
        for c0, c1 in [(s[0], s[1]) for s in deals]:
            x = p0_state(c0, c1)
            if a0 == noop and a1 == noop:
                y = p1_state(c0, c1)
                lines.append(f"T: {a0} {a1} : {x} : {y} : 1.000000")
            else:
                lines.append(f"T: {a0} {a1} : {x} : {x} : 1.000000")

        # Phase-1: only (a0_real, noop) is valid → Phase-2
        for c0, c1 in [(s[0], s[1]) for s in deals]:
            x = p1_state(c0, c1)
            if a0 < n_real and a1 == noop:
                y = p2_state(c0, c1, a0)
                lines.append(f"T: {a0} {a1} : {x} : {y} : 1.000000")
            else:
                lines.append(f"T: {a0} {a1} : {x} : {x} : 1.000000")

        # Phase-2: only (noop, a1_real) is valid → uniform reset to Phase-0
        for c0, c1 in [(s[0], s[1]) for s in deals]:
            for a0_taken in range(n_real):
                x = p2_state(c0, c1, a0_taken)
                if a0 == noop and a1 < n_real:
                    for c0_, c1_ in [(s[0], s[1]) for s in deals]:
                        y = p0_state(c0_, c1_)
                        lines.append(f"T: {a0} {a1} : {x} : {y} : {p0_prob:.6f}")
                else:
                    lines.append(f"T: {a0} {a1} : {x} : {x} : 1.000000")

    # ---- observations ------------------------------------------------------
    # O: a0 a1 : next_state : o0 o1 : prob
    for a0, a1 in joint_actions:
        # Entering Phase-0 (from reset or self-loop): dummy obs
        for c0, c1 in [(s[0], s[1]) for s in deals]:
            y = p0_state(c0, c1)
            lines.append(f"O: {a0} {a1} : {y} : {dummy_obs_0} {dummy_obs_1} : 1.000000")

        # Entering Phase-1 (from (noop,noop) or self-loop): real initial obs
        for c0, c1 in [(s[0], s[1]) for s in deals]:
            y  = p1_state(c0, c1)
            o0 = obs0_phase1(c0, c1)    # c1 for agent 0
            o1 = obs1_phase1(c0, c1)    # c0 for agent 1
            lines.append(f"O: {a0} {a1} : {y} : {o0} {o1} : 1.000000")

        # Entering Phase-2 (from (a0,noop) or self-loop): a0 revealed to agent 1
        for c0, c1 in [(s[0], s[1]) for s in deals]:
            for a0_taken in range(n_real):
                y  = p2_state(c0, c1, a0_taken)
                o0 = obs0_phase2(c0, c1)              # c1 again for agent 0
                o1 = obs1_phase2(c0, c1, a0_taken)    # c0 + a0 for agent 1
                lines.append(f"O: {a0} {a1} : {y} : {o0} {o1} : 1.000000")

    # ---- rewards -----------------------------------------------------------
    # R: a0 a1 : state : * : * : reward
    # Only non-zero for valid Phase-2 exits: (noop, a1) from a Phase-2 state
    for c0, c1 in [(s[0], s[1]) for s in deals]:
        for a0_taken in range(n_real):
            x = p2_state(c0, c1, a0_taken)
            for a1 in range(n_real):
                r = float(payoffs[c0][c1][a0_taken][a1])
                lines.append(f"R: {noop} {a1} : {x} : * : * : {r:.6f}")

    # ---- write file --------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    filename = os.path.join(OUT_DIR, f"Hanabi_{game_letter}_light.dpomdp")
    with open(filename, "w") as f:
        f.write("\n".join(lines) + "\n")

    opt = OPTIMAL_RETURNS.get(game_letter, "?")
    print(f"  {game_letter}: {n_states} states, {n_act} actions/agent, "
          f"{n_obs_0}/{n_obs_1} obs (a0/a1), optimal={opt}  -> {filename}")


def main():
    games = sys.argv[1:] if len(sys.argv) > 1 else list("ABCDEF")
    print(f"Exporting games: {games}")
    for g in games:
        g = g.upper()
        if g not in list("ABCDEF"):
            print(f"  {g}: skipped (only A-F; G is MyHanabi with 4-card deals, different structure)")
            continue
        export_game(g)

    print("\nDone. Run from osarsa-aaai-25/code/ with e.g.:")
    for g in games:
        g = g.upper()
        if g in list("ABCDEF"):
            print(f"  build/sdms.exe -f Hanabi_{g} -N 2 -p 9 -n oSarsa-seq -m 2 -t 60 -s 0")


if __name__ == "__main__":
    main()
