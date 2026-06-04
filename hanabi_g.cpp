/// hanabi_g.cpp  —  MyHanabi (Game G) 2-agent Dec-POMDP generator
///
/// Encoding (noop trick for sequential turns):
///   - Agent 0 acts on EVEN history lengths (0,2,4…); agent 1 on ODD (1,3,5…).
///   - At each simultaneous step: one agent takes a real action (0–3),
///     the other plays NOOP (4).  Invalid joint actions → self-loop.
///   - On the action that terminates the game → uniform reset over all 24 deals.
///
/// Compact incremental observations (Hanabi convention: see partner's hand):
///   At the INITIAL state (no actions yet):
///     Agent 0 obs = deal[2]*4 + deal[3]   (partner cards p1c0, p1c1 → 0..15)
///     Agent 1 obs = deal[0]*4 + deal[1]   (partner cards p0c0, p0c1 → 0..15)
///   At any subsequent state:
///     Both agents observe the LAST (action, card_or_NO_CARD) entry in history.
///     Play actions  (a ∈ {0,1}, c ∈ {0..3})  encoded as  16 + a*4 + c   (16..23)
///     Declare actions (a ∈ {2,3}, c = NO_CARD) encoded as  24 + (a-2)   (24..25)
///   Total real obs per agent = 26 (see Phase-0 section below for full N_OBS).
///
/// The agent's private HISTORY (sequence of incremental obs) carries everything:
///   - Initial obs reveals partner's hand.
///   - Subsequent obs reveal each play action's card + all declare-safe indices.
///   - Own initial hand is never directly revealed (correct Hanabi semantics).
///
/// Phase-0 states (24 total, one per deal):
///   Agents play (NOOP,NOOP) → transition to game initial state, delivering real
///   initial obs.  Any other action self-loops in phase-0 with dummy obs (26).
///   belief_init is uniform over phase-0, so agents always receive initial obs.
///   Terminating game actions reset to phase-0 (not game initial states).
///   Total obs per agent = 27 (0..15 initial, 16..23 play, 24..25 declare, 26 dummy).
///   5 bits per obs (vs 6 bits for 37), enabling truncation=4 within 64-bit Support.
///
/// Solver settings:  -N 2 -p 13 -n oSarsa-seq -m 4 -t 300 -s 0

#include "../../core/_module.hpp"
#include "hanabi_g.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <numeric>
#include <vector>


namespace problem_examples {
namespace hanabi_g {

using namespace std;
using namespace common;


// ---- constants ----------------------------------------------------------------

static const int N_CARDS        = 4;   // card values: 0 1 2 3
static const int HAND_SIZE      = 2;   // cards per player
static const int N_REAL_ACTIONS = 4;   // 0=PlayCard0  1=PlayCard1  2=DeclPartner0  3=DeclPartner1
static const int NOOP           = 4;   // noop action index
static const int N_ACTIONS      = 5;   // real + noop
static const int NO_CARD        = 4;   // sentinel in history for declare-safe steps
static const int PILE_CAPACITY  = 4;   // game ends when 4 cards in pile
static const int MAX_ACTIONS    = 8;   // horizon(12) − deal_size(4)
static const double DISCOUNT    = 0.99;

// Observation encoding sizes  (compressed: 27 values → 5 bits per obs)
// Initial obs:   c0 * N_CARDS + c1              →  [0, 16)  = 0..15
// Play obs:      16 + action*N_CARDS + card     →  [16, 24) = 16..23  (action ∈ {0,1})
// Declare obs:   24 + (action - HAND_SIZE)      →  [24, 26) = 24..25  (action ∈ {2,3})
// Dummy obs:     26  (delivered on entering a phase-0 state — agents have no info yet)
static const int OBS_INIT_BASE    = 0;                                    // 0..15
static const int OBS_PLAY_BASE    = N_CARDS * N_CARDS;                    // = 16
static const int OBS_DECLARE_BASE = OBS_PLAY_BASE + HAND_SIZE * N_CARDS;  // = 24
static const int OBS_DUMMY        = OBS_DECLARE_BASE + HAND_SIZE;         // = 26
static const int N_OBS            = OBS_DUMMY + 1;                        // = 27


// ---- compact state types ------------------------------------------------------

using Deal    = array<int, 4>;          // (p0c0, p0c1, p1c0, p1c1)
using HistEnt = pair<int, int>;         // (action, card_or_NO_CARD)
using History = vector<HistEnt>;

struct State {
    Deal    deal;
    History hist;
    bool operator<(const State& o) const {
        if (deal != o.deal) return deal < o.deal;
        return hist < o.hist;
    }
};


// ---- game logic helpers -------------------------------------------------------

static int pile_size(const History& h) {
    int c = 0;
    for (auto& [a, card] : h) if (card != NO_CARD) ++c;
    return c;
}

static bool is_terminal(const History& h) {
    return pile_size(h) >= PILE_CAPACITY || (int)h.size() >= MAX_ACTIONS;
}

static int current_player(const History& h) { return (int)h.size() % 2; }

/// Track which card slots each player still holds (true = not yet played).
static void get_hands(const History& h, array<bool,2>& p0, array<bool,2>& p1) {
    p0 = {true, true};
    p1 = {true, true};
    for (int i = 0; i < (int)h.size(); ++i) {
        auto [a, c] = h[i];
        if (a < HAND_SIZE) {
            if (i % 2 == 0) p0[a] = false;
            else             p1[a] = false;
        }
    }
}

/// Legal actions for the current player.
static vector<int> legal_actions(const Deal& deal, const History& h) {
    if (is_terminal(h)) return {};
    int player = current_player(h);
    array<bool,2> p0, p1;
    get_hands(h, p0, p1);

    vector<int> acts;
    const array<bool,2>& my   = (player == 0) ? p0 : p1;
    const array<bool,2>& part = (player == 0) ? p1 : p0;

    for (int i = 0; i < HAND_SIZE; ++i) if (my[i])   acts.push_back(i);          // play own card i
    for (int i = 0; i < HAND_SIZE; ++i) if (part[i]) acts.push_back(HAND_SIZE+i); // declare partner i safe
    return acts;
}

/// Apply action; return the new history entry (action, revealed_card_or_NO_CARD).
static HistEnt apply_action(const Deal& deal, const History& h, int action) {
    int player = current_player(h);
    if (action < HAND_SIZE) {
        int card = (player == 0) ? deal[action] : deal[HAND_SIZE + action];
        return {action, card};
    }
    return {action, NO_CARD};
}

/// Payoff: count cards that extend the sequence from the last counted card.
/// last advances only on a hit; misses are skipped without stopping the loop.
/// Matches Python calc_final_reward exactly.
static double compute_payoff(const History& h) {
    double count = 0.0;
    int last = -1;
    for (auto& [a, c] : h) {
        if (c != NO_CARD) {
            if ((last == -1 && c == 0) || (last != -1 && c == last + 1)) {
                count += 1.0;
                last = c;
            }
        }
    }
    return count;
}


// ---- compact incremental observations ----------------------------------------
//
//  Each agent receives one observation per time step:
//    • At the initial state (hist empty): partner's starting cards.
//    • After any action: the last (action, card_or_NO_CARD) that just happened.
//
//  This is all that changes at each step — the agent accumulates a private
//  HISTORY of these incremental observations which reconstructs full information.

/// Observation received by each agent upon entering state s.
static int encode_action_obs(int a, int c) {
    if (a < HAND_SIZE)                                    // play action (a ∈ {0,1})
        return OBS_PLAY_BASE + a * N_CARDS + c;           // 16..23
    return OBS_DECLARE_BASE + (a - HAND_SIZE);            // declare action (a ∈ {2,3}) → 24..25
}

static int make_obs0(const State& s) {
    if (s.hist.empty())
        return OBS_INIT_BASE + s.deal[2] * N_CARDS + s.deal[3]; // sees p1c0, p1c1
    auto [a, c] = s.hist.back();
    return encode_action_obs(a, c);
}

static int make_obs1(const State& s) {
    if (s.hist.empty())
        return OBS_INIT_BASE + s.deal[0] * N_CARDS + s.deal[1]; // sees p0c0, p0c1
    auto [a, c] = s.hist.back();
    return encode_action_obs(a, c);
}


// ---- generator ----------------------------------------------------------------
//
// Phase-0 states (IDs 0..23): one per deal.  Agents play (NOOP,NOOP) → transition
// to the game initial state, which delivers real initial observations.  Any other
// joint action self-loops back to the phase-0 state with dummy observations.
//
// Game states (IDs 24+): BFS-enumerated non-terminal histories.
//   • Entering an initial game state (empty hist) → real initial obs (0..15)
//   • Entering a non-initial game state → last-action obs (16..25)
//   • Entering a phase-0 state (from reset or self-loop) → dummy obs (26)
//
// On a terminating action from a game state: reward delivered, uniform reset
// to phase-0 (NOT to game initial states, so agents always get initial obs next).
//
// Solver settings:  -N 2 -p 11 -n oSarsa-seq -m 2 -t 300 -s 0
//   horizon=11 covers: 1 phase-0 step + up to 8 game actions + 2 buffer cycles

void generate() {

    // ---- 1. All 24 start deals (permutations of {0,1,2,3}) -------------------

    vector<Deal> deals;
    Deal base = {0, 1, 2, 3};
    do { deals.push_back(base); } while (next_permutation(base.begin(), base.end()));
    assert((int)deals.size() == 24);

    const int n_deals       = (int)deals.size();
    const double reset_prob = 1.0 / (double)n_deals;

    // Phase-0 global IDs: 0 .. n_deals-1
    // Game state global IDs: n_deals + (BFS index)

    // ---- 2. BFS — enumerate all non-terminal game states ---------------------

    vector<State>   game_states;
    map<State, int> game_state_to_local;  // 0-based within game_states

    for (auto& d : deals) {
        State s; s.deal = d; s.hist = {};
        game_state_to_local[s] = (int)game_states.size();
        game_states.push_back(s);
    }

    game_states.reserve(60000);
    for (int qi = 0; qi < (int)game_states.size(); ++qi) {
        const State s = game_states[qi]; // copy — vector may reallocate
        for (int a : legal_actions(s.deal, s.hist)) {
            History nh = s.hist;
            nh.push_back(apply_action(s.deal, s.hist, a));
            if (is_terminal(nh)) continue;
            State ns; ns.deal = s.deal; ns.hist = nh;
            if (!game_state_to_local.count(ns)) {
                game_state_to_local[ns] = (int)game_states.size();
                game_states.push_back(ns);
            }
        }
    }

    const int n_game_states = (int)game_states.size();
    const int n_states      = n_deals + n_game_states;

    // Helper: global ID for a game state
    auto game_id = [&](const State& s) -> int {
        return n_deals + game_state_to_local.at(s);
    };

    // ---- 3. Fill PROBLEM header ----------------------------------------------

    PROBLEM.discount         = DISCOUNT;
    PROBLEM.criterion_reward = true;
    PROBLEM.agents_number    = 2;
    PROBLEM.last_agent       = 1;
    PROBLEM.states_number    = n_states;

    // belief_init: uniform over phase-0 states
    PROBLEM.belief_init = Vector(n_states, 0.0);
    for (int di = 0; di < n_deals; ++di)
        PROBLEM.belief_init(di) = reset_prob;

    PROBLEM.actions_number_byAgent  = {N_ACTIONS, N_ACTIONS};
    PROBLEM.actions_joint_number    = N_ACTIONS * N_ACTIONS; // 25

    // N_OBS = 27: 0..15 initial obs, 16..23 play obs, 24..25 declare obs, 26 dummy
    PROBLEM.observations_number_byAgent = {N_OBS, N_OBS};
    PROBLEM.observations_joint_number   = N_OBS * N_OBS;    // 729

    // ---- 4. Transitions + rewards --------------------------------------------

    PROBLEM.rewards_matrix = algebra::Matrix<double>(n_states, PROBLEM.actions_joint_number, 0.0);

    // Phase-0 transitions
    for (int di = 0; di < n_deals; ++di) {
        int p0_id = di;
        State init_s; init_s.deal = deals[di]; init_s.hist = {};
        int game_init_id = game_id(init_s);

        for (int u0 = 0; u0 < N_ACTIONS; ++u0) {
            for (int u1 = 0; u1 < N_ACTIONS; ++u1) {
                int u = u1 + u0 * N_ACTIONS;
                if (u0 == NOOP && u1 == NOOP)
                    PROBLEM.dynamics_T[u][p0_id][game_init_id] = 1.0;
                else
                    PROBLEM.dynamics_T[u][p0_id][p0_id] = 1.0;
            }
        }
    }

    // Game state transitions
    for (int gi = 0; gi < n_game_states; ++gi) {
        const State& s = game_states[gi];
        int si         = n_deals + gi;
        int player     = current_player(s.hist);
        auto legal     = legal_actions(s.deal, s.hist);

        for (int u0 = 0; u0 < N_ACTIONS; ++u0) {
            for (int u1 = 0; u1 < N_ACTIONS; ++u1) {
                int u = u1 + u0 * N_ACTIONS;

                // Determine valid real action for this player
                int real_action = -1;
                if (player == 0 && u0 < N_REAL_ACTIONS && u1 == NOOP) {
                    if (find(legal.begin(), legal.end(), u0) != legal.end())
                        real_action = u0;
                } else if (player == 1 && u0 == NOOP && u1 < N_REAL_ACTIONS) {
                    if (find(legal.begin(), legal.end(), u1) != legal.end())
                        real_action = u1;
                }

                if (real_action < 0) {
                    // Invalid or illegal action: self-loop
                    PROBLEM.dynamics_T[u][si][si] = 1.0;
                    continue;
                }

                History nh = s.hist;
                nh.push_back(apply_action(s.deal, s.hist, real_action));

                if (is_terminal(nh)) {
                    // Terminating action → reward + uniform reset to phase-0
                    PROBLEM.rewards_matrix(si, u) = compute_payoff(nh);
                    for (int di = 0; di < n_deals; ++di)
                        PROBLEM.dynamics_T[u][si][di] += reset_prob;
                } else {
                    State ns; ns.deal = s.deal; ns.hist = nh;
                    PROBLEM.dynamics_T[u][si][game_id(ns)] = 1.0;
                }
            }
        }
    }

    // ---- 5. Observations (deterministic, indexed by destination state) -------
    // Joint obs index: z = o1 + o0 * N_OBS  (matches get_jointIndex convention)

    for (int u = 0; u < PROBLEM.actions_joint_number; ++u) {
        // Phase-0 destination states → dummy obs for both agents
        for (int di = 0; di < n_deals; ++di) {
            int z = OBS_DUMMY + OBS_DUMMY * N_OBS;
            PROBLEM.dynamics_O[u][di][z] = 1.0;
        }
        // Game destination states → real incremental obs
        for (int gi = 0; gi < n_game_states; ++gi) {
            int yi = n_deals + gi;
            int o0 = make_obs0(game_states[gi]);
            int o1 = make_obs1(game_states[gi]);
            int z  = o1 + o0 * N_OBS;
            PROBLEM.dynamics_O[u][yi][z] = 1.0;
        }
    }

    cout << "[hanabi_g] n_phase0="    << n_deals
         << "  n_game_states="        << n_game_states
         << "  n_states="             << n_states
         << "  n_obs_per_agent="      << N_OBS
         << "  joint_obs="            << (N_OBS * N_OBS)
         << endl;
    cout.flush();

    PROBLEM.finalize();
    cout << "[hanabi_g] finalize() done" << endl; cout.flush();
}


}} // namespace problem_examples::hanabi_g
