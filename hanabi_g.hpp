#pragma once

namespace problem_examples {
namespace hanabi_g {

    /// Generate the MyHanabi (Game G) Dec-POMDP for 2 agents.
    ///
    /// MyHanabi structure:
    ///   4 unique cards {0,1,2,3}, 2 cards per player.
    ///   24 start deals (all permutations of 4 distinct cards).
    ///   Actions: 0=PlayCard0, 1=PlayCard1, 2=DeclarePartnerCard0Safe, 3=DeclarePartnerCard1Safe
    ///   Each agent sees the PARTNER'S hand but not their own (Hanabi convention).
    ///   Sequential: agent 0 acts first each round, then agent 1 (alternating turns).
    ///   Game ends when pile holds 4 cards OR action count reaches 8.
    ///   Reward = length of the longest prefix of {0,1,2,3} played in order (0–4).
    ///
    /// Encoded as simultaneous Dec-POMDP with noop actions (action 4):
    ///   - When it is agent i's real turn: agent i takes a real action, the other noops.
    ///   - Invalid joint actions (wrong-phase actor) → self-loop, reward 0.
    ///   - On the action that terminates the game → uniform reset to any of the 24 start deals.
    ///
    void generate();

}} // namespace problem_examples::hanabi_g
