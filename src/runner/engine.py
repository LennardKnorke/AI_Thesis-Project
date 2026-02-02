import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy

from agents import AgentList
from tiny_game import DecPOMDP, OPTIMAL_RETURNS

def run_training(env: DecPOMDP, agents: AgentList, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, AgentList]:
    """
    Runs either model-free training or model-based planning based on the type of agents provided.
    """
    if agents[0].MODEL_BASED and not agents[1].MODEL_BASED:
        raise ValueError("All agents must be either model-based or model-free.")
    
    if agents[0].MODEL_BASED:
        return run_model_based_planning(env, agents, *args, **kwargs)
    else:
        return run_model_free_training(env, agents, *args, **kwargs)


def run_model_free_training(
        env: DecPOMDP, 
        agents: AgentList,
        train_episodes: int = 100_000,
        *args, **kwargs
) -> tuple[np.ndarray, np.ndarray, AgentList]:
    """
    Runs model-free training (Standard RL Loop).
    """
    loss_results = []
    reward_results = []

    for i in range(train_episodes):
        env.reset()
        _ = run_episode(env, agents)
        
        loss = agents.train()
        avg_test_reward = test_on_all_start_states(env, agents)

        loss_results.append(loss)
        reward_results.append(avg_test_reward)

        if "pbar" in kwargs and kwargs["pbar"] is not None:
            kwargs["pbar"].set_postfix({
                "G": kwargs.get('game_name', ''),
                "Ep": f"{i+1}/{train_episodes}",
                "Loss": f"{loss:.4f}",
                "Rew": f"{avg_test_reward:.2f}"
            })

        if avg_test_reward >= OPTIMAL_RETURNS[kwargs.get('game_name')]:
            break

    return np.array(reward_results), np.array(loss_results), agents


def run_model_based_planning(
    env: DecPOMDP, 
    agents: AgentList,
    max_iterations: int = None,
    convergence_threshold: float = 0.0001,
    attempts: int = 1, # <--- NEW PARAMETER
    *args, **kwargs
) -> tuple[np.ndarray, np.ndarray, AgentList]:
    """
    Runs model-based planning (Value Iteration).
    Supports multiple 'attempts' (random restarts) to avoid local optima.
    Returns results from the Best Attempt.
    """
    assert convergence_threshold is not None

    best_final_reward = -float('inf')
    best_results = (np.array([]), np.array([]), deepcopy(agents))

    game_name = kwargs.get('game_name')
    optimal_return = OPTIMAL_RETURNS.get(game_name, float('inf'))

    # --- ATTEMPTS LOOP ---
    for attempt in range(attempts):
        agents.reset()
        
        # Buffers for THIS attempt
        current_loss_history = []
        current_reward_history = []
        
        current_iteration = 0
        if max_iterations is None:
            max_iterations = -1

        is_optimal = False
        converged = False

        # --- PLANNING LOOP ---
        while not converged and (max_iterations == -1 or current_iteration < max_iterations):
            loss = agents.train()
            avg_test_reward = test_on_all_start_states(env, agents)

            current_loss_history.append(loss)
            current_reward_history.append(avg_test_reward)

            if avg_test_reward >= optimal_return:
                is_optimal = True
                converged = True

            #if convergence_threshold is not None and loss < convergence_threshold:
            #    converged = True

            if "pbar" in kwargs and kwargs["pbar"] is not None:
                kwargs["pbar"].set_postfix({
                    "G": kwargs.get('game_name', ''),
                    "Att": f"{attempt+1}/{attempts}",
                    "Iter": f"{current_iteration+1}",
                    "Delta": f"{loss:.5f}",
                    "Rew": f"{avg_test_reward:.2f}"
                })
            current_iteration += 1

        # Calculate  final reward
        if len(current_reward_history) > 0:
            final_reward = current_reward_history[-1]
        else:
            final_reward = -float('inf')

        # Save best results
        if final_reward > best_final_reward or attempt == 0:
            best_final_reward = final_reward
            best_agents_state = deepcopy(agents)
            best_results = (
                np.array(current_reward_history), 
                np.array(current_loss_history), 
                best_agents_state
            )

        # End if already optimal
        if is_optimal:
            break
    
    return best_results


def test_on_all_start_states(env: DecPOMDP, agents: AgentList) -> float:
    start_states = env.start_states()
    total_test_reward = 0.0
    for start_state in start_states:
        episode_reward = run_episode(env, agents, start_state=start_state, test_episode=True)
        total_test_reward += episode_reward
    return total_test_reward / len(start_states)


def run_episode(
    env: DecPOMDP, 
    agents: AgentList,
    start_state: list | None = None,
    test_episode: bool = False,
):
    if start_state is None: 
        env.reset() 
    else:
        env.reset(history=list(start_state))

    total_reward = 0.0
    done = False
    
    while not done:
        # Determine current player
        # Tiny Hanabi: Start (Len 2) -> P0. Middle (Len 3) -> P1.
        if len(env.context()) == 2:
            player_id = 0
        elif len(env.context()) == 3:
            player_id = 1
        else:
            # Should not happen if loop checks 'done' correctly, but safe fallback
            break
            
        agent = agents[player_id]

        # Construct Observation (Masked)
        # Note: env.context() returns tuples in original game, cast to list for mutability
        observation = list(env.context())
        observation[player_id] = -1 # Mask Own Card
        observation = tuple(observation) # Cast back for Agent Act
        
        action = agent.act(observation)
        env.step(action)

        # Build Next Observation for Training Storage
        next_observation = list(env.context())
        if len(next_observation) <= 4:
            next_observation[player_id] = -1
        next_observation = tuple(next_observation)

        done = env.is_terminal()
        reward = env.payoff() if done else 0.0
        total_reward += reward

        if not test_episode:
            agent.save_transition(
                observation,
                action,
                next_observation,
                reward,
                done
            )
            
    return total_reward