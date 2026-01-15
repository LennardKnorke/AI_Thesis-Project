# runner/engine.py

import numpy as np
from tqdm import tqdm

from agents import AgentList, BaseAgent
from tiny_game import * 
from config import *


def run_training(env : DecPOMP_Rework, agents : AgentList, *args, **kwargs)-> tuple[np.ndarray, np.ndarray, AgentList]:
    """
    Runs either model-free training or model-based planning based on the type of agents provided.
    Args:
        env (DecPOMP_Rework): The environment in which the agents will be trained or planned.
        agents (AgentList): A list of agents to be trained or planned.
    Returns:
        tuple[np.ndarray, np.ndarray, AgentList]: A tuple containing:
            - loss_results (np.ndarray): Array of training loss values recorded at each test interval (avg if both agents training).
            - test_results (np.ndarray): Array of average test rewards recorded at each test interval.
            - agents (AgentList): The trained or planned agents after the process.
    """

    # Pre condition Checks
    if agents[0].MODEL_BASED and not agents[1].MODEL_BASED:
        raise ValueError("All agents must be either model-based or model-free.")
    
    # Re-Route to appropriate training function
    if agents[0].MODEL_BASED:
        return run_model_based_planning(
            env,
            agents,
            *args,
            **kwargs
        )
    else:
        return run_model_free_training(
            env,
            agents,
            *args,
            **kwargs
        )


def run_model_free_training(
        env : DecPOMP_Rework, 
        agents : AgentList,
        train_episodes : int = 100_000,
        train_test_freq : int = 10,
        *args,
        **kwargs
)->tuple[np.ndarray, np.ndarray, AgentList]:
    """
    This function runs model-free training for the given agents in the specified environment.
    Adheres to general model-free training loop of collecting experiences, learning from these, and period testing.
    Args:
        env (DecPOMP_Rework): The environment in which the agents will be trained.
        agents (AgentList): A list of model-free agents to be trained.
        train_episodes (int): The total number of episode collected for training.
        train_test_freq (int): Frequency of testing (in episodes).
    Returns:
        tuple[np.ndarray, np.ndarray, AgentList]: A tuple containing:
            - loss_results (np.ndarray): Array of training loss values recorded at each test interval (avg if both agents training).
            - test_results (np.ndarray): Array of average test rewards recorded at each test interval.
            - agents (AgentList): The trained agents after the training process.
    """
    # Pre condition Checks
    assert train_test_freq > 0, "train_test_freq must be positive"
    
    for agent in agents:
        assert not agent.MODEL_BASED, "All agents must be model-free for this training function."
    
    assert train_episodes >= 1, "train_episodes must be at least 1"
    assert train_episodes % train_test_freq == 0 or train_episodes == 1, \
        "train_episodes must be divisible by train_test_freq or equal to 1"

    # Set up Results
    loss_results = []
    reward_results = []

    # START - MAIN TRAINING LOOP
    for i in range(train_episodes):        
        # collect episode experience
        env.reset() # Random start state
        _ = run_episode(
            env,
            agents
        )
        # Periodic Training and Testing
        if i == 0 or (i+1) % train_test_freq == 0 or i == train_episodes - 1:
            # Training Step
            loss = agents.train()
            
            # Testing Step
            avg_test_reward = test_on_all_start_states(env, agents)

            # Update Results 
            loss_results.append(loss)
            reward_results.append(avg_test_reward)

            if "pbar" in kwargs.keys() and type(kwargs["pbar"]) == tqdm:
                kwargs["pbar"].set_postfix({
                    "G" : kwargs['game_name'],
                    "Ep": f"{i+1}/{train_episodes}",
                    "Loss": f"{np.mean(loss_results):.4f}",
                    "Rew": f"{np.mean(reward_results):.2f}"
                })

    # END - MAIN TRAINING LOOP
    return np.array(reward_results), np.array(loss_results), agents


def run_model_based_planning(
    env : DecPOMP_Rework, 
    agents : AgentList,

    max_iterations : int = None,
    convergence_threshold : float = 0.0001,
    *args,
    **kwargs
)-> tuple[np.ndarray, np.ndarray, AgentList]:
    """
    This function runs model-based planning for the given agents in the specified environment.
    Adheres to general model-based planning loop of iteratively improving the policy via planning and testing
    Args:
        env (DecPOMP_Rework): The environment in which the agents will be planned.
        agents (AgentList): A list of model-based agents to be planned.
        max_iterations (int): Maximum number of planning iterations. If None, runs until convergence.
        convergence_threshold (float): Threshold for convergence based on change in loss.
    Returns:
        tuple[np.ndarray, np.ndarray, AgentList]: A tuple containing:
            - loss_results (np.ndarray): Array of planning loss values recorded at each iteration.
            - test_results (np.ndarray): Array of average test rewards recorded at each iteration.
            - agents (AgentList): The planned agents after the planning process.
    """

    # Pre condition Checks
    for agent in agents:
        assert agent.MODEL_BASED, "All agents must be model-based for this planning function."

    assert convergence_threshold is not None, "convergence_threshold must be specified for model-based planning."

    # Set up Results
    loss_results = []
    reward_results = []

    # Set up Loop Conditions
    current_iteration = 0
    if max_iterations is None:
        max_iterations = -1  # Infinite loop until convergence
    converged = False

    # START - MAIN PLANNING LOOP
    while not converged or current_iteration != max_iterations:
        # Planning Step
        loss = agents.train()
        # Testing Step
        avg_test_reward = test_on_all_start_states(env, agents)

        # Store Results
        loss_results.append(loss)
        reward_results.append(avg_test_reward)

        # Update Loop Conditions
        if loss < convergence_threshold:
            converged = True

        if "pbar" in kwargs.keys() and type(kwargs["pbar"]) == tqdm:
            kwargs["pbar"].set_postfix({
                "G" : kwargs['game_name'],
                "Iter": f"{current_iteration+1}",
                "Delta": f"{loss:.5f}",
                "Rew": f"{np.mean(reward_results):.2f}"
            })
        
        current_iteration += 1

    # END - MAIN PLANNING LOOP
    return np.array(reward_results), np.array(loss_results), agents


def test_on_all_start_states(
    env : DecPOMP_Rework, 
    agents : AgentList,
)-> float:
    """
    Loops over all start states, run policies greedyil and returns average reward.
    Args:
        env (DecPOMP_Rework): The environment in which the agents will be tested.
        agents (AgentList): A list of agents to be tested.
    Returns:
        float: The average test reward over all start states.
    """

    start_states = env.start_states()

    total_test_reward = 0.0

    # Loop over all start states
    for start_state in start_states:

        # Run Episode from this start state
        episode_reward = run_episode(
            env,
            agents,
            start_state=start_state,
            test_episode=True
        )

        # Accumulate Reward over all start states
        total_test_reward += episode_reward
    return total_test_reward / len(start_states)


def run_episode(
    env : DecPOMP_Rework, 
    agents : AgentList,
    start_state : list|None = None,
    test_episode : bool = False,
):
    """
    Runs a single episode in the environment with the given agents.
    Args:
        env (DecPOMP_Rework): The environment in which the episode will be run.
        agents (AgentList): A list of agents participating in the episode.
        start_state (list|None): Specific start state to initialize the environment. If None, random start state is used.
        test_episode (bool): If True, no transitions are saved (for testing only).
    """
    if start_state is None: 
        env.reset() # Random start state
    else:
        # Set specific start state
        current_history = list(start_state) 
        env.reset(history=current_history)

    # Set up reward and step counter
    total_reward = 0.0
    agent_indices = np.arange(len(agents))
    np.random.shuffle(agent_indices)

    # START - EPISODE LOOP
    while not env.is_terminal():
        # Fix current agent
        player_id = env.get_playerId()
        assigned_agent_idx = agent_indices[player_id]
        agent = agents[assigned_agent_idx]
        
        needs_tensor = agent.requires_tensor

        # Pick action - Execute - Observe
        observation = env.get_observation(player_id, as_tensor=needs_tensor)

        action = agent.act(observation)

        env.step(action)

        next_observation = env.get_observation(
            agent_id=env.get_playerId(),
            as_tensor=needs_tensor
        )
        done = env.is_terminal()
        reward = env.payoff() if done else 0.0
        total_reward += reward

        # Safe transition if Training
        if not test_episode:
            agent.save_transition(
                observation,
                action,
                next_observation,
                reward,
                done
            )
        observation = next_observation
    return total_reward