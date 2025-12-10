

import numpy as np


from agents import AgentList, BaseAgent
from tiny_game import * 
from config import EPISODES_TEST



def run_episode(
    env : Game, 
    agents : AgentList,
    start_state : list,
    test_episode : bool = False,
):
    current_history = list(start_state) 
    env.reset(history=current_history)
    total_reward = 0.0
    current_step = 0
    while not env.is_terminal():
        # Fix current agent
        agent = agents[current_step]

        # Observation
        observation = env.context()
        # First Player only Observes the other agents cards
        if current_step == 0:
            observation = [observation[0]]
        elif current_step == 1:
            observation = observation[1:]
        else:
            raise ValueError

        action = agent.act(observation)

        # Environment Step
        env.step(action)

        # Get follow up History + Reward
        next_observation = observation
        next_observation.append(action)

        done = env.is_terminal() or current_step == 1
        if done:
            reward = env.payoff()
        else:
            reward = 0.0
            
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

    