# runner/__ini__.py
from .engine import (
    run_episode, run_training, 
    run_model_free_training, run_model_based_planning, 
    test_on_all_start_states
)

__all__ = [
    "run_episode", "run_training", 
    "run_model_free_training", "run_model_based_planning",
    "test_on_all_start_states"
]


"""
def get_agent_list(agent_type, game_instance):
    agents = []
    num_agents = game_instance.num_agents
    num_actions = game_instance.num_actions
    
    # Extract Payoff Matrix for Heuristic Agent
    # Depending on the class (DecPOMDP vs Game), it might be .payoff_matrix or .R
    payoff_matrix = getattr(game_instance, 'payoff_matrix', None)
    if payoff_matrix is None and hasattr(game_instance, 'R'):
         payoff_matrix = game_instance.R
         
    config = {'payoff_matrix': payoff_matrix}

    for i in range(num_agents):
        if agent_type == "random":
            agents.append(RandomAgent(i, num_actions))
        
        elif agent_type == "heuristic":
            agents.append(HeuristicAgent(i, num_actions, config=config))
        
        elif agent_type == "decpomdp":
            agents.append(DecPOMDPAgent(i, num_actions, config=config))

    return AgentList(agents)


def run_episode(game, agent_list):

    if hasattr(agent_list, 'reset_all'):
        agent_list.reset_all()
    
    total_reward = 0
    history = [] 

    for step in range(game.horizon):
        # 1. Observation: History of joint actions
        observations = [tuple(history) for _ in range(game.num_agents)]
        
        # 2. Valid Actions (Usually all actions 0 to N are valid in matrix games)
        valid_actions = [list(range(game.num_actions)) for _ in range(game.num_agents)]

        # 3. Agents Act
        if hasattr(agent_list, 'take_joint_action'):
            joint_action = agent_list.take_joint_action(observations, valid_actions)
        else:
            # Fallback if using the Actor-based AgentList which uses .act()
            joint_action = agent_list.act(observations)
        
        # 4. Calculate Reward
        try:
            action_tuple = tuple(joint_action)
            if hasattr(game, 'payoff_matrix'):
                reward = game.payoff_matrix[action_tuple]
            elif hasattr(game, 'R'):
                 reward = game.R[action_tuple]
            else:
                reward = 0
        except Exception as e:
            reward = 0 
        
        total_reward += reward
        history.append(joint_action)

    return total_reward
"""
