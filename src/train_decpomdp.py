import numpy as np
import os
import matplotlib.pyplot as plt

from tiny_game import GAMES, Settings, get_game, GameNames
from runner import run_episode
from config import EPISODES_TRAIN, RESULTS_DIR

# Agents
from agents import AgentList, VDN, DecPOMDPAgent

ALGORITHMS = {
    "VDN" : VDN
}

def train_decpomdp(algorithm : str = "VDN"):
    # --- CONFIGURATION ---
    # Change this to "VDN" or "IQL"
    ALGORITHM = "VDN" 
    
    print(f"\n--- Training DecPOMDP Agents using {ALGORITHM} ---")
    
    # Setup Directories
    base_dir = os.path.join(RESULTS_DIR, ALGORITHM)
    os.makedirs(base_dir, exist_ok=True)

    for game_name in GAMES:
        print(f"Training on {game_name}...")
        
        game = get_game(GameNames(game_name), Settings.decpomdp)
        
        # 1. Create Agents (The "Tabular Agents")
        agent_instances = [
            DecPOMDPAgent(0, game.num_actions),
            DecPOMDPAgent(1, game.num_actions)
        ]
        
        # 2. Create Controller based on Algorithm
        if ALGORITHM == "VDN":
            # Centralized Training Controller
            agents = VDN(agent_instances)
        else:
            # Decentralized Default Controller (IQL)
            agents = AgentList(agent_instances)

        # 3. Training Loop
        rewards = []
        start_states = game.start_states()

        for ep in range(EPISODES_TRAIN):
            # Random card deal
            s = start_states[np.random.randint(len(start_states))]
            
            # Run Episode
            r = run_episode(game, agents, start_state=s, test_episode=False)
            rewards.append(r)
            
            # TRIGGER TRAINING
            # If agents=AgentList, it calls agent.train() -> IQL
            # If agents=VDN, it calls VDN.train() -> Centralized Update
            agents.train()
            
            # Decay
            for a in agent_instances:
                a.decay_epsilon(0.9995, 0.05)

        # Save Plot
        plt.figure()
        plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
        plt.title(f"{ALGORITHM} on {game_name}")
        plt.savefig(os.path.join(base_dir, f"{game_name}_train.png"))
        plt.close()
        
    print("Done.")