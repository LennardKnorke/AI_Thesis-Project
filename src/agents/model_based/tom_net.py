# /agents/model_based/tom_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterNet(nn.Module):
    """
    Encodes 'Who the agent is' based on past behavior.
    Auxiliary Task: Classify the Agent Type.
    """
    def __init__(self, input_dim, embedding_dim, num_agent_types):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        # Processes a sequence of (Observation, Action) pairs
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
        
        # Auxiliary Head: Predict Agent Type (e.g., 0=Random, 1=Rational)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_agent_types)
        )

    def forward(self, past_episodes):
        """
        Args:
            past_episodes: (Batch, Num_Episodes, Seq_Len, Input_Dim)
        """
        batch_size, num_eps, seq_len, feat_dim = past_episodes.size()
        
        if past_episodes.size(1) == 0:
            return torch.zeros(batch_size, self.embedding_dim).to(past_episodes.device), None

        # Merge Batch and Num_Episodes for parallel processing
        flat_input = past_episodes.view(-1, seq_len, feat_dim)
        
        # Run LSTM
        # We take the final hidden state as the summary of that episode
        _, (h_n, _) = self.lstm(flat_input)
        episode_embeddings = h_n[-1] # (Batch*Num_Episodes, Emb_Dim)
        
        # Reshape back
        episode_embeddings = episode_embeddings.view(batch_size, num_eps, -1)
        
        # Average pooling across all past episodes to get a stable Character profile
        e_char = torch.mean(episode_embeddings, dim=1) # (Batch, Emb_Dim)
        
        # Auxiliary Prediction
        type_logits = self.classifier(e_char)
        
        return e_char, type_logits


class MentalNet(nn.Module):
    """
    Encodes 'What the agent believes right now'.
    Auxiliary Task: Reconstruct the history (Auto-encoder objective).
    """
    def __init__(self, input_dim, embedding_dim, max_seq_len):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, embedding_dim, batch_first=True)
        
        # Auxiliary Head: Reconstruct the input history from the embedding
        # This ensures e_mental captures specific details (like "Did I hint 0?").
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_seq_len * input_dim) # Flattened reconstruction
        )
        return
    
    def forward(self, current_history):
        """
        Args:
            current_history: (Batch, Seq_Len, Input_Dim)
        """
        # Run LSTM
        _, (h_n, _) = self.lstm(current_history)
        e_mental = h_n[-1] # (Batch, Emb_Dim)
        
        # Auxiliary Reconstruction
        reconstruction_flat = self.decoder(e_mental)
        return e_mental, reconstruction_flat


class ToM_WorldModel(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 num_agent_types, 
                 embedding_dim=32,
                 max_history_len=4):
        super().__init__()
        
        # Input dim is usually State + Action (one-hot)
        self.input_dim = state_dim + action_dim
        self.embedding_dim = embedding_dim
        
        # --- 1. Character Head ---
        self.char_net = CharacterNet(self.input_dim, embedding_dim, num_agent_types)
        
        # --- 2. Mental Head ---
        self.mental_net = MentalNet(self.input_dim, embedding_dim, max_history_len)
        
        # --- 3. Prediction Head ---
        # Input: Char_Emb + Mental_Emb + Current_State
        self.pred_input_dim = embedding_dim + embedding_dim + state_dim
        
        self.pred_net = nn.Sequential(
            nn.Linear(self.pred_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Logits for next action
        )
        return

    def forward(self, past_episodes, current_history, current_state):
        """
        Full Forward Pass.
        
        Args:
            past_episodes: (Batch, N_Eps, Seq, Feat) - Context for Character
            current_history: (Batch, Seq, Feat) - Context for Mental State
            current_state: (Batch, State_Dim) - Immediate Context (e.g. Partner Hand)
            
        Returns:
            action_logits: Prediction of partner's next move.
            aux_data: Dict containing (type_logits, reconstruction) for loss calculation.
        """
        if past_episodes.size(1) == 0:
            # No past knowledge: Use zero vector (or learnable parameter)
            batch_size = current_history.size(0)
            e_char = torch.zeros(batch_size, self.embedding_dim).to(current_history.device)
            type_logits = None # Cannot classify without data
        else:
            e_char, type_logits = self.char_net(past_episodes)
        # 1. Get Character
        e_char, type_logits = self.char_net(past_episodes)
        
        # 2. Get Mental State
        e_mental, history_recon = self.mental_net(current_history)
        
        # 3. Predict Action
        combined = torch.cat([e_char, e_mental, current_state], dim=1)
        action_logits = self.pred_net(combined)
        
        return action_logits, {
            "agent_type_logits": type_logits,
            "history_recon": history_recon
        }