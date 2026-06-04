# tiny_game/my_hanabi.py
import numpy as np
from typing import Any

from tiny_hanabi.game.settings import Game 


class MyHanabi(Game):
    """
    Playing Hanabi with one color and of N numbers.
    """
    def __init__(self, payoffs: np.ndarray = None, normalize : bool = False):
        # Ignore payoffs and optimal return parameter
        self.num_cards = 4          # 4 unique cards in game (0, 1, 2, 3)
        self.cards_in_hand = 2          # Two cards per player
        self.NO_CARD = 4
        self.NO_ACTION = 4
        self.num_actions = 4        # (PlayC0, PlayC1, DelcareSaveC0, DeclareSaveC1)
        self.normalize = normalize

        # 
        self.horizon = 4 + 4 + 4 #Max horizone (4 start cards + four save declerations + 4 cards played)
        self.payoffs = None
        self.optimal_return = 4.0
        
        # Relevant attributes during the game
        self.current_pile = []  # Discarded Card (Actual Integers discarded)
        self.history = []       # (P0C0, P0C1, P1C0, P1C1, P0A0, P1A0, P0A1...)
        self.start_state = ()   # (P0C0, P0C1, P1C0, P1C1)
        self.cards_p0 = ()      # (P0C0, P0C1)
        self.cards_p1 = ()      # (P1C0, P1C1)
        self.timestep = 0
        return

    def start_states(self) -> list:
        """Return tuple of all possible start states for the game"""
        start_states = []
        for p1_c1 in range(self.num_cards):
            for p1_c2 in range(self.num_cards):
                for p2_c1 in range(self.num_cards):
                    for p2_c2 in range(self.num_cards):
                        start_cards = (p1_c1, p1_c2, p2_c1, p2_c2)
                        # Now duplicate cards allowed
                        if len(start_cards) != len(set(start_cards)): continue
                        # Avoid duplicates
                        if start_cards in start_states: continue
                        start_states.append(start_cards)
        return start_states
    
    def random_start(self):
        """ Init environment with a random start state """
        # Pick random start
        start_states = self.start_states()
        self.start_state = start_states[np.random.choice(range(len(start_states)))]
        
        # Copy Start
        self.history = [v for v in self.start_state]
        self.cards_p0 = tuple(self.history[:2])
        self.cards_p1 = tuple(self.history[2:])

        # Reset
        self.timestep = 0
        self.current_pile.clear()
        return
    
    def is_terminal(self) -> bool:
        # Game done pile is full or history is too long
        if len(self.current_pile) == self.cards_in_hand * 2 or len(self.history) == self.horizon:
            return True
        else:
            return False
    
    def payoff(self) -> float:
        # Reward only at the end
        if self.is_terminal():
            return self.calc_final_reward()
        else:
            return 0.0
        
    def calc_final_reward(self)->float:
        """ Calc Reward. +1 for every two cards in correct order"""
        last_card = None
        count = 0.0
        for card in self.current_pile:
            # Increase Reward
            if (last_card is None and card == 0) or (last_card is not None and card == last_card + 1):
                count += 1.0
                last_card = card
            #else:
            #    break
        return float(count/self.optimal_return) if self.normalize else count

    
    def step(self, action : int)->None:
        """Transition"""
        # Determine the current Player
        p0_plays = (len(self.history) % 2 == 0)

        # Plays a card
        if 0 <= action < self.cards_in_hand:

            # Determine the card played
            if p0_plays:
                card_played = self.cards_p0[action]
            else:
                card_played = self.cards_p1[action]
            
            # Error catch - Card already Played
            if card_played in self.current_pile:
                raise ValueError("Invalid action provided! Card Already Played. Can't play card again.")

            obs = (action, card_played)
            # Put card in pile and save action in history
            self.current_pile.append(card_played)
            #self.history.append(action)

        # Declares Partner Card as Save
        elif self.cards_in_hand <= action < self.cards_in_hand * 2:
            # Determine the declared card
            save_card_idx = action - 2
            if p0_plays:
                save_card = self.cards_p1[save_card_idx]
            else:
                save_card = self.cards_p0[save_card_idx]

            # Error catch - Card not on hand but already in pile!
            if save_card in self.current_pile:
                raise ValueError("Invalid action provided! Card already played. Can't declare card as save.")
            
            # Save action - No card added to pile
            obs = (action, self.NO_CARD)
            #self.history.append(action)
        # Invalid action provided
        else:
            raise ValueError("Invalid action provided! Action index out of bound.")
        self.timestep += 1
        self.history.append(obs)
        return
    
    def num_legal_actions(self, full_history : tuple|None = None):
        if full_history is None:
            full_history = self.history
        
        actions_taken = full_history[4:]
        
        # Recreate which cards still exist on hand
        p0_hand = [True, True]
        p1_hand = [True, True]
        is_p0_turn = True
        for a in actions_taken:
            if type(a) == tuple:
                a = a[0]
            if a < 2:
                if is_p0_turn:
                    p0_hand[a] = False
                else:
                    p1_hand[a] = False
            is_p0_turn = not is_p0_turn

        current_player_is_p0 = is_p0_turn

        if current_player_is_p0:
            my_hand = p0_hand
            partner_hand = p1_hand
        else:
            my_hand = p1_hand
            partner_hand = p0_hand

        actions_mask = [0,0,0,0]
        actions_list = []

        for i in range(2):
            if my_hand[i]: 
                actions_mask[i] = 1
                actions_list.append(i)
        
        for i in range(2):
            action_idx = 2 + i
            if partner_hand[i]: 
                actions_mask[action_idx] = 1
                actions_list.append(action_idx)

        return tuple(actions_mask), tuple(actions_list)
    
    def context(self):
        """Used later to get the current step"""
        return tuple(self.history)
    
    def episode(self) -> list:
        """Return the history and the payoff"""
        return self.history + [self.payoff()]
    
    def reset(self, history : list|None = None) -> None:
        if history is None:
            self.random_start()
        else:
            assert len(history) >= 4
            assert not -1 in history[:4], "Can't initialize with unknown cards"
            start_state_list = history[:4]

            self.history = list(history)

            self.start_state = tuple(start_state_list)

            self.cards_p0 = tuple(start_state_list[:2])
            self.cards_p1 = tuple(start_state_list[2:])

            self.current_pile.clear()
            self.timestep = 0

            observations = self.history[4:]
            p0_turn = True 
            
            for action, card_revealed in observations:
                # Play Actions (0 or 1)
                if action < 2:
                    if p0_turn:
                        card = self.cards_p0[action]
                    else:
                        card = self.cards_p1[action]

                    if card != card_revealed:
                        raise ValueError("Faulty History Provided. Revealed card + card on hand do not match")
                    
                    self.current_pile.append(card)
                
                # Turn passes
                p0_turn = not p0_turn
                self.timestep += 1
            