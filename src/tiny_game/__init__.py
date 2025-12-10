# tiny_game/__init__.py

#A Supper Simple Wrapper for the tiny-hanabi github repo.
#Easier to import and to get an overview of relevant code


from tiny_hanabi.game.payoff_matrices import (
    GameNames,PAYOFFS,OPTIMAL_RETURNS
)

from tiny_hanabi.game.settings import (
    statetype, Settings, Game,
    DecPOMDP
)

from tiny_hanabi.game.assemblers import (
    get_game, normalize_payoffs
)

GAMES = ["A", "B", "C", "D", "E", "F"]

__all__ = [
    "GameNames", "PAYOFFS","OPTIMAL_RETURNS",
    "statetype", "Settings", "Game",
    "DecPOMDP",
    "get_game", "normalize_payoffs",

    "GAMES"
]