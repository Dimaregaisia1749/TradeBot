from app.strategies.ABCStrategy import ABCStrategy
from app.strategies.base import BaseStrategy

from typing import Dict, AnyStr

class UnsupportedStrategyError(Exception):
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name

    def __str__(self):
        return f"Strategy {self.strategy_name} is not supported"

strategies: Dict[str, BaseStrategy.__class__] = {
    "abc": ABCStrategy,
}

def resolve_strategy(strategy_name: str, figi: str, *args, **kwargs) -> BaseStrategy:
    """
    Creates strategy instance by strategy name. Passes all arguments to strategy constructor.
    
    :return: strategy instance.
    """
    if strategy_name not in strategies:
        raise UnsupportedStrategyError(strategy_name)
    return strategies[strategy_name](figi=figi, *args, **kwargs)