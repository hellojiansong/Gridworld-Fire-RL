import numpy as np


class RandomAgent:
    """Selects uniformly random actions. Useful as a baseline opponent."""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def select_action(self, state=None, info=None) -> int:
        return int(self.rng.integers(0, 4))
