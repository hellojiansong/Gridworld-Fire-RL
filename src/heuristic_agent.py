"""Greedy heuristic agent for GridWorldFireGame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
import numpy as np


Position = Tuple[int, int]


@dataclass
class HeuristicWeights:
    """Tunable weights for the greedy move scorer."""

    # Hard constraints / strong penalties
    invalid_move: float = -10_000.0
    predicted_collision: float = -200.0

    # Direct rewards / penalties from the environment
    booster: float = 5.0
    enemy_fire: float = -20.0
    own_fire: float = -5.0
    cliff: float = -20.0

    # Extra heuristic terms
    mobility: float = 2.0
    avoid_enemy_position: float = -100.0
    prefer_center: float = 0.1
    prefer_high_ground: float = 0.25
    chase_when_safe: float = 0.0


class HeuristicAgent:
    """
    Greedy heuristic agent.

    Usage:
        agent = HeuristicAgent(player_id=1)
        action = agent.select_action(state, info)

    The agent does not modify the environment. It only reads the observation.
    """

    ACTIONS: Dict[int, Position] = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    ACTION_NAMES: Dict[int, str] = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(
        self,
        player_id: int,
        weights: Optional[HeuristicWeights] = None,
        seed: Optional[int] = None,
    ):
        if player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2.")

        self.player_id = player_id
        self.opponent_id = 2 if player_id == 1 else 1
        self.weights = weights or HeuristicWeights()
        self.rng = np.random.default_rng(seed)

    def select_action(
        self,
        state: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        opponent_action_estimate: Optional[int] = None,
    ) -> int:
        """
        Choose the greedy action.

        Args:
            state:
                4-channel observation from env.get_state().
            info:
                Optional env.info() dictionary. If provided, positions are read from it.
                If omitted, positions are inferred from state[1].
            opponent_action_estimate:
                Optional predicted opponent action. If given, the scorer can penalize
                predicted same-tile collisions or swaps.

        Returns:
            Action id: 0, 1, 2, or 3.
        """
        self._validate_state(state)

        scores = {
            action: self.score_action(
                state=state,
                action=action,
                info=info,
                opponent_action_estimate=opponent_action_estimate,
            )
            for action in self.ACTIONS
        }

        best_score = max(scores.values())
        best_actions = [a for a, s in scores.items() if s == best_score]

        # Random tie-break avoids deterministic loops.
        return int(self.rng.choice(best_actions))

    def score_action(
        self,
        state: np.ndarray,
        action: int,
        info: Optional[Dict[str, Any]] = None,
        opponent_action_estimate: Optional[int] = None,
    ) -> float:
        """
        Score a single action from the current observation.

        This is intentionally simple and easy to extend.
        """
        self._validate_state(state)

        if action not in self.ACTIONS:
            return self.weights.invalid_move

        terrain, _, fire, boosters = state
        grid_size = terrain.shape[0]

        my_pos = self.get_player_position(state, self.player_id, info)
        opp_pos = self.get_player_position(state, self.opponent_id, info)

        if my_pos is None:
            return self.weights.invalid_move

        target = self.apply_action(my_pos, action)

        # Border check
        if not self.inside_grid(target, grid_size):
            return self.weights.invalid_move

        r, c = my_pos
        nr, nc = target

        current_height = int(terrain[r, c])
        target_height = int(terrain[nr, nc])
        elevation_diff = target_height - current_height

        # Wall check: cannot climb more than one elevation step.
        if elevation_diff > 1:
            return self.weights.invalid_move

        score = 0.0

        # Hazard and reward signals
        if elevation_diff < -1:
            score += self.weights.cliff

        fire_owner = self.fire_owner_at(fire[nr, nc])

        if fire_owner == self.opponent_id:
            # In the environment, timer-1 fire still counts for the move
            # because fire ownership is snapshotted before decay.
            score += self.weights.enemy_fire
        elif fire_owner == self.player_id:
            score += self.weights.own_fire

        if boosters[nr, nc] > 0.5:
            score += self.weights.booster

        # Avoid stepping into the opponent's current position.
        if opp_pos is not None and target == opp_pos:
            score += self.weights.avoid_enemy_position

        # Optional predicted simultaneous-collision penalty.
        if opponent_action_estimate is not None and opp_pos is not None:
            opp_target = self.apply_action(opp_pos, opponent_action_estimate)
            if self.inside_grid(opp_target, grid_size):
                # Same target tile
                if target == opp_target:
                    score += self.weights.predicted_collision

                # Swap positions
                if target == opp_pos and opp_target == my_pos:
                    score += self.weights.predicted_collision

        # Mobility and positioning
        score += self.weights.mobility * self.count_safe_neighbors(state, target)
        score += self.weights.prefer_high_ground * target_height
        score += self.weights.prefer_center * self.center_score(target, grid_size)

        if self.weights.chase_when_safe != 0.0 and opp_pos is not None:
            score += self.weights.chase_when_safe * self.negative_manhattan(target, opp_pos)

        return float(score)

    def get_player_position(
        self,
        state: np.ndarray,
        player_id: int,
        info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Position]:
        """
        Get player position.

        Prefer info["players"][player_id]["position"] because it is exact.
        Fall back to state[1] if info is not provided.
        """
        if info is not None:
            try:
                return tuple(info["players"][player_id]["position"])  # type: ignore[return-value]
            except (KeyError, TypeError):
                pass

        player_channel = state[1]

        if player_id == 1:
            positions = np.argwhere(player_channel > 0)
        else:
            positions = np.argwhere(player_channel < 0)

        if len(positions) == 0:
            return None

        row, col = positions[0]
        return int(row), int(col)

    def count_safe_neighbors(self, state: np.ndarray, pos: Position) -> int:
        """
        Count how many neighboring moves look passable and not immediately dangerous.

        This is a shallow mobility estimate, not a full simulator.
        """
        terrain, _, fire, _ = state
        grid_size = terrain.shape[0]
        r, c = pos
        current_height = int(terrain[r, c])

        safe_count = 0

        for action in self.ACTIONS:
            nr, nc = self.apply_action(pos, action)

            if not self.inside_grid((nr, nc), grid_size):
                continue

            elevation_diff = int(terrain[nr, nc]) - current_height
            if elevation_diff > 1:
                continue

            owner = self.fire_owner_at(fire[nr, nc])
            if owner == self.opponent_id:
                continue

            safe_count += 1

        return safe_count

    def fire_owner_at(self, fire_value: float) -> Optional[int]:
        """
        Decode fire channel value.

        Positive fire belongs to Player 1.
        Negative fire belongs to Player 2.
        Zero means no active visible fire.
        """
        if fire_value > 0:
            return 1
        if fire_value < 0:
            return 2
        return None

    def apply_action(self, pos: Position, action: int) -> Position:
        dr, dc = self.ACTIONS[action]
        return pos[0] + dr, pos[1] + dc

    @staticmethod
    def inside_grid(pos: Position, grid_size: int) -> bool:
        r, c = pos
        return 0 <= r < grid_size and 0 <= c < grid_size

    @staticmethod
    def center_score(pos: Position, grid_size: int) -> float:
        """
        Higher near center, lower near corners.
        """
        center = (grid_size - 1) / 2.0
        r, c = pos
        dist = abs(r - center) + abs(c - center)
        max_dist = 2 * center
        return 1.0 - (dist / max_dist if max_dist > 0 else 0.0)

    @staticmethod
    def negative_manhattan(a: Position, b: Position) -> float:
        """
        Returns negative Manhattan distance.
        Higher value means closer to the target.
        """
        return -float(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    @staticmethod
    def _validate_state(state: np.ndarray) -> None:
        if not isinstance(state, np.ndarray):
            raise TypeError("state must be a numpy.ndarray.")

        if state.ndim != 3:
            raise ValueError("state must have shape (4, grid_size, grid_size).")

        if state.shape[0] != 4:
            raise ValueError("state must have 4 channels.")

        if state.shape[1] != state.shape[2]:
            raise ValueError("state grid must be square.")


