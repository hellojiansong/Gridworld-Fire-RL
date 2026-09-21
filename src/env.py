import os
import numpy as np
from generate_hill_map import HillMapGenerator

class GridWorldFireGame:
    """
    Two-player simultaneous Grid World fire-trail game.
    Both players execute an action simultaneously every turn.

    Actions:
        0 = up
        1 = down
        2 = left
        3 = right

    State shape:
        (4, grid_size, grid_size)

    State channels:
        state[0] = terrain elevations in range [0, 5]

        state[1] = player positions with normalized lives
            0.0                 = empty
            lives / max_lives   = player 1
            -lives / max_lives  = player 2

        state[2] = fire trails
            0      = no fire
            10..1  = player 1 fire timer
            -10..-1 = player 2 fire timer

        state[3] = boosters
            0.0 = no booster
            1.0 = booster
    """

    # Movement is encoded as row/column deltas. Moving up means the row
    # index decreases, moving down means it increases, and columns behave
    # the same way for left/right.
    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    # Human-readable names for logging, rendering, and debugging.
    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(
        self,
        grid_size=10,
        player_lives=1,
        booster_prob=0.001,
        max_episode_length=500,
        render=False,
        player1_type="agent",
        player2_type="agent",
        seed=None,
        debug=False,
        reward_method = "default",
        validate_maps=True,
    ):
        # Keep the environment inside the project requirements. Many parts of
        # the renderer and terrain generator assume a square grid of this size.
        if not (10 <= grid_size <= 50):
            raise ValueError("grid_size must be between 10 and 50.")

        # Both players always start with the same positive number of lives.
        if player_lives < 1:
            raise ValueError("player_lives must be at least 1.")

        # The episode must have at least one simultaneous step.
        if max_episode_length < 1:
            raise ValueError("max_episode_length must be at least 1.")

        # A player can either be controlled through terminal input or by code.
        if player1_type not in ["agent", "human"]:
            raise ValueError("player1_type must be 'agent' or 'human'.")

        if player2_type not in ["agent", "human"]:
            raise ValueError("player2_type must be 'agent' or 'human'.")

        valid_reward_methods = ["default", "offensive", "defensive", "territorial"]

        if reward_method not in valid_reward_methods:
            raise ValueError(
                f"reward_method must be one of {valid_reward_methods}."
            )

        if booster_prob < 0:
            raise ValueError("booster_prob must be non-negative.")
        if booster_prob*5 > 1:
            raise ValueError("booster_prob is too high. Booster_prob*elevation_range will exceed 1.")
        
        # The players location state channel values are the ratio of current lives to maximum lives.
        # Fixed at 3 to keep the player-channel values in a consistent range regardless of starting_lives.
        self.max_lives = 3
        if player_lives > self.max_lives:
            raise ValueError("player_lives must be at most 3.")

        # Store configuration values in normalized integer form.
        self.grid_size = int(grid_size)
        self.starting_lives = int(player_lives)
        self.booster_prob = float(booster_prob)
        self.max_episode_length = int(max_episode_length)
        self.debug = bool(debug)
        self.reward_method = reward_method
        # When False, skip the (expensive) map traversability check in
        # generate_terrain(). Used for faster training when seeds are not baked.
        self.validate_maps = bool(validate_maps)
        self.map_engine = HillMapGenerator(grid_size=self.grid_size, starting_lives=self.starting_lives)
        
        # starting burn time for fire trails used in _burn_time_for_cell_P1/2, are increased by 1 with each eaten booster.
        self.base_burn_time_P1 = 10
        self.burn_time_P1 = 10
        self.base_burn_time_P2 = 10
        self.burn_time_P2 = 10

        # Load the seed registry into memory once
        self.registry_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "map_seeds", f"valid_seeds_{self.grid_size}.npy"
        )
        if os.path.exists(self.registry_path):
            self.valid_seeds_pool = np.load(self.registry_path)
        else:
            # Only warn when debugging — otherwise this spams once per env created
            # (every imitation episode / vec-env worker) when seeds are not baked.
            if self.debug:
                print(f"Warning: {self.registry_path} not found! Run the map baker script first. Falling back to real-time generation.")
            self.valid_seeds_pool = None

        # NumPy's Generator gives reproducible terrain, booster, and random
        # action behavior when a seed is provided.
        self.rng = np.random.default_rng(seed)

        # Remember whether each player should ask for keyboard input or choose
        # an automatic random action when no action is passed to step().
        self.player_types = {
            1: player1_type,
            2: player2_type,
        }

        # Human games always render so the player can see the board. Agent-only
        # games render only if requested explicitly.
        self.render_enabled = (
            render
            or player1_type == "human"
            or player2_type == "human"
        )

        # Player 1 starts top-left. Player 2 starts bottom-right.
        self.start_positions = {
            1: (0, 0),
            2: (self.grid_size - 1, self.grid_size - 1),
        }

        # These arrays/dictionaries are created in reset(). Keeping them as
        # None before reset lets _check_has_been_reset() catch invalid use.
        self.terrain = None
        self.trails = None
        self.powerups = None
        self.player_positions = None
        self.lives = None

        # Episode-level status fields.
        self.current_player = "both"
        self.done = False
        self.winner = None
        self.turn_count = 0

        # Last-step debug fields. These are useful for render(), info(), and
        # training/debugging scripts that inspect the most recent transition.
        self.last_actor = None
        self.last_actions = {1: None, 2: None}
        self.last_action = None
        self.last_events = {1: None, 2: None}
        self.last_event = None
        self.last_rewards = {1: 0, 2: 0}
        self.last_reward = {1: 0, 2: 0}
        self.last_move_details = {1: None, 2: None}

        # Per-player cumulative statistics.
        self.powerups_eaten = {1: 0, 2: 0}
        self.reward_history = {1: [], 2: []}

        # More detailed counters for analysis after an episode.
        self.wall_or_border_hits = {1: 0, 2: 0}
        self.cliff_hits = {1: 0, 2: 0}
        self.enemy_fire_hits = {1: 0, 2: 0}
        self.player_collision_hits = {1: 0, 2: 0}
        self.own_fire_conversions = {1: 0, 2: 0}
        self.enemy_fire_conversions = {1: 0, 2: 0}

        # Tracks whose fire a player is currently standing on. This matters
        # because leaving a fire tile can convert ownership of that tile.
        self.standing_on_fire_owner = {
            1: None,
            2: None,
        }

    def generate_terrain(self):
        """Pulls a random valid map seed from the pre-generated registry pool."""
        if getattr(self, 'valid_seeds_pool', None) is not None:
            # Pick a random verified seed out of pre-computed seeds
            chosen_seed = int(self.rng.choice(self.valid_seeds_pool))
            self.map_seed = chosen_seed
            return self.map_engine.generate(chosen_seed)
        else:
            # Fallback dynamic loop if registry file is missing
            while True:
                fallback_seed = int(self.rng.integers(0, 99999999))
                terrain = self.map_engine.generate(fallback_seed)
                # Bypass the O(grid^4) traversability check when validation is
                # disabled — the first generated map is accepted immediately.
                if not self.validate_maps or self.map_engine.is_valid(terrain):
                    self.map_seed = fallback_seed
                    return terrain

    def reset(self):
        """
        Resets the entire environment to the starting state.
        Must be called at the beginning of every new episode.
        """
        # Build a fresh terrain map for each new episode.
        self.terrain = self.generate_terrain()

        # Validate the terrain generator output. These checks make it easier to
        # catch mistakes if generate_terrain() is changed later.
        if self.terrain.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"generate_terrain() must return shape ({self.grid_size}, {self.grid_size})."
            )

        if np.any(self.terrain < 0) or np.any(self.terrain > 5):
            raise ValueError("Terrain values must be in the range [0, 5].")

        # Each player has a separate positive timer matrix for their own fire.
        self.trails = {
            1: np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
            2: np.zeros((self.grid_size, self.grid_size), dtype=np.int32),
        }

        # Restore players and lives to the configured start state.
        self.player_positions = dict(self.start_positions)

        self.lives = {
            1: self.starting_lives,
            2: self.starting_lives,
        }

        # Clear episode status.
        self.current_player = "both"
        self.done = False
        self.winner = None
        self.turn_count = 0

        # Reset burn times to base values at the start of each episode.
        self.burn_time_P1 = self.base_burn_time_P1
        self.burn_time_P2 = self.base_burn_time_P2

        # Clear last-step debug data.
        self.last_actor = None
        self.last_actions = {1: None, 2: None}
        self.last_action = None
        self.last_events = {1: None, 2: None}
        self.last_event = "reset"
        self.last_rewards = {1: 0, 2: 0}
        self.last_reward = {1: 0, 2: 0}
        self.last_move_details = {1: None, 2: None}

        # Clear cumulative statistics.
        self.powerups_eaten = {1: 0, 2: 0}
        self.reward_history = {1: [], 2: []}

        self.wall_or_border_hits = {1: 0, 2: 0}
        self.cliff_hits = {1: 0, 2: 0}
        self.enemy_fire_hits = {1: 0, 2: 0}
        self.player_collision_hits = {1: 0, 2: 0}
        self.own_fire_conversions = {1: 0, 2: 0}
        self.enemy_fire_conversions = {1: 0, 2: 0}

        self.standing_on_fire_owner = {
            1: None,
            2: None,
        }

        # Create initial boosters according to terrain elevation probabilities.
        self.powerups = self._generate_powerups()

        # Do not allow a booster to appear underneath a player at spawn.
        for position in self.player_positions.values():
            r, c = position
            self.powerups[r, c] = False

        # Show the initial board when rendering is enabled.
        if self.render_enabled:
            self.render()

        # Return the observation expected by agents.
        return self.get_state()

    def step(self, action_player1=None, action_player2=None):
        """
        Executes one simultaneous game step.

        Returns:
            state, reward_player_1, reward_player_2, done, info
        """
        # The environment must be initialized with reset() before it can step.
        self._check_has_been_reset()

        # Once an episode is done, repeated calls do not change the state.
        if self.done:
            return self.get_state(), 0, 0, True, self.info()

        # Store both submitted actions by player id. A value of None means that
        # the environment should choose or ask for the action below.
        actions = {
            1: action_player1,
            2: action_player2,
        }

        for player in [1, 2]:
            # Missing human actions are requested from the terminal. Missing
            # agent actions are sampled randomly so step() remains usable.
            if actions[player] is None:
                if self.player_types[player] == "human":
                    actions[player] = self._ask_human_action(player)
                else:
                    actions[player] = self._random_action()

            # Reject actions outside the four movement ids.
            if actions[player] not in self.ACTIONS:
                raise ValueError(
                    f"Invalid action for player {player}. "
                    "Use 0:up, 1:down, 2:left, 3:right."
                )

        # Save the chosen actions immediately for debugging and rendering.
        self.last_actor = "both"
        self.last_actions = dict(actions)
        self.last_action = dict(actions)

        # Reward signals are filled during the step.
        # The selected reward method converts them into numerical rewards later.
        reward_signals = self._new_reward_signals()
        rewards = {1: 0, 2: 0}

        # Events summarize the outcome of this step for each player.
        events = {1: None, 2: None}

        # Keep a copy of old positions because simultaneous movement decisions
        # must be resolved from the same starting state.
        old_positions = {
            1: self.player_positions[1],
            2: self.player_positions[2],
        }

        move_details = {}

        for player in [1, 2]:
            old_r, old_c = old_positions[player]

            # This dictionary captures a detailed transition trace for render()
            # and info(). Values are filled in as each pass resolves.
            move_details[player] = {
                "action": actions[player],
                "action_name": self.ACTION_NAMES[actions[player]],
                "old_position": old_positions[player],
                "old_elevation": int(self.terrain[old_r, old_c]),
                "intended_position": None,
                "target_elevation": None,
                "elevation_difference": None,
                "final_position": old_positions[player],
                "final_elevation": int(self.terrain[old_r, old_c]),
                "blocked": False,
                "blocked_reason": None,
                "moved": False,
                "stepped_off_cliff": False,
                "stepped_on_enemy_fire": False,
                "stepped_on_own_fire": False,
                "picked_booster": False,
                "lives_before": int(self.lives[player]),
                "lives_lost": 0,
                "lives_after": int(self.lives[player]),
                "reward": None,
                "event": None,
            }
        # Snapshot fire ownership before timers tick down.
        # This lets fire with timer 1 still affect a player who moves onto it this step,
        # even though it expires before the returned next state.
        fire_owner_before_decay = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.int8,
        )
        fire_owner_before_decay[self.trails[1] > 0] = 1
        fire_owner_before_decay[self.trails[2] > 0] = 2

        # Both fire trails tick down once per simultaneous environment step.
        self._decrement_player_trail(1)
        self._decrement_player_trail(2)

        intended_positions = {}
        final_positions = {}
        blocked = {1: False, 2: False}
        moved = {1: False, 2: False}
        fire_owner_on_entry = {1: None, 2: None}
        blocked_fire_damage_owner = {1: None, 2: None}

        elevation_differences = {1: 0, 2: 0}
        current_elevations = {1: None, 2: None}
        target_elevations = {1: None, 2: None}

        # ---------------------------------------------------------
        # PASS 1: Determine intended positions and check borders/walls
        # ---------------------------------------------------------
        for player in [1, 2]:
            old_r, old_c = old_positions[player]
            dr, dc = self.ACTIONS[actions[player]]
            new_r = old_r + dr
            new_c = old_c + dc

            # The intended position is the raw target before collision checks.
            intended_positions[player] = (new_r, new_c)
            move_details[player]["intended_position"] = (new_r, new_c)

            # Moving outside the grid is blocked and penalized.
            if not self._inside_grid(new_r, new_c):
                blocked[player] = True
                final_positions[player] = old_positions[player]
                self.wall_or_border_hits[player] += 1
                reward_signals[player]["blocked_move"] = True
                events[player] = f"player_{player}_hit_grid_border"

                move_details[player]["blocked"] = True
                move_details[player]["blocked_reason"] = "grid_border"
                continue

            # Elevation difference determines whether the target is a wall,
            # flat/uphill/downhill, or a cliff.
            current_elevation = int(self.terrain[old_r, old_c])
            target_elevation = int(self.terrain[new_r, new_c])
            elevation_difference = target_elevation - current_elevation

            current_elevations[player] = current_elevation
            target_elevations[player] = target_elevation
            elevation_differences[player] = elevation_difference

            move_details[player]["target_elevation"] = target_elevation
            move_details[player]["elevation_difference"] = elevation_difference

            # Climbing more than one level is a wall, so the player stays put.
            if elevation_difference > 1:
                blocked[player] = True
                final_positions[player] = old_positions[player]
                self.wall_or_border_hits[player] += 1
                reward_signals[player]["blocked_move"] = True
                events[player] = (
                    f"player_{player}_hit_wall_"
                    f"from_{current_elevation}_to_{target_elevation}"
                )

                move_details[player]["blocked"] = True
                move_details[player]["blocked_reason"] = "wall"
                continue

            # If border and wall checks pass, the move is tentatively valid.
            final_positions[player] = intended_positions[player]

        # ---------------------------------------------------------
        # PASS 2: Process simultaneous collisions
        # ---------------------------------------------------------
        # If both unblocked players target the same tile, neither moves.
        if (
            not blocked[1]
            and not blocked[2]
            and intended_positions[1] == intended_positions[2]
        ):
            for player in [1, 2]:
                blocked[player] = True
                final_positions[player] = old_positions[player]
                self.player_collision_hits[player] += 1
                reward_signals[player]["blocked_move"] = True
                events[player] = "both_players_tried_to_enter_same_tile"

                move_details[player]["blocked"] = True
                move_details[player]["blocked_reason"] = "same_tile_collision"

        # If both unblocked players try to trade places, neither moves.
        elif (
            not blocked[1]
            and not blocked[2]
            and intended_positions[1] == old_positions[2]
            and intended_positions[2] == old_positions[1]
        ):
            for player in [1, 2]:
                blocked[player] = True
                final_positions[player] = old_positions[player]
                self.player_collision_hits[player] += 1
                reward_signals[player]["blocked_move"] = True
                events[player] = "both_players_tried_to_swap_positions"

                move_details[player]["blocked"] = True
                move_details[player]["blocked_reason"] = "swap_collision"

        for player in [1, 2]:
            opponent = self._opponent(player)

            # If the opponent is blocked and remains in their old tile, the
            # current player cannot move into that still-occupied tile.
            if (
                not blocked[player]
                and blocked[opponent]
                and intended_positions[player] == old_positions[opponent]
            ):
                blocked[player] = True
                final_positions[player] = old_positions[player]
                self.player_collision_hits[player] += 1
                reward_signals[player]["blocked_move"] = True
                events[player] = (
                    f"player_{player}_tried_to_move_into_"
                    f"player_{opponent}_blocked_position"
                )

                move_details[player]["blocked"] = True
                move_details[player]["blocked_reason"] = (
                    f"opponent_{opponent}_blocked_position"
                )

        # ---------------------------------------------------------
        # PASS 3: Process movement and fire generation
        # ---------------------------------------------------------
        for player in [1, 2]:
            old_r, old_c = old_positions[player]

            # A blocked move is still an attempt to leave the tile, so it can
            # create or transform fire under the player.
            if blocked[player]:
                previous_owner = int(fire_owner_before_decay[old_r, old_c])
                previous_owner = previous_owner if previous_owner != 0 else None
                blocked_fire_damage_owner[player] = self._apply_blocked_leave_fire(
                    player,
                    old_r,
                    old_c,
                    previous_owner,
                )
                continue

            self._leave_current_tile(player, old_r, old_c)
            moved[player] = True
            move_details[player]["moved"] = True

        # Commit all final positions after all leave-tile fire has been created.
        for player in [1, 2]:
            self.player_positions[player] = final_positions[player]

            final_r, final_c = final_positions[player]
            move_details[player]["final_position"] = final_positions[player]
            move_details[player]["final_elevation"] = int(self.terrain[final_r, final_c])

        for player in [1, 2]:
            # Only update entry fire memory for players who actually moved.
            # Blocked players stay on their current tile and keep their previous memory.
            if not moved[player]:
                continue

            r, c = self.player_positions[player]

            current_owner = self._fire_owner_at(r, c)

            previous_owner = int(fire_owner_before_decay[r, c])
            previous_owner = previous_owner if previous_owner != 0 else None

            # Current fire has priority. This includes fire newly created this step.
            if current_owner is not None:
                entry_owner = current_owner
            else:
                entry_owner = previous_owner

            fire_owner_on_entry[player] = entry_owner

            # This owner is remembered for future leave-conversion behavior.
            self.standing_on_fire_owner[player] = entry_owner

            # Stepping onto own fire still consumes it. Enemy fire remains active
            # and keeps counting down normally while a player stands on it.
            if entry_owner == player:
                self.trails[1][r, c] = 0
                self.trails[2][r, c] = 0

        # ---------------------------------------------------------
        # PASS 4: Consequences of movement
        # ---------------------------------------------------------
        for player in [1, 2]:
            opponent = self._opponent(player)

            if blocked[player]:
                damage_owner = blocked_fire_damage_owner[player]

                if damage_owner == opponent:
                    self.enemy_fire_hits[player] += 1
                    self.lives[player] = max(0, self.lives[player] - 1)

                    reward_signals[player]["stepped_on_enemy_fire"] = True
                    reward_signals[opponent]["opponent_stepped_on_my_fire"] = True

                    move_details[player]["stepped_on_enemy_fire"] = True
                    move_details[player]["lives_lost"] = 1

                    blocked_fire_event = f"blocked_on_player_{opponent}_fire"
                    if events[player] is None:
                        events[player] = f"player_{player}_{blocked_fire_event}"
                    else:
                        events[player] += f"_and_{blocked_fire_event}"

                continue

            r, c = self.player_positions[player]

            # Downhill drops greater than one elevation count as cliffs.
            stepped_off_cliff = moved[player] and elevation_differences[player] < -1

            # Enemy fire causes damage. Own fire is safe for lives but has a
            # separate reward penalty.
            stepped_on_enemy_fire = (
                moved[player]
                and fire_owner_on_entry[player] == opponent
            )

            stepped_on_own_fire = (
                moved[player]
                and fire_owner_on_entry[player] == player
            )

            move_details[player]["stepped_off_cliff"] = stepped_off_cliff
            move_details[player]["stepped_on_enemy_fire"] = stepped_on_enemy_fire
            move_details[player]["stepped_on_own_fire"] = stepped_on_own_fire

            picked_booster = False

            # Boosters disappear when collected and refresh all active fire
            # timers belonging to the collecting player.
            if self.powerups[r, c]:
                self.powerups[r, c] = False
                self.powerups_eaten[player] += 1

                if player == 1:
                    self.burn_time_P1 += 1
                else:
                    self.burn_time_P2 += 1

                self._refresh_all_player_fire(player)

                picked_booster = True
                reward_signals[player]["picked_booster"] = True

            move_details[player]["picked_booster"] = picked_booster

            # Cliff and enemy-fire hazards stack. A player can lose two lives
            # in one step if both conditions are true. The selected reward method
            # later decides the numerical penalty from reward_signals.
            # If the player steps onto enemy fire, the fire owner receives a
            # reward signal through opponent_stepped_on_my_fire.
            if stepped_off_cliff or stepped_on_enemy_fire:
                lives_lost = 0

                if stepped_off_cliff:
                    self.cliff_hits[player] += 1
                    lives_lost += 1
                    reward_signals[player]["stepped_off_cliff"] = True

                if stepped_on_enemy_fire:
                    self.enemy_fire_hits[player] += 1
                    lives_lost += 1

                    # Penalty for stepping onto enemy fire.
                    reward_signals[player]["stepped_on_enemy_fire"] = True

                    # Reward for the opponent whose fire was stepped on.
                    reward_signals[opponent]["opponent_stepped_on_my_fire"] = True

                self.lives[player] = max(0, self.lives[player] - lives_lost)
                move_details[player]["lives_lost"] = lives_lost

                # Build a readable event label that preserves which hazard or
                # hazards happened this step.
                if stepped_off_cliff and stepped_on_enemy_fire:
                    events[player] = (
                        f"player_{player}_stepped_off_cliff_"
                        f"from_{current_elevations[player]}_"
                        f"to_{target_elevations[player]}_"
                        f"and_on_player_{opponent}_fire"
                    )
                elif stepped_off_cliff:
                    events[player] = (
                        f"player_{player}_stepped_off_cliff_"
                        f"from_{current_elevations[player]}_"
                        f"to_{target_elevations[player]}"
                    )
                else:
                    events[player] = (
                        f"player_{player}_stepped_on_player_{opponent}_fire"
                    )

                if picked_booster:
                    events[player] += "_and_picked_booster"

            # Own fire does not cost lives, but it does apply a reward penalty.
            elif stepped_on_own_fire:
                reward_signals[player]["stepped_on_own_fire"] = True
                events[player] = f"player_{player}_stepped_on_own_fire"

                if picked_booster:
                    events[player] += "_and_picked_booster"

            # If there was no hazard, the event is either booster collection or
            # a normal successful movement.
            else:
                if picked_booster:
                    events[player] = f"player_{player}_picked_booster"
                else:
                    events[player] = (
                        f"player_{player}_moved_"
                        f"{self.ACTION_NAMES[actions[player]]}"
                    )

        # ---------------------------------------------------------
        # PASS 5: Process Win/Loss and Final Terminal Rewards
        # ---------------------------------------------------------
        dead_players = []

        # A player is dead once lives reach zero or below.
        for player in [1, 2]:
            if self.lives[player] == 0:
                dead_players.append(player)

        # Draw / Both players die at the same time
        if len(dead_players) == 2:
            self.done = True
            self.winner = None

            reward_signals[1]["terminal"] = "draw"
            reward_signals[2]["terminal"] = "draw"

            for player in [1, 2]:
                if events[player] is None:
                    events[player] = f"player_{player}_lost_final_life"
                else:
                    events[player] += "_and_lost_final_life"

        # One player dies, the other wins
        elif len(dead_players) == 1:
            loser = dead_players[0]
            winner = self._opponent(loser)

            # If exactly one player dies, the other player wins immediately.
            self.done = True
            self.winner = winner

            reward_signals[loser]["terminal"] = "loss"
            reward_signals[winner]["terminal"] = "win"

            if events[loser] is None:
                events[loser] = f"player_{loser}_lost_final_life"
            else:
                events[loser] += "_and_lost_final_life"

            # Do NOT append "_and_won" to the winner's movement event.
            # The winner may also have stepped on fire or off a cliff but survived.
            # So we keep their movement event separate from the terminal win reason.
            if events[winner] is None:
                events[winner] = (
                    f"player_{winner}_won_because_player_{loser}_lost_final_life"
                )
            else:
                events[winner] = (
                    f"{events[winner]} | "
                    f"player_{winner}_won_because_player_{loser}_lost_final_life"
                )
        # Time-limit result. Use self.turn_count + 1 because the current step
        # has not yet been counted by _finish_turn().
        elif self.turn_count + 1 >= self.max_episode_length:
            self.done = True
            if self.lives[1] > self.lives[2]:
                self.winner = 1
            elif self.lives[2] > self.lives[1]:
                self.winner = 2
            else:
                self.winner = None

            if self.winner is None:
                reward_signals[1]["terminal"] = "draw"
                reward_signals[2]["terminal"] = "draw"
            else:
                loser = self._opponent(self.winner)
                reward_signals[self.winner]["terminal"] = "win"
                reward_signals[loser]["terminal"] = "loss"

            for player in [1, 2]:
                if events[player] is None:
                    events[player] = "max_episode_length_reached"
                else:
                    events[player] += " | max_episode_length_reached"

            if self.winner is not None:
                events[self.winner] += "_and_won_on_lives"
                loser = self._opponent(self.winner)
                events[loser] += "_and_lost_on_lives"

        # Calculate final rewards based on the selected reward method and the
        # reward signals collected during this step.
        rewards = self._calculate_rewards(reward_signals)

        # Finish the per-player move trace now that terminal reward overrides
        # and final lives are known.
        for player in [1, 2]:
            move_details[player]["reward"] = int(rewards[player])
            move_details[player]["event"] = events[player]
            move_details[player]["lives_after"] = int(self.lives[player])

        # Store move_details before _finish_turn() so render() and info() can
        # show complete transition details for this step.
        self.last_move_details = dict(move_details)

        return self._finish_turn(rewards, events)

    def info(self):
        """
        Returns game information and extensive statistics for both players.
        """
        self._check_has_been_reset()

        players_info = {}

        for player in [1, 2]:
            opponent = self._opponent(player)

            # Collect per-player state, score history, and diagnostic counters.
            # These values are useful for training logs and post-game analysis.
            players_info[player] = {
                "position": tuple(self.player_positions[player]),
                "lives_left": int(self.lives[player]),
                "num_powerups_eaten": int(self.powerups_eaten[player]),
                "rewards": list(self.reward_history[player]),
                "total_reward": int(sum(self.reward_history[player])),
                "num_wall_or_border_hits": int(self.wall_or_border_hits[player]),
                "num_cliff_hits": int(self.cliff_hits[player]),
                "num_enemy_fire_hits": int(self.enemy_fire_hits[player]),
                "num_player_collision_hits": int(self.player_collision_hits[player]),
                "num_own_fire_conversions": int(self.own_fire_conversions[player]),
                "num_enemy_fire_conversions": int(self.enemy_fire_conversions[player]),
                "num_burning_tiles": int(np.sum(self.trails[player] > 0)),
                "standing_on_fire_owner": self.standing_on_fire_owner[player],
                "standing_on_own_fire": self.standing_on_fire_owner[player] == player,
                "standing_on_enemy_fire": self.standing_on_fire_owner[player] == opponent,
            }

        # Return a single dictionary containing both global episode state and
        # nested player-specific information.
        return {
            "grid_size": self.grid_size,
            "max_episode_length": self.max_episode_length,
            "turn_count": self.turn_count,
            "current_player": self.current_player,
            "last_actor": self.last_actor,
            "last_action": self.last_action,
            "last_actions": dict(self.last_actions),
            "last_event": self.last_event,
            "last_events": dict(self.last_events),
            "last_reward": dict(self.last_reward),
            "last_rewards": dict(self.last_rewards),
            "last_move_details": dict(self.last_move_details),
            "done": self.done,
            "winner": self.winner,
            "num_powerups_on_grid": int(np.sum(self.powerups)),
            "players": players_info,
        }

    def get_state(self):
        """
        Constructs the 4-channel state.
        Shape: (4, grid_size, grid_size)

        State channels:
            state[0] = terrain elevations in range [0, 5]

            state[1] = player positions with normalized lives
                0.0                 = empty
                lives / max_lives   = player 1
                -lives / max_lives  = player 2

            state[2] = fire trails
                0      = no fire
                10..1  = player 1 fire timer
                -10..-1 = player 2 fire timer

            state[3] = boosters
                0.0 = no booster
                1.0 = booster
        """
        self._check_has_been_reset()

        # Channel 0: terrain elevations
        terrain_channel = self.terrain.astype(np.float32).copy()

        # Channel 1: player positions and normalized lives
        player_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        p1_r, p1_c = self.player_positions[1]
        p2_r, p2_c = self.player_positions[2]

        player_channel[p1_r, p1_c] = self.lives[1] / self.max_lives
        player_channel[p2_r, p2_c] = -self.lives[2] / self.max_lives

        # Channel 2: signed fire trails
        fire_channel = (
            self.trails[1].astype(np.float32)
            - self.trails[2].astype(np.float32)
        )

        # Channel 3: booster mask
        booster_channel = self.powerups.astype(np.float32)

        return np.stack(
            [
                terrain_channel,
                player_channel,
                fire_channel,
                booster_channel,
            ],
            axis=0,
        )

    def render(self):
        """
        Simple terminal renderer for debugging.
        """
        self._check_has_been_reset()

        # Print a header with episode status and the previous step outcome.
        print()
        print("=" * 90)
        print(
            f"Round: {self.turn_count}/{self.max_episode_length} | "
            f"Mode: simultaneous | "
            f"Done: {self.done} | "
            f"Winner: {self.winner} | "
            f"Seed: {getattr(self, 'map_seed', 'Unknown')}"
        )
        print(
            f"Lives -> Player 1: {self.lives[1]} | "
            f"Player 2: {self.lives[2]}"
        )
        print(f"Last actions: {self.last_actions}")
        print(f"Last events: {self.last_events}")
        print(f"Last rewards: {self.last_rewards}")
        print(
            f"Total rewards -> Player 1: {sum(self.reward_history[1])} | "
            f"Player 2: {sum(self.reward_history[2])}"
        )

        # After the first step, print detailed movement diagnostics for each
        # player so blocked moves, hazards, boosters, and rewards are visible.
        if self.last_move_details[1] is not None:
            print("Move details:")

            for player in [1, 2]:
                detail = self.last_move_details[player]

                target_position = detail["intended_position"]
                target_elevation = detail["target_elevation"]

                # Out-of-bounds targets do not have an elevation.
                if target_elevation is None:
                    target_text = f"{target_position} h=OUT"
                else:
                    target_text = f"{target_position} h={target_elevation}"

                print(
                    f"  P{player}: "
                    f"{detail['old_position']} h={detail['old_elevation']} "
                    f"-> {target_text} | "
                    f"final {detail['final_position']} h={detail['final_elevation']} | "
                    f"blocked={detail['blocked']} "
                    f"reason={detail['blocked_reason']} | "
                    f"cliff={detail['stepped_off_cliff']} "
                    f"enemy_fire={detail['stepped_on_enemy_fire']} "
                    f"own_fire={detail['stepped_on_own_fire']} "
                    f"booster={detail['picked_booster']} | "
                    f"lives_lost={detail['lives_lost']} | "
                    f"reward={detail['reward']}"
                )

        print("-" * 90)

        p1_pos = self.player_positions[1]
        p2_pos = self.player_positions[2]

        # Draw each grid cell. Later checks intentionally overwrite earlier
        # ones so visible priority is: player > booster > fire > terrain.
        for r in range(self.grid_size):
            row_items = []

            for c in range(self.grid_size):
                # Start with terrain elevation as the default cell display.
                cell = str(int(self.terrain[r, c]))

                # Player 1 fire is shown as aN, where N is remaining burn time.
                if self.trails[1][r, c] > 0:
                    cell = f"a{int(self.trails[1][r, c])}"

                # Player 2 fire is shown as bN.
                if self.trails[2][r, c] > 0:
                    cell = f"b{int(self.trails[2][r, c])}"

                # Boosters are shown as an asterisk.
                if self.powerups[r, c]:
                    cell = "*"

                # Players overwrite all other symbols at their current location.
                if (r, c) == p1_pos:
                    cell = "A"

                if (r, c) == p2_pos:
                    cell = "B"

                # Right-align cells so single-character and two-character
                # symbols form a readable grid.
                row_items.append(f"{cell:>4}")

            print("".join(row_items))

        print("=" * 90)
        print()

    def debug_print_state(self):
        """
        Prints the full 4-channel state for debugging.
        """

        state = self.get_state()

        channel_names = [
            "terrain elevations",
            "players + normalized lives",
            "fire trails",
            "boosters",
        ]

        print()
        print("=" * 50, "DEBUG", "=" * 50)

        print(f"Turn: {self.turn_count}")
        print(f"Done: {self.done}")
        print(f"Winner: {self.winner}")
        print(f"Lives: P1={self.lives[1]}, P2={self.lives[2]}")
        print(f"Positions: P1={self.player_positions[1]}, P2={self.player_positions[2]}")
        print(f"Last actions: {self.last_actions}")
        print(f"Last events: {self.last_events}")
        print(f"Last rewards: {self.last_rewards}")

        with np.printoptions(precision=2, suppress=True, linewidth=200):
            for i, name in enumerate(channel_names):
                print()
                print(f"Channel {i}: {name}")
                print(state[i])

        print("=" * 100)
        print()

    def _finish_turn(self, rewards, events):
        """
        Handles administration at the end of a turn and applies the time-limit check.
        """
        # Store event and reward summaries for the just-completed step.
        self.last_events = dict(events)
        self.last_event = f"P1: {events[1]} | P2: {events[2]}"

        self.last_rewards = dict(rewards)
        self.last_reward = dict(rewards)

        # Append step rewards to cumulative histories.
        self.reward_history[1].append(int(rewards[1]))
        self.reward_history[2].append(int(rewards[2]))

        # One simultaneous pair of actions counts as one environment turn.
        self.turn_count += 1

        # Spawn boosters after non-terminal turns only.
        if not self.done:
            self._spawn_powerups_after_turn()

        # Render after all state updates so the board matches the returned info.
        if self.render_enabled:
            self.render()

        # Print the full state for debugging
        if self.debug:
            self.debug_print_state()

        # Standard environment return: observation, two rewards, done flag, info.
        return (
            self.get_state(),
            self.last_rewards[1],
            self.last_rewards[2],
            self.done,
            self.info(),
        )

    def _generate_powerups(self):
        """
        Generates a boolean mask for new boosters based on terrain elevation.
        Higher terrain has a greater chance of spawning a booster.
        """

        # The base probability of a booster appearing on a tile based on its elevation.
        probabilities = self.terrain.astype(np.float32) * self.booster_prob

        # A booster appears wherever a uniform random value is below the tile's
        # elevation-based probability.
        random_values = self.rng.random((self.grid_size, self.grid_size))
        return random_values < probabilities

    def _spawn_powerups_after_turn(self):
        """
        Adds newly generated powerups to the existing ones.
        Ensures they do not spawn directly on players.
        """
        # Generate a fresh mask of possible new booster locations.
        new_powerups = self._generate_powerups()

        # Remove any new booster that would appear under a player.
        for position in self.player_positions.values():
            r, c = position
            new_powerups[r, c] = False

        # Existing boosters remain until collected.
        self.powerups = self.powerups | new_powerups

    def _inside_grid(self, r, c):
        # True when a row/column coordinate is on the board.
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def _opponent(self, player):
        # There are exactly two players, so the opponent id is the other id.
        return 2 if player == 1 else 1

    def _burn_time_for_cell_P1(self, r, c):
        # Higher elevation burns for fewer turns: elevation 0 -> 10, 5 -> 5.
        elevation = int(self.terrain[r, c])
        return self.burn_time_P1 - elevation
    
    def _burn_time_for_cell_P2(self, r, c):
        # Higher elevation burns for fewer turns: elevation 0 -> 10, 5 -> 5.
        elevation = int(self.terrain[r, c])
        return self.burn_time_P2 - elevation

    def _fire_owner_at(self, r, c):
        if self.trails[1][r, c] > 0:
            return 1

        if self.trails[2][r, c] > 0:
            return 2

        return None

    def _set_fire(self, owner, r, c):
        # A cell can only show one player's fire at a time. Clear both matrices
        # first, then set the owner's timer based on the cell elevation.
        self.trails[1][r, c] = 0
        self.trails[2][r, c] = 0
        self.trails[owner][r, c] = self._burn_time_for_cell_P1(r, c) if owner == 1 else self._burn_time_for_cell_P2(r, c)

    def _leave_current_tile(self, player, r, c):
        # Leaving a tile creates or converts fire depending on whose fire the
        # player was standing on before moving away.
        opponent = self._opponent(player)
        standing_fire_owner = self.standing_on_fire_owner[player]

        # Empty tile: the moving player leaves their own fire.
        if standing_fire_owner is None:
            self._set_fire(owner=player, r=r, c=c)

        # Own fire tile: leaving it converts the tile into enemy fire.
        elif standing_fire_owner == player:
            self._set_fire(owner=opponent, r=r, c=c)
            self.own_fire_conversions[player] += 1

        # Enemy fire remains enemy fire. It is not extinguished or claimed by
        # the player who stepped onto it.
        elif standing_fire_owner == opponent:
            pass

        # The player is no longer standing on the old tile.
        self.standing_on_fire_owner[player] = None

    def _apply_blocked_leave_fire(self, player, r, c, previous_owner):
        opponent = self._opponent(player)
        current_owner = self._fire_owner_at(r, c)
        standing_fire_owner = self.standing_on_fire_owner[player]

        # Active enemy fire keeps burning and damages the blocked player.
        if current_owner == opponent:
            self.standing_on_fire_owner[player] = opponent
            return opponent

        # Active own fire flips to enemy fire on a repeated blocked attempt.
        if current_owner == player:
            self._set_fire(owner=opponent, r=r, c=c)
            self.own_fire_conversions[player] += 1
            self.standing_on_fire_owner[player] = opponent
            return opponent

        # If visible fire expired at the start of this turn, the tile is empty
        # for this blocked attempt. The next blocked attempt will create fire.
        if previous_owner is not None:
            self.standing_on_fire_owner[player] = None
            return None

        # Own fire that was consumed on entry still converts when the player
        # tries and fails to leave it.
        if standing_fire_owner == player:
            self._set_fire(owner=opponent, r=r, c=c)
            self.own_fire_conversions[player] += 1
            self.standing_on_fire_owner[player] = opponent
            return opponent

        # Stale enemy-fire memory means the visible fire is gone; clear it.
        if standing_fire_owner == opponent:
            self.standing_on_fire_owner[player] = None
            return None

        # Empty tile: the first blocked attempt creates the player's own fire.
        self._set_fire(owner=player, r=r, c=c)
        self.standing_on_fire_owner[player] = player
        return None

    def _decrement_player_trail(self, player):
        # All active timers for this player's fire lose one turn.
        self.trails[player] -= 1

        # Clamp expired timers to zero so the matrix never goes negative.
        self.trails[player] = np.maximum(self.trails[player], 0)

    def _refresh_all_player_fire(self, player):
        # Find every currently burning tile owned by this player.
        burning_cells = self.trails[player] > 0
        rows, cols = np.where(burning_cells)

        # Reset each active timer to the full duration for its elevation.
        for r, c in zip(rows, cols):
            self.trails[player][r, c] = self._burn_time_for_cell_P1(r, c) if player == 1 else self._burn_time_for_cell_P2(r, c)

    def _own_fire_reward(self, player):
        # Reward is +1 for each active fire tile owned by the player.
        return int(np.sum(self.trails[player] > 0))
    
    def _new_reward_signals(self):
        """
        Create the reward-signal dictionary for one environment step.

        The movement and hazard logic fills these boolean signals during step().
        The selected reward method then converts the signals into numerical rewards.

        own_fire_tiles is measured at the beginning of the step. Current lives are
        read later during reward calculation, after cliff/fire damage has already
        been applied to self.lives[player].
        """
        return {
            1: {
                "own_fire_tiles": self._own_fire_reward(1),
                "blocked_move": False,
                "picked_booster": False,
                "stepped_off_cliff": False,
                "stepped_on_enemy_fire": False,
                "stepped_on_own_fire": False,
                "opponent_stepped_on_my_fire": False,
                "terminal": None,
            },
            2: {
                "own_fire_tiles": self._own_fire_reward(2),
                "blocked_move": False,
                "picked_booster": False,
                "stepped_off_cliff": False,
                "stepped_on_enemy_fire": False,
                "stepped_on_own_fire": False,
                "opponent_stepped_on_my_fire": False,
                "terminal": None,
            },
        }


    def _calculate_default_rewards(self, reward_signals):
        """
        Balanced reward method.

        This method gives moderate rewards for boosters, active fire control and
        making the opponent step onto the player's fire. It also gives moderate
        penalties for blocked moves, cliffs, enemy fire and own fire.

        Hazard penalties are weighted by the number of lives missing after the step.
        A player with fewer remaining lives receives a larger penalty for risky
        movement. Terminal rewards replace all other rewards from the same step.
        """
        rewards = {1: 0, 2: 0}

        for player in [1, 2]:
            signals = reward_signals[player]

            # If this is a terminal step, skip normal reward shaping.
            # The terminal loop below will overwrite the reward.
            if signals["terminal"] is not None:
                break

            # self.lives[player] is already updated by the movement/hazard logic.
            missing_lives = self.starting_lives - self.lives[player]

            # Use at least 1 so penalties such as own-fire are not zero at full lives.
            life_multiplier = max(1, missing_lives)

            # Blocked movement penalized, but fire-control rewards still apply.
            if signals["blocked_move"]:
                rewards[player] -= 10

            if signals["picked_booster"]:
                rewards[player] += 5

            # Cliff and enemy fire can both happen in one move, so use -= instead
            # of = to allow penalties to stack.
            if signals["stepped_off_cliff"]:
                rewards[player] -= 25 * life_multiplier

            if signals["stepped_on_enemy_fire"]:
                rewards[player] -= 25 * life_multiplier

            # Own fire is not lethal, but it is still discouraged.
            if signals["stepped_on_own_fire"]:
                rewards[player] -= 10 * life_multiplier

            # Reward damaging the opponent with this player's fire.
            if signals["opponent_stepped_on_my_fire"]:
                rewards[player] += 50

            # Reward maintaining active fire tiles.
            rewards[player] += 0.1 * signals["own_fire_tiles"]

        # Terminal rewards replace all non-terminal rewards.
        for player in [1, 2]:
            terminal = reward_signals[player]["terminal"]

            if terminal == "win":
                rewards[player] = 500
            elif terminal == "loss":
                rewards[player] = -500
            elif terminal == "draw":
                rewards[player] = 0

        return rewards


    def _calculate_offensive_rewards(self, reward_signals):
        """
        Offensive reward method.

        This method encourages aggressive play, fire pressure and direct damage to
        the opponent. It gives stronger rewards for boosters, active fire tiles and
        making the opponent step onto the player's fire.

        Hazard penalties are lower than in the defensive method, so the agent is
        more willing to take risks. Losing is also less bad than drawing, so the
        agent should prefer risky attempts to win over passive draws.

        Terminal rewards replace all other rewards from the same step.
        """
        rewards = {1: 0, 2: 0}

        for player in [1, 2]:
            signals = reward_signals[player]

            # If this is a terminal step, skip normal reward shaping.
            if signals["terminal"] is not None:
                break

            missing_lives = self.starting_lives - self.lives[player]
            life_multiplier = max(1, missing_lives)

            # Blocked movement wastes tempo and offensive pressure.
            if signals["blocked_move"]:
                rewards[player] -= 15

            # Boosters are valuable because they refresh this player's fire timers.
            if signals["picked_booster"]:
                rewards[player] += 5

            # Lower hazard penalties encourage risk-taking.
            if signals["stepped_off_cliff"]:
                rewards[player] -= 20 * life_multiplier

            if signals["stepped_on_enemy_fire"]:
                rewards[player] -= 20 * life_multiplier

            # Own fire is bad, but not very severe for the offensive style.
            if signals["stepped_on_own_fire"]:
                rewards[player] -= 5 * life_multiplier

            # Main offensive success signal.
            if signals["opponent_stepped_on_my_fire"]:
                rewards[player] += 20

            # Small reward for active fire pressure.
            rewards[player] += 0.1 * signals["own_fire_tiles"]

        # Terminal rewards replace all non-terminal rewards.
        for player in [1, 2]:
            terminal = reward_signals[player]["terminal"]

            if terminal == "win":
                rewards[player] = 500
            elif terminal == "loss":
                rewards[player] = -500
            elif terminal == "draw":
                rewards[player] = 0

        return rewards


    def _calculate_defensive_rewards(self, reward_signals):
        """
        Defensive reward method.

        This method encourages survival, safe movement and avoiding unnecessary
        damage. It gives smaller rewards for fire pressure than the offensive method,
        but it punishes cliffs, enemy fire and own fire more strongly.

        Losing is worse than drawing, so the agent should prefer survival or a draw
        over risky play that may lead to death.

        Terminal rewards replace all other rewards from the same step.
        """
        rewards = {1: 0, 2: 0}

        for player in [1, 2]:
            signals = reward_signals[player]

            # If this is a terminal step, skip normal reward shaping.
            if signals["terminal"] is not None:
                break

            missing_lives = self.starting_lives - self.lives[player]
            life_multiplier = max(1, missing_lives)

            # Defensive agents should avoid bad positioning and failed movement.
            if signals["blocked_move"]:
                rewards[player] -= 10

            # Boosters are useful, but less central than in offensive play.
            if signals["picked_booster"]:
                rewards[player] += 5

            # Stronger hazard penalties encourage safer movement.
            if signals["stepped_off_cliff"]:
                rewards[player] -= 30 * life_multiplier

            if signals["stepped_on_enemy_fire"]:
                rewards[player] -= 30 * life_multiplier

            # Own fire is not lethal, but defensive play should still avoid it.
            if signals["stepped_on_own_fire"]:
                rewards[player] -= 15 * life_multiplier

            # Opponent damage is useful, but less emphasized than in offensive play.
            if signals["opponent_stepped_on_my_fire"]:
                rewards[player] += 10

            # Small reward for maintaining fire control.
            rewards[player] += 0.1 * signals["own_fire_tiles"]

        # Terminal rewards replace all non-terminal rewards.
        for player in [1, 2]:
            terminal = reward_signals[player]["terminal"]

            if terminal == "win":
                rewards[player] = 500
            elif terminal == "loss":
                rewards[player] = -500
            elif terminal == "draw":
                rewards[player] = 0

        return rewards


    def _calculate_territorial_rewards(self, reward_signals):
        """
        Territorial reward method.

        Rewards are driven by territory advantage: the difference between the
        player's own burning tiles and the opponent's burning tiles each step.
        This encourages active map coverage rather than purely reactive play.

        Hazard penalties are kept but scaled down relative to the territory
        signal so that survival supports — rather than overrides — territorial
        dominance. Boosters are highly valued because refreshing fire timers
        directly expands territory. A draw is penalised less than a loss because
        territory-even games reflect competitive play.

        Terminal rewards replace all step rewards for the final step.
        """
        rewards = {1: 0, 2: 0}

        for player in [1, 2]:
            opponent = self._opponent(player)
            signals = reward_signals[player]

            if signals["terminal"] is not None:
                break

            missing_lives = self.starting_lives - self.lives[player]
            life_multiplier = max(1, missing_lives)

            # Core territorial signal: own fire tiles minus opponent fire tiles.
            # Positive when ahead, negative when behind.
            own_tiles = signals["own_fire_tiles"]
            opponent_tiles = reward_signals[opponent]["own_fire_tiles"]
            rewards[player] += 3 * (own_tiles - opponent_tiles)

            # Survival penalties — lighter than other methods so that the
            # agent stays aggressive rather than retreating to safety.
            if signals["blocked_move"]:
                rewards[player] -= 5

            if signals["stepped_off_cliff"]:
                rewards[player] -= 20 * life_multiplier

            if signals["stepped_on_enemy_fire"]:
                rewards[player] -= 20 * life_multiplier

            if signals["stepped_on_own_fire"]:
                rewards[player] -= 5 * life_multiplier

            # Boosters are the most impactful pickup: they refresh all active
            # fire timers, immediately extending territorial coverage.
            if signals["picked_booster"]:
                rewards[player] += 15

            # Small reward for direct fire damage; territory already captures
            # most of the benefit indirectly.
            if signals["opponent_stepped_on_my_fire"]:
                rewards[player] += 5

        # Terminal rewards replace all step rewards.
        for player in [1, 2]:
            terminal = reward_signals[player]["terminal"]

            if terminal == "win":
                rewards[player] = 100
            elif terminal == "loss":
                rewards[player] = -100
            elif terminal == "draw":
                rewards[player] = -50

        return rewards

    def _calculate_rewards(self, reward_signals):
        """
        Dispatch reward calculation to the reward method selected during environment
        initialization.
        """
        if self.reward_method == "default":
            return self._calculate_default_rewards(reward_signals)

        if self.reward_method == "offensive":
            return self._calculate_offensive_rewards(reward_signals)

        if self.reward_method == "defensive":
            return self._calculate_defensive_rewards(reward_signals)

        if self.reward_method == "territorial":
            return self._calculate_territorial_rewards(reward_signals)

        raise ValueError(f"Unknown reward method: {self.reward_method}")

    def _ask_human_action(self, player):
        # Terminal input helper used when a player type is set to "human".
        print(f"Player {player}'s action.")
        print("Choose action: W: up, S: down, A: left, D: right.")

        while True:
            value = input("Action: ").upper()

            # Keep asking until the user enters a valid movement key.
            if value not in {"W", "S", "A", "D"}:
                print("Invalid input. Enter W, S, A, or D.")
                continue

            # Convert keyboard controls to the environment's numeric action ids.
            if value == "W":
                return 0
            if value == "S":
                return 1
            if value == "A":
                return 2
            if value == "D":
                return 3

    def _random_action(self):
        # Simple fallback policy for agent players when no action is provided.
        return int(self.rng.choice([0, 1, 2, 3]))

    def _check_has_been_reset(self):
        # Terrain is created by reset(), so it is a compact sentinel for whether
        # the environment has been initialized.
        if self.terrain is None:
            raise RuntimeError(
                "Call reset() before using step(), info(), get_state(), or render()."
            )


if __name__ == "__main__":
    # When this file is run directly, start a simple two-human terminal game.
    env = GridWorldFireGame(
        grid_size=10,
        player_lives=3,
        max_episode_length=50,
        render=True,
        player1_type="human",
        player2_type="human",
        seed=42,
    )

    # Reset creates the first terrain, players, trails, boosters, and state.
    state = env.reset()
    done = False

    # Keep asking both humans for simultaneous actions until the episode ends.
    while not done:
        state, reward_1, reward_2, done, info = env.step()
