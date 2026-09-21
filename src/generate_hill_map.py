import numpy as np
from collections import deque


class HillMapGenerator:
    """
    Generates hill landscapes with optional ridges.

    Main hill rules:
    - Start from a flat height-0 map.
    - For each edge, choose 0, 1, or 2 starting blocks with equal probability.
    - Each selected starting block gets a fixed hill height: 3, 4, or 5.
    - A path always alternates:
        forward move -> diagonal move -> forward move -> diagonal move -> ...
    - A forward move goes 1 or 2 blocks toward the opposite edge.
    - A diagonal move chooses one diagonal side or both diagonal sides.
    - If both diagonal sides are chosen, the path branches.
    - For every chosen diagonal side, the number of diagonal steps is chosen
      independently as 1, 2, or 3.
    - A path stops when it reaches the map edge or encounters a block with
      height equal to or greater than the path height.

    Ridge rules:
    - Ridge placement is decided during path movement, but applied after smoothing.
    - Each taken forward/diagonal block has a 10% chance to start a ridge.
    - If a ridge starts, one side is chosen randomly.
    - The side-neighbour block is made 2 or 3 levels lower than the path center.
    - The same ridge side and drop are repeated for a total length of 3, 4, or 5
      taken blocks.
    - While a ridge is active, the 10% chance is not checked.
    """

    MIN_HEIGHT = 0
    MAX_HEIGHT = 5

    RIDGE_START_PROBABILITY = 0.20

    def __init__(self, grid_size=12, starting_lives=3):
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")

        self.grid_size = int(grid_size)
        self.starting_lives = int(starting_lives)

        # Player 1 starts top-left, player 2 starts bottom-right. The validity
        # check guarantees both corners can traverse the whole map.
        self.start_p1 = (0, 0)
        self.start_p2 = (self.grid_size - 1, self.grid_size - 1)

    MIN_TOTAL_STARTS = 2

    def generate(self, seed=None):
        rng = np.random.default_rng(seed)

        terrain = np.zeros(
            (self.grid_size, self.grid_size),
            dtype=np.int32,
        )

        ridge_operations = []

        edges = self._edge_definitions()

        # First roll the per-edge number of starts independently (0, 1, or 2).
        starts_per_edge = [int(rng.integers(0, 3)) for _ in edges]

        # The per-edge rolls can all land on 0, which would produce an empty,
        # all-zero map. Enforce a minimum total number of starting blocks across
        # the whole map by topping up on randomly chosen edges until the minimum
        # is met. Each edge is capped at the number of cells it has so the later
        # replace=False selection stays valid.
        edge_capacity = [len(self._edge_cells(name)) for name, _, _ in edges]

        while sum(starts_per_edge) < self.MIN_TOTAL_STARTS:
            available = [
                i for i in range(len(edges))
                if starts_per_edge[i] < edge_capacity[i]
            ]
            if not available:
                break
            choice = int(rng.choice(available))
            starts_per_edge[choice] += 1

        for (edge_name, forward_dir, side_dir), num_starts in zip(edges, starts_per_edge):
            if num_starts == 0:
                continue

            edge_cells = self._edge_cells(edge_name)

            selected_indices = rng.choice(
                len(edge_cells),
                size=num_starts,
                replace=False,
            )

            for index in selected_indices:
                start_cell = edge_cells[int(index)]
                hill_height = int(rng.integers(3, 6))

                self._grow_hill_path(
                    terrain=terrain,
                    start_cell=start_cell,
                    hill_height=hill_height,
                    forward_dir=forward_dir,
                    side_dir=side_dir,
                    rng=rng,
                    ridge_operations=ridge_operations,
                )

        self._smooth(terrain)
        self._apply_ridges(terrain, ridge_operations)

        return np.clip(
            terrain,
            self.MIN_HEIGHT,
            self.MAX_HEIGHT,
        ).astype(np.int32)

    def is_valid(self, terrain):
        terrain = np.asarray(terrain)

        if terrain.shape != (self.grid_size, self.grid_size):
            return False

        if np.any(terrain < self.MIN_HEIGHT) or np.any(terrain > self.MAX_HEIGHT):
            return False

        return self._is_traversable(terrain)

    def _is_traversable(self, terrain):
        """
        Validates that the map conforms to the connectivity rules of the game.

        Both start corners must be able to reach each other, every cell must be
        reachable by at least one player within the lives budget, and from every
        reachable cell a player must be able to get back to a start corner.
        """
        if not (self._can_reach(terrain, self.start_p1, self.start_p2) and
                self._can_reach(terrain, self.start_p2, self.start_p1)):
            return False

        max_lives_p1 = self._get_reachable_cells(terrain, self.start_p1, self.starting_lives)
        max_lives_p2 = self._get_reachable_cells(terrain, self.start_p2, self.starting_lives)

        union_reachable = (max_lives_p1 > 0) | (max_lives_p2 > 0)
        if not union_reachable.all():
            return False

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                budgets = {int(max_lives_p1[r, c]), int(max_lives_p2[r, c])} - {0}
                for player_lives in budgets:
                    if not (self._can_reach(terrain, (r, c), self.start_p1, player_lives) or
                            self._can_reach(terrain, (r, c), self.start_p2, player_lives)):
                        return False

        return True

    def _get_reachable_cells(self, terrain, start, starting_lives):
        queue = deque([(start[0], start[1], starting_lives)])
        max_lives_at_tile = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        max_lives_at_tile[start[0], start[1]] = starting_lives

        while queue:
            r, c, current_lives = queue.popleft()
            curr_height = terrain[r, c]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not self._inside(nr, nc):
                    continue

                target_height = terrain[nr, nc]
                elevation_diff = target_height - curr_height

                if elevation_diff > 1:
                    continue

                next_lives = current_lives
                if elevation_diff < -1:
                    next_lives -= 1

                if next_lives <= 0:
                    continue

                if next_lives > max_lives_at_tile[nr, nc]:
                    max_lives_at_tile[nr, nc] = next_lives
                    queue.append((nr, nc, next_lives))

        return max_lives_at_tile

    def _can_reach(self, terrain, start, goal, starting_lives=None):
        if starting_lives is None:
            starting_lives = self.starting_lives

        queue = deque([(start[0], start[1], starting_lives)])
        max_lives_at_tile = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        max_lives_at_tile[start[0], start[1]] = starting_lives

        while queue:
            r, c, current_lives = queue.popleft()

            if (r, c) == goal:
                return True

            curr_height = terrain[r, c]

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if not self._inside(nr, nc):
                    continue

                target_height = terrain[nr, nc]
                elevation_diff = target_height - curr_height

                if elevation_diff > 1:
                    continue

                next_lives = current_lives
                if elevation_diff < -1:
                    next_lives -= 1

                if next_lives <= 0:
                    continue

                if next_lives > max_lives_at_tile[nr, nc]:
                    max_lives_at_tile[nr, nc] = next_lives
                    queue.append((nr, nc, next_lives))

        return False

    def max_neighbour_difference(self, terrain):
        max_diff = 0

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                for nr, nc in self._neighbours(r, c):
                    diff = abs(int(terrain[r, c]) - int(terrain[nr, nc]))
                    max_diff = max(max_diff, diff)

        return int(max_diff)

    def _grow_hill_path(
        self,
        terrain,
        start_cell,
        hill_height,
        forward_dir,
        side_dir,
        rng,
        ridge_operations,
    ):
        """
        Grow one hill path.

        active stores:
            (current_cell, next_move_type, ridge_state)

        ridge_state is either:
            None
        or:
            (remaining_blocks, side_sign, drop)

        side_sign is -1 or 1 and determines on which side of the path
        the ridge is placed.
        """
        placed = self._place_hill_cell(
            terrain=terrain,
            cell=start_cell,
            hill_height=hill_height,
            side_dir=side_dir,
        )

        if not placed:
            return

        active = [(start_cell, "forward", None)]
        max_steps = self.grid_size * self.grid_size

        for _ in range(max_steps):
            if not active:
                break

            next_active = []

            for cell, next_move_type, ridge_state in active:
                if next_move_type == "forward":
                    new_states = self._take_forward_step(
                        terrain=terrain,
                        cell=cell,
                        hill_height=hill_height,
                        forward_dir=forward_dir,
                        side_dir=side_dir,
                        rng=rng,
                        ridge_state=ridge_state,
                        ridge_operations=ridge_operations,
                    )
                else:
                    new_states = self._take_diagonal_step(
                        terrain=terrain,
                        cell=cell,
                        hill_height=hill_height,
                        forward_dir=forward_dir,
                        side_dir=side_dir,
                        rng=rng,
                        ridge_state=ridge_state,
                        ridge_operations=ridge_operations,
                    )

                next_active.extend(new_states)

            active = next_active

    def _take_forward_step(
        self,
        terrain,
        cell,
        hill_height,
        forward_dir,
        side_dir,
        rng,
        ridge_state,
        ridge_operations,
    ):
        """
        Take a straight step toward the opposite edge.

        The path moves either 1 or 2 blocks forward.
        After this, the next movement must be diagonal.
        """
        steps = int(rng.integers(1, 3))  # 1 or 2

        endpoint, visited_cells = self._move_and_raise(
            terrain=terrain,
            start_cell=cell,
            direction=forward_dir,
            steps=steps,
            hill_height=hill_height,
            side_dir=side_dir,
        )

        if endpoint is None:
            return []

        ridge_state = self._update_ridge_state_for_cells(
            visited_cells=visited_cells,
            hill_height=hill_height,
            side_dir=side_dir,
            rng=rng,
            ridge_state=ridge_state,
            ridge_operations=ridge_operations,
        )

        return [(endpoint, "diagonal", ridge_state)]

    def _take_diagonal_step(
        self,
        terrain,
        cell,
        hill_height,
        forward_dir,
        side_dir,
        rng,
        ridge_state,
        ridge_operations,
    ):
        """
        Take a diagonal step.

        One diagonal side or both diagonal sides are chosen.
        If both sides are chosen, the path branches.

        The number of steps for each selected diagonal side is chosen
        independently as 1, 2, or 3.
        """
        diagonal_dirs = self._choose_diagonal_directions(
            forward_dir=forward_dir,
            side_dir=side_dir,
            rng=rng,
        )

        new_states = []

        for diagonal_dir in diagonal_dirs:
            steps = int(rng.integers(1, 4))  # 1, 2, or 3

            endpoint, visited_cells = self._move_and_raise(
                terrain=terrain,
                start_cell=cell,
                direction=diagonal_dir,
                steps=steps,
                hill_height=hill_height,
                side_dir=side_dir,
            )

            if endpoint is None:
                continue

            branch_ridge_state = self._copy_ridge_state(ridge_state)

            branch_ridge_state = self._update_ridge_state_for_cells(
                visited_cells=visited_cells,
                hill_height=hill_height,
                side_dir=side_dir,
                rng=rng,
                ridge_state=branch_ridge_state,
                ridge_operations=ridge_operations,
            )

            new_states.append((endpoint, "forward", branch_ridge_state))

        return new_states

    def _move_and_raise(self, terrain, start_cell, direction, steps, hill_height, side_dir):
        """
        Move 1, 2, or 3 blocks in the given direction and raise every visited block.

        The path stops if the next block is outside the map or already has
        height equal to or greater than hill_height.

        Returns:
            endpoint, visited_cells
        """
        r, c = start_cell
        dr, dc = direction

        endpoint = None
        visited_cells = []

        for _ in range(steps):
            nr = r + dr
            nc = c + dc

            if not self._inside(nr, nc):
                return None, []

            if terrain[nr, nc] >= hill_height:
                return None, []

            self._place_hill_cell(
                terrain=terrain,
                cell=(nr, nc),
                hill_height=hill_height,
                side_dir=side_dir,
            )

            r, c = nr, nc
            endpoint = (r, c)
            visited_cells.append(endpoint)

        return endpoint, visited_cells

    def _update_ridge_state_for_cells(
        self,
        visited_cells,
        hill_height,
        side_dir,
        rng,
        ridge_state,
        ridge_operations,
    ):
        """
        Update ridge state for all blocks visited during one movement.

        If a ridge is already active:
            - apply it to the current block,
            - decrease remaining length,
            - do not perform the 10% ridge-start check.

        If no ridge is active:
            - perform the 10% ridge-start check.
        """
        for cell in visited_cells:
            if ridge_state is not None:
                remaining_blocks, side_sign, drop = ridge_state

                ridge_operations.append(
                    {
                        "cell": cell,
                        "hill_height": hill_height,
                        "side_dir": side_dir,
                        "side_sign": side_sign,
                        "drop": drop,
                    }
                )

                remaining_blocks -= 1

                if remaining_blocks <= 0:
                    ridge_state = None
                else:
                    ridge_state = (remaining_blocks, side_sign, drop)

                continue

            if rng.random() < self.RIDGE_START_PROBABILITY:
                side_sign = int(rng.choice([-1, 1]))
                drop = int(rng.integers(2, 4))          # 2 or 3 lower
                ridge_length = int(rng.integers(3, 6))  # total length 3, 4, or 5

                ridge_operations.append(
                    {
                        "cell": cell,
                        "hill_height": hill_height,
                        "side_dir": side_dir,
                        "side_sign": side_sign,
                        "drop": drop,
                    }
                )

                remaining_blocks = ridge_length - 1

                if remaining_blocks > 0:
                    ridge_state = (remaining_blocks, side_sign, drop)
                else:
                    ridge_state = None

        return ridge_state

    def _apply_ridges(self, terrain, ridge_operations):
        """
        Apply recorded ridge operations after smoothing.

        A ridge lowers one side-neighbour of a path block so that:
            - going from the main path block to the ridge block is a cliff,
            - going from the ridge block back to the main path block is a wall,
            - but the ridge block still has an outward escape route.

        Therefore, the ridge block is only lowered if the next block farther
        outward exists and remains within height difference <= 1.
        """
        for operation in ridge_operations:
            r, c = operation["cell"]
            side_r, side_c = operation["side_dir"]
            side_sign = operation["side_sign"]
            drop = operation["drop"]

            ridge_r = r + side_sign * side_r
            ridge_c = c + side_sign * side_c

            outward_r = ridge_r + side_sign * side_r
            outward_c = ridge_c + side_sign * side_c

            if not self._inside(ridge_r, ridge_c):
                continue

            if not self._inside(outward_r, outward_c):
                continue

            center_height = int(terrain[r, c])
            current_ridge_height = int(terrain[ridge_r, ridge_c])
            outward_height = int(terrain[outward_r, outward_c])

            target_ridge_height = max(
                self.MIN_HEIGHT,
                center_height - drop,
            )

            # Only lower the side block. Never raise it because of a ridge operation.
            new_ridge_height = min(
                current_ridge_height,
                target_ridge_height,
            )

            # The main block must actually become a wall/cliff relation.
            if center_height - new_ridge_height <= 1:
                continue

            # The lowered ridge block must still have an outward escape route.
            if abs(new_ridge_height - outward_height) > 1:
                continue

            terrain[ridge_r, ridge_c] = new_ridge_height

    def _copy_ridge_state(self, ridge_state):
        if ridge_state is None:
            return None

        remaining_blocks, side_sign, drop = ridge_state
        return int(remaining_blocks), int(side_sign), int(drop)

    def _place_hill_cell(self, terrain, cell, hill_height, side_dir):
        """
        Raise the center cell and its side neighbours.

        The center is raised to hill_height.
        Side neighbours parallel to the corresponding map edge are raised with
        decreasing height.

        Example for hill_height 5:
            1 2 3 4 5 4 3 2 1

        Existing higher terrain is never lowered.
        """
        r, c = cell

        if not self._inside(r, c):
            return False

        if terrain[r, c] >= hill_height:
            return False

        terrain[r, c] = hill_height

        side_r, side_c = side_dir

        for sign in [-1, 1]:
            height = hill_height - 1
            sr = r + sign * side_r
            sc = c + sign * side_c

            while self._inside(sr, sc) and height >= 1:
                if terrain[sr, sc] >= height:
                    break

                terrain[sr, sc] = height

                height -= 1
                sr += sign * side_r
                sc += sign * side_c

        return True

    def _choose_diagonal_directions(self, forward_dir, side_dir, rng):
        """
        Choose one diagonal side or both diagonal sides.

        The three outcomes have equal probability:
            one side,
            the other side,
            both sides.

        If both sides are chosen, the path branches.
        """
        fr, fc = forward_dir
        sr, sc = side_dir

        diagonal_a = (fr + sr, fc + sc)
        diagonal_b = (fr - sr, fc - sc)

        choice = int(rng.integers(0, 3))

        if choice == 0:
            return [diagonal_a]

        if choice == 1:
            return [diagonal_b]

        return [diagonal_a, diagonal_b]

    def _smooth(self, terrain):
        """
        Raise lower neighbours until every adjacent height difference is <= 1.

        This only raises low cells. It does not lower hill peaks.
        Ridges are added after this pass, so ridges are not smoothed away.
        """
        changed = True

        while changed:
            changed = False

            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    current_height = int(terrain[r, c])

                    for nr, nc in self._neighbours(r, c):
                        neighbour_height = int(terrain[nr, nc])

                        if current_height - neighbour_height > 1:
                            terrain[nr, nc] = current_height - 1
                            changed = True

    def _edge_definitions(self):
        return [
            ("top", (1, 0), (0, 1)),
            ("bottom", (-1, 0), (0, 1)),
            ("left", (0, 1), (1, 0)),
            ("right", (0, -1), (1, 0)),
        ]

    def _edge_cells(self, edge_name):
        n = self.grid_size

        if edge_name == "top":
            return [(0, c) for c in range(n)]

        if edge_name == "bottom":
            return [(n - 1, c) for c in range(n)]

        if edge_name == "left":
            return [(r, 0) for r in range(n)]

        if edge_name == "right":
            return [(r, n - 1) for r in range(n)]

        raise ValueError(f"Unknown edge name: {edge_name}")

    def _neighbours(self, r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc

            if self._inside(nr, nc):
                yield nr, nc

    def _inside(self, r, c):
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size