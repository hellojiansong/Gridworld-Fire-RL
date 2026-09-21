import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from generate_hill_map import HillMapGenerator


# Fill in the parameters here.
GRID_SIZE = 10
STARTING_LIVES = 3

# Use None for a random map every run.
# Use an integer if you want to reproduce one specific map.
SEED = 108

SHOW_HEIGHT_NUMBERS = True


def visualize_hill_map():
    generator = HillMapGenerator(
        grid_size=GRID_SIZE,
        starting_lives=STARTING_LIVES,
    )

    terrain = generator.generate(seed=SEED)

    is_valid = generator.is_valid(terrain)
    max_diff = generator.max_neighbour_difference(terrain)

    print("Generated terrain:")
    print(terrain)
    print(f"Valid map: {'YES' if is_valid else 'NO'}")
    print(f"Maximum neighbouring height difference: {max_diff}")

    elevation_colors = [
        "#F5DEB3",  # height 0
        "#D2B48C",  # height 1
        "#CD853F",  # height 2
        "#A0522D",  # height 3
        "#8B4513",  # height 4
        "#5C4033",  # height 5
    ]

    cmap = ListedColormap(elevation_colors)

    plt.figure(figsize=(8, 8))

    image = plt.imshow(
        terrain,
        cmap=cmap,
        vmin=0,
        vmax=5,
        interpolation="nearest",
    )

    colorbar = plt.colorbar(
        image,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label("Elevation")

    # Player starting positions
    plt.scatter(
        [0],
        [0],
        color="red",
        marker="o",
        s=150,
        edgecolor="white",
        linewidth=2,
        label="Player 1 Start",
    )

    plt.scatter(
        [GRID_SIZE - 1],
        [GRID_SIZE - 1],
        color="blue",
        marker="o",
        s=150,
        edgecolor="white",
        linewidth=2,
        label="Player 2 Start",
    )

    if SHOW_HEIGHT_NUMBERS:
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                plt.text(
                    c,
                    r,
                    str(int(terrain[r, c])),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=9,
                    fontweight="bold",
                )

    ax = plt.gca()

    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)

    ax.grid(
        which="minor",
        color="black",
        linestyle="-",
        linewidth=1,
        alpha=0.3,
    )

    ax.tick_params(
        which="major",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    seed_text = "random" if SEED is None else str(SEED)

    plt.title(
        f"Hill Map ({GRID_SIZE}x{GRID_SIZE}) | "
        f"Seed: {seed_text} | "
        f"Valid: {is_valid} | "
        f"Max diff: {max_diff}"
    )

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_hill_map()