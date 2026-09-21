import random
import numpy as np
import torch


def normalize_state(state_np: np.ndarray) -> np.ndarray:
    """
    Normalize raw env state.
    ch0 (elevation [0,5]) → /5   ch2 (fire [-10,10]) → /10  others unchanged.
    """
    out = state_np.astype(np.float32, copy=True)
    out[0] /= 5.0
    out[2] /= 10.0
    return out


def state_for_player(normalized_state_np: np.ndarray, player_id: int) -> np.ndarray:
    """
    Convert an already-normalized state to a policy model's perspective.

    Policy models always see themselves as player 1:
    - self position/lives are positive in channel 1
    - self fire is positive in channel 2
    """
    if player_id == 1:
        return normalized_state_np

    if player_id == 2:
        out = normalized_state_np.copy()
        out[1] *= -1
        out[2] *= -1
        return out

    raise ValueError("player_id must be 1 or 2.")


def state_to_tensor(state_np: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a (4, N, N) numpy state to a (1, 4, N, N) float tensor on device."""
    return torch.from_numpy(state_np).unsqueeze(0).float().to(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_returns(
    rewards: list,
    gamma: float,
    final_value: float = 0.0,
    dones: list = None,
) -> list:
    """
    Compute discounted returns for a trajectory.

    G_t = r_t + gamma * G_{t+1}  (bootstrapped from final_value if episode not done)
    """
    returns = []
    G = final_value
    if dones is None:
        dones = [False] * len(rewards)
    for r, done in zip(reversed(rewards), reversed(dones)):
        G = r + gamma * G * (1.0 - float(done))
        returns.insert(0, G)
    return returns


def save_checkpoint(network: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str) -> None:
    torch.save({"model": network.state_dict(), "optimizer": optimizer.state_dict()}, path)


def load_checkpoint(path: str, network: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    network.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
