import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def _orthogonal_init(layer: nn.Module, gain: float) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)


class PPONetwork(nn.Module):
    """Shared conv trunk with GroupNorm instead of BatchNorm.

    GroupNorm behaves identically in train() and eval(), avoiding the importance-ratio
    corruption that BatchNorm causes when PPO rolls out in eval() but updates in train().
    """

    def __init__(self, pool_size: int = 8, num_actions: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, 32)
        self.norm2 = nn.GroupNorm(8, 64)
        self.norm3 = nn.GroupNorm(8, 64)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc1_actor = nn.Linear(64 * pool_size * pool_size, 256)
        self.fc2_actor = nn.Linear(256, 128)
        self.fc1_critic = nn.Linear(64 * pool_size * pool_size, 256)
        self.fc2_critic = nn.Linear(256, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        # PPO-style orthogonal init: sqrt(2) gain on hidden layers (preserves
        # variance through ReLU), small gain on the heads so the initial policy
        # is near-uniform and the value head starts near zero.
        gain_relu = np.sqrt(2)
        for layer in [self.conv1, self.conv2, self.conv3,
                      self.fc1_actor, self.fc2_actor,
                      self.fc1_critic, self.fc2_critic]:
            _orthogonal_init(layer, gain_relu)
        _orthogonal_init(self.actor, 0.01)
        _orthogonal_init(self.critic, 1.0)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.norm3(self.conv3(x)))
        x = x.flatten(start_dim=1)
        features_actor = F.relu(self.fc1_actor(x))
        features_actor = F.relu(self.fc2_actor(features_actor))
        features_critic = F.relu(self.fc1_critic(x.detach()))
        features_critic = F.relu(self.fc2_critic(features_critic))
        action_logits = self.actor(features_actor)
        value = self.critic(features_critic)
        return action_logits, value


if __name__ == "__main__":
    import torch
    net = PPONetwork()
    dummy = torch.zeros(2, 4, 12, 12)
    logits, value = net(dummy)
    print(f"logits shape: {logits.shape}")  # (2, 4)
    print(f"value shape:  {value.shape}")   # (2, 1)
