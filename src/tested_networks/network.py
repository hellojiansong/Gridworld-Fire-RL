import torch.nn as nn
import torch.nn.functional as F


class PPONetwork(nn.Module):
    """Baseline: shared conv trunk with BatchNorm, separate actor and critic linear heads."""

    def __init__(self, pool_size: int = 8, num_actions: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.fc1_actor = nn.Linear(64 * pool_size * pool_size, 256)
        self.fc2_actor = nn.Linear(256, 128)
        self.fc1_critic = nn.Linear(64 * pool_size * pool_size, 256)
        self.fc2_critic = nn.Linear(256, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
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
    print(f"logits shape: {logits.shape}")
    print(f"value shape:  {value.shape}")
