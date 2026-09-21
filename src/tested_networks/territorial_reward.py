import torch.nn as nn
import torch.nn.functional as F


class TerritorialRewardPPONetwork(nn.Module):
    """Same architecture as GridFullSeparate, trained with the territorial reward method."""

    def __init__(self, pool_size: int = 8, num_actions: int = 4):
        super().__init__()

        self.actor_conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.actor_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.actor_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.actor_batch_norm1 = nn.BatchNorm2d(32)
        self.actor_batch_norm2 = nn.BatchNorm2d(64)
        self.actor_batch_norm3 = nn.BatchNorm2d(64)

        self.critic_conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.critic_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.critic_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.critic_batch_norm1 = nn.BatchNorm2d(32)
        self.critic_batch_norm2 = nn.BatchNorm2d(64)
        self.critic_batch_norm3 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        flattened_size = 64 * pool_size * pool_size

        self.fc1_actor = nn.Linear(flattened_size, 256)
        self.fc2_actor = nn.Linear(256, 128)
        self.fc1_critic = nn.Linear(flattened_size, 256)
        self.fc2_critic = nn.Linear(256, 128)
        self.actor = nn.Linear(128, num_actions)
        self.critic = nn.Linear(128, 1)

    def actor_features(self, x):
        x = F.relu(self.actor_batch_norm1(self.actor_conv1(x)))
        x = F.relu(self.actor_batch_norm2(self.actor_conv2(x)))
        x = self.pool(x)
        x = F.relu(self.actor_batch_norm3(self.actor_conv3(x)))
        return x.flatten(start_dim=1)

    def critic_features(self, x):
        x = F.relu(self.critic_batch_norm1(self.critic_conv1(x)))
        x = F.relu(self.critic_batch_norm2(self.critic_conv2(x)))
        x = self.pool(x)
        x = F.relu(self.critic_batch_norm3(self.critic_conv3(x)))
        return x.flatten(start_dim=1)

    def forward(self, x):
        actor_x = self.actor_features(x)
        critic_x = self.critic_features(x)

        features_actor = F.relu(self.fc1_actor(actor_x))
        features_actor = F.relu(self.fc2_actor(features_actor))
        features_critic = F.relu(self.fc1_critic(critic_x))
        features_critic = F.relu(self.fc2_critic(features_critic))

        action_logits = self.actor(features_actor)
        value = self.critic(features_critic)
        return action_logits, value
