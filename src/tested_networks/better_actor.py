import torch.nn as nn
import torch.nn.functional as F


class BetterActorPPONetwork(nn.Module):
    """Deeper actor: 6 conv layers with a residual block, separate critic trunk."""

    def __init__(self, pool_size: int = 8, num_actions: int = 4):
        super().__init__()

        self.actor_conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.actor_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.actor_conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.actor_conv4 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.actor_conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.actor_conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.critic_conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.critic_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.critic_conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.critic_conv4 = nn.Conv2d(96, 128, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        flattened_size = 128 * pool_size * pool_size

        self.fc1_actor = nn.Linear(flattened_size, 1024)
        self.actor_ln1 = nn.LayerNorm(1024)
        self.fc2_actor = nn.Linear(1024, 256)
        self.actor_ln2 = nn.LayerNorm(256)
        self.actor = nn.Linear(256, num_actions)

        self.fc1_critic = nn.Linear(flattened_size, 512)
        self.fc2_critic = nn.Linear(512, 128)
        self.critic = nn.Linear(128, 1)

    def actor_features(self, x):
        x = F.relu(self.actor_conv1(x))
        x = F.relu(self.actor_conv2(x))
        x = self.pool(x)
        x = F.relu(self.actor_conv3(x))
        x = F.relu(self.actor_conv4(x))

        residual = x
        x = F.relu(self.actor_conv5(x))
        x = self.actor_conv6(x)
        x = F.relu(x + residual)

        return x.flatten(start_dim=1)

    def critic_features(self, x):
        x = F.relu(self.critic_conv1(x))
        x = F.relu(self.critic_conv2(x))
        x = self.pool(x)
        x = F.relu(self.critic_conv3(x))
        x = F.relu(self.critic_conv4(x))
        return x.flatten(start_dim=1)

    def forward(self, x):
        actor_x = self.actor_features(x)
        critic_x = self.critic_features(x)

        features_actor = F.relu(self.actor_ln1(self.fc1_actor(actor_x)))
        features_actor = F.relu(self.actor_ln2(self.fc2_actor(features_actor)))

        features_critic = F.relu(self.fc1_critic(critic_x))
        features_critic = F.relu(self.fc2_critic(features_critic))

        action_logits = self.actor(features_actor)
        value = self.critic(features_critic)
        return action_logits, value
