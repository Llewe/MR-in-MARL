import numpy as np


class RewardNormalization:
    """
    Reward normalization class

    This can be helpful if mpe is used. The rewards there can be ]-inf, inf[.
    """

    current_episode = 1
    mean_reward = 0.0
    std_reward = 1.0

    def normalize(self, reward: float) -> float:
        # Reward normalization
        self.mean_reward = (self.mean_reward * self.current_episode + reward) / (
            self.current_episode + 1
        )
        up = (self.std_reward**2) * self.current_episode + (
            reward - self.mean_reward
        ) ** 2
        down = self.current_episode + 1
        dif = up / down
        root = np.sqrt(dif)

        self.std_reward = np.sqrt(
            (
                (self.std_reward**2) * self.current_episode
                + (reward - self.mean_reward) ** 2
            )
            / (self.current_episode + 1)
        )

        # Normalize the reward
        normalized_reward = (reward - self.mean_reward) / self.std_reward

        self.current_episode += 1
        return normalized_reward
