import gymnasium as gym
from gymnasium import RewardWrapper

class ClassicMountainCarRewardWrapper(RewardWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        # If the car has reached or passed the goal
        if self.env.unwrapped.state[0] >= self.env.unwrapped.goal_position:
            return 100  # reward for success (optional, you can set it to 0)
        return -1  # constant penalty per step like MountainCar-v0

