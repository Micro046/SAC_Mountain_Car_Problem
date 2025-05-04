# Soft Actor-Critic (SAC) on MountainCarContinuous-v0

This project implements the **Soft Actor-Critic (SAC)** reinforcement learning algorithm on the classic **MountainCarContinuous-v0** environment from OpenAI Gym.

## 🚗 Problem Description

The **Mountain Car Continuous** environment involves driving a car up a steep hill. The challenge is that the engine is not strong enough to climb the hill directly — instead, the agent must build momentum by moving back and forth. The objective is to reach the flag at the top with minimal steps.

- **Observation space**: Continuous `[position, velocity]`
- **Action space**: Continuous throttle force
- **Reward**: +100 for reaching the goal, otherwise -1 per timestep

---

## 🤖 Algorithm: Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm that incorporates **maximum entropy reinforcement learning** to promote exploration by adding an entropy term to the reward.

Key features:
- Stochastic policy
- Twin Q-networks to mitigate overestimation bias
- Automatic entropy tuning

---
