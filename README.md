---

# 🧠 Soft Actor-Critic (SAC) on MountainCarContinuous-v0

This project implements the **Soft Actor-Critic (SAC)** reinforcement learning algorithm on the classic `MountainCarContinuous-v0` environment using [Gymnasium](https://gymnasium.farama.org/). The codebase is written in **PyTorch** and includes full training, evaluation, and logging functionality via **TensorBoard**.

---

## 🚗 Problem Description

The **Mountain Car Continuous** environment involves driving a car up a steep hill. The car’s engine alone is not powerful enough to climb the hill directly, so the agent must learn to build momentum by moving back and forth.

* **Observation space**: Continuous `[position, velocity]`
* **Action space**: Continuous throttle force
* **Reward**: +100 for reaching the goal, otherwise -1 per timestep
* **Max steps per episode**: 2000

---

## 🤖 Algorithm: Soft Actor-Critic (SAC)

SAC is an off-policy actor-critic algorithm based on the **maximum entropy framework**, encouraging both reward maximization and exploration.

### 🔑 Key Features

* **Stochastic policy** for better exploration
* **Twin Q-networks** to address overestimation bias
* **Automatic entropy tuning** for stability
* **Replay buffer** and **target networks**
* **Gradient updates per step**, customizable

---

## 🏗️ Project Structure

```plaintext
.
├── train.py                # Main training loop
├── sac.py                  # SAC algorithm class
├── network.py              # Policy and Q-value networks
├── buffer.py               # Experience replay buffer
├── utils.py                # Helper functions for plotting and updates
├── reward_wrapper.py       # Custom reward reshaping
├── checkpoints/            # Model checkpoints
├── plots/                  # Learning curve plots
├── runs/                   # TensorBoard logs
```

---

## 🧪 Training Details

| Hyperparameter            | Value      |
| ------------------------- | ---------- |
| Optimizer                 | Adam       |
| Learning Rate             | 0.0003     |
| Discount Factor (γ)       | 0.999      |
| Soft Target Update (τ)    | 0.005      |
| Batch Size                | 256        |
| Replay Buffer Size        | 100,000    |
| Start Steps (Exploration) | 10,000     |
| Policy Type               | Gaussian   |
| Updates per Step          | 1          |
| Gradient Steps            | 8          |
| Entropy Coefficient (α)   | Auto-tuned |
| Max Episode Steps         | 2000       |

---

## 📈 Results

During training, we log key metrics including:

* Actor & Critic losses
* Entropy temperature
* Episode rewards
* Evaluation rewards

Final performance is visualized in the learning curve and gameplay video.

### 🎥 Demo Video (SAC Agent Solving the Task)

> 🚩 The agent successfully reaches the flag by learning to build momentum!

[![Watch Video](https://img.youtube.com/vi_webhook/0.jpg)](https://github.com/Micro046/SAC_Mountain_Car_Problem/blob/main/mountaincar_sac.mp4)
👉 [View Video on GitHub](https://github.com/Micro046/SAC_Mountain_Car_Problem/blob/main/mountaincar_sac.mp4)


## 📚 References

* Haarnoja et al. (2018). *Soft Actor-Critic Algorithms for Deep Reinforcement Learning*
* OpenAI Gymnasium Docs: [MountainCarContinuous-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/)
* PyTorch Documentation: [pytorch.org](https://pytorch.org)

---
