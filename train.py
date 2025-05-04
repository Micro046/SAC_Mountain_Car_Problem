import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
from buffer import ReplayMemory
from utils import plot_learning_curve
from torch.utils.tensorboard import SummaryWriter
import os
import time
import warnings
from reward_wrapper import ClassicMountainCarRewardWrapper
from gym.wrappers import TimeLimit

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="Soft Actor-Critic for Mountain Car  using Gymnasium")

parser.add_argument('--env-name', default="MountainCarContinuous-v0")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.999)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=400000)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--gradient_steps', type=int, default=8)
parser.add_argument('--start_steps', type=int, default=10000)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--cuda', action="store_true")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Gymnasium Environment
base_env = gym.make(args.env_name)
wrapped_env = ClassicMountainCarRewardWrapper(base_env)
env = TimeLimit(wrapped_env, max_episode_steps=2000)
env.action_space.seed(args.seed)
state, _ = env.reset(seed=args.seed)

# Agent and components
agent = SAC(env.observation_space.shape[0], env.action_space, args)
writer = SummaryWriter(f"runs/SAC_{args.env_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
memory = ReplayMemory(args.replay_size, args.seed)

total_steps = 0
updates = 0
episode_rewards = []
episode_lengths = []
episode_timesteps = []
start_time = time.time()

# Training Loop
for i_episode in itertools.count(1):
    state, _ = env.reset(seed=args.seed)
    episode_reward = 0
    episode_steps = 0
    done = False

    while not done:
        if total_steps < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_steps += 1
        total_steps += 1
        episode_reward += reward

        non_terminal = float(not done)
        memory.push(state, action, reward, next_state, non_terminal)
        state = next_state

        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                for _ in range(args.gradient_steps):
                    c1_loss, c2_loss, p_loss, e_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', c1_loss, updates)
                    writer.add_scalar('loss/critic_2', c2_loss, updates)
                    writer.add_scalar('loss/policy', p_loss, updates)
                    writer.add_scalar('loss/entropy_loss', e_loss, updates)
                    writer.add_scalar('entropy_temp/alpha', alpha, updates)
                    updates += 1

    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_steps)
    episode_timesteps.append(total_steps)

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print(f"Episode {i_episode} | Steps: {episode_steps} | Total Steps: {total_steps} | Reward: {episode_reward:.2f}")

    if total_steps > args.num_steps:
        break

    if i_episode % 4 == 0:
        print("----------------------------------------")
        print(f"| Episode {i_episode}")
        print(f"| Mean Reward (Last 100): {np.mean(episode_rewards[-100:]):.2f}")
        print(f"| Mean Length: {np.mean(episode_lengths[-100:]):.1f}")
        print(f"| FPS: {int(total_steps / (time.time() - start_time))}")
        print(f"| Actor Loss: {p_loss:.3f}, Critic Loss: {(c1_loss + c2_loss):.3f}, Alpha: {alpha:.4f}")
        print("----------------------------------------")

    if i_episode % 10 == 0 and args.eval:
        eval_reward = 0
        for _ in range(10):
            state, _ = env.reset(seed=args.seed)
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                eval_reward += reward
                state = next_state
        avg_eval = eval_reward / 10
        writer.add_scalar('avg_reward/test', avg_eval, i_episode)
        print(f"--- Evaluation | Episode {i_episode} | Avg Reward: {avg_eval:.2f} ---")

# Save final results
os.makedirs("plots", exist_ok=True)
plot_path = f"plots/{args.env_name}_SAC_learning_curve.png"
plot_learning_curve(episode_timesteps, episode_rewards, plot_path)
print(f"Learning curve saved to {plot_path}")

agent.save_checkpoint(args.env_name, suffix="final")
env.close()
writer.flush()
writer.close()
