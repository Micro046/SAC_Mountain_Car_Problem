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
import single_valley_mountain_car

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="Soft Actor-Critic for Mountain Car  using Gymnasium")

parser.add_argument('--env-name', default="SingleValleyMountainCarContinuous-v0")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.00005)
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=400000)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--gradient_steps', type=int, default=1)
parser.add_argument('--start_steps', type=int, default=1000)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--cuda', action="store_true")

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Gymnasium Environment
base_env = gym.make(args.env_name)
env = base_env
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

state_low  = torch.tensor(env.observation_space.low, dtype=torch.float32)
state_high = torch.tensor(env.observation_space.high, dtype=torch.float32)
state_dim = env.observation_space.shape[0]
device = "cuda" if args.cuda else "cpu"


def make_window_edges(pos_low=-1.2, pos_high=0.6, nb_windows=10):
    edges = np.linspace(pos_low, pos_high, nb_windows + 1)
    return [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

def in_goal(state_np, env):
    pos, vel = state_np
    return (pos >= env.unwrapped.goal_position
            and vel >= getattr(env.unwrapped, "goal_velocity", 0.0))

def sample_umbrella_state(env, state_low, state_high, windows, p_goal=0, device="cpu"):
    if np.random.rand() < p_goal:
        # Near-goal sample
        pos = np.random.uniform(env.unwrapped.goal_position, state_high[0])
        vel = np.random.uniform(0, state_high[1])
    else:
        # Uniformly across state-space windows (full space!)
        w_lo, w_hi = windows[np.random.randint(len(windows))]
        pos = np.random.uniform(w_lo, w_hi)
        vel = np.random.uniform(state_low[1], state_high[1])

    return np.array([pos, vel], dtype=np.float32)



state_low  = torch.tensor(env.observation_space.low,  dtype=torch.float32)
state_high = torch.tensor(env.observation_space.high, dtype=torch.float32)
windows = make_window_edges(pos_low=state_low[0].item(),
                             pos_high=env.unwrapped.goal_position,
                             nb_windows=10)

device = agent.device


# Training Loop
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0

    for _ in range(args.batch_size):
        state = sample_umbrella_state(env, state_low, state_high,
                                      windows=windows, device=device)

        env.reset()
        env.unwrapped.state = state.copy()

        action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        if reward == 100:
            print(f"Batch {i_episode}, Step {episode_steps}: "
            f"Sampled state={state}, action={action}, "
            f"next_state={next_state}, reward=+100 (GOAL!)")


        non_terminal = float(not (terminated or truncated))

        if terminated:
            reward = 100.0
        else:
            reward = -1.0

        episode_steps += 1
        total_steps += 1
        episode_reward += reward


        # Store this single-step transition
        memory.push(state, action, reward, next_state, non_terminal)


        # Update SAC agent as usual
        if len(memory) > args.batch_size:
            for _ in range(args.updates_per_step):
                for _ in range(args.gradient_steps):
                    c1_loss, c2_loss, p_loss, e_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates)
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

    # Standard evaluation with rollouts
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
