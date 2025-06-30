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
import matplotlib.pyplot as plt
import imageio
from utils import (plot_value_function, plot_reward_and_done, ensure_rgb, pad_frame_to_shape)

# Suppress warnings (optional)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="Soft Actor-Critic for Mountain Car using Gymnasium")
parser.add_argument('--env-name', default="SingleValleyMountainCarContinuous-v0")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--updates_per_step', type=int, default=1)
parser.add_argument('--gradient_steps', type=int, default=1)
parser.add_argument('--start_steps', type=int, default=1000)
parser.add_argument('--target_update_interval', type=int, default=1)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--video_interval_episodes', type=int, default=5, help="capture a frame every N episodes")
parser.add_argument('--video_fps', type=int, default=5, help="FPS for output MP4")
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

# Matplotlib visualization setup
fig, (ax_val, ax_rew, ax_done) = plt.subplots(1, 3, figsize=(15, 5))
plt.ion()
frames_data = []
target_shape = None  # to be determined by first frame

total_steps = 0
updates = 0
episode_rewards = []
episode_lengths = []
episode_timesteps = []
start_time = time.time()

state_low = torch.tensor(env.observation_space.low, dtype=torch.float32)
state_high = torch.tensor(env.observation_space.high, dtype=torch.float32)
state_dim = env.observation_space.shape[0]
device = agent.device

def sample_umbrella_state(env, state_low, state_high):
    pos = np.random.uniform(state_low[0], state_high[0])
    vel = np.random.uniform(state_low[1], state_high[1])
    return np.array([pos, vel], dtype=np.float32)

# For average value computation
x = np.linspace(state_low[0], state_high[0], 30)
y = np.linspace(state_low[1], state_high[1], 30)
P, V = np.meshgrid(x, y)
states_grid = np.column_stack([P.ravel(), V.ravel()])

@torch.no_grad()
def average_value_function(states, agent):
    agent.critic.eval(); agent.policy.eval()
    s = torch.as_tensor(states, device=agent.device, dtype=torch.float32)
    _, _, mean_a = agent.policy.sample(s)
    q1, q2 = agent.critic(s, mean_a)
    agent.critic.train(); agent.policy.train()
    return torch.min(q1, q2).mean().item()

# Training Loop
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0

    for _ in range(args.batch_size):
        state = sample_umbrella_state(env, state_low, state_high)
        env.reset()
        env.unwrapped.state = state.copy()
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        next_state[0] = np.clip(next_state[0], state_low[0], state_high[0])
        next_state[1] = np.clip(next_state[1], state_low[1], state_high[1])

        # if reward == 100:
        #     print(f"Batch {i_episode}, Step {episode_steps}: Sampled state={state}, action={action}, next_state={next_state}, reward=+100 (GOAL!)")

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
        avg_value = average_value_function(states_grid, agent)
        print("----------------------------------------")
        print(f"| Episode {i_episode}")
        print(f"| Mean Reward (Last 100): {np.mean(episode_rewards[-100:]):.2f}")
        print(f"| Mean Length: {np.mean(episode_lengths[-100:]):.1f}")
        print(f"| FPS: {int(total_steps / (time.time() - start_time))}")
        print(f"| Actor Loss: {p_loss:.3f}, Critic Loss: {(c1_loss + c2_loss):.3f}, Alpha: {alpha:.4f}")
        print(f"| Avg Value Function: {avg_value:.2f}")
        print("----------------------------------------")
        writer.add_scalar('value/avg_q', avg_value, i_episode)

    # Visualization capture
    if i_episode % args.video_interval_episodes == 0:
        print(f"[Viz] Episode {i_episode} – capturing frame")
        plot_value_function(agent, ax_val)
        plot_reward_and_done(env, ax_rew, ax_done)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.tostring_rgb()
        frame = np.frombuffer(buf, dtype=np.uint8)
        expected = w * h * 3
        if frame.size == expected:
            frame = frame.reshape((h, w, 3))
        else:
            tmp = f"temp_ep{i_episode}.png"
            fig.savefig(tmp, dpi=100)
            frame = plt.imread(tmp)
            os.remove(tmp)
            frame = (frame * 255).astype(np.uint8)
        if target_shape is None:
            target_shape = frame.shape
        frames_data.append(frame)

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

# Build video
print("Building video …")
os.makedirs("videos", exist_ok=True)
if frames_data:
    video_path = f"videos/SAC_{args.env_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4"
    with imageio.get_writer(video_path, fps=args.video_fps, codec='libx264') as writer:
        for idx, frm in enumerate(frames_data):
            frm_rgb = ensure_rgb(frm)
            if frm_rgb.shape != target_shape:
                frm_rgb = pad_frame_to_shape(frm_rgb, target_shape)
            writer.append_data(frm_rgb)
    print("Saved video to", video_path)
else:
    print("No frames captured, skipping video.")

agent.save_checkpoint(args.env_name, suffix="final")
env.close()
writer.close()
