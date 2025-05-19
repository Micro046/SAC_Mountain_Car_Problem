import argparse
import datetime
import os
import random
import time
from reward_wrapper import ClassicMountainCarRewardWrapper
from gym.wrappers import TimeLimit
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sac import SAC
from utils import plot_learning_curve
import matplotlib
matplotlib.use("TkAgg")  # or "Qt5Agg" if you have Qt5 installed
import matplotlib.pyplot as plt


# ------------- Replay Buffer -------------
class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        state      = np.asarray(state,      dtype=np.float32).flatten()
        next_state = np.asarray(next_state, dtype=np.float32).flatten()
        action     = np.asarray(action,     dtype=np.float32).flatten()
        reward     = float(reward)
        done       = float(done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        idx = random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in idx]
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ------------- Umbrella Sampler -------------
def sample_random_states(batch_size):
    positions  = np.random.uniform(-1.2, 0.44, size=(batch_size,1))
    velocities = np.random.uniform(-0.07,0.07, size=(batch_size,1))
    return np.hstack([positions, velocities])

# ------------- Visualization Helper -------------
def visualize_2d_samples(data, title, ax, xlabel, ylabel):
    ax.clear()
    ax.scatter(data[:,0], data[:,1], s=4, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def main():
    parser = argparse.ArgumentParser(
        description="Umbrella + SAC on MountainCarContinuous"
    )
    parser.add_argument('--env-name',      default="MountainCarContinuous-v0")
    parser.add_argument('--policy',        default="Gaussian", type=str)
    parser.add_argument('--eval',          type=bool, default=True)
    parser.add_argument('--gamma',         type=float, default=0.999)
    parser.add_argument('--tau',           type=float, default=0.005)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--alpha',         type=float, default=0.01)
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
    parser.add_argument('--seed',          type=int, default=1234)
    parser.add_argument('--batch_size',    type=int, default=1024)
    parser.add_argument('--replay_size',   type=int, default=100_000)
    parser.add_argument('--hidden_size',   type=int, default=256)
    parser.add_argument('--updates_per_round', type=int, default=1)
    parser.add_argument('--gradient_steps', type=int, default=1)
    parser.add_argument('--num_steps',     type=int, default=200_000)
    parser.add_argument('--eval_interval', type=int, default=2_000)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--cuda',          action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_env = gym.make(args.env_name)
    wrapped_env = ClassicMountainCarRewardWrapper(base_env)
    env = TimeLimit(wrapped_env, max_episode_steps=2000)
    env.action_space.seed(args.seed)
    state, _ = env.reset(seed=args.seed)

    agent  = SAC(env.observation_space.shape[0], env.action_space, args)
    memory = ReplayMemory(args.replay_size, args.seed)

    log_dir = f"runs/umbrella_sac_{args.env_name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(log_dir)
    start_time = time.time()

    plt.ion()
    fig, (ax_s, ax_r) = plt.subplots(1,2, figsize=(12,5))
    visited_states = []
    eval_rewards   = []

    total_steps = 0
    updates     = 0
    rounds = args.num_steps // args.batch_size

    for rnd in range(1, rounds+1):
        # Umbrella batch statistics
        batch_rewards = []
        batch_steps = 0
        batch_start_time = time.time()

        # Umbrella sampling: collect a batch
        states = sample_random_states(args.batch_size)
        for s in states:
            obs, _ = env.reset()
            env.unwrapped.state = s.copy()
            a = agent.select_action(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            memory.push(s, a, r, s2, done)
            visited_states.append(s)
            batch_rewards.append(r)
            total_steps += 1
            batch_steps += 1

        # SAC updates
        if len(memory) >= args.batch_size:
            for _ in range(args.updates_per_round):
                for _ in range(args.gradient_steps):
                    c1, c2, pol, ent, alpha = \
                        agent.update_parameters(memory, args.batch_size, updates)
                    writer.add_scalar('loss/critic1', c1, updates)
                    writer.add_scalar('loss/critic2', c2, updates)
                    writer.add_scalar('loss/policy',   pol, updates)
                    writer.add_scalar('loss/entropy',  ent, updates)
                    writer.add_scalar('alpha',         alpha, updates)
                    updates += 1

        # Logging per umbrella batch ("pseudo-episode")
        batch_reward_mean = np.mean(batch_rewards)
        batch_reward_std  = np.std(batch_rewards)
        time_elapsed = time.time() - batch_start_time
        print(
            f"Round {rnd} | Steps Collected: {batch_steps} | Buffer Size: {len(memory)} | "
            f"batch_reward_mean: {batch_reward_mean:.3f}  | "
        )

        writer.add_scalar('reward/train', batch_reward_mean, rnd)
        writer.add_scalar('steps/train', total_steps, rnd)

        # Evaluation
        if total_steps % args.eval_interval < args.batch_size:
            R = 0.0
            for _ in range(5):
                s, _ = env.reset(seed=args.seed)
                done = False
                while not done:
                    a = agent.select_action(s, evaluate=True)
                    s, r, term, trunc, _ = env.step(a)
                    done = term or trunc
                    R   += r
            R /= 5.0
            eval_rewards.append(R)
            writer.add_scalar('eval/reward', R, total_steps)
            print(f"[{total_steps:6d}] Eval Reward: {R:.2f}")

        # Live plot every 10 rounds
        if rnd % 10 == 0:
            ax_s = visualize_2d_samples(
                np.vstack(visited_states),
                title=f"Visited States after {total_steps} steps",
                ax=ax_s,
                xlabel="Position", ylabel="Velocity"
            )
            ax_r.clear()
            ax_r.plot(eval_rewards, '-o')
            ax_r.set_title("Eval Reward")
            plt.pause(0.01)

    agent.save_checkpoint(args.env_name, suffix="final")
    os.makedirs("plots", exist_ok=True)
    plot_learning_curve(
        np.arange(len(eval_rewards))*args.eval_interval,
        eval_rewards,
        f"plots/{args.env_name}_umbrella_sac.png"
    )
    print("Training done. Total time:", time.time()-start_time)
    plt.ioff()
    plt.show()
    writer.close()
    env.close()

if __name__=="__main__":
    main()