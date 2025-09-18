import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from URL_continous.utils import soft_update, hard_update
from discrete_sac_network import DiscreteQNetwork, DiscretePolicy
from discrete_mountain_car import vectorized_discrete_mountain_car_step
from multi_valley_mountain_car_disc import *
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)


# Argument Parser (mirrors continuous-style trainer)
parser = argparse.ArgumentParser(description="Discrete SAC training (uniform batch sampling)")

parser.add_argument('--env-name', default="DiscreteMountainCarBinary-v0")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.99)  # Higher for sparse rewards
parser.add_argument('--tau', type=float, default=0.001)   # Slower target updates
parser.add_argument('--lr', type=float, default=1e-4)    # Lower learning rate
parser.add_argument('--alpha', type=float, default=0.5)   # Higher entropy for exploration
parser.add_argument('--automatic_entropy_tuning', type=bool, default=Tru1)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=2000)  # Smaller batch
parser.add_argument('--num_steps', type=int, default=200000)  # More training steps
parser.add_argument('--hidden_size', type=int, default=512)   # Larger network
parser.add_argument('--cuda', action="store_true")
parser.add_argument('--eval_freq', type=int, default=10000, help='Evaluation frequency (steps)')
parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Gymnasium Environment
env = gym.make(args.env_name)
env.action_space.seed(args.seed)
state, _ = env.reset(seed=args.seed)

# Initialize SAC components directly
device = torch.device("cuda" if args.cuda else "cpu")
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

# Hyperparameters
gamma = args.gamma
tau = args.tau
alpha = torch.tensor(args.alpha, device=device)
automatic_entropy_tuning = args.automatic_entropy_tuning

# Initialize networks
critic = DiscreteQNetwork(num_inputs, num_actions, args.hidden_size).to(device=device)
critic_optim = Adam(critic.parameters(), lr=args.lr)

critic_target = DiscreteQNetwork(num_inputs, num_actions, args.hidden_size).to(device)
hard_update(critic_target, critic)

# Policy and entropy tuning
policy = DiscretePolicy(num_inputs, num_actions, args.hidden_size).to(device)
policy_optim = Adam(policy.parameters(), lr=args.lr)

if automatic_entropy_tuning:
    target_entropy = np.log(num_actions)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = Adam([log_alpha], lr=args.lr)


def select_action(state, evaluate=False):
    s = torch.FloatTensor(state).to(device).unsqueeze(0)
    with torch.no_grad():
        if evaluate:
            probs = policy.get_action_probs(s)
            action = probs.argmax(dim=-1)
            return int(action.item())
        else:
            actions, _, _ = policy.sample(s)
            return int(actions.item())


def evaluate_policy(env, num_episodes=10):
    """
    Evaluate the current policy by running actual episodes and collecting rewards
    """
    episode_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done and episode_length < 1000:  # Max episode length
            action = select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state
            episode_length += 1

        episode_rewards.append(episode_reward)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards
    }


def sample_uniform_batch_discrete(env, batch_size):
    """
    Vectorized sampling of batch transitions B={(s,a,r,s',d)} where:
    - s: states sampled uniformly from observation space (clipped)
    - a: actions sampled uniformly from discrete action space
    - r: rewards from vectorized environment step
    - s': next states from vectorized environment step
    - d: done flags
    """
    # Sample states from extended box then clip back (like continuous script)
    boundary_extension = 0.1
    low = env.observation_space.low
    high = env.observation_space.high
    extended_low = low - boundary_extension * (high - low)
    extended_high = high + boundary_extension * (high - low)

    state_batch = np.random.uniform(
        low=extended_low,
        high=extended_high,
        size=(batch_size, env.observation_space.shape[0])
    )
    state_batch = np.clip(state_batch, low, high).astype(np.float32)

    # Sample discrete actions uniformly: {0,1,2}
    action_batch = np.random.randint(0, num_actions, size=(batch_size,)).astype(np.int64)

    if args.env_name == "MultiValleyMountainCarDiscrete-v0":
        next_state_batch, reward_batch, terminated_batch = vectorized_multi_valley_mountain_car_step_discrete(
            state_batch, action_batch
        )
    else:
        next_state_batch, reward_batch, terminated_batch = vectorized_discrete_mountain_car_step(
            state_batch, action_batch
        )

    done_batch = terminated_batch.astype(np.float32)

    return (state_batch,
            action_batch,
            reward_batch.astype(np.float32),
            next_state_batch.astype(np.float32),
            done_batch)


# Training tracking
best_eval_reward = -float('inf')
training_logs = []

# Training loop
for i in range(args.num_steps):
    # Sample batch
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = sample_uniform_batch_discrete(
        env, args.batch_size
    )

    # Convert to tensors for SAC update
    state_batch_t = torch.FloatTensor(state_batch).to(device)
    action_batch_t = torch.LongTensor(action_batch).to(device)
    reward_batch_t = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
    next_state_batch_t = torch.FloatTensor(next_state_batch).to(device)
    done_batch_t = torch.FloatTensor(done_batch).to(device).unsqueeze(1)

    with torch.no_grad():
        # Discrete SAC target uses expectation over actions
        next_probs = policy.get_action_probs(next_state_batch_t)
        next_log_probs = torch.log(next_probs + 1e-8)
        q1_next_t, q2_next_t = critic_target(next_state_batch_t)
        min_q_next_t = torch.min(q1_next_t, q2_next_t)
        v_next = (next_probs * (min_q_next_t - (alpha if not automatic_entropy_tuning else log_alpha.exp()) * next_log_probs)).sum(dim=1, keepdim=True)
        next_q_value = reward_batch_t + (1 - done_batch_t) * gamma * v_next

    # Current Q-values for taken actions
    q1_current, q2_current = critic(state_batch_t)
    q1_sa = q1_current.gather(1, action_batch_t.unsqueeze(1))
    q2_sa = q2_current.gather(1, action_batch_t.unsqueeze(1))

    qf1_loss = F.mse_loss(q1_sa, next_q_value)
    qf2_loss = F.mse_loss(q2_sa, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # Update critic
    critic_optim.zero_grad()
    qf_loss.backward()
    critic_optim.step()

    # Policy update
    probs = policy.get_action_probs(state_batch_t)
    log_probs = torch.log(probs + 1e-8)
    q1_pi, q2_pi = critic(state_batch_t)
    min_q_pi = torch.min(q1_pi, q2_pi)

    temperature = (alpha if not automatic_entropy_tuning else log_alpha.exp())
    policy_loss = (probs * (temperature * log_probs - min_q_pi)).sum(dim=1).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Entropy tuning
    if automatic_entropy_tuning:
        # Encourage entropy to target
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
        alpha_loss = -(log_alpha * (entropy - target_entropy))
        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

    # Soft update target critic
    soft_update(critic_target, critic, tau)

    if i % 1000 == 0:
        with torch.no_grad():
            # Compute V(S) for current batch (expectation)
            probs_b = policy.get_action_probs(state_batch_t)
            log_probs_b = torch.log(probs_b + 1e-8)
            q1_b, q2_b = critic(state_batch_t)
            min_q_b = torch.min(q1_b, q2_b)
            v_b = (probs_b * (min_q_b - (alpha if not automatic_entropy_tuning else log_alpha.exp()) * log_probs_b)).sum(dim=1).mean().item()
        current_alpha = (alpha if not automatic_entropy_tuning else log_alpha.exp()).item()
        print(f"Step {i}: V(S) = {v_b:.3f}, Q-loss = {qf_loss.item():.3f}, Policy loss = {policy_loss.item():.3f}, alpha={current_alpha:.4f}")
        
        # Log training metrics
        training_logs.append({
            'step': i,
            'v_s': v_b,
            'q_loss': qf_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': current_alpha
        })

    # Evaluation
    if i % args.eval_freq == 0 and i > 0:
        print(f"\n--- Evaluation at Step {i} ---")
        eval_results = evaluate_policy(env, args.eval_episodes)
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print("--- End Evaluation ---\n")
        
        # Save best model
        if eval_results['mean_reward'] > best_eval_reward:
            best_eval_reward = eval_results['mean_reward']
            # Save best model (overwrite previous)
            save_dir = "./results_discrete"
            os.makedirs(save_dir, exist_ok=True)
            
            model_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'step': i,
                'critic_state_dict': critic.state_dict(),
                'critic_target_state_dict': critic_target.state_dict(),
                'policy_state_dict': policy.state_dict(),
                'critic_optim_state_dict': critic_optim.state_dict(),
                'policy_optim_state_dict': policy_optim.state_dict(),
                'log_alpha': log_alpha if automatic_entropy_tuning else None,
                'alpha_optim_state_dict': alpha_optim.state_dict() if automatic_entropy_tuning else None,
                'eval_reward': best_eval_reward,
                'args': vars(args)
            }, model_path)
            print(f"Best model saved to: {model_path}")
        
        # Log evaluation results
        training_logs.append({
            'step': i,
            'eval_mean_reward': eval_results['mean_reward'],
            'eval_std_reward': eval_results['std_reward'],
            'eval_episode_rewards': eval_results['episode_rewards']
        })


# Save final logs
print("Training completed!")
save_dir = "./results_discrete"
os.makedirs(save_dir, exist_ok=True)

# Save training logs as text file
logs_path = os.path.join(save_dir, "training_logs.txt")
with open(logs_path, 'w') as f:
    f.write("Step\tV(S)\tQ-Loss\tPolicy-Loss\tAlpha\tEval-Mean\tEval-Std\n")
    for log in training_logs:
        if 'eval_mean_reward' in log:
            f.write(f"{log['step']}\t{log.get('v_s', 'N/A')}\t{log.get('q_loss', 'N/A')}\t{log.get('policy_loss', 'N/A')}\t{log.get('alpha', 'N/A')}\t{log.get('eval_mean_reward', 'N/A')}\t{log.get('eval_std_reward', 'N/A')}\n")
        else:
            f.write(f"{log['step']}\t{log.get('v_s', 'N/A')}\t{log.get('q_loss', 'N/A')}\t{log.get('policy_loss', 'N/A')}\t{log.get('alpha', 'N/A')}\tN/A\tN/A\n")
print(f"Training logs saved to: {logs_path}")

print(f"Best evaluation reward: {best_eval_reward:.2f}")
print(f"Best model and logs saved in: {save_dir}")
