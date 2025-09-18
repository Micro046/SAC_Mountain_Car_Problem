import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from network import GaussianPolicy, QNetwork, DeterministicPolicy
from utils import soft_update, hard_update
import warnings
from single_valley_mountain_car import *
from multi_valley_mountain_car_cont import *


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Argument Parser
parser = argparse.ArgumentParser(description="offline Umbrella Reinforcement learning")

parser.add_argument('--env-name', default="MultiValleyMountainCar-v0")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--eval', type=bool, default=True)
parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.000001)
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--hidden_size', type=int, default=256)
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
num_actions = env.action_space.shape[0]

# Hyperparameters
gamma = args.gamma
tau = args.tau
alpha = args.alpha
policy_type = args.policy
automatic_entropy_tuning = args.automatic_entropy_tuning

# Initialize networks
critic = QNetwork(num_inputs, num_actions, args.hidden_size).to(device=device)
critic_optim = Adam(critic.parameters(), lr=args.lr)

critic_target = QNetwork(num_inputs, num_actions, args.hidden_size).to(device)
hard_update(critic_target, critic)

if policy_type == "Gaussian":
    # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
    if automatic_entropy_tuning is True:
        target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = Adam([log_alpha], lr=args.lr)

    policy = GaussianPolicy(num_inputs, num_actions, args.hidden_size, env.action_space).to(device)
    policy_optim = Adam(policy.parameters(), lr=args.lr)

else:
    alpha = 0
    automatic_entropy_tuning = False
    policy = DeterministicPolicy(num_inputs, num_actions, args.hidden_size, env.action_space).to(device)
    policy_optim = Adam(policy.parameters(), lr=args.lr)

def select_action(state, evaluate=False):
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    if evaluate is False:
        action, _, _ = policy.sample(state)
    else:
        _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]

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
            # Take action and step environment
            action = select_action(state, evaluate=True)  # Use deterministic policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
            episode_length += 1
        
        episode_rewards.append(episode_reward)
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'episode_rewards': episode_rewards
    }


def sample_uniform_batch(env, batch_size):
    """
    Vectorized sampling of batch transitions B={(s,a,r,s',a',d)} where:
    - s: states sampled uniformly from observation space
    - a: actions sampled uniformly from action space  
    - r: rewards from vectorized environment step
    - s': next states from vectorized environment step
    - a': next actions from policy
    - d: done flags
    """
    # Sample states beyond boundaries and then clip back (vectorized)
    # Sample from extended range (e.g., 20% beyond boundaries)
    boundary_extension = 0.2
    extended_low = env.observation_space.low - boundary_extension * (env.observation_space.high - env.observation_space.low)
    extended_high = env.observation_space.high + boundary_extension * (env.observation_space.high - env.observation_space.low)
    
    state_batch = np.random.uniform(
        low=extended_low,
        high=extended_high,
        size=(batch_size, env.observation_space.shape[0])
    )
    
    # Clip back to original boundaries
    state_batch = np.clip(state_batch, env.observation_space.low, env.observation_space.high)
    
    # Sample actions uniformly from action space (vectorized)
    action_batch = np.random.uniform(
        low=env.action_space.low,
        high=env.action_space.high,
        size=(batch_size, env.action_space.shape[0])
    )
    
    # Vectorized environment step
    next_state_batch, reward_batch, terminated_batch = vectorized_mountain_car_step(
        state_batch, action_batch
    )
    
    # Get next actions from policy (vectorized)
    next_state_tensor = torch.FloatTensor(next_state_batch).to(device)
    with torch.no_grad():
        next_action_tensor, _, _ = policy.sample(next_state_tensor)
        next_action_batch = next_action_tensor.detach().cpu().numpy()
    
    # Done flags (no truncation in this environment)
    done_batch = terminated_batch
    
    return (state_batch, 
            action_batch, 
            reward_batch, 
            next_state_batch, 
            next_action_batch,
            done_batch)

# Training loop
for i in range(args.num_steps):
    # Sample batch with uniform action sampling
    state_batch, action_batch, reward_batch, next_state_batch, next_action_batch, done_batch = sample_uniform_batch(
        env, args.batch_size
    )
    
    # Convert to tensors for SAC update
    state_batch = torch.FloatTensor(state_batch).to(device)
    action_batch = torch.FloatTensor(action_batch).to(device)
    reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
    next_action_batch = torch.FloatTensor(next_action_batch).to(device)
    done_batch = torch.FloatTensor(done_batch).to(device).unsqueeze(1)

    with torch.no_grad():
        # Use the pre-computed next actions from policy (a')
        # Get log probability for the next actions
        _, next_state_log_pi, _ = policy.sample(next_state_batch)
        qf1_next_target, qf2_next_target = critic_target(next_state_batch, next_action_batch)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = reward_batch + (1 - done_batch) * gamma * (min_qf_next_target)
    qf1, qf2 = critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias
    qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = ð”¼(st,at)~D[0.5(Q1(st,at) - r(st,at) - Î³(ð”¼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = ð”¼(st,at)~D[0.5(Q1(st,at) - r(st,at) - Î³(ð”¼st+1~p[V(st+1)]))^2]
    qf_loss = qf1_loss + qf2_loss

    # Update critic
    critic_optim.zero_grad()
    qf_loss.backward()
    critic_optim.step()

    # Policy update
    pi, log_pi, _ = policy.sample(state_batch)
    qf1_pi, qf2_pi = critic(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Entropy tuning
    if automatic_entropy_tuning:
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()
        alpha = log_alpha.exp()

    # Soft update target critic
    soft_update(critic_target, critic, tau)

    if i % 1000 == 0:
        # Compute V(S) for current batch
        with torch.no_grad():
            batch_actions, _, _ = policy.sample(state_batch)
            q1_batch, q2_batch = critic(state_batch, batch_actions)
            v_batch = torch.min(q1_batch, q2_batch)
            mean_v_batch = v_batch.mean().item()
        
        print(f"Step {i}: V(S) = {mean_v_batch:.3f}, Q-loss = {qf_loss.item():.3f}, Policy loss = {policy_loss.item():.3f}")
    
    # Evaluation
    if i % args.eval_freq == 0 and i > 0:
        print(f"\n--- Evaluation at Step {i} ---")
        eval_results = evaluate_policy(env, args.eval_episodes)
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print("--- End Evaluation ---\n")
