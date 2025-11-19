import argparse
import datetime
import math
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from utils import (
    soft_update,
    hard_update,
    log_training_iteration,
    evaluate_policy,
    prepare_visualization_grid,
    visualize_state_distribution,
    visualize_policy,
    visualize_value_function,
    estimate_density_total_mass,
)
from network import GaussianPolicy, QNetwork, CategoricalPolicy, DiscreteQNetwork
import os
import warnings
import csv

from env import MultiValleyMountainCarEnv 
from density_estimator import DensityEstimator

warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_args():
    """Return command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description="Soft Actor-Critic for Mountain Car using Gymnasium")
    parser.add_argument('--env-name', default="MultiValleyMountainCarDiscrete-v0")
    parser.add_argument('--policy', default=None)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--alpha', type=float, default=0.2, help="Action entropy coefficient (SAC)")
    parser.add_argument('--beta', type=float, default=0.2, help="State entropy coefficient (URL)")
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--updates_per_step', type=int, default=1)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--log_freq', type=int, default=1000, help='Logging frequency (iterations)')
    parser.add_argument('--eval_freq', type=int, default=5000, help='Evaluation frequency (iterations)')
    parser.add_argument('--run_name', type=str, default=None, help='Run name for CSV logs (default: auto-generated)')
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--density_batch_size', type=int, default=10000, help='Batch size N for density net')
    parser.add_argument('--density_test_funcs', type=int, default=1024, help='Batch size B for RFF test functions')
    parser.add_argument('--weight_decay_pi', type=float, default=5e-6, help='Weight decay for policy optimizer')
    parser.add_argument('--visualize_freq', type=int, default=500, help='Iteration frequency for saving visualizations (0 to disable)')
    parser.add_argument('--visualize_grid_size', type=int, default=500, help='Resolution of the state grid for visualizations')
    return parser.parse_args()


def create_environment(args):
    """Instantiate the gym environment and derive space information."""
    env = gym.make(args.env_name)
    env_unwrapped = env.unwrapped
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)

    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if args.policy is None:
        args.policy = "Categorical" if is_discrete else "Gaussian"
    print(f"is_discrete: {is_discrete}")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if is_discrete else env.action_space.shape[0]
    return env, env_unwrapped, is_discrete, state_dim, action_dim


def get_device(args):
    return torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")


def build_networks(state_dim, action_dim, args, env, device, is_discrete):
    if is_discrete:
        policy = CategoricalPolicy(state_dim, action_dim, args.hidden_size)
        critic = DiscreteQNetwork(state_dim, action_dim, args.hidden_size)
        critic_target = DiscreteQNetwork(state_dim, action_dim, args.hidden_size)
    else:
        policy = GaussianPolicy(state_dim, action_dim, args.hidden_size, env.action_space)
        critic = QNetwork(state_dim, action_dim, args.hidden_size)
        critic_target = QNetwork(state_dim, action_dim, args.hidden_size)

    hard_update(critic_target, critic)
    policy = policy.to(device)
    critic = critic.to(device)
    critic_target = critic_target.to(device)
    return policy, critic, critic_target


def initialize_optimizers(policy, critic, lr, policy_weight_decay=0.0):
    policy_optim = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=policy_weight_decay)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
    return policy_optim, critic_optim


def setup_entropy_config(args, env, device, is_discrete, action_dim):
    if args.automatic_entropy_tuning:
        if is_discrete:
            target_entropy = -math.log(action_dim)
        else:
            target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.lr)
        alpha = args.alpha
    else:
        alpha = args.alpha
        target_entropy = None
        log_alpha = None
        alpha_optim = None
    return alpha, target_entropy, log_alpha, alpha_optim


def create_density_module(policy, env_unwrapped, args, device):
    return DensityEstimator(
        policy_net=policy,
        env_helpers=env_unwrapped,
        hidden_size=args.hidden_size,
        lr=args.lr,
        device=device
    )


def create_csv_logger(run_name):
    os.makedirs("logs", exist_ok=True)
    csv_path = f"logs/{run_name}_metrics.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        'total_steps', 'loss_critic_1', 'loss_critic_2', 'loss_policy',
        'loss_entropy_alpha', 'loss_density_beta', 'entropy_temp_alpha',
        'policy_entropy', 'density_total_mass', 'avg_reward_test', 'std_reward_test',
        'avg_discounted_return_test'
    ])
    csv_writer.writeheader()
    csv_file.flush()
    return csv_file, csv_writer, csv_path


args = parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)

env, env_unwrapped, is_discrete, state_dim, action_dim = create_environment(args)
device = get_device(args)
box_area = env_unwrapped.BOX_AREA
policy, critic, critic_target = build_networks(state_dim, action_dim, args, env, device, is_discrete)
policy_optim, critic_optim = initialize_optimizers(
    policy,
    critic,
    args.lr,
    policy_weight_decay=args.weight_decay_pi
)
density_estimator = create_density_module(policy, env_unwrapped, args, device)
alpha, target_entropy, log_alpha, alpha_optim = setup_entropy_config(args, env, device, is_discrete, action_dim)
run_name = args.run_name if args.run_name else f"SAC_{args.env_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
csv_file, csv_writer, csv_path = create_csv_logger(run_name)

if args.visualize_freq > 0:
    grid_states, grid_shape, plot_extent = prepare_visualization_grid(
        env_unwrapped.STATE_LOW.to(device),
        env_unwrapped.STATE_HIGH.to(device),
        grid_size=args.visualize_grid_size,
        device=device
    )
    os.makedirs("plots/state_distribution", exist_ok=True)
    os.makedirs("plots/policy", exist_ok=True)
    os.makedirs("plots/value", exist_ok=True)
else:
    grid_states = grid_shape = plot_extent = None

# Dictionary to store current row values
csv_row = {}

total_steps = 0
updates = 0
p_loss = c1_loss = c2_loss = d_loss = 0.0
if args.automatic_entropy_tuning and log_alpha is not None:
    alpha_tlogs = log_alpha.exp().detach()
else:
    alpha_tlogs = torch.tensor(alpha, device=device)

latest_metrics = {}

for i in range(args.num_steps):
    
    # --- 5. Use the env's sampling methods ---
    # Sample from proposal distribution m(s)
    state = env_unwrapped.sample_proposal(args.batch_size, device=device)
    
    if is_discrete:
        # Sample random actions for the batch
        action = torch.randint(0, env.action_space.n, (args.batch_size,), device=device)
    else:
        # Sample uniformly within the continuous action bounds
        action_low = torch.tensor(env.action_space.low, device=device, dtype=torch.float32)
        action_high = torch.tensor(env.action_space.high, device=device, dtype=torch.float32)
        action = torch.rand(args.batch_size, action_dim, device=device) * (action_high - action_low) + action_low
    
    # Use the env's vectorized step
    with torch.no_grad():
        next_state, reward, terminated = env_unwrapped.vectorized_step(state, action)
    
    total_steps += 1

    for _ in range(args.updates_per_step):
        state_batch = state
        action_batch = action
        reward_batch = reward.unsqueeze(1)
        next_state_batch = next_state
        terminated_batch = terminated.bool()
        mask_batch = (~terminated_batch).float().unsqueeze(1)

        current_alpha = (log_alpha.exp() 
                         if args.automatic_entropy_tuning else alpha)

        # --- 6. UMBRELLA-SAC CRITIC UPDATE ---
        if is_discrete:
            action_batch = action_batch.to(device).long().unsqueeze(1)

            with torch.no_grad():
                # --- Get state-entropy bonus ---
                log_d_s = density_estimator.density_net.log_prob(state_batch).unsqueeze(1)
                reward_batch_url = reward_batch - args.beta * log_d_s
                # ---

                _, probs_n, log_probs_n = policy.forward(next_state_batch)
                q1_next, q2_next = critic_target(next_state_batch)
                min_q_next = torch.min(q1_next, q2_next)
                v_next = (probs_n * (min_q_next - current_alpha * log_probs_n)).sum(dim=1, keepdim=True)
                
                # Use the modified reward_batch_url
                next_q_value = reward_batch_url + mask_batch * args.gamma * v_next

            q1, q2 = critic(state_batch)
            q1_a = q1.gather(1, action_batch)
            q2_a = q2.gather(1, action_batch)

            qf1_loss = F.smooth_l1_loss(q1_a, next_q_value)
            qf2_loss = F.smooth_l1_loss(q2_a, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            critic_optim.zero_grad()
            qf_loss.backward()
            critic_optim.step()

            # --- POLICY UPDATE ---
            q1_pi, q2_pi = critic(state_batch)
            min_q_pi = torch.min(q1_pi, q2_pi)
            _, probs, log_probs = policy.forward(state_batch)
            policy_loss = (probs * (current_alpha * log_probs - min_q_pi.detach())).sum(dim=1).mean()

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            # --- ALPHA UPDATE ---
            if args.automatic_entropy_tuning:
                entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
                alpha_loss = -(log_alpha * (entropy.detach() + target_entropy)).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                alpha = log_alpha.exp()
                alpha_tlogs = alpha.clone()
            else:
                alpha_loss = torch.tensor(0., device=device)
                alpha_tlogs = float(alpha)

        else: # Continuous actions
            action_batch = action_batch.float()

            with torch.no_grad():
                # --- Get state-entropy bonus ---
                log_d_s = density_estimator.density_net.log_prob(state_batch).unsqueeze(1)
                reward_batch_url = reward_batch - args.beta * log_d_s
                # ---

                next_state_action, next_state_log_pi, _ = policy.sample(next_state_batch)
                qf1_next_target, qf2_next_target = critic_target(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - current_alpha * next_state_log_pi
                
                # Use the modified reward_batch_url
                next_q_value = reward_batch_url + mask_batch * args.gamma * min_qf_next_target

            qf1, qf2 = critic(state_batch, action_batch)
            qf1_loss = F.smooth_l1_loss(qf1, next_q_value)
            qf2_loss = F.smooth_l1_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            critic_optim.zero_grad()
            qf_loss.backward()
            critic_optim.step()

            pi, log_pi, _ = policy.sample(state_batch)
            qf1_pi, qf2_pi = critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, q2_pi)
            policy_loss = ((current_alpha * log_pi) - min_qf_pi).mean()

            policy_optim.zero_grad()
            policy_loss.backward()
            policy_optim.step()

            if args.automatic_entropy_tuning:
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                alpha = log_alpha.exp()
                alpha_tlogs = alpha.clone()
            else:
                alpha_loss = torch.tensor(0., device=device)
                alpha_tlogs = float(alpha)

        # --- 7. DENSITY ESTIMATOR UPDATE ---
        # This update uses its *own* sampling (sample_proposal, etc.)
        # and references the now-updated `policy` network.
        d_loss = density_estimator.update(
            N=args.density_batch_size,
            B=args.density_test_funcs,
            gamma=args.gamma
        )
        # ---

        c1_loss = qf1_loss.item()
        c2_loss = qf2_loss.item()
        p_loss = policy_loss.item()
        e_loss = alpha_loss.item()

        updates += 1
        if updates % args.target_update_interval == 0:
            soft_update(critic_target, critic, args.tau)

            with torch.no_grad():
                if is_discrete:
                    _, p_curr, lp_curr = policy.forward(state_batch)
                    entropy = (-(p_curr * lp_curr).sum(dim=1)).mean().item()
                else:
                    _, log_pi_curr, _ = policy.sample(state_batch)
                    entropy = (-log_pi_curr).mean().item()
            alpha_val = alpha_tlogs.item() if hasattr(alpha_tlogs, 'item') else float(alpha_tlogs)
            latest_metrics = {
                'loss_critic_1': c1_loss,
                'loss_critic_2': c2_loss,
                'loss_policy': p_loss,
                'loss_entropy_alpha': e_loss,
                'loss_density_beta': d_loss,
                'entropy_temp_alpha': alpha_val,
                'policy_entropy': entropy,
                'density_total_mass': csv_row.get('density_total_mass', '')
            }

    if i % args.log_freq == 0 and updates > 0 and latest_metrics:
        alpha_value = None
        if updates > 0:
            alpha_value = alpha_tlogs.item() if hasattr(alpha_tlogs, 'item') else float(alpha_tlogs)
        log_training_iteration(i, updates, p_loss, c1_loss + c2_loss, alpha_value)
        csv_row.update({'total_steps': total_steps, **latest_metrics})
        csv_writer.writerow(csv_row.copy())
        csv_file.flush()

    if i % args.eval_freq == 0 and i > 0 and args.eval:
        avg_eval, std_eval, avg_discounted = evaluate_policy(
            env,
            policy,
            args,
            device,
            is_discrete
        )
        csv_row['avg_reward_test'] = avg_eval
        csv_row['std_reward_test'] = std_eval
        csv_row['avg_discounted_return_test'] = avg_discounted
        csv_row['total_steps'] = total_steps
        csv_writer.writerow(csv_row.copy())
        csv_file.flush()
        print(
            f"--- Evaluation | Iteration {i} | Avg Reward: {avg_eval:.2f} ± {std_eval:.2f} "
            f"| Avg Discounted Return: {avg_discounted:.2f} ---"
        )

    if args.visualize_freq > 0 and (i % args.visualize_freq == 0):
        visualize_state_distribution(
            density_estimator.density_net,
            grid_states,
            grid_shape,
            plot_extent,
            os.path.join("plots", "state_distribution", f"state_dist_{i:06d}.png")
        )
        visualize_policy(
            policy,
            grid_states,
            grid_shape,
            plot_extent,
            os.path.join("plots", "policy", f"policy_{i:06d}.png"),
            is_discrete
        )
        visualize_value_function(
            critic,
            policy,
            grid_states,
            grid_shape,
            plot_extent,
            os.path.join("plots", "value", f"value_{i:06d}.png"),
            is_discrete
        )
        if updates > 0:
            density_mass = estimate_density_total_mass(
                density_estimator.density_net,
                grid_states,
                box_area
            )
            csv_row['density_total_mass'] = density_mass
            csv_row['total_steps'] = total_steps
            csv_writer.writerow(csv_row.copy())
            csv_file.flush()

# --- 9. CHECKPOINT DENSITY NET ---
os.makedirs("checkpoints", exist_ok=True)
ckpt_path = f"checkpoints/url_checkpoint_{args.env_name}_final"
print(f"Saving models to {ckpt_path}")
torch.save({
    'policy_state_dict': policy.state_dict(),
    'critic_state_dict': critic.state_dict(),
    'critic_target_state_dict': critic_target.state_dict(),
    'critic_optimizer_state_dict': critic_optim.state_dict(),
    'policy_optimizer_state_dict': policy_optim.state_dict(),
    'density_state_dict': density_estimator.density_net.state_dict(),
    'density_optimizer_state_dict': density_estimator.density_optim.state_dict(),
    'log_alpha': log_alpha,
    'alpha': log_alpha.exp().item() if args.automatic_entropy_tuning else float(alpha),
}, ckpt_path)

env.close()
csv_file.close()
print(f"CSV logs saved to {csv_path}")
