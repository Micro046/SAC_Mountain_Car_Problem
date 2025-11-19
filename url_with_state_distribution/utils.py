import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

matplotlib.use('Agg')

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def log_training_iteration(iteration, updates, actor_loss, critic_loss, alpha_value):
    """Pretty-print the current training status."""
    print("----------------------------------------")
    print(f"| Iteration {iteration}")
    if updates > 0 and alpha_value is not None:
        print(f"| Actor Loss: {actor_loss:.3f}, Critic Loss: {critic_loss:.3f}, Alpha: {alpha_value:.4f}")
    else:
        print("| Actor Loss: N/A, Critic Loss: N/A, Alpha: N/A (No updates yet)")
    print("----------------------------------------")


def evaluate_policy(env, policy, args, device, is_discrete, eval_episodes=10, max_steps=1000):
    """Run evaluation rollouts and return reward statistics."""
    eval_rewards = []
    eval_discounted_returns = []

    for ep in range(eval_episodes):
        state_np, _ = env.reset(seed=args.seed + 20_000 + ep)
        state = torch.as_tensor(state_np, device=device, dtype=torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0.0
        discounted_return = 0.0
        discount = 1.0
        step_count = 0

        while not done and step_count < max_steps:
            with torch.no_grad():
                if is_discrete:
                    _, probs, _ = policy.forward(state)
                    action_np = int(probs.argmax(dim=-1, keepdim=True).item())
                else:
                    _, _, mean_action = policy.sample(state)
                    action_np = mean_action.squeeze().cpu().numpy()

            next_state_np, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += float(reward)
            discounted_return += discount * float(reward)
            discount *= args.gamma
            state = torch.as_tensor(next_state_np, device=device, dtype=torch.float32).unsqueeze(0)
            step_count += 1

        eval_rewards.append(episode_reward)
        eval_discounted_returns.append(discounted_return)

    avg_eval = float(np.mean(eval_rewards))
    std_eval = float(np.std(eval_rewards))
    avg_discounted = float(np.mean(eval_discounted_returns))
    return avg_eval, std_eval, avg_discounted


def prepare_visualization_grid(state_low, state_high, grid_size=500, device=None):
    device = device or state_low.device
    pos = torch.linspace(state_low[0].item(), state_high[0].item(), grid_size, device=device)
    vel = torch.linspace(state_low[1].item(), state_high[1].item(), grid_size, device=device)
    mesh_pos, mesh_vel = torch.meshgrid(pos, vel, indexing='ij')
    grid_states = torch.stack([mesh_pos, mesh_vel], dim=-1).reshape(-1, 2)
    extent = (
        state_low[0].item(),
        state_high[0].item(),
        state_low[1].item(),
        state_high[1].item(),
    )
    return grid_states, (grid_size, grid_size), extent


def visualize_state_distribution(density_net, grid_states, grid_shape, extent, save_path):
    with torch.no_grad():
        logd = density_net.log_prob(grid_states)
        d_values = torch.exp(logd).cpu().numpy().reshape(grid_shape)

    vmin = max(d_values.min(), 1e-6)
    vmax = d_values.max()
    norm = LogNorm(vmin=vmin, vmax=vmax if vmax > vmin else vmin * 10)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        d_values.T,
        extent=extent,
        origin='lower',
        cmap='gray',
        aspect='auto',
        norm=norm,
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Estimated State Distribution')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def visualize_policy(policy, grid_states, grid_shape, extent, save_path, is_discrete):
    with torch.no_grad():
        if is_discrete:
            _, probs, _ = policy.forward(grid_states)
            action_index = min(1, probs.shape[1] - 1)
            metric = probs[:, action_index]
            title = f'Policy π(a={action_index}|s)'
        else:
            _, _, mean_action = policy.sample(grid_states)
            metric = mean_action.squeeze(-1)
            title = 'Policy Mean Action'

        heatmap = metric.detach().cpu().numpy().reshape(grid_shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        heatmap.T,
        extent=extent,
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def visualize_value_function(critic, policy, grid_states, grid_shape, extent, save_path, is_discrete):
    with torch.no_grad():
        if is_discrete:
            q1, q2 = critic(grid_states)
            q_min = torch.min(q1, q2)
            values = q_min.max(dim=1).values
        else:
            pi_actions, _, _ = policy.sample(grid_states)
            q1, q2 = critic(grid_states, pi_actions)
            values = torch.min(q1, q2).squeeze(-1)

        value_map = values.cpu().numpy().reshape(grid_shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        value_map.T,
        extent=extent,
        origin='lower',
        cmap='plasma',
        aspect='auto'
    )
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_title('Value Function (min Q)')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def estimate_density_total_mass(density_net, grid_states, box_area):
    with torch.no_grad():
        logd = density_net.log_prob(grid_states)
        density = torch.exp(logd)
        avg_density = density.mean()
    return float(avg_density.item() * box_area)
