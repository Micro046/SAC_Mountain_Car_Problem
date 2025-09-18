import argparse
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
from discrete_sac_agent import DiscreteSACAgent
from discrete_mountain_car import DiscreteMountainCarEnv
from URL_continous.utils import plot_learning_curve
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Discrete SAC on Mountain Car")
    
    # Environment settings
    parser.add_argument('--env-name', type=str, default='DiscreteMountainCar-v0',
                        choices=['DiscreteMountainCar-v0', 'DiscreteMountainCarDense-v0'],
                        help='Environment name')
    parser.add_argument('--max-episode-steps', type=int, default=200,
                        help='Maximum steps per episode')
    
    # Training settings
    parser.add_argument('--num-episodes', type=int, default=1500,
                        help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=300000,
                        help='Maximum training steps')
    parser.add_argument('--eval-freq', type=int, default=50,
                        help='Evaluation frequency (episodes)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of episodes for evaluation')
    
    # Agent hyperparameters (optimized for Mountain Car)
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Soft update parameter')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Temperature parameter')
    parser.add_argument('--automatic-entropy-tuning', action='store_true', default=False,
                        help='Enable automatic entropy tuning')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden layer dimension')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--start-steps', type=int, default=5000,
                        help='Random action steps before training')
    parser.add_argument('--update-freq', type=int, default=1,
                        help='Update frequency (steps)')
    
    # System settings
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Logging and saving
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='Save trained model')
    parser.add_argument('--save-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation (disabled by default)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed training information')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device(device_arg):
    """Get the appropriate device for training"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_arg)


def evaluate_agent(agent, env, num_episodes=10, render=False):
    """
    Evaluate the agent's performance.
    
    Args:
        agent: The SAC agent to evaluate
        env: The environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        dict: Evaluation results
    """
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Count successful episodes (reached goal)
        if episode_reward > 50:  # Assuming success gives +100 reward
            success_count += 1
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
        'episode_rewards': episode_rewards
    }


def train_discrete_sac(args):
    """Main training function"""
    
    # Set up device and seed
    device = get_device(args.device)
    set_seed(args.seed)
    print(f"Using device: {device}")
    
    # Create environment
    env = gym.make(args.env_name)
    env.action_space.seed(args.seed)
    
    # Create evaluation environment (no rendering by default)
    eval_env = gym.make(args.env_name)
    eval_env.action_space.seed(args.seed + 1)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {args.env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Print hyperparameters
    print(f"\nHyperparameters:")
    print(f"  Learning rate: {args.lr}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Tau: {args.tau}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Automatic entropy tuning: {args.automatic_entropy_tuning}")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size}")
    print(f"  Start steps: {args.start_steps}")
    print(f"  Render during evaluation: {args.render}")
    
    # Create agent
    agent = DiscreteSACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        automatic_entropy_tuning=args.automatic_entropy_tuning,
        hidden_dim=args.hidden_dim,
        buffer_size=args.buffer_size,
        device=device
    )
    
    # Training tracking
    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    eval_episodes = []
    training_stats = []
    
    # Training loop
    episode = 0
    total_steps = 0
    state, _ = env.reset()
    
    print("Starting training...")
    print(f"Random exploration steps: {args.start_steps}")
    print(f"Training episodes: {args.num_episodes}")
    print(f"Max training steps: {args.max_steps}")
    
    while episode < args.num_episodes and total_steps < args.max_steps:
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and total_steps < args.max_steps:
            # Select action
            if total_steps < args.start_steps:
                action = env.action_space.sample()  # Random action
            else:
                action = agent.select_action(state, evaluate=False)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update counters
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            state = next_state
            
            # Update agent
            if total_steps > args.start_steps and total_steps % args.update_freq == 0:
                stats = agent.update(args.batch_size)
                if stats and args.verbose:
                    training_stats.append(stats)
        
        # End of episode
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1
        
        # Reset environment for next episode
        if not done:  # Only reset if we didn't finish naturally
            state, _ = env.reset()
        else:
            state, _ = env.reset()
        
        # Evaluation
        if episode % args.eval_freq == 0:
            eval_results = evaluate_agent(agent, eval_env, args.eval_episodes, args.render)
            eval_rewards.append(eval_results['mean_reward'])
            eval_episodes.append(episode)
            
            print(f"Episode {episode}/{args.num_episodes}, Steps: {total_steps}")
            print(f"  Training - Mean Reward: {np.mean(episode_rewards[-args.eval_freq:]):.2f}")
            print(f"  Evaluation - Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Success Rate: {eval_results['success_rate']:.2%}")
            print(f"  Alpha: {agent.alpha.item():.4f}")
            if training_stats:
                recent_stats = training_stats[-100:] if len(training_stats) > 100 else training_stats
                avg_q_loss = np.mean([s['q_loss'] for s in recent_stats if 'q_loss' in s])
                avg_policy_loss = np.mean([s['policy_loss'] for s in recent_stats if 'policy_loss' in s])
                print(f"  Q-Loss: {avg_q_loss:.4f}, Policy Loss: {avg_policy_loss:.4f}")
            print("-" * 50)
    
    print("Training completed!")
    
    # Final evaluation
    print("Running final evaluation...")
    final_eval = evaluate_agent(agent, eval_env, args.eval_episodes * 2, False)
    print(f"Final Evaluation Results:")
    print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Success Rate: {final_eval['success_rate']:.2%}")
    
    # Save results
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    if args.save_model:
        model_path = os.path.join(args.save_dir, f"discrete_sac_{args.env_name}_{timestamp}.pth")
        agent.save(model_path)
        print(f"Model saved to: {model_path}")
    
    # Plot and save learning curves
    if episode_rewards:
        # Training curve
        episodes_range = list(range(1, len(episode_rewards) + 1))
        train_plot_path = os.path.join(args.save_dir, f"training_curve_{timestamp}.png")
        plot_learning_curve(episodes_range, episode_rewards, train_plot_path, window=50)
        print(f"Training curve saved to: {train_plot_path}")
        
        # Evaluation curve
        if eval_rewards:
            eval_plot_path = os.path.join(args.save_dir, f"evaluation_curve_{timestamp}.png")
            plt.figure(figsize=(10, 6))
            plt.plot(eval_episodes, eval_rewards, 'b-', linewidth=2, label='Evaluation Reward')
            plt.xlabel('Episode')
            plt.ylabel('Mean Reward')
            plt.title('Evaluation Performance')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(eval_plot_path)
            plt.close()
            print(f"Evaluation curve saved to: {eval_plot_path}")
    
    # Save training data
    results = {
        'args': vars(args),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'eval_rewards': eval_rewards,
        'eval_episodes': eval_episodes,
        'final_evaluation': final_eval,
        'training_stats': agent.get_statistics(),
        'total_steps': total_steps,
        'total_episodes': episode
    }
    
    results_path = os.path.join(args.save_dir, f"results_{timestamp}.npz")
    np.savez(results_path, **results)
    print(f"Training data saved to: {results_path}")
    
    env.close()
    eval_env.close()
    
    return results


if __name__ == "__main__":
    args = parse_args()
    results = train_discrete_sac(args)
