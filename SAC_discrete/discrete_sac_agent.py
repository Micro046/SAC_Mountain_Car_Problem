import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from discrete_sac_network import DiscreteQNetwork, DiscretePolicy
from URL_continous.utils import soft_update, hard_update


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done)
        )
    
    def __len__(self):
        return len(self.buffer)


class DiscreteSACAgent:
    """
    Soft Actor-Critic agent for discrete action spaces.
    
    This implementation uses the discrete SAC formulation where:
    - Policy outputs action probabilities
    - Q-networks output Q-values for each action
    - Entropy regularization is applied to action probabilities
    """
    
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 automatic_entropy_tuning=True,
                 # Exploration aids for sparse rewards
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_steps=50000,
                 # Prevent alpha collapse when auto-tuning
                 min_alpha=0.02,
                 hidden_dim=256,
                 buffer_size=1000000,
                 device='cpu'):
        """
        Initialize the discrete SAC agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of discrete actions
            lr: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update parameter
            alpha: Temperature parameter for entropy regularization
            automatic_entropy_tuning: Whether to automatically tune alpha
            hidden_dim: Hidden layer size for networks
            buffer_size: Size of replay buffer
            device: Device to run computations on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.automatic_entropy_tuning = automatic_entropy_tuning
        # Epsilon-greedy schedule (training-time exploration)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.total_env_steps = 0
        # Minimum alpha when auto tuning to avoid premature determinism
        self.min_alpha = float(min_alpha)
        
        # Initialize networks
        self.q_network = DiscreteQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_target = DiscreteQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy = DiscretePolicy(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize target network
        hard_update(self.q_target, self.q_network)
        
        # Optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Entropy tuning
        if automatic_entropy_tuning:
            # Target entropy is -log(1/|A|) = log(|A|) for discrete actions
            self.target_entropy = np.log(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=device)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.q_loss_history = []
        self.policy_loss_history = []
        self.alpha_loss_history = []
        self.alpha_history = []
    
    def select_action(self, state, evaluate=False):
        """
        Select action using the current policy.
        
        Args:
            state: Current state
            evaluate: If True, use deterministic action selection
            
        Returns:
            int: Selected action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if evaluate:
                # Use deterministic policy (argmax)
                action_probs = self.policy.get_action_probs(state)
                action = action_probs.argmax(dim=-1)
                return action.cpu().numpy()[0]

            # Training: epsilon-greedy over policy
            # Linear decay of epsilon from start to end over decay steps
            if self.epsilon_decay_steps > 0:
                frac = min(1.0, max(0.0, self.total_env_steps / float(self.epsilon_decay_steps)))
                epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac
            else:
                epsilon = self.epsilon_end

            if np.random.rand() < epsilon:
                return int(np.random.randint(0, self.action_dim))
            else:
                action, _, _ = self.policy.sample(state)
                return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
        # Count environment interaction steps for epsilon schedule
        self.total_env_steps += 1
    
    def update(self, batch_size=256):
        """
        Update the agent's networks using a batch of experiences.
        
        Args:
            batch_size: Size of the batch to sample from replay buffer
            
        Returns:
            dict: Training statistics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Update Q-networks
        q_loss = self._update_q_networks(states, actions, rewards, next_states, dones)
        
        # Update policy
        policy_loss = self._update_policy(states)
        
        # Update temperature parameter
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = self._update_alpha(states)
        
        # Update target networks
        soft_update(self.q_target, self.q_network, self.tau)
        
        # Store statistics
        stats = {
            'q_loss': q_loss,
            'policy_loss': policy_loss,
            'alpha': self.alpha.item()
        }
        
        if alpha_loss is not None:
            stats['alpha_loss'] = alpha_loss
        
        self.q_loss_history.append(q_loss)
        self.policy_loss_history.append(policy_loss)
        self.alpha_history.append(self.alpha.item())
        if alpha_loss is not None:
            self.alpha_loss_history.append(alpha_loss)
        
        return stats
    
    def _update_q_networks(self, states, actions, rewards, next_states, dones):
        """Update Q-networks using Bellman equation"""
        with torch.no_grad():
            # Get next state action probabilities
            next_action_probs = self.policy.get_action_probs(next_states)
            next_log_probs = torch.log(next_action_probs + 1e-8)
            
            # Get target Q-values
            next_q1_target, next_q2_target = self.q_target(next_states)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Compute target value using expectation over actions
            next_v_target = (next_action_probs * (next_q_target - self.alpha * next_log_probs)).sum(dim=-1, keepdim=True)
            
            # Compute target Q-value
            q_target = rewards + (1 - dones) * self.gamma * next_v_target
        
        # Get current Q-values
        q1_current, q2_current = self.q_network(states)
        q1_current = q1_current.gather(1, actions.unsqueeze(1))
        q2_current = q2_current.gather(1, actions.unsqueeze(1))
        
        # Compute Q-losses
        q1_loss = F.mse_loss(q1_current, q_target)
        q2_loss = F.mse_loss(q2_current, q_target)
        q_loss = q1_loss + q2_loss
        
        # Update Q-networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        return q_loss.item()
    
    def _update_policy(self, states):
        """Update policy network"""
        # Get action probabilities and log probabilities
        action_probs = self.policy.get_action_probs(states)
        log_probs = torch.log(action_probs + 1e-8)
        
        # Get Q-values
        q1, q2 = self.q_network(states)
        q_values = torch.min(q1, q2)
        
        # Compute policy loss using expectation over actions
        policy_loss = (action_probs * (self.alpha * log_probs - q_values)).sum(dim=-1).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def _update_alpha(self, states):
        """Update temperature parameter alpha"""
        with torch.no_grad():
            action_probs = self.policy.get_action_probs(states)
            log_probs = torch.log(action_probs + 1e-8)
            entropy = -(action_probs * log_probs).sum(dim=-1).mean()
        
        alpha_loss = -self.log_alpha * (entropy - self.target_entropy)
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Clamp alpha to avoid collapse
        self.alpha = torch.clamp(self.log_alpha.exp(), min=self.min_alpha)
        
        return alpha_loss.item()
    
    def save(self, filepath):
        """Save agent's networks and parameters"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'q_optimizer_state_dict': self.q_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'automatic_entropy_tuning': self.automatic_entropy_tuning,
                'target_entropy': self.target_entropy if self.automatic_entropy_tuning else None
            }
        }, filepath)
    
    def load(self, filepath):
        """Load agent's networks and parameters"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        
        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            self.alpha = self.log_alpha.exp()
    
    def get_statistics(self):
        """Get training statistics"""
        return {
            'q_loss_history': self.q_loss_history,
            'policy_loss_history': self.policy_loss_history,
            'alpha_loss_history': self.alpha_loss_history,
            'alpha_history': self.alpha_history
        }
