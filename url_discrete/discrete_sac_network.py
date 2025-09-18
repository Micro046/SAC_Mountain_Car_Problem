import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


def weights_init_(m):
    """Initialize network weights using Xavier uniform initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class DiscreteQNetwork(nn.Module):
    """
    Q-Network for discrete action spaces.
    Outputs Q-values for each discrete action.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DiscreteQNetwork, self).__init__()
        
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        
        # Q2 architecture (for double Q-learning)
        self.linear4 = nn.Linear(num_inputs, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Forward pass through both Q-networks
        
        Args:
            state: State tensor of shape (batch_size, num_inputs)
            
        Returns:
            tuple: (q1_values, q2_values) where each has shape (batch_size, num_actions)
        """
        # Q1 forward pass
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        q1 = self.linear3(x1)
        
        # Q2 forward pass
        x2 = F.relu(self.linear4(state))
        x2 = F.relu(self.linear5(x2))
        q2 = self.linear6(x2)
        
        return q1, q2


class DiscretePolicy(nn.Module):
    """
    Policy network for discrete action spaces.
    Outputs action probabilities using softmax.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DiscretePolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Forward pass to get action logits
        
        Args:
            state: State tensor of shape (batch_size, num_inputs)
            
        Returns:
            torch.Tensor: Action logits of shape (batch_size, num_actions)
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits
    
    def sample(self, state):
        """
        Sample actions from the policy
        
        Args:
            state: State tensor of shape (batch_size, num_inputs)
            
        Returns:
            tuple: (actions, log_probs, probs)
                - actions: Sampled actions (batch_size,)
                - log_probs: Log probabilities of sampled actions (batch_size,)
                - probs: Action probabilities (batch_size, num_actions)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        # Create categorical distribution
        dist = Categorical(probs)
        actions = dist.sample()
        
        # Get log probabilities of sampled actions
        log_probs = dist.log_prob(actions)
        
        return actions, log_probs, probs
    
    def get_log_probs(self, state, actions):
        """
        Get log probabilities for given state-action pairs
        
        Args:
            state: State tensor of shape (batch_size, num_inputs)
            actions: Action tensor of shape (batch_size,)
            
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size,)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        return log_probs
    
    def get_action_probs(self, state):
        """
        Get action probabilities for all actions
        
        Args:
            state: State tensor of shape (batch_size, num_inputs)
            
        Returns:
            torch.Tensor: Action probabilities of shape (batch_size, num_actions)
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        return probs


class ValueNetwork(nn.Module):
    """
    Value network for state value estimation (optional, can be derived from Q-values)
    """
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init_)
    
    def forward(self, state):
        """
        Forward pass to get state values
        
        Args:
            state: State tensor of shape (batch_size, num_inputs)
            
        Returns:
            torch.Tensor: State values of shape (batch_size, 1)
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value
