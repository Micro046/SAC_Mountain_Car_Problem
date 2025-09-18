import math
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.error import DependencyNotInstalled


class DiscreteMountainCarEnv(gym.Env):
    """
    Discrete-action Mountain Car environment based on the classic control problem.
    
    The agent needs to drive a car up a steep hill. The car cannot accelerate
    directly up the hill, so it must build momentum by driving back and forth.
    
    Action space: Discrete(3) - [0: push left, 1: no push, 2: push right]
    Observation space: Box([position, velocity])
    
    Rewards:
        - Sparse reward: +100 when reaching the goal, -1 for each step
        - Dense reward option: reward based on progress towards goal
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self,
                 render_mode: Optional[str] = None,
                 goal_velocity: float = 0.0,
                 reward_type: str = "sparse"):
        """
        Initialize the discrete Mountain Car environment.
        
        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            goal_velocity: Minimum velocity required at goal position
            reward_type: "sparse" for sparse rewards, "dense" for dense rewards
        """
        super().__init__()
        
        # Physics parameters
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.gravity = 0.0025
        
        # Reward configuration
        self.reward_type = reward_type
        
        # Action space: 0=push left, 1=no push, 2=push right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [position, velocity]
        low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(low_state, high_state, dtype=np.float32)
        
        # Rendering
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        
        self.state = None
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0, 1, or 2)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        pos, vel = self.state
        
        # Convert discrete action to continuous force
        # 0: push left (-1), 1: no push (0), 2: push right (+1)
        force = action - 1  # Maps 0->-1, 1->0, 2->+1
        
        # Physics update
        vel += force * self.power - self.gravity * math.cos(3 * pos)
        vel = np.clip(vel, -self.max_speed, self.max_speed)
        
        pos += vel
        pos = np.clip(pos, self.min_position, self.max_position)
        
        # Handle left boundary condition
        if pos == self.min_position and vel < 0:
            vel = 0.0
        
        # Check termination
        terminated = bool(pos >= self.goal_position and vel >= self.goal_velocity)
        
        # Calculate reward
        if self.reward_type == "sparse":
            reward = 100.0 if terminated else -1.0
        else:  # dense reward
            reward = self._dense_reward(pos, vel, terminated)
        
        self.state = np.array([pos, vel], dtype=np.float32)
        
        if self.render_mode == "human":
            self.render()
        
        return self.state, reward, terminated, False, {}
    
    def _dense_reward(self, pos, vel, terminated):
        """
        Calculate dense reward based on progress towards goal.
        
        Args:
            pos: Current position
            vel: Current velocity
            terminated: Whether episode is terminated
            
        Returns:
            float: Dense reward value
        """
        if terminated:
            return 100.0
        
        # Progress reward based on position
        progress = (pos - self.min_position) / (self.goal_position - self.min_position)
        progress_reward = progress * 10  # Scale progress reward
        
        # Velocity reward when moving towards goal
        velocity_reward = max(0, vel) * 10 if pos > -0.5 else 0
        
        # Small negative reward for each step to encourage efficiency
        step_penalty = -1.0
        
        return progress_reward + velocity_reward + step_penalty
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Initialize position randomly in the valley
        pos = self.np_random.uniform(low=-0.6, high=-0.4)
        self.state = np.array([pos, 0.0], dtype=np.float32)
        
        if self.render_mode == "human":
            self.render()
        
        return self.state.copy(), {}
    
    @staticmethod
    def _height(x):
        """Calculate height of the hill at position x"""
        return np.sin(3 * x) * 0.45 + 0.55
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            gym.logger.warn("Render mode was not set!")
            return
        
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed. Run `pip install "gymnasium[classic-control]"`'
            )
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        world_width = self.max_position - self.min_position
        scale = self.screen_width / world_width
        carw, carh = 40, 20
        
        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255, 255, 255))
        
        # Draw track
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        pts = list(zip((xs - self.min_position) * scale, ys * scale))
        pygame.draw.aalines(surf, (0, 0, 0), False, pts)
        
        # Draw car
        pos = self.state[0]
        clearance = 10
        rot = math.cos(3 * pos)
        l, r, t, b = -carw / 2, carw / 2, carh, 0
        corners = []
        for cx, cy in ((l, b), (l, t), (r, t), (r, b)):
            vec = pygame.math.Vector2(cx, cy).rotate_rad(rot)
            corners.append((
                vec.x + (pos - self.min_position) * scale,
                vec.y + clearance + self._height(pos) * scale
            ))
        gfxdraw.aapolygon(surf, corners, (0, 0, 0))
        gfxdraw.filled_polygon(surf, corners, (0, 0, 0))
        
        # Draw wheels
        for wx, wy in ((carw / 4, 0), (-carw / 4, 0)):
            vec = pygame.math.Vector2(wx, wy).rotate_rad(rot)
            wheel = (
                int(vec.x + (pos - self.min_position) * scale),
                int(vec.y + clearance + self._height(pos) * scale)
            )
            gfxdraw.aacircle(surf, wheel[0], wheel[1], int(carh / 2.5), (80, 80, 80))
            gfxdraw.filled_circle(surf, wheel[0], wheel[1], int(carh / 2.5), (80, 80, 80))
        
        # Draw goal flag
        gx = int((self.goal_position - self.min_position) * scale)
        gy = int(self._height(self.goal_position) * scale)
        gfxdraw.vline(surf, gx, gy, gy + 50, (0, 0, 0))
        gfxdraw.filled_polygon(
            surf,
            [(gx, gy + 50), (gx, gy + 40), (gx + 25, gy + 45)],
            (204, 204, 0),
        )
        
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Close the environment and cleanup resources"""
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def vectorized_discrete_mountain_car_step(states, actions):
    """
    Vectorized step function for discrete Mountain Car environment.
    
    Args:
        states (np.ndarray): Array of shape (batch_size, 2) containing [position, velocity]
        actions (np.ndarray): Array of shape (batch_size,) containing discrete actions (0, 1, 2)
    
    Returns:
        tuple: (next_states, rewards, terminated)
            - next_states (np.ndarray): Array of shape (batch_size, 2) with next states
            - rewards (np.ndarray): Array of shape (batch_size,) with rewards
            - terminated (np.ndarray): Array of shape (batch_size,) with termination flags
    """
    # Extract position and velocity
    pos = states[:, 0]
    vel = states[:, 1]
    
    # Convert discrete actions to continuous forces
    # 0: push left (-1), 1: no push (0), 2: push right (+1)
    forces = actions.astype(np.float32) - 1.0
    
    # Environment parameters
    power = 0.0015
    gravity = 0.0025
    max_speed = 0.07
    min_position = -1.2
    max_position = 0.6
    goal_position = 0.45
    goal_velocity = 0.0
    
    # Physics update (vectorized)
    vel_new = vel + forces * power - gravity * np.cos(3.0 * pos)
    vel_new = np.clip(vel_new, -max_speed, max_speed)
    
    pos_new = pos + vel_new
    pos_new = np.clip(pos_new, min_position, max_position)
    
    # Handle boundary condition: if at left boundary and velocity negative, set to 0
    boundary_mask = (pos_new == min_position) & (vel_new < 0.0)
    vel_new[boundary_mask] = 0.0
    
    # Check termination condition
    terminated = (pos_new >= goal_position) & (vel_new >= goal_velocity)
    
    # Compute sparse rewards: +100 for success, -1 for each step
    rewards = np.where(terminated, 100.0, -1.0)
    
    # Create next states array
    next_states = np.column_stack([pos_new, vel_new]).astype(np.float32)
    
    return next_states, rewards, terminated


# Register the environment
register(
    id="DiscreteMountainCar-v0",
    entry_point="discrete_mountain_car:DiscreteMountainCarEnv",
    max_episode_steps=200,
    kwargs={"reward_type": "sparse"}
)

register(
    id="DiscreteMountainCarDense-v0",
    entry_point="discrete_mountain_car:DiscreteMountainCarEnv",
    max_episode_steps=200,
    kwargs={"reward_type": "dense"}
)
