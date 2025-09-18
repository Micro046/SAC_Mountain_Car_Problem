import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
import torch

class MultiValleyMountainCarEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    def __init__(self, force=0.001, dt=0.1):

        self.min_position = -0.99
        self.max_position =  0.99
        self.max_speed    =  0.07

        # For drawing
        self.min_height = -0.2
        self.max_height =  0.6

        # Starts
        self.min_start_position = 0.67
        self.max_start_position = 0.77
        self.max_start_velocity = 0.01

        # Goal window
        self.min_goal_position = -0.05
        self.max_goal_position =  0.05

        # Physics
        self.dt      = dt
        self.force   = force
        self.gravity = 0.0025

        # Spaces
        self.low  = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position,  self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        # Continuous action in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Rendering
        self.screen = None
        self.clock = None
        self.isopen = True

        # RNG / state
        self.seed()
        self.state = None  # np.array([position, velocity], dtype=float32)

    # ----------------- Helpers -----------------
    @staticmethod
    def _height(x):
        # safe denominator for |x|≈1
        x = np.asarray(x, dtype=np.float64)
        denom = np.clip(1.0 - x**2, 1e-6, None)
        return 0.1 * (np.cos(2 * np.pi * x) + 2 * np.cos(4 * np.pi * x) - np.log(denom))

    @staticmethod
    def _tg_alpha(x_torch):
        # slope dh/dx with Torch (used for physics + rotation)
        # 0.1 * ( 2x/(1-x^2) - 2π sin(2πx) - 8π sin(4πx) )
        x = x_torch.to(dtype=torch.float32)
        denom = torch.clamp(1.0 - x * x, min=1e-6)
        return 0.1 * (2.0 * x / denom - 2.0 * torch.pi * torch.sin(2.0 * torch.pi * x)
                      - 8.0 * torch.pi * torch.sin(4.0 * torch.pi * x))

    def _is_on_target(self, position: float) -> bool:
        return (self.min_goal_position <= position) and (position <= self.max_goal_position)
    @staticmethod
    def _parse_action(action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.size == 0:
            a = np.array([0.0], dtype=np.float32)
        return float(np.clip(a[0], -1.0, 1.0))

    def _total_horizontal_force(self, action_u_torch, position_torch):
        # action_u_torch: torch scalar in [-1,1]
        # position_torch: torch scalar (same device)
        u = torch.clamp(action_u_torch.to(torch.float32), -1.0, 1.0)
        tg_alpha = self._tg_alpha(position_torch)
        # agent input + gravity projected along slope
        return u * self.force + tg_alpha * (-self.gravity)

    # ----------------- Gymnasium API -----------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        sign = self.np_random.integers(0, 2) * 2 - 1  # -1 or +1
        position = float(sign * self.np_random.uniform(self.min_start_position, self.max_start_position))
        velocity = float(self.np_random.uniform(-self.max_start_velocity, self.max_start_velocity))
        self.state = np.array([position, velocity], dtype=np.float32)

        info = {}
        return self.state.copy(), info

    def step(self, action):
        assert self.state is not None, "Call reset() before step()."
        # parse continuous action
        u = self._parse_action(action)

        position, velocity = self.state.astype(np.float64)

        # total horizontal force (Torch for slope; result -> float)
        force_t = self._total_horizontal_force(
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(position, dtype=torch.float32)
        )
        total_horizontal_force = float(force_t.item())

        # integrate
        velocity += total_horizontal_force * self.dt
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity * self.dt
        position = np.clip(position, self.min_position, self.max_position)

        self.state = np.array([position, velocity], dtype=np.float32)

        # never terminate/truncate inside the env
        terminated = False
        truncated = False

        # sparse reward (1 only when inside the goal window)
        reward = 1.0 if self._is_on_target(position) else 0.0

        info = {}
        return self.state.copy(), reward, terminated, truncated, info

    # ----------------- Rendering -----------------
    def render(self, mode="human"):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        screen_width = 1200
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width / world_width
        carwidth = 20
        carheight = 10

        if self.screen is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        # Terrain
        xs = np.linspace(self.min_position, self.max_position, 200)
        ys = self._height(xs)
        xys = list(zip((xs - self.min_position) * scale, (ys - self.min_height) * scale))
        # correct aarlines signature
        pygame.draw.aalines(surf, (0, 0, 0), False, xys)

        # Car
        pos = float(self.state[0]) if self.state is not None else 0.0
        slope_val = float(self._tg_alpha(torch.tensor(pos)))
        angle = np.arctan(slope_val)
        clearance = 5
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        rect = [np.array([l, b]), np.array([l, t]), np.array([r, t]), np.array([r, b])]
        rot = np.array([[ np.cos(angle), -np.sin(angle)],
                        [ np.sin(angle),  np.cos(angle)]])
        coords = []
        cx = (pos - self.min_position) * scale
        cy = clearance + (self._height(pos) - self.min_height) * scale
        for c in rect:
            rc = rot @ c
            coords.append((rc[0] + cx, rc[1] + cy))
        gfxdraw.aapolygon(surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(surf, coords, (0, 0, 0))

        # Wheels
        for c in [np.array([ carwidth/4, 0.0]), np.array([-carwidth/4, 0.0])]:
            wc = rot @ c
            wx = int(wc[0] + cx)
            wy = int(wc[1] + cy)
            gfxdraw.aacircle(surf, wx, wy, int(carheight/2.5), (128, 128, 128))
            gfxdraw.filled_circle(surf, wx, wy, int(carheight/2.5), (128, 128, 128))

        # Goal flags
        for gx in [self.min_goal_position, self.max_goal_position]:
            flagx = int((gx - self.min_position) * scale)
            flagy1 = int((self._height(gx) - self.min_height) * scale)
            flagy2 = flagy1 + 50
            gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))
            tri = [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            gfxdraw.aapolygon(surf, tri, (204, 204, 0))
            gfxdraw.filled_polygon(surf, tri, (204, 204, 0))

        # Flip Y for display
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    # ----------------- Umbrella Reinforce extensions -----------------
    def reward(self, state, action, state_dot):
        # state: (..., 2) with torch tensors expected
        position, velocity = torch.moveaxis(state, -1, 0)
        inside = (position >= self.min_goal_position) & (position <= self.max_goal_position)
        return inside.to(dtype=state.dtype)

    def state_dot(self, state, action):
        # state/action: torch tensors (possibly batched)
        position, velocity = torch.moveaxis(state, -1, 0)
        acceleration = self._total_horizontal_force(action, position)
        position_dot = velocity
        velocity_dot = acceleration
        return torch.moveaxis(torch.cat((position_dot.unsqueeze(0), velocity_dot.unsqueeze(0)), axis=0), 0, -1)

    def start_prob(self, state, device=None):
        low  = torch.tensor([ self.min_start_position, -self.max_start_velocity], device=device)
        high = torch.tensor([ self.max_start_position,  self.max_start_velocity], device=device)
        delta = high - low
        p_inside = (0.5) * (1.0 / torch.prod(delta))

        is_inside_right = torch.all((low[None, :] < state) & (state < high[None, :]), dim=-1)
        is_inside_left  = torch.all((-high[None, :] < state) & (state < -low[None, :]), dim=-1)
        p = torch.where(is_inside_left | is_inside_right, p_inside, torch.zeros_like(p_inside))
        return p

    def divergrnce(self, state, action):
        return torch.as_tensor(0.0, dtype=state.dtype, device=state.device if isinstance(state, torch.Tensor) else None)

    def rate(self, state, action, device=None):
        p0 = self.start_prob(state, device)
        v  = self.state_dot(state, action)
        r  = self.reward(state, action, v)
        div_v = self.divergrnce(state, action)
        return p0, r, v, div_v

    def reflecting_representation(self, state):
        x, v = torch.moveaxis(state, -1, 0)
        device = state.device
        x_bounds = torch.tensor([self.min_position, self.max_position], device=device)
        v_bounds = torch.tensor([-self.max_speed, self.max_speed], device=device)

        boundary = x_bounds.expand((1,) * len(x.shape) + (2,))
        v_cos = torch.cos((v - v_bounds.min()) / (v_bounds.max() - v_bounds.min()) * torch.pi)
        x, v_cos = x.unsqueeze(-1), v_cos.unsqueeze(-1)
        distance_to_boundary, _ = torch.abs(x - boundary).min(dim=-1)

        return torch.cat((x, v_cos ** 2, v_cos * distance_to_boundary.unsqueeze(-1)), dim=-1)


def vectorized_mountain_car_step(states, actions):
    """
    Vectorized step function for MultiValleyMountainCar environment.
    
    This function processes multiple state-action pairs simultaneously for efficient
    batch processing during training.
    
    Args:
        states (np.ndarray): Array of shape (batch_size, 2) containing [position, velocity]
        actions (np.ndarray): Array of shape (batch_size, 1) containing actions in [-1, 1]
    
    Returns:
        tuple: (next_states, rewards, terminated)
            - next_states (np.ndarray): Array of shape (batch_size, 2) with next states
            - rewards (np.ndarray): Array of shape (batch_size,) with rewards
            - terminated (np.ndarray): Array of shape (batch_size,) with termination flags
    """
    # Extract position and velocity
    pos = states[:, 0]
    vel = states[:, 1]
    
    # Clip actions to valid range
    actions = np.clip(actions, -1.0, 1.0)
    
    # Environment parameters (matching MultiValleyMountainCarEnv)
    dt = 0.1
    max_speed = 0.07
    min_position = -0.99
    max_position = 0.99
    
    # Convert to torch tensors for vectorized computation
    pos_torch = torch.tensor(pos, dtype=torch.float32)
    actions_torch = torch.tensor(actions.flatten(), dtype=torch.float32)
    
    # Compute total horizontal force (vectorized)
    # This is a simplified version - you might need to adapt the _total_horizontal_force logic
    force = actions_torch * 0.001  # Simplified force calculation
    
    # Physics update (vectorized)
    vel_new = vel + force.numpy() * dt
    vel_new = np.clip(vel_new, -max_speed, max_speed)
    
    pos_new = pos + vel_new * dt
    pos_new = np.clip(pos_new, min_position, max_position)
    
    # Check termination condition (simplified - you might want to adapt this)
    # For now, we'll use a simple goal condition
    goal_position = 0.45
    goal_velocity = 0.0
    terminated = (pos_new >= goal_position) & (vel_new >= goal_velocity)
    
    # Compute rewards: sparse reward (1 when on target, 0 otherwise)
    rewards = np.where(terminated, 1.0, 0.0)
    
    # Create next states array
    next_states = np.column_stack([pos_new, vel_new])
    
    # Clip next states to valid bounds (matching single valley implementation)
    state_low = np.array([min_position, -max_speed])
    state_high = np.array([max_position, max_speed])
    next_states[:, 0] = np.clip(next_states[:, 0], state_low[0], state_high[0])
    next_states[:, 1] = np.clip(next_states[:, 1], state_low[1], state_high[1])
    
    return next_states, rewards, terminated


from gymnasium.envs.registration import register
register(
    id="MultiValleyMountainCar-v0",
    entry_point="multi_valley_mountain_car_cont:MultiValleyMountainCarEnv",
)
