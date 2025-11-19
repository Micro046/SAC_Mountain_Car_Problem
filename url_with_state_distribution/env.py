import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.registration import register

# --- Module-level device setup ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiValleyMountainCarEnv(gym.Env):
    """
    Multi-valley Mountain Car with discrete actions (2 actions: left/right).
    This class now contains all environment logic, constants, and helper
    functions for sampling and vectorized operations.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, force: float = 0.001, dt: float = 0.1, render_mode: str | None = None):
        super().__init__()

        # --- Device ---
        self.device = DEVICE

        # --- World/physics ---
        self.min_position = -0.99
        self.max_position = 0.99
        self.max_speed = 0.07
        self.dt = dt
        self.force = force
        self.gravity = 0.0025
        
        # --- Goal ---
        self.min_goal_position = -0.05
        self.max_goal_position = 0.05
        self.min_height = -0.2
        self.max_height = 0.6 # For rendering

        # --- Start State ---
        self.min_start_position = 0.67
        self.max_start_position = 0.77
        self.max_start_velocity = 0.01

        # --- Spaces ---
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        
        # --- Torch-based State Bounds (for proposal sampling) ---
        self.STATE_LOW = torch.tensor([self.min_position, -self.max_speed], device=self.device, dtype=torch.float32)
        self.STATE_HIGH = torch.tensor([self.max_position, self.max_speed], device=self.device, dtype=torch.float32)
        self.BOX_AREA = (self.max_position - self.min_position) * (2 * self.max_speed)
        self.LOG_M_S = -math.log(self.BOX_AREA)  # log(1/V) for uniform proposal m(s)

        # --- Rendering ---
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        # --- RNG/state ---
        self.seed()
        self.state = None  # np.array([position, velocity], dtype=float32)

    # -----------------
    # --- TORCH-BASED DYNAMICS HELPERS (for training) ---
    # -----------------
    
    @staticmethod
    @torch.no_grad()
    def _tg_alpha_torch(x_torch: torch.Tensor) -> torch.Tensor:
        """ Computes the terrain slope (torch)"""
        x = x_torch.to(dtype=torch.float32)
        denom = torch.clamp(1.0 - x * x, min=1e-6)
        return 0.1 * (2.0 * x / denom - 2.0 * torch.pi * torch.sin(2.0 * torch.pi * x)
                      - 8.0 * torch.pi * torch.sin(4.0 * torch.pi * x))

    @torch.no_grad()
    def _total_horizontal_force_torch(self, a_torch: torch.Tensor, x_torch: torch.Tensor) -> torch.Tensor:
        """ Computes the net force (torch) """
        # Map action {0,1} -> u in {-1,+1}
        u_torch = a_torch.float() * 2.0 - 1.0
        return u_torch * self.force + self._tg_alpha_torch(x_torch) * (-self.gravity)

    @torch.no_grad()
    def next_state(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Computes the next state given current state(s) and action(s). (torch)
        s: (N, 2)
        a: (N,) or (N, 1)
        """
        s = s.to(self.device)
        a = a.to(self.device)
        
        x = s[:, 0]
        v = s[:, 1]
        
        f = self._total_horizontal_force_torch(a.squeeze(-1), x)
        
        v_new = torch.clamp(v + f * self.dt, -self.max_speed, self.max_speed)
        x_new = torch.clamp(x + v * self.dt, self.min_position, self.max_position)
        
        # Handle the "wall bounce" logic
        v_new = torch.where((x_new == self.min_position) & (v_new < 0), torch.tensor(0.0, device=self.device), v_new)
        v_new = torch.where((x_new == self.max_position) & (v_new > 0), torch.tensor(0.0, device=self.device), v_new)
        
        return torch.stack([x_new, v_new], dim=1)

    @torch.no_grad()
    def vectorized_step(self, s: torch.Tensor, a: torch.Tensor):
        """
        The vectorized step function for your train.py loop. (torch)
        Returns next_state, reward, terminated (as Tensors)
        """
        next_s = self.next_state(s, a)
        
        # Sparse reward: 1 if in goal, 0 otherwise
        pos_new = next_s[:, 0]
        reward = ((pos_new >= self.min_goal_position) & (pos_new <= self.max_goal_position)).to(dtype=torch.float32)
        
        terminated = torch.full((s.shape[0],), False, device=self.device, dtype=torch.bool)
        
        return next_s, reward, terminated

    # -----------------
    # --- SAMPLING HELPERS (for training) ---
    # -----------------
    
    @torch.no_grad()
    def sample_proposal(self, n: int, device=None) -> torch.Tensor:
        """ Samples N states uniformly from the state space m(s) """
        dev = device or self.device
        u = torch.rand(n, 2, device=dev)
        s = self.STATE_LOW.to(dev) + u * (self.STATE_HIGH.to(dev) - self.STATE_LOW.to(dev))
        return s

    @torch.no_grad()
    def sample_start(self, n: int, device=None) -> torch.Tensor:
        """ Samples N states from the initial distribution d_0(s) """
        dev = device or self.device
        sign = torch.where(torch.rand(n, device=dev) < 0.5, -torch.ones(n, device=dev), torch.ones(n, device=dev))
        pos = sign * torch.rand(n, device=dev) * (self.max_start_position - self.min_start_position) + sign * self.min_start_position
        vel = (torch.rand(n, device=dev) * 2 - 1) * self.max_start_velocity
        return torch.stack([pos, vel], dim=1)

    # -----------------
    # --- NUMPY HELPERS (for rendering & gym API) ---
    # -----------------
    
    @staticmethod
    def _height_np(x: np.ndarray | float) -> np.ndarray | float:
        """ Numpy version of height function, for rendering only. """
        x = np.asarray(x, dtype=np.float64)
        denom = np.clip(1.0 - x**2, 1e-6, None)
        return 0.1 * (np.cos(2 * np.pi * x) + 2 * np.cos(4 * np.pi * x) - np.log(denom))

    def _is_on_target(self, position: float) -> bool:
        return (self.min_goal_position <= position) and (position <= self.max_goal_position)
        
    # -----------------
    # --- Gymnasium API ---
    # -----------------
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        # Use the class's torch-based sampler
        self.state = self.sample_start(1, device='cpu').squeeze(0).numpy()
        
        info = {}
        if self.render_mode == "human":
            self.render()
        return self.state.copy(), info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self.state is not None, "Call reset() before step()."

        # --- Use the class's module-level torch functions ---
        state_tensor = torch.tensor(self.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_tensor = torch.tensor([int(action)], dtype=torch.long, device=self.device)

        next_state_tensor = self.next_state(state_tensor, action_tensor)
        self.state = next_state_tensor.squeeze(0).cpu().numpy()
        # ---
        
        position, velocity = self.state

        terminated = False
        truncated = False
        reward = 1.0 if self._is_on_target(position) else 0.0
        info = {}
        
        if self.render_mode == "human":
            self.render()
        
        return self.state.copy(), reward, terminated, truncated, info

    def render(self):
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
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()

        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        # Terrain (use numpy helper)
        xs = np.linspace(self.min_position, self.max_position, 200)
        ys = self._height_np(xs)
        xys = list(zip((xs - self.min_position) * scale, (ys - self.min_height) * scale))
        pygame.draw.aalines(surf, (0, 0, 0), False, xys)

        # Car
        pos = float(self.state[0]) if self.state is not None else 0.0
        # Use torch tg_alpha helper, convert to float
        slope_val = float(self._tg_alpha_torch(torch.tensor(pos, device=self.device)).item())
        angle = np.arctan(slope_val)
        clearance = 5
        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        rect = [np.array([l, b]), np.array([l, t]), np.array([r, t]), np.array([r, b])]
        rot = np.array([[ np.cos(angle), -np.sin(angle)],
                        [ np.sin(angle),  np.cos(angle)]])
        coords = []
        cx = (pos - self.min_position) * scale
        cy = clearance + (self._height_np(pos) - self.min_height) * scale
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
            flagy1 = int((self._height_np(gx) - self.min_height) * scale)
            flagy2 = flagy1 + 50
            gfxdraw.vline(surf, flagx, flagy1, flagy2, (0, 0, 0))
            tri = [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
            gfxdraw.aapolygon(surf, tri, (204, 204, 0))
            gfxdraw.filled_polygon(surf, tri, (204, 204, 0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

# Register so gym.make() works once this module is imported
register(
    id="MultiValleyMountainCarDiscrete-v0",
    entry_point="env:MultiValleyMountainCarEnv",
)

