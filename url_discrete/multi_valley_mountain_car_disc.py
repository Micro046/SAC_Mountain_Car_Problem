import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.registration import register
import torch


class MultiValleyMountainCarEnv(gym.Env):
    """
    Multi-valley Mountain Car with discrete actions (2 actions: left/right).

    Observation (Box(2)):
      0: position  in [-0.99, 0.99]
      1: velocity  in [-0.07, 0.07]

    Actions (Discrete(2)):
      0: push left  (maps to u = -1)
      1: push right (maps to u = +1)

    Reward:
      1 when position is inside goal window (|position| <= 0.05), else 0.

    Episode termination/truncation:
      Never (handled externally if desired).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, force: float = 0.001, dt: float = 0.1, render_mode: str | None = None):
        super().__init__()

        # World/physics
        self.min_position = -0.99
        self.max_position = 0.99
        self.max_speed = 0.07

        self.min_height = -0.2
        self.max_height = 0.6

        self.min_goal_position = -0.05
        self.max_goal_position = 0.05

        self.min_start_position = 0.67
        self.max_start_position = 0.77
        self.max_start_velocity = 0.01

        self.dt = dt
        self.force = force
        self.gravity = 0.0025

        # Spaces
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.isopen = True

        # RNG/state
        self.seed()
        self.state = None  # np.array([position, velocity], dtype=float32)

    # ----------------- Helpers -----------------
    @staticmethod
    def _height(x: np.ndarray | float) -> np.ndarray | float:
        x = np.asarray(x, dtype=np.float64)
        denom = np.clip(1.0 - x**2, 1e-6, None)
        return 0.1 * (np.cos(2 * np.pi * x) + 2 * np.cos(4 * np.pi * x) - np.log(denom))

    @staticmethod
    def _tg_alpha(x_torch: torch.Tensor) -> torch.Tensor:
        x = x_torch.to(dtype=torch.float32)
        denom = torch.clamp(1.0 - x * x, min=1e-6)
        return 0.1 * (2.0 * x / denom - 2.0 * torch.pi * torch.sin(2.0 * torch.pi * x)
                      - 8.0 * torch.pi * torch.sin(4.0 * torch.pi * x))

    def _is_on_target(self, position: float) -> bool:
        return (self.min_goal_position <= position) and (position <= self.max_goal_position)

    def _total_horizontal_force(self, u_torch: torch.Tensor, position_torch: torch.Tensor) -> torch.Tensor:
        # u_torch in {-1, +1}
        u = torch.sign(u_torch).to(torch.float32)
        tg_alpha = self._tg_alpha(position_torch)
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
        assert self.action_space.contains(action), f"Invalid action: {action}"
        assert self.state is not None, "Call reset() before step()."

        # Map action {0,1} -> u in {-1,+1}
        u = -1.0 if int(action) == 0 else +1.0

        position, velocity = self.state.astype(np.float64)

        # Compute total horizontal force in Torch
        force_t = self._total_horizontal_force(
            torch.tensor(u, dtype=torch.float32),
            torch.tensor(position, dtype=torch.float32)
        )
        total_horizontal_force = float(force_t.item())

        # Integrate dynamics
        velocity += total_horizontal_force * self.dt
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity * self.dt
        position = np.clip(position, self.min_position, self.max_position)

        self.state = np.array([position, velocity], dtype=np.float32)

        # Never terminate/truncate internally
        terminated = False
        truncated = False

        # Sparse reward
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

        # Goal flags (left and right edges of goal window)
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

    # ----------------- Optional RL helpers -----------------
    def reward(self, state, action, state_dot):
        # state: (..., 2)
        position, _ = torch.moveaxis(state, -1, 0)
        inside = (position >= self.min_goal_position) & (position <= self.max_goal_position)
        return inside.to(dtype=state.dtype)


def _tg_alpha_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    denom = np.clip(1.0 - x * x, 1e-6, None)
    return 0.1 * (2.0 * x / denom - 2.0 * np.pi * np.sin(2.0 * np.pi * x) - 8.0 * np.pi * np.sin(4.0 * np.pi * x))


def vectorized_multi_valley_mountain_car_step_discrete(states: np.ndarray,
                                                       actions: np.ndarray,
                                                       dt: float = 0.1,
                                                       force: float = 0.001,
                                                       gravity: float = 0.0025) -> tuple:
    """
    Vectorized step for the discrete Multi-Valley Mountain Car.

    Args:
        states:  (batch, 2) -> [position, velocity]
        actions: (batch,)  -> integers {0,1} (0:left, 1:right)
        dt:      integration time step
        force:   action force scale
        gravity: gravity parameter

    Returns:
        next_states: (batch, 2)
        rewards:     (batch,)
        terminated:  (batch,) all False (env does not terminate internally)
    """
    pos = states[:, 0].astype(np.float64)
    vel = states[:, 1].astype(np.float64)

    # Map actions -> u in {-1,+1}
    u = np.where(actions.astype(np.int64) == 0, -1.0, 1.0)

    # Horizontal force: agent input + slope gravity projection
    tg_alpha = _tg_alpha_np(pos)
    total_force = u * force + tg_alpha * (-gravity)

    vel_new = vel + total_force * dt
    vel_new = np.clip(vel_new, -0.07, 0.07)

    pos_new = pos + vel_new * dt
    pos_new = np.clip(pos_new, -0.99, 0.99)

    # Rewards: 1 if inside goal window, else 0
    rewards = ((pos_new >= -0.05) & (pos_new <= 0.05)).astype(np.float32)

    next_states = np.column_stack([pos_new.astype(np.float32), vel_new.astype(np.float32)])
    terminated = np.zeros_like(rewards, dtype=bool)
    return next_states, rewards, terminated


# Register so gym.make() works once this module is imported
register(
    id="MultiValleyMountainCarDiscrete-v0",
    entry_point="multi_valley_mountain_car_disc:MultiValleyMountainCarEnv",
)


