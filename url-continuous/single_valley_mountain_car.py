# single_valley_mountain_car.py
#
# Usage example
# -------------
# >>> import gymnasium as gym
# >>> import single_valley_mountain_car      # just import once to register
# >>> env = gym.make("SingleValleyMountainCarContinuous-v0")
# >>> s, _ = env.reset()
# >>> for _ in range(5):
# ...     ns, r, term, trunc, _ = env.step(env.action_space.sample())
# ...     print(ns, r, term)
# ...

import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from gymnasium.error import DependencyNotInstalled


class SingleValleyMountainCarEnv(gym.Env):
    """
    Continuous-action Mountain-Car (one valley) with
        * reward = -1 at every step
        * reward = +100, terminated = True when goal reached
    Action space:  Box([-1, 1], shape=(1,))
    Observation :  Box([position, velocity])
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 goal_velocity: float = 0.0):
        super().__init__()

        # ----- physics / limits -------------------------------------------------
        self.min_position = -1.2
        self.max_position =  0.6
        self.max_speed    =  0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power         = 0.0015

        # gymnasium spaces
        self.action_space      = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(1,),
                                            dtype=np.float32)
        low_state  = np.array([self.min_position, -self.max_speed],
                              dtype=np.float32)
        high_state = np.array([self.max_position,  self.max_speed],
                              dtype=np.float32)
        self.observation_space = spaces.Box(low_state, high_state,
                                            dtype=np.float32)

        # render-related
        self.render_mode   = render_mode
        self.screen_width  = 600
        self.screen_height = 400
        self.screen = None
        self.clock  = None
        self.isopen = True

        self.state = None                    # will be set in reset()

    # --------------------------------------------------------------------- step
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)[0]

        pos, vel = self.state
        vel += action * self.power - 0.0025 * math.cos(3.0 * pos)
        vel = np.clip(vel, -self.max_speed, self.max_speed)

        pos += vel
        pos = np.clip(pos, self.min_position, self.max_position)
        if pos == self.min_position and vel < 0.0:
            vel = 0.0

        terminated = bool(pos >= self.goal_position and
                          vel >= self.goal_velocity)

        # ------------- reward: -1 each step, +100 on success ---------------
        reward = 100.0 if terminated else -1.0

        self.state = np.array([pos, vel], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    # -------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        pos = self.np_random.uniform(low=-0.6, high=-0.4)      # standard init
        self.state = np.array([pos, 0.0], dtype=np.float32)

        if self.render_mode == "human":
            self.render()

        return self.state.copy(), {}

    # -------------------------------------------------------------- helpers ----
    @staticmethod
    def _height(x):
        return np.sin(3.0 * x) * 0.45 + 0.55

    # ------------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            gym.logger.warn("Render mode was not set!")
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                'pygame is not installed.  Run `pip install "gymnasium[classic-control]"`'
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

        # ----- draw track
        xs = np.linspace(self.min_position, self.max_position, 100)
        ys = self._height(xs)
        pts = list(zip((xs - self.min_position) * scale, ys * scale))
        pygame.draw.aalines(surf, (0, 0, 0), False, pts)

        # ----- draw car
        pos = self.state[0]
        clearance = 10
        rot = math.cos(3 * pos)
        l, r, t, b = -carw / 2, carw / 2, carh, 0
        corners = []
        for cx, cy in ((l, b), (l, t), (r, t), (r, b)):
            vec = pygame.math.Vector2(cx, cy).rotate_rad(rot)
            corners.append((vec.x + (pos - self.min_position) * scale,
                            vec.y + clearance + self._height(pos) * scale))
        gfxdraw.aapolygon(surf, corners, (0, 0, 0))
        gfxdraw.filled_polygon(surf, corners, (0, 0, 0))

        # wheels
        for wx, wy in ((carw / 4, 0), (-carw / 4, 0)):
            vec = pygame.math.Vector2(wx, wy).rotate_rad(rot)
            wheel = (int(vec.x + (pos - self.min_position) * scale),
                     int(vec.y + clearance + self._height(pos) * scale))
            gfxdraw.aacircle(surf, wheel[0], wheel[1], int(carh / 2.5), (80, 80, 80))
            gfxdraw.filled_circle(surf, wheel[0], wheel[1], int(carh / 2.5), (80, 80, 80))

        # goal flag
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

    # ------------------------------------------------------------------- close
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def vectorized_mountain_car_step(states, actions):
    """
    Vectorized step function for Mountain Car environment.
    
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
    
    # Environment parameters (matching SingleValleyMountainCarEnv)
    power = 0.0015
    max_speed = 0.07
    min_position = -1.2
    max_position = 0.6
    goal_position = 0.45
    goal_velocity = 0.0
    
    # Physics update (vectorized)
    vel_new = vel + actions.flatten() * power - 0.0025 * np.cos(3.0 * pos)
    vel_new = np.clip(vel_new, -max_speed, max_speed)
    
    pos_new = pos + vel_new
    pos_new = np.clip(pos_new, min_position, max_position)
    
    # Handle boundary condition: if at left boundary and velocity negative, set to 0
    boundary_mask = (pos_new == min_position) & (vel_new < 0.0)
    vel_new[boundary_mask] = 0.0
    
    # Check termination condition
    terminated = (pos_new >= goal_position) & (vel_new >= goal_velocity)
    
    # Compute rewards: +100 for success, -1 for each step
    rewards = np.where(terminated, 100.0, -1.0)
    
    # Create next states array
    next_states = np.column_stack([pos_new, vel_new]).astype(np.float32)
    
    # Clip next states to observation space bounds
    state_low = np.array([min_position, -max_speed])
    state_high = np.array([max_position, max_speed])
    next_states[:, 0] = np.clip(next_states[:, 0], state_low[0], state_high[0])
    next_states[:, 1] = np.clip(next_states[:, 1], state_low[1], state_high[1])
    
    return next_states, rewards, terminated


# --------------- register with gymnasium so `gym.make()` works -------------
register(
    id="SingleValleyMountainCarContinuous-v0",
    entry_point="single_valley_mountain_car:SingleValleyMountainCarEnv",
    max_episode_steps=999,      # truncation limit
)
