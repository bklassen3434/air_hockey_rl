import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import AirHockey


class AirHockeyEnv(gym.Env):
    """Gym environment wrapper for Air Hockey game with self-play support."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, opponent=None):
        super().__init__()

        self.render_mode = render_mode
        self.opponent = opponent

        # Use external opponent if provided
        use_external = opponent is not None
        self.game = AirHockey(render_mode=render_mode, use_external_opponent=use_external)

        # Define action space: 9 discrete actions
        # 0=stay, 1=up, 2=down, 3=left, 4=right,
        # 5=up-left, 6=up-right, 7=down-left, 8=down-right
        self.action_space = spaces.Discrete(9)

        # Define observation space: 12 values
        # [puck_x, puck_y, puck_vx, puck_vy, agent_x, agent_y, opponent_x, opponent_y,
        #  rel_puck_x, rel_puck_y, puck_dist, puck_speed]
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(12,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        if self.opponent is not None:
            self.opponent.reset()
        obs = self.game.reset()
        return obs, {}

    def step(self, action):
        """Take one step in the environment"""
        # Get opponent action if we have an opponent controller
        opponent_action = None
        if self.opponent is not None:
            current_obs = self.game._get_obs()
            opponent_action = self.opponent.get_action(current_obs)

        obs, reward, terminated, truncated, info = self.game.step(action, opponent_action=opponent_action)
        return obs, reward, terminated, truncated, info

    def set_opponent(self, opponent):
        """Change the opponent controller."""
        self.opponent = opponent
        self.game.use_external_opponent = opponent is not None

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            self.game.render()

    def close(self):
        """Close the environment"""
        self.game.close()
