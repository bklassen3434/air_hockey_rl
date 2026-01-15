"""Opponent controller abstraction for self-play training."""

from abc import ABC, abstractmethod
import os
import glob
import numpy as np
from typing import Optional


def get_checkpoint_timestep(path: str) -> int:
    """Extract timestep number from checkpoint filename."""
    name = os.path.basename(path)
    try:
        return int(name.replace("checkpoint_", "").replace(".zip", ""))
    except:
        return 0


class OpponentController(ABC):
    """Abstract base class for opponent controllers."""

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> int:
        """Get opponent action based on observation."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state."""
        pass


class RuleBasedOpponent(OpponentController):
    """The original rule-based opponent that follows puck x-position."""

    def __init__(self, speed_factor: float = 0.75):
        self.speed_factor = speed_factor

    def get_action(self, obs: np.ndarray) -> int:
        """Simple AI: follow puck x-position only."""
        puck_x = obs[0]
        opponent_x = obs[6]

        threshold = 0.0125  # ~10 pixels when width=800

        if opponent_x < puck_x - threshold:
            return 4  # Move right
        elif opponent_x > puck_x + threshold:
            return 3  # Move left
        return 0  # Stay

    def reset(self) -> None:
        pass


class ModelOpponent(OpponentController):
    """Opponent controlled by a trained model."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load a model from checkpoint."""
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)

    def _mirror_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Mirror the observation for opponent's perspective (12 values).

        Original: [puck_x, puck_y, puck_vx, puck_vy, agent_x, agent_y, opp_x, opp_y,
                   rel_puck_x, rel_puck_y, puck_dist, puck_speed]
        """
        # Mirror positions
        puck_x = obs[0]
        puck_y = 1.0 - obs[1]
        puck_vx = obs[2]
        puck_vy = -obs[3]
        # Swap agent and opponent (opponent becomes "agent" in their view)
        new_agent_x = obs[6]
        new_agent_y = 1.0 - obs[7]
        new_opp_x = obs[4]
        new_opp_y = 1.0 - obs[5]

        # Recalculate relative features from opponent's perspective
        rel_puck_x = puck_x - new_agent_x
        rel_puck_y = puck_y - new_agent_y

        return np.array([
            puck_x,
            puck_y,
            puck_vx,
            puck_vy,
            new_agent_x,
            new_agent_y,
            new_opp_x,
            new_opp_y,
            rel_puck_x,
            rel_puck_y,
            obs[10],  # puck_dist stays same
            obs[11],  # puck_speed stays same
        ], dtype=np.float32)

    def _mirror_action(self, action: int) -> int:
        """Mirror action from opponent's perspective back to game perspective.
        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right,
                 5=up-left, 6=up-right, 7=down-left, 8=down-right
        """
        if action == 1:  # up -> down
            return 2
        elif action == 2:  # down -> up
            return 1
        elif action == 5:  # up-left -> down-left
            return 7
        elif action == 6:  # up-right -> down-right
            return 8
        elif action == 7:  # down-left -> up-left
            return 5
        elif action == 8:  # down-right -> up-right
            return 6
        return action  # 0, 3, 4 stay same

    def get_action(self, obs: np.ndarray) -> int:
        """Get action from the model using mirrored observation."""
        if self.model is None:
            return 0

        mirrored_obs = self._mirror_observation(obs)
        action, _ = self.model.predict(mirrored_obs, deterministic=True)
        return self._mirror_action(int(action))

    def reset(self) -> None:
        pass


class SelfPlayOpponent(OpponentController):
    """Opponent for self-play training with checkpoint management."""

    def __init__(self,
                 checkpoint_dir: str = "./checkpoints",
                 fallback_to_rule_based: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.fallback_to_rule_based = fallback_to_rule_based

        self.model_opponent = ModelOpponent()
        self.rule_opponent = RuleBasedOpponent()
        self.current_checkpoint: Optional[str] = None

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load a specific checkpoint."""
        try:
            self.model_opponent.load_model(checkpoint_path)
            self.current_checkpoint = checkpoint_path
            return True
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def get_available_checkpoints(self) -> list:
        """Get sorted list of available checkpoints."""
        if not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint_*.zip"))
        return sorted(checkpoints, key=get_checkpoint_timestep)

    def get_action(self, obs: np.ndarray) -> int:
        """Get action, using model if available, else rule-based."""
        if self.model_opponent.model is not None:
            return self.model_opponent.get_action(obs)
        elif self.fallback_to_rule_based:
            return self.rule_opponent.get_action(obs)
        return 0

    def reset(self) -> None:
        pass
