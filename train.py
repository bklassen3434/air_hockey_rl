"""Self-play training script for Air Hockey RL."""

import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from env import AirHockeyEnv
from opponent import SelfPlayOpponent, get_checkpoint_timestep


class SelfPlayCallback(BaseCallback):
    """Callback for saving checkpoints and updating opponent during training."""

    def __init__(self,
                 opponent: SelfPlayOpponent,
                 checkpoint_dir: str = "./checkpoints",
                 save_freq: int = 10000,
                 opponent_update_freq: int = 20000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.opponent = opponent
        self.checkpoint_dir = checkpoint_dir
        self.save_freq = save_freq
        self.opponent_update_freq = opponent_update_freq
        self.last_save_step = 0
        self.last_opponent_update = 0

        os.makedirs(checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Save checkpoint periodically
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_{self.num_timesteps}"
            )
            self.model.save(checkpoint_path)
            if self.verbose:
                print(f"\nSaved checkpoint: {checkpoint_path}.zip")
            self.last_save_step = self.num_timesteps

        # Update opponent periodically
        if self.num_timesteps - self.last_opponent_update >= self.opponent_update_freq:
            self._update_opponent()
            self.last_opponent_update = self.num_timesteps

        return True

    def _update_opponent(self):
        """Update opponent to use a past checkpoint."""
        checkpoints = self.opponent.get_available_checkpoints()

        if len(checkpoints) < 2:
            return

        # Use second-to-last checkpoint for stability
        target_checkpoint = checkpoints[-2]

        if target_checkpoint != self.opponent.current_checkpoint:
            if self.opponent.load_checkpoint(target_checkpoint):
                if self.verbose:
                    print(f"\nUpdated opponent to: {target_checkpoint}")


def train_selfplay(
    total_timesteps: int = 500000,
    checkpoint_freq: int = 10000,
    opponent_update_freq: int = 20000,
    checkpoint_dir: str = "./checkpoints",
    n_envs: int = 1
):
    """
    Train agent with self-play.

    Args:
        total_timesteps: Total training steps
        checkpoint_freq: How often to save checkpoints
        opponent_update_freq: How often to update opponent checkpoint
        checkpoint_dir: Directory to save checkpoints
        n_envs: Number of parallel environments
    """
    print("=" * 60)
    print("Self-Play Training for Air Hockey")
    print("=" * 60)

    # Create shared opponent controller
    opponent = SelfPlayOpponent(
        checkpoint_dir=checkpoint_dir,
        fallback_to_rule_based=True
    )

    # Create environment factory
    def make_env():
        return AirHockeyEnv(render_mode=None, opponent=opponent)

    # Create vectorized environment
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    print(f"Created {n_envs} environment(s)")

    # Check for existing checkpoints to resume from
    existing_checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.zip"))

    if existing_checkpoints:
        # Resume from latest checkpoint
        latest = max(existing_checkpoints, key=get_checkpoint_timestep)
        print(f"Resuming from checkpoint: {latest}")
        model = PPO.load(latest, env=env)

        # Load opponent from older checkpoint
        sorted_checkpoints = sorted(existing_checkpoints, key=get_checkpoint_timestep)
        if len(sorted_checkpoints) >= 2:
            opponent.load_checkpoint(sorted_checkpoints[-2])
    else:
        print("Starting fresh training (opponent uses rule-based AI initially)")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # More exploration
            max_grad_norm=0.5,
            tensorboard_log="./logs"
        )

    # Create callback
    callback = SelfPlayCallback(
        opponent=opponent,
        checkpoint_dir=checkpoint_dir,
        save_freq=checkpoint_freq,
        opponent_update_freq=opponent_update_freq,
        verbose=1
    )

    print(f"\nTraining for {total_timesteps} timesteps")
    print(f"Checkpoints every {checkpoint_freq} steps")
    print(f"Opponent updates every {opponent_update_freq} steps")
    print("-" * 60)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}.zip")

    # Also save to standard location for play.py compatibility
    model.save("air_hockey_agent")
    print("Also saved to: air_hockey_agent.zip")

    env.close()
    return model


if __name__ == "__main__":
    train_selfplay(
        total_timesteps=500000,
        checkpoint_freq=10000,
        opponent_update_freq=20000,
        checkpoint_dir="./checkpoints",
        n_envs=4
    )
