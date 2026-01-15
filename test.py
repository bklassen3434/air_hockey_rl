from stable_baselines3 import PPO
from env import AirHockeyEnv
from opponent import RuleBasedOpponent, ModelOpponent
import pygame
import os


def play_game(model_path="air_hockey_agent.zip", goals_to_win=5, opponent_model_path=None):
    """
    Watch the trained agent play a full game (first to N goals).

    Args:
        model_path: Path to the agent model
        goals_to_win: Number of goals needed to win (default 5)
        opponent_model_path: Optional path to opponent model (uses rule-based if None)
    """

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python train.py' first to train an agent!")
        return

    print("Loading trained agent...")
    model = PPO.load(model_path)

    # Setup opponent
    if opponent_model_path and os.path.exists(opponent_model_path):
        print(f"Loading opponent model from: {opponent_model_path}")
        opponent = ModelOpponent(opponent_model_path)
    else:
        print("Using rule-based opponent")
        opponent = RuleBasedOpponent()

    print("Creating environment...")
    env = AirHockeyEnv(render_mode="human", opponent=opponent)

    print(f"\nFirst to {goals_to_win} goals wins!")
    print("Press Q or close window to quit")
    print("-" * 60)

    # Track cumulative scores across rounds
    agent_score = 0
    opponent_score = 0
    total_steps = 0

    try:
        while agent_score < goals_to_win and opponent_score < goals_to_win:
            obs, _ = env.reset()
            done = False

            # Sync cumulative scores to game for display
            env.game.score_agent = agent_score
            env.game.score_opponent = opponent_score

            print(f"\nScore: Agent {agent_score} - {opponent_score} Opponent")

            while not done:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nWindow closed. Exiting...")
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            print("\nQ pressed. Exiting...")
                            env.close()
                            return

                # Agent chooses action
                action, _ = model.predict(obs, deterministic=True)

                # Take step in environment
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_steps += 1

                # Render the game
                env.render()

            # Check who scored this point (game.score is 1 for whoever just scored)
            if env.game.score_agent > agent_score:
                agent_score += 1
                print("  GOAL! Agent scores!")
            else:
                opponent_score += 1
                print("  GOAL! Opponent scores!")

            # Update display immediately
            env.game.score_agent = agent_score
            env.game.score_opponent = opponent_score
            env.render()

        # Game over - announce winner
        print("\n" + "=" * 60)
        if agent_score >= goals_to_win:
            print(f"AGENT WINS {agent_score}-{opponent_score}!")
        else:
            print(f"OPPONENT WINS {opponent_score}-{agent_score}!")
        print(f"Total steps: {total_steps}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print(f"Final score: Agent {agent_score} - {opponent_score} Opponent")

    finally:
        env.close()
        print("\nDone!")


if __name__ == "__main__":
    play_game(goals_to_win=5)
