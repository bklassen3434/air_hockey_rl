"""Play against the trained AI agent."""

from game import AirHockey
from opponent import ModelOpponent
import pygame
import os


def get_human_action(keys):
    """Convert keyboard state to action.

    Actions: 0=stay, 1=up, 2=down, 3=left, 4=right,
             5=up-left, 6=up-right, 7=down-left, 8=down-right
    """
    w = keys[pygame.K_w]
    s = keys[pygame.K_s]
    a = keys[pygame.K_a]
    d = keys[pygame.K_d]

    # Diagonal movements
    if w and a:
        return 5  # up-left
    if w and d:
        return 6  # up-right
    if s and a:
        return 7  # down-left
    if s and d:
        return 8  # down-right

    # Cardinal movements
    if w:
        return 1  # up
    if s:
        return 2  # down
    if a:
        return 3  # left
    if d:
        return 4  # right

    return 0  # stay


def play_game(model_path="air_hockey_agent.zip", goals_to_win=5):
    """
    Play against the trained AI.

    You control the BLUE paddle (bottom) with WASD keys.
    The AI controls the RED paddle (top).

    Args:
        model_path: Path to the trained agent model
        goals_to_win: Number of goals needed to win (default 5)
    """
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python train_selfplay.py' first to train an agent!")
        return

    print("Loading AI opponent...")
    ai_opponent = ModelOpponent(model_path)

    print("Creating game...")
    game = AirHockey(render_mode="human", use_external_opponent=True)

    print(f"\n{'='*60}")
    print("HUMAN VS AI - Air Hockey")
    print(f"{'='*60}")
    print("You: BLUE paddle (bottom) - Use WASD to move")
    print("AI:  RED paddle (top)")
    print(f"First to {goals_to_win} goals wins!")
    print("Press Q or ESC to quit")
    print(f"{'='*60}\n")

    # Track scores
    human_score = 0
    ai_score = 0

    try:
        while human_score < goals_to_win and ai_score < goals_to_win:
            obs = game.reset()
            done = False

            # Set display scores
            game.score_agent = human_score
            game.score_opponent = ai_score

            print(f"Score: You {human_score} - {ai_score} AI")

            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nWindow closed. Thanks for playing!")
                        game.close()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            print("\nThanks for playing!")
                            game.close()
                            return

                # Get human action from keyboard
                keys = pygame.key.get_pressed()
                human_action = get_human_action(keys)

                # Get AI action (AI plays from top, so use mirrored obs)
                ai_action = ai_opponent.get_action(obs)

                # Step game: human controls agent (bottom), AI controls opponent (top)
                obs, _, done, _, _ = game.step(human_action, opponent_action=ai_action)

                # Render
                game.render()

            # Check who scored
            if game.score_agent > human_score:
                human_score += 1
                print("  YOU SCORED!")
            else:
                ai_score += 1
                print("  AI scored!")

            # Update display
            game.score_agent = human_score
            game.score_opponent = ai_score
            game.render()
            pygame.time.wait(500)  # Brief pause after goal

        # Game over
        print(f"\n{'='*60}")
        if human_score >= goals_to_win:
            print(f"YOU WIN {human_score}-{ai_score}!")
        else:
            print(f"AI WINS {ai_score}-{human_score}!")
        print(f"{'='*60}")

        # Keep window open briefly
        pygame.time.wait(2000)

    except KeyboardInterrupt:
        print("\n\nGame interrupted")
        print(f"Final score: You {human_score} - {ai_score} AI")

    finally:
        game.close()
        print("\nThanks for playing!")


if __name__ == "__main__":
    play_game(goals_to_win=5)
