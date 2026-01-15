import pygame
import numpy as np

class AirHockey:
    """Super simple air hockey game"""

    def __init__(self, width=800, height=600, render_mode=None, use_external_opponent=False):
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.use_external_opponent = use_external_opponent

        # Game objects sizes
        self.puck_radius = 15
        self.paddle_radius = 30
        self.goal_width = 200

        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 50, 50)
        self.BLUE = (50, 50, 255)
        self.GRAY = (150, 150, 150)

        # Initialize pygame if rendering
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Air Hockey RL")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        self.reset()

    def reset(self):
        """Reset the game to starting position"""
        # Puck starts on agent's side, moving slowly toward opponent
        self.puck_pos = np.array([self.width / 2, self.height * 0.7], dtype=np.float32)
        self.puck_vel = np.array([np.random.uniform(-50, 50), -100], dtype=np.float32)

        # Agent paddle (bottom)
        self.agent_pos = np.array([self.width / 2, self.height - 100], dtype=np.float32)

        # Opponent paddle (top) - simple AI that follows puck
        self.opponent_pos = np.array([self.width / 2, 100], dtype=np.float32)

        self.score_agent = 0
        self.score_opponent = 0

        return self._get_obs()

    def _get_obs(self):
        """Return normalized observations (12 values)"""
        # Relative position of puck to agent
        rel_x = (self.puck_pos[0] - self.agent_pos[0]) / self.width
        rel_y = (self.puck_pos[1] - self.agent_pos[1]) / self.height

        # Distance to puck (normalized by diagonal)
        dist = np.linalg.norm(self.puck_pos - self.agent_pos)
        max_dist = np.sqrt(self.width**2 + self.height**2)
        norm_dist = dist / max_dist

        # Puck speed (normalized)
        puck_speed = np.linalg.norm(self.puck_vel) / 500

        return np.array([
            self.puck_pos[0] / self.width,
            self.puck_pos[1] / self.height,
            self.puck_vel[0] / 500,
            self.puck_vel[1] / 500,
            self.agent_pos[0] / self.width,
            self.agent_pos[1] / self.height,
            self.opponent_pos[0] / self.width,
            self.opponent_pos[1] / self.height,
            rel_x,
            rel_y,
            norm_dist,
            puck_speed,
        ], dtype=np.float32)

    def _move_paddle(self, position, action: int, y_min: float, y_max: float, dt: float):
        """Move a paddle based on action and keep within bounds.
        Actions: 0=stay, 1=up, 2=down, 3=left, 4=right,
                 5=up-left, 6=up-right, 7=down-left, 8=down-right
        """
        paddle_speed = 400 * dt
        diag_speed = paddle_speed * 0.707  # 1/sqrt(2) for diagonal

        if action == 1:  # up
            position[1] -= paddle_speed
        elif action == 2:  # down
            position[1] += paddle_speed
        elif action == 3:  # left
            position[0] -= paddle_speed
        elif action == 4:  # right
            position[0] += paddle_speed
        elif action == 5:  # up-left
            position[0] -= diag_speed
            position[1] -= diag_speed
        elif action == 6:  # up-right
            position[0] += diag_speed
            position[1] -= diag_speed
        elif action == 7:  # down-left
            position[0] -= diag_speed
            position[1] += diag_speed
        elif action == 8:  # down-right
            position[0] += diag_speed
            position[1] += diag_speed

        position[0] = np.clip(position[0], self.paddle_radius, self.width - self.paddle_radius)
        position[1] = np.clip(position[1], y_min, y_max)

    def _apply_opponent_action(self, action: int, dt: float):
        """Apply an action to move the opponent paddle."""
        self._move_paddle(self.opponent_pos, action,
                         y_min=self.paddle_radius,
                         y_max=self.height/2 - self.paddle_radius, dt=dt)

    def _rule_based_opponent_ai(self, dt: float):
        """Original simple AI that follows puck x-position."""
        opponent_speed = 300 * dt
        if self.opponent_pos[0] < self.puck_pos[0] - 10:
            self.opponent_pos[0] += opponent_speed
        elif self.opponent_pos[0] > self.puck_pos[0] + 10:
            self.opponent_pos[0] -= opponent_speed

        # Keep opponent paddle on top half and in bounds
        self.opponent_pos[0] = np.clip(self.opponent_pos[0], self.paddle_radius,
                                        self.width - self.paddle_radius)
        self.opponent_pos[1] = np.clip(self.opponent_pos[1], self.paddle_radius,
                                        self.height/2 - self.paddle_radius)

    def step(self, action, opponent_action=None, dt=1/60):
        """
        Update game state
        action: 0=stay, 1=up, 2=down, 3=left, 4=right
        opponent_action: Optional action for opponent (if None, uses AI or does nothing)
        """
        reward = 0
        done = False

        # Move agent paddle
        self._move_paddle(self.agent_pos, action,
                         y_min=self.height/2 + self.paddle_radius,
                         y_max=self.height - self.paddle_radius, dt=dt)

        # Handle opponent movement
        if opponent_action is not None:
            self._apply_opponent_action(opponent_action, dt)
        elif not self.use_external_opponent:
            self._rule_based_opponent_ai(dt)

        # Update puck position
        self.puck_pos += self.puck_vel * dt

        # Bounce off left/right walls
        if self.puck_pos[0] <= self.puck_radius or self.puck_pos[0] >= self.width - self.puck_radius:
            self.puck_vel[0] *= -1
            self.puck_pos[0] = np.clip(self.puck_pos[0], self.puck_radius,
                                        self.width - self.puck_radius)

        # Check goals
        goal_left = self.width / 2 - self.goal_width / 2
        goal_right = self.width / 2 + self.goal_width / 2

        # Agent scores (puck at top)
        if self.puck_pos[1] <= self.puck_radius:
            if goal_left <= self.puck_pos[0] <= goal_right:
                reward = 1.0
                self.score_agent += 1
                done = True
            else:
                # Bounce off top wall
                self.puck_vel[1] *= -1
                self.puck_pos[1] = self.puck_radius

        # Opponent scores (puck at bottom)
        if self.puck_pos[1] >= self.height - self.puck_radius:
            if goal_left <= self.puck_pos[0] <= goal_right:
                reward = -1.0
                self.score_opponent += 1
                done = True
            else:
                # Bounce off bottom wall
                self.puck_vel[1] *= -1
                self.puck_pos[1] = self.height - self.puck_radius

        # Max puck speed to prevent exponential speedup
        max_puck_speed = 600

        # Check collision with agent paddle
        dist_agent = np.linalg.norm(self.puck_pos - self.agent_pos)
        if dist_agent <= self.puck_radius + self.paddle_radius:
            direction = (self.puck_pos - self.agent_pos) / dist_agent
            new_speed = min(np.linalg.norm(self.puck_vel) * 1.1, max_puck_speed)
            self.puck_vel = direction * new_speed
            self.puck_pos = self.agent_pos + direction * (self.puck_radius + self.paddle_radius)

            # Simple reward: +0.1 for hitting toward opponent, -0.05 for hitting toward self
            if self.puck_vel[1] < 0:
                reward += 0.1
            else:
                reward -= 0.05

        # Check collision with opponent paddle
        dist_opponent = np.linalg.norm(self.puck_pos - self.opponent_pos)
        if dist_opponent <= self.puck_radius + self.paddle_radius:
            direction = (self.puck_pos - self.opponent_pos) / dist_opponent
            new_speed = min(np.linalg.norm(self.puck_vel) * 1.1, max_puck_speed)
            self.puck_vel = direction * new_speed
            self.puck_pos = self.opponent_pos + direction * (self.puck_radius + self.paddle_radius)

        # Simple positioning reward: get close to puck when it's in our half
        if self.puck_pos[1] > self.height / 2:
            dist_to_puck = np.linalg.norm(self.puck_pos - self.agent_pos)
            reward += 0.01 * (1.0 - min(dist_to_puck / 300, 1.0))

        return self._get_obs(), reward, done, False, {}

    def render(self):
        """Draw the game"""
        if self.screen is None:
            return

        # Clear screen
        self.screen.fill(self.WHITE)

        # Draw center line
        pygame.draw.line(self.screen, self.GRAY, (0, self.height//2),
                        (self.width, self.height//2), 2)

        # Draw goals
        goal_left = self.width // 2 - self.goal_width // 2
        goal_right = self.width // 2 + self.goal_width // 2

        # Top goal (opponent's)
        pygame.draw.line(self.screen, self.RED, (goal_left, 0), (goal_right, 0), 5)

        # Bottom goal (agent's)
        pygame.draw.line(self.screen, self.BLUE, (goal_left, self.height),
                        (goal_right, self.height), 5)

        # Draw puck
        pygame.draw.circle(self.screen, self.BLACK, self.puck_pos.astype(int),
                          self.puck_radius)

        # Draw paddles
        pygame.draw.circle(self.screen, self.BLUE, self.agent_pos.astype(int),
                          self.paddle_radius)
        pygame.draw.circle(self.screen, self.RED, self.opponent_pos.astype(int),
                          self.paddle_radius)

        # Draw score
        font = pygame.font.Font(None, 48)
        score_text = font.render(f"You: {self.score_agent}  Opponent: {self.score_opponent}",
                                True, self.BLACK)
        self.screen.blit(score_text, (20, 20))

        pygame.display.flip()
        if self.clock:
            self.clock.tick(60)

    def close(self):
        """Clean up"""
        if self.screen is not None:
            pygame.quit()
