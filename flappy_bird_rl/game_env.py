
import pygame
import random
import config

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird RL')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('monospace', 15)
        self.game_width = 288  # The original game width

        # Game objects
        self.bird_y = config.SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self.score = 0

    def reset(self):
        """Resets the game to the initial state."""
        self.bird_y = config.SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.pipes = [self._get_new_pipe()]
        self.score = 0
        return self._get_state()

    def step(self, action, debug_info=None):
        """
        Perform one step in the environment.
        Action 0: Do nothing
        Action 1: Jump
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Bird movement
        if action == 1:
            self.bird_vel = config.BIRD_JUMP
        
        self.bird_vel += config.GRAVITY
        self.bird_y += self.bird_vel

        # Pipe movement
        pipe_passed = False
        for pipe in self.pipes:
            pipe['x'] -= config.PIPE_SPEED

        # Add new pipe
        if self.pipes[-1]['x'] < self.game_width - config.PIPE_SPACING:
            self.pipes.append(self._get_new_pipe())

        # Remove old pipes and update score
        if self.pipes[0]['x'] < -52: # 52 is pipe width
            self.pipes.pop(0)
            pipe_passed = True
            self.score += 1

        # Check for collisions
        done, reward = self._check_collision(pipe_passed)

        # Update the display
        self._update_screen(debug_info)

        return self._get_state(), reward, done

    def _get_state(self):
        """
        Get the current state for the RL agent.
        State is a tuple: (bird_y, bird_vel, pipe_top_y, pipe_x)
        """
        # Find the upcoming pipe
        upcoming_pipe = None
        for p in self.pipes:
            if p['x'] + 52 > 50: # 50 is bird's x position, 52 is pipe width
                upcoming_pipe = p
                break
        
        if upcoming_pipe is None:
            # Should not happen in normal gameplay, but as a fallback
            upcoming_pipe = self.pipes[0]

        state = (
            self.bird_y,
            self.bird_vel,
            upcoming_pipe['y'], # y of the top of the gap
            upcoming_pipe['x']
        )
        return state

    def _check_collision(self, pipe_passed):
        """Check for collision with ground or pipes."""
        bird_rect = pygame.Rect(50, self.bird_y, 34, 24) # Bird dimensions

        # Ground and ceiling collision
        if self.bird_y > config.SCREEN_HEIGHT - 24 or self.bird_y < 0:
            return True, -100

        # Pipe collision
        for pipe in self.pipes:
            pipe_upper_rect = pygame.Rect(pipe['x'], 0, 52, pipe['y'])
            pipe_lower_rect = pygame.Rect(pipe['x'], pipe['y'] + config.PIPE_GAP_SIZE, 52, config.SCREEN_HEIGHT)
            if bird_rect.colliderect(pipe_upper_rect) or bird_rect.colliderect(pipe_lower_rect):
                return True, -100

        # Reward logic
        if pipe_passed:
            return False, 5 # Big reward for passing a pipe
        
        return False, 0.1 # Small reward for surviving

    def _get_new_pipe(self):
        """Generates a new pipe with a random gap position."""
        gap_y = random.randrange(100, config.SCREEN_HEIGHT - 200)
        return {'x': self.game_width, 'y': gap_y}

    def _update_screen(self, debug_info=None):
        """Updates the game display."""
        self.screen.fill(config.BLACK)

        # --- Game Area ---
        game_surface = self.screen.subsurface((0, 0, self.game_width, config.SCREEN_HEIGHT))
        game_surface.fill((10, 10, 40)) # Dark blue background for game area

        # Draw bird
        pygame.draw.rect(game_surface, config.WHITE, (50, self.bird_y, 34, 24))

        # Draw pipes
        for pipe in self.pipes:
            pygame.draw.rect(game_surface, config.GREEN, (pipe['x'], 0, 52, pipe['y']))
            pygame.draw.rect(game_surface, config.GREEN, (pipe['x'], pipe['y'] + config.PIPE_GAP_SIZE, 52, config.SCREEN_HEIGHT))

        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, config.WHITE)
        game_surface.blit(score_text, (10, 10))

        # --- Debug Area ---
        if debug_info:
            self.draw_debug_info(debug_info)

        pygame.display.flip()
        self.clock.tick(config.FPS)

    def draw_debug_info(self, info):
        """Draws the debug information on the right side of the screen."""
        x_offset = self.game_width + 10
        y_offset = 10
        line_height = 20

        # Title
        title_text = self.font.render("--- AGENT MONITOR ---", True, config.WHITE)
        self.screen.blit(title_text, (x_offset, y_offset))
        y_offset += line_height * 2

        # ROUND (formerly Episode)
        ep_text = self.font.render(f"ROUND   : {info['episode']}", True, config.WHITE)
        self.screen.blit(ep_text, (x_offset, y_offset))
        y_offset += line_height

        # State (formatted to remove dtype)
        formatted_state = tuple(int(x) for x in info['state'])
        state_text = self.font.render(f"State   : {formatted_state}", True, config.WHITE)
        self.screen.blit(state_text, (x_offset, y_offset))
        y_offset += line_height * 1.5

        # Q-values
        q_title = self.font.render("Q-Values:", True, config.WHITE)
        self.screen.blit(q_title, (x_offset, y_offset))
        y_offset += line_height

        # Action 0 (No Jump)
        q0_color = config.WHITE
        if info['action'] == 0:
            q0_color = (255, 255, 0) # Yellow if chosen
        q0_text = self.font.render(f"  NO JUMP: {info['q_values'][0]:.4f}", True, q0_color)
        self.screen.blit(q0_text, (x_offset, y_offset))
        y_offset += line_height

        # Action 1 (Jump)
        q1_color = config.WHITE
        if info['action'] == 1:
            q1_color = (255, 255, 0) # Yellow if chosen
        q1_text = self.font.render(f"  JUMP   : {info['q_values'][1]:.4f}", True, q1_color)
        self.screen.blit(q1_text, (x_offset, y_offset))
        y_offset += line_height * 1.5

        # Epsilon
        epsilon_text = self.font.render(f"Epsilon : {info['epsilon']:.4f}", True, config.WHITE)
        self.screen.blit(epsilon_text, (x_offset, y_offset))
        y_offset += line_height

