import pygame
import numpy as np
from env import StickWalkEnv, STICK_LENGTH, TARGET_DISTANCE

# Display settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 400
FPS = 50

# Camera settings
PIXELS_PER_METER = 50  # Scale for rendering
CAMERA_FOLLOW_SPEED = 0.1  # How fast camera follows stick

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
GRAY = (150, 150, 150)
YELLOW = (255, 255, 0)


class Renderer:
    def __init__(self, env):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Stick Walker")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        self.env = env
        self.camera_x = 0  # Camera position in world coordinates
        
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = (x - self.camera_x) * PIXELS_PER_METER + WINDOW_WIDTH / 2
        screen_y = WINDOW_HEIGHT - 50 - y * PIXELS_PER_METER  # Flip y-axis, add ground offset
        return int(screen_x), int(screen_y)
    
    def render(self):
        """Render the current state."""
        state = self.env.get_state()
        
        # Smooth camera follow
        target_camera_x = state['x']
        self.camera_x += (target_camera_x - self.camera_x) * CAMERA_FOLLOW_SPEED
        
        # Clear screen
        self.screen.fill(WHITE)
        
        # Draw ground
        ground_y = WINDOW_HEIGHT - 50
        pygame.draw.line(self.screen, GRAY, (0, ground_y), (WINDOW_WIDTH, ground_y), 3)
        
        # Draw starting line
        start_screen_x, _ = self.world_to_screen(0, 0)
        if 0 <= start_screen_x <= WINDOW_WIDTH:
            pygame.draw.line(self.screen, GREEN, (start_screen_x, ground_y - 100), 
                           (start_screen_x, ground_y), 2)
            label = self.small_font.render("START", True, GREEN)
            self.screen.blit(label, (start_screen_x - 25, ground_y - 120))
        
        # Draw target line
        target_screen_x, _ = self.world_to_screen(TARGET_DISTANCE, 0)
        if 0 <= target_screen_x <= WINDOW_WIDTH:
            pygame.draw.line(self.screen, BLUE, (target_screen_x, ground_y - 100), 
                           (target_screen_x, ground_y), 2)
            label = self.small_font.render("TARGET", True, BLUE)
            self.screen.blit(label, (target_screen_x - 30, ground_y - 120))
        
        # Calculate stick endpoints
        half_length = STICK_LENGTH / 2
        cos_angle = np.cos(state['angle'])
        sin_angle = np.sin(state['angle'])
        
        end1_x = state['x'] - half_length * cos_angle
        end1_y = state['y'] - half_length * sin_angle
        end2_x = state['x'] + half_length * cos_angle
        end2_y = state['y'] + half_length * sin_angle
        
        # Draw stick
        end1_screen = self.world_to_screen(end1_x, end1_y)
        end2_screen = self.world_to_screen(end2_x, end2_y)
        pygame.draw.line(self.screen, RED, end1_screen, end2_screen, 8)
        
        # Draw stick center
        center_screen = self.world_to_screen(state['x'], state['y'])
        pygame.draw.circle(self.screen, YELLOW, center_screen, 8)
        
        # Draw endpoints
        pygame.draw.circle(self.screen, BLACK, end1_screen, 6)
        pygame.draw.circle(self.screen, BLACK, end2_screen, 6)
        
        # Draw info text
        distance = state['x']
        info_texts = [
            f"Step: {state['step']}",
            f"Distance: {distance:.2f}m",
            f"Velocity: {state['vx']:.2f} m/s",
            f"Angle: {np.degrees(state['angle']):.1f}°",
        ]
        
        y_offset = 10
        for text in info_texts:
            surface = self.small_font.render(text, True, BLACK)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25
        
        # Draw controls
        controls = [
            "Controls:",
            "A - Rotate Left",
            "D - Rotate Right",
            "SPACE - Do Nothing",
            "R - Reset",
            "Q - Quit",
        ]
        y_offset = 10
        for text in controls:
            surface = self.small_font.render(text, True, BLACK)
            self.screen.blit(surface, (WINDOW_WIDTH - 200, y_offset))
            y_offset += 25
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def close(self):
        pygame.quit()


def main():
    """Run manual control mode."""
    env = StickWalkEnv()
    renderer = Renderer(env)
    
    obs = env.reset()
    running = True
    action = 1  # Default: do nothing
    
    print("Manual control mode")
    print("A/D to rotate, SPACE to do nothing, R to reset, Q to quit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    print("Environment reset")
        
        # Get current key state for continuous control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            action = 0  # Rotate left
        elif keys[pygame.K_d]:
            action = 2  # Rotate right
        else:
            action = 1  # Do nothing
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        # Render
        renderer.render()
        
        if done:
            print(f"Episode finished! Distance: {info['distance']:.2f}m in {info['steps']} steps")
            obs = env.reset()
    
    renderer.close()


if __name__ == "__main__":
    main()