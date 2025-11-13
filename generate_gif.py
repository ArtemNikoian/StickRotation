import pygame
import numpy as np
from network import NeuralNetwork
from env import StickWalkEnv
from render import Renderer
from PIL import Image

# Model settings
MODEL_PATH = "models_ga/model_ga.pkl"

def generate_gif(model_path=MODEL_PATH, output_path="stick_walker_demo.gif", max_steps=500):
    """Generate a GIF of the trained model in action."""
    print(f"Loading model from {model_path}...")
    network = NeuralNetwork.load(model_path)
    
    env = StickWalkEnv()
    
    # Initialize pygame in headless mode for frame capture
    pygame.init()
    screen = pygame.Surface((1200, 400))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 100, 255)
    GRAY = (150, 150, 150)
    YELLOW = (255, 255, 0)
    
    # Render settings
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 400
    PIXELS_PER_METER = 50
    CAMERA_FOLLOW_SPEED = 0.1
    STICK_LENGTH = 1.0
    TARGET_DISTANCE = 10.0
    
    def world_to_screen(x, y, camera_x):
        screen_x = (x - camera_x) * PIXELS_PER_METER + WINDOW_WIDTH / 2
        screen_y = WINDOW_HEIGHT - 50 - y * PIXELS_PER_METER
        return int(screen_x), int(screen_y)
    
    print("Generating GIF frames...")
    obs = env.reset()
    frames = []
    camera_x = 0
    step_count = 0
    frame_skip = 5  # Capture every Nth frame to speed up animation (higher = faster)
    
    while step_count < max_steps:
        # Get action from model
        action = network.predict(obs)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Only capture every Nth frame to speed up the GIF
        if step_count % frame_skip != 0:
            continue
        
        # Get state for rendering
        state = env.get_state()
        
        # Smooth camera follow
        target_camera_x = state['x']
        camera_x += (target_camera_x - camera_x) * CAMERA_FOLLOW_SPEED
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw ground
        ground_y = WINDOW_HEIGHT - 50
        pygame.draw.line(screen, GRAY, (0, ground_y), (WINDOW_WIDTH, ground_y), 3)
        
        # Draw starting line
        start_screen_x, _ = world_to_screen(0, 0, camera_x)
        if 0 <= start_screen_x <= WINDOW_WIDTH:
            pygame.draw.line(screen, GREEN, (start_screen_x, ground_y - 100), 
                           (start_screen_x, ground_y), 2)
            label = small_font.render("START", True, GREEN)
            screen.blit(label, (start_screen_x - 25, ground_y - 120))
        
        # Draw target line
        target_screen_x, _ = world_to_screen(TARGET_DISTANCE, 0, camera_x)
        if 0 <= target_screen_x <= WINDOW_WIDTH:
            pygame.draw.line(screen, BLUE, (target_screen_x, ground_y - 100), 
                           (target_screen_x, ground_y), 2)
            label = small_font.render("TARGET", True, BLUE)
            screen.blit(label, (target_screen_x - 30, ground_y - 120))
        
        # Calculate stick endpoints
        half_length = STICK_LENGTH / 2
        cos_angle = np.cos(state['angle'])
        sin_angle = np.sin(state['angle'])
        
        end1_x = state['x'] - half_length * cos_angle
        end1_y = state['y'] - half_length * sin_angle
        end2_x = state['x'] + half_length * cos_angle
        end2_y = state['y'] + half_length * sin_angle
        
        # Draw stick
        end1_screen = world_to_screen(end1_x, end1_y, camera_x)
        end2_screen = world_to_screen(end2_x, end2_y, camera_x)
        pygame.draw.line(screen, RED, end1_screen, end2_screen, 8)
        
        # Draw stick center
        center_screen = world_to_screen(state['x'], state['y'], camera_x)
        pygame.draw.circle(screen, YELLOW, center_screen, 8)
        
        # Draw endpoints
        pygame.draw.circle(screen, BLACK, end1_screen, 6)
        pygame.draw.circle(screen, BLACK, end2_screen, 6)
        
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
            surface = small_font.render(text, True, BLACK)
            screen.blit(surface, (10, y_offset))
            y_offset += 25
        
        # Convert pygame surface to PIL Image
        frame_str = pygame.image.tostring(screen, 'RGB')
        frame_img = Image.frombytes('RGB', (WINDOW_WIDTH, WINDOW_HEIGHT), frame_str)
        frames.append(frame_img)
        
        if done:
            break
    
    pygame.quit()
    
    print(f"Saving GIF with {len(frames)} frames to {output_path}...")
    # Save as GIF using PIL for better GitHub compatibility
    # Duration in milliseconds (200ms = slower, more compatible with GitHub)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=200,  # Duration in milliseconds per frame (higher = slower)
        loop=0,
        disposal=2  # Clear to background between frames for better compatibility
    )
    print(f"GIF saved successfully!")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        output_path = sys.argv[1]
    else:
        output_path = "stick_walker_demo.gif"
    
    generate_gif(output_path=output_path)

