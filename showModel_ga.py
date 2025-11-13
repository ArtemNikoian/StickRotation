import pygame
import numpy as np
from network import NeuralNetwork
from env import StickWalkEnv
from render import Renderer

# Model settings
MODEL_PATH = "models_ga/model_ga.pkl"

def show_model_performance(model_path=MODEL_PATH, n_episodes=5):
    """Show trained model performance with visualization."""
    print(f"Loading model from {model_path}...")
    network = NeuralNetwork.load(model_path)
    
    env = StickWalkEnv()
    renderer = Renderer(env)
    
    print(f"\nShowing model performance for {n_episodes} episodes")
    print("Press R to reset episode, Q to quit")
    print("-" * 50)
    
    episode = 0
    obs = env.reset()
    running = True
    episode_reward = 0
    
    while running and episode < n_episodes:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    # Skip to next episode
                    obs = env.reset()
                    episode += 1
                    episode_reward = 0
                    print(f"\nStarting episode {episode + 1}/{n_episodes}")
        
        # Get action from model
        action = network.predict(obs)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Render
        renderer.render()
        
        if done:
            print(f"Episode {episode + 1} finished!")
            print(f"  Distance: {info['distance']:.2f}m")
            print(f"  Steps: {info['steps']}")
            print(f"  Total reward: {episode_reward:.2f}")
            
            episode += 1
            if episode < n_episodes:
                print(f"\nStarting episode {episode + 1}/{n_episodes}")
                obs = env.reset()
                episode_reward = 0
    
    renderer.close()
    print("\nVisualization complete!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Custom model path
        show_model_performance(model_path=sys.argv[1])
    else:
        # Default model path
        show_model_performance()