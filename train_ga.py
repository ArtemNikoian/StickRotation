import os
from datetime import datetime

import numpy as np
from env import StickWalkEnv
from network import NeuralNetwork
from torch.utils.tensorboard import SummaryWriter

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
ELITE_SIZE = 5  # Keep top performers
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.3
CROSSOVER_RATE = 0.7

# Evaluation Parameters
EPISODES_PER_EVAL = 3  # Average over multiple episodes
MAX_STEPS = 1500

MODEL_SAVE_PATH = "models_ga/model_ga.pkl"
TENSORBOARD_LOG_BASE_DIR = "runs"


def evaluate_network(network, n_episodes=EPISODES_PER_EVAL):
    """Evaluate a network's fitness - returns distance, steps, and reward."""

    env = StickWalkEnv()

    total_distance = 0
    total_steps = 0
    total_reward = 0

    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = network.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        total_distance += info['distance']
        total_steps += info['steps']
        total_reward += episode_reward

    return (total_distance / n_episodes,
            total_steps / n_episodes,
            total_reward / n_episodes)


def select_parents(population, reward_scores, n_parents):
    """Tournament selection."""
    parents = []
    for _ in range(n_parents):
        # Random tournament
        tournament_size = 5
        tournament_idx = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [reward_scores[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        parents.append(population[winner_idx].copy())
    return parents


def crossover(parent1, parent2):
    """Uniform crossover of two networks."""
    child = parent1.copy()
    
    if np.random.random() < CROSSOVER_RATE:
        weights1 = parent1.get_weights_flat()
        weights2 = parent2.get_weights_flat()
        
        # Uniform crossover
        mask = np.random.random(len(weights1)) < 0.5
        child_weights = np.where(mask, weights1, weights2)
        
        child.set_weights_flat(child_weights)
    
    return child


def mutate(network):
    """Mutate network weights."""
    weights = network.get_weights_flat()
    
    # Add gaussian noise to random subset of weights
    mutation_mask = np.random.random(len(weights)) < MUTATION_RATE
    noise = np.random.randn(len(weights)) * MUTATION_STRENGTH
    weights += mutation_mask * noise
    
    network.set_weights_flat(weights)
    return network


def train_ga():
    """Train using genetic algorithm."""
    print("Initializing population...")

    # Create model directory
    os.makedirs("models_ga", exist_ok=True)

    # Initialize TensorBoard writer
    run_name = datetime.now().strftime("stick_walker_ga_%Y%m%d_%H%M%S")
    tensorboard_log_dir = os.path.join(TENSORBOARD_LOG_BASE_DIR, run_name)
    writer = SummaryWriter(tensorboard_log_dir)

    # Create initial population
    population = [NeuralNetwork() for _ in range(POPULATION_SIZE)]

    best_network = None
    best_distance_ever = -np.inf
    best_reward_ever = -np.inf

    print(f"Starting genetic algorithm training...")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Generations: {GENERATIONS}")
    print(f"Elite size: {ELITE_SIZE}")
    print(f"Mutation rate: {MUTATION_RATE}")
    print(f"TensorBoard logging to: {tensorboard_log_dir}")
    print("-" * 60)

    for generation in range(GENERATIONS):
        # Evaluate fitness
        distances = []
        steps_list = []
        rewards = []

        for i, network in enumerate(population):
            distance, steps, reward = evaluate_network(network)
            distances.append(distance)
            steps_list.append(steps)
            rewards.append(reward)

            if (i + 1) % 10 == 0:
                print(f"Gen {generation+1}: Evaluating {i+1}/{POPULATION_SIZE}...", end='\r')

        # Track statistics - use reward as fitness
        best_reward = np.max(rewards)
        mean_reward = np.mean(rewards)

        best_idx = np.argmax(rewards)  # Changed to use rewards
        best_distance = distances[best_idx]
        mean_distance = np.mean(distances)

        best_steps = steps_list[best_idx]
        mean_steps = np.mean(steps_list)

        # Log to TensorBoard
        writer.add_scalar('Distance/Best', best_distance, generation)
        writer.add_scalar('Distance/Mean', mean_distance, generation)
        writer.add_scalar('Steps/Best', best_steps, generation)
        writer.add_scalar('Steps/Mean', mean_steps, generation)
        writer.add_scalar('Reward/Best', best_reward, generation)
        writer.add_scalar('Reward/Mean', mean_reward, generation)
        writer.add_histogram('Population/Rewards', np.array(rewards), generation)
        writer.add_histogram('Population/Distances', np.array(distances), generation)
        writer.add_histogram('Population/Steps', np.array(steps_list), generation)
        writer.flush()

        # Track best network ever (by reward)
        if best_reward > best_reward_ever:
            best_reward_ever = best_reward
            best_network = population[best_idx].copy()

        print(f"Gen {generation+1}/{GENERATIONS} | "
              f"Distance: {best_distance:.2f}m (mean: {mean_distance:.2f}m) | "
              f"Steps: {best_steps:.0f} (mean: {mean_steps:.0f}) | "
              f"Reward: {best_reward:.2f} (mean: {mean_reward:.2f})")

        # Sort population by reward (fitness)
        sorted_indices = np.argsort(rewards)[::-1]
        population = [population[i] for i in sorted_indices]
        rewards = [rewards[i] for i in sorted_indices]

        # Create next generation
        next_generation = []

        # Elitism: keep best performers
        for i in range(ELITE_SIZE):
            next_generation.append(population[i].copy())

        # Generate offspring
        while len(next_generation) < POPULATION_SIZE:
            # Select parents
            parents = select_parents(population, rewards, 2)

            # Crossover
            child = crossover(parents[0], parents[1])

            # Mutate
            child = mutate(child)

            next_generation.append(child)

        population = next_generation

        # Save best model periodically
        if (generation + 1) % 10 == 0 and best_network is not None:
            best_network.save(MODEL_SAVE_PATH)
            print(f"  -> Model saved to {MODEL_SAVE_PATH}")

    writer.close()

    print("\nTraining complete!")
    print(f"Best reward achieved: {best_reward_ever:.2f}")
    print(f"Saving final model to {MODEL_SAVE_PATH}...")
    if best_network is not None:
        best_network.save(MODEL_SAVE_PATH)
        print("Model saved!")

    return best_network


def evaluate_model(model_path=MODEL_SAVE_PATH, n_episodes=10):
    """Evaluate the trained model."""
    print(f"Loading model from {model_path}...")
    network = NeuralNetwork.load(model_path)

    env = StickWalkEnv()

    print(f"\nEvaluating for {n_episodes} episodes...")
    episode_distances = []
    episode_steps = []
    episode_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = network.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        episode_distances.append(info['distance'])
        episode_steps.append(info['steps'])
        episode_rewards.append(episode_reward)

        print(f"Episode {ep+1}: Distance={info['distance']:.2f}m, Steps={info['steps']}, Reward={episode_reward:.2f}")

    print("\n" + "="*50)
    print(f"Mean distance: {np.mean(episode_distances):.2f}m")
    print(f"Max distance: {np.max(episode_distances):.2f}m")
    print(f"Mean steps: {np.mean(episode_steps):.0f}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Evaluation mode
        if len(sys.argv) > 2:
            evaluate_model(model_path=sys.argv[2])
        else:
            evaluate_model()
    else:
        # Training mode
        train_ga()
        
        # Quick evaluation
        print("\n" + "="*50)
        print("Running quick evaluation...")
        evaluate_model(n_episodes=5)