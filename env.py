import numpy as np

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

# Physics
STICK_LENGTH = 1.0  # meters
MASS = 1.0  # kg (normalized to 1 for simplicity)
MOMENT_OF_INERTIA = (1/12) * MASS * STICK_LENGTH**2  # Rod rotating around center
GRAVITY = 9.81  # m/s^2
TORQUE_STRENGTH = 2.5  # N*m (rotational force from actions)
MAX_ANGULAR_VELOCITY = 10.0  # rad/s (maximum angular speed)
GROUND_FRICTION = 0.8  # Friction coefficient
GROUND_RESTITUTION = 0.2  # Bounciness (0 = no bounce, 1 = perfect bounce)
NORMAL_FORCE_STRENGTH = 500.0  # How strongly ground pushes back

# Simulation
DT = 0.02  # Timestep (50 FPS)
MAX_STEPS = 1500  # Maximum steps per episode
TARGET_DISTANCE = 10.0  # meters

# Rewards
STEP_PENALTY = 0.2  # Penalty per step to encourage speed

# ============================================================================
# ENVIRONMENT
# ============================================================================

class StickWalkEnv:
    def __init__(self):
        # Action space: 0=left rotation, 1=nothing, 2=right rotation
        self.action_space_n = 3
        
        # Observation space: [angle, angular_velocity, x, y, vx, vy]
        self.observation_space_shape = (6,)
        
        # State variables
        self.x = None  # Center x position
        self.y = None  # Center y position
        self.vx = None  # Linear velocity x
        self.vy = None  # Linear velocity y
        self.angle = None  # Radians (0 = horizontal)
        self.angular_velocity = None  # Radians per second
        
        self.step_count = None
        self.start_x = None
        self.prev_x = None
        
    def reset(self):
        """Reset environment to initial state."""
        # Start with stick vertical, slightly off-center to break symmetry
        self.x = 0.0
        self.y = STICK_LENGTH / 2 + 0.1  # Slightly above ground
        self.vx = 0.0
        self.vy = 0.0
        self.angle = np.pi / 2 + 0.1  # Nearly vertical, slight tilt
        self.angular_velocity = 0.0
        
        self.step_count = 0
        self.start_x = self.x
        self.prev_x = self.x
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one timestep."""
        self.step_count += 1
        
        # Apply action torque
        if action == 0:  # Rotate left (counter-clockwise)
            torque = TORQUE_STRENGTH
        elif action == 2:  # Rotate right (clockwise)
            torque = -TORQUE_STRENGTH
        else:  # Do nothing
            torque = 0.0
        
        # Apply torque to angular velocity
        angular_acceleration = torque / MOMENT_OF_INERTIA
        self.angular_velocity += angular_acceleration * DT
        self.angular_velocity = np.clip(self.angular_velocity, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        
        # Apply gravity
        self.vy -= GRAVITY * DT
        
        # Update angle
        self.angle += self.angular_velocity * DT
        
        # Calculate endpoint positions
        half_length = STICK_LENGTH / 2
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        
        # End 1 (left end when horizontal)
        end1_x = self.x - half_length * cos_angle
        end1_y = self.y - half_length * sin_angle
        
        # End 2 (right end when horizontal)
        end2_x = self.x + half_length * cos_angle
        end2_y = self.y + half_length * sin_angle
        
        # Ground collision for both endpoints
        total_force_x = 0.0
        total_force_y = 0.0
        total_torque = 0.0
        
        for end_x, end_y in [(end1_x, end1_y), (end2_x, end2_y)]:
            if end_y < 0:  # Below ground
                # Calculate relative position from center
                rx = end_x - self.x
                ry = end_y - self.y
                
                # Normal force (pushes up)
                penetration_depth = -end_y
                normal_force = NORMAL_FORCE_STRENGTH * penetration_depth
                
                # Friction force (opposes horizontal motion of endpoint)
                # Endpoint velocity = center velocity + rotational contribution
                endpoint_vx = self.vx - ry * self.angular_velocity
                endpoint_vy = self.vy + rx * self.angular_velocity
                
                if abs(endpoint_vx) > 0.01:
                    friction_direction = -np.sign(endpoint_vx)
                    friction_force = GROUND_FRICTION * normal_force * friction_direction
                else:
                    friction_force = 0.0
                
                # Apply forces
                total_force_x += friction_force
                total_force_y += normal_force
                
                # Calculate torques (cross product in 2D: rx * Fy - ry * Fx)
                torque_from_normal = rx * normal_force
                torque_from_friction = -ry * friction_force
                total_torque += torque_from_normal + torque_from_friction
                
                # Small bounce effect
                if endpoint_vy < 0:
                    self.vy += -endpoint_vy * GROUND_RESTITUTION
        
        # Apply forces to linear motion
        ax = total_force_x / MASS
        ay = total_force_y / MASS
        self.vx += ax * DT
        self.vy += ay * DT
        
        # Apply torque to angular motion
        angular_accel = total_torque / MOMENT_OF_INERTIA
        self.angular_velocity += angular_accel * DT
        self.angular_velocity = np.clip(self.angular_velocity, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        
        # Update position
        self.x += self.vx * DT
        self.y += self.vy * DT
        
        # Reward based on forward movement minus time cost
        delta_x = self.x - self.prev_x  # Positive for right, negative for left
        reward = delta_x - STEP_PENALTY
        self.prev_x = self.x
        
        # Check if done
        done = False
        if self.step_count >= MAX_STEPS:
            done = True
        if self.x - self.start_x >= TARGET_DISTANCE:
            done = True
        
        # Info
        info = {
            'distance': self.x - self.start_x,
            'steps': self.step_count,
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get current observation (normalized)."""
        # Normalize values to roughly [-1, 1] range for better learning
        obs = np.array([
            self.angle / np.pi,  # Angle normalized by pi
            self.angular_velocity / 10.0,  # Angular velocity
            self.x / 10.0,  # Position x
            self.y / 2.0,  # Position y
            self.vx / 5.0,  # Velocity x
            self.vy / 5.0,  # Velocity y
        ], dtype=np.float32)
        return obs
    
    def get_state(self):
        """Get raw state for rendering (unnormalized)."""
        return {
            'x': self.x,
            'y': self.y,
            'angle': self.angle,
            'vx': self.vx,
            'vy': self.vy,
            'angular_velocity': self.angular_velocity,
            'step': self.step_count,
        }


if __name__ == "__main__":
    # Test the environment
    env = StickWalkEnv()
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Action space size: {env.action_space_n}")
    print(f"Observation space shape: {env.observation_space_shape}")
    
    # Run a few random steps
    total_reward = 0
    for i in range(100):
        action = np.random.randint(0, env.action_space_n)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 20 == 0:
            state = env.get_state()
            print(f"Step {i}: x={state['x']:.2f}, y={state['y']:.2f}, angle={state['angle']:.2f}, reward={reward:.3f}")
        
        if done:
            print(f"\nEpisode finished after {info['steps']} steps")
            print(f"Total distance: {info['distance']:.2f} meters")
            print(f"Total reward: {total_reward:.2f}")
            break