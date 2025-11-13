import numpy as np
import pickle

class NeuralNetwork:
    """Simple feedforward neural network for the stick walker."""
    
    def __init__(self, input_size=6, hidden_sizes=[32, 32], output_size=3):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.5)
        self.biases.append(np.zeros(hidden_sizes[0]))
        
        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * 0.5)
            self.biases.append(np.zeros(hidden_sizes[i+1]))
        
        # Last hidden to output
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.5)
        self.biases.append(np.zeros(output_size))
    
    def forward(self, x):
        """Forward pass through the network."""
        activation = x
        
        # Pass through all layers except last
        for i in range(len(self.weights) - 1):
            activation = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = np.tanh(activation)  # Activation function
        
        # Output layer
        output = np.dot(activation, self.weights[-1]) + self.biases[-1]
        return output
    
    def predict(self, observation):
        """Predict action from observation."""
        output = self.forward(observation)
        return np.argmax(output)  # Return action with highest value
    
    def get_weights_flat(self):
        """Get all weights and biases as a flat array."""
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.extend(w.flatten())
            flat.extend(b.flatten())
        return np.array(flat)
    
    def set_weights_flat(self, flat_weights):
        """Set all weights and biases from a flat array."""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            
            self.weights[i] = flat_weights[idx:idx+w_size].reshape(self.weights[i].shape)
            idx += w_size
            
            self.biases[i] = flat_weights[idx:idx+b_size].reshape(self.biases[i].shape)
            idx += b_size
    
    def copy(self):
        """Create a copy of this network."""
        new_net = NeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)
        new_net.set_weights_flat(self.get_weights_flat())
        return new_net
    
    def save(self, filepath):
        """Save network to file."""
        data = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'weights': self.weights,
            'biases': self.biases
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath):
        """Load network from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        net = NeuralNetwork(data['input_size'], data['hidden_sizes'], data['output_size'])
        net.weights = data['weights']
        net.biases = data['biases']
        return net