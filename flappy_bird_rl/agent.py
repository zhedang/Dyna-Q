
import numpy as np
import config

class QLearningAgent:
    def __init__(self):
        # Discretization bins
        # We simplify the state to be the bird's relative position to the next pipe gap
        self.y_bins = np.linspace(-config.SCREEN_HEIGHT, config.SCREEN_HEIGHT, 20)
        self.x_bins = np.linspace(0, config.SCREEN_WIDTH, 20)
        self.vel_bins = np.linspace(-10, 10, 10)

        # State space size
        self.state_space_dims = (len(self.y_bins)+1, len(self.x_bins)+1, len(self.vel_bins)+1)
        self.q_table = np.zeros(self.state_space_dims + (2,)) # 2 actions: 0 (do nothing), 1 (jump)

        # Hyperparameters
        self.learning_rate = config.LEARNING_RATE
        self.discount_factor = config.DISCOUNT_FACTOR
        self.epsilon = config.EPSILON_START
        self.epsilon_decay = config.EPSILON_DECAY
        self.epsilon_min = config.EPSILON_END

    def _discretize_state(self, state):
        """Converts a continuous state into a discrete one."""
        bird_y, bird_vel, pipe_y, pipe_x = state

        # State representation: relative position to the pipe gap center
        relative_y = bird_y - (pipe_y + config.PIPE_GAP_SIZE / 2)
        relative_x = pipe_x - 50 # 50 is the bird's fixed x position

        # Digitize to find the bin index
        y_idx = np.digitize(relative_y, self.y_bins)
        x_idx = np.digitize(relative_x, self.x_bins)
        vel_idx = np.digitize(bird_vel, self.vel_bins)

        return (y_idx, x_idx, vel_idx)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            # Explore, with a bias towards doing nothing to make exploration more stable
            return np.random.choice([0, 1], p=[0.9, 0.1])
        else:
            discrete_state = self._discretize_state(state)
            return np.argmax(self.q_table[discrete_state]) # Exploit

    def update(self, state, action, reward, next_state, done):
        """Update Q-table using the Q-learning formula."""
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)

        if done:
            target = reward
        else:
            best_next_action_q_value = np.max(self.q_table[discrete_next_state])
            target = reward + self.discount_factor * best_next_action_q_value

        # Q-learning formula
        self.q_table[discrete_state][action] = (
            (1 - self.learning_rate) * self.q_table[discrete_state][action] +
            self.learning_rate * target
        )

    def decay_epsilon(self):
        """Decay epsilon to reduce exploration over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
