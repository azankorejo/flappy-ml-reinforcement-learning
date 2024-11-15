import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import random
import numpy as np
from collections import deque
sys.path.append("game/")
import flappy as game

class DQNAgent:
    def __init__(self):
        # Game parameters
        self.game_name = 'bird'
        self.action_count = 2  # Number of possible actions
        self.gamma = 0.99  # Discount factor for future rewards
        self.initial_epsilon = 0.0001
        self.final_epsilon = 0.0001
        self.observe_phase = 100000.  # Time steps to observe before training
        self.explore_phase = 2000000.  # Time steps to anneal epsilon
        self.frame_per_action = 1
        self.replay_memory_size = 50000  # Replay memory size
        self.batch_size = 32  # Mini-batch size for training
        self.epsilon = self.initial_epsilon  # Initial exploration probability
        self.replay_memory = deque()
        
        # Network placeholders and weights
        self.session = tf.InteractiveSession()
        self.state_input, self.q_values, self.hidden_fc = self.build_network()
        self.optimizer = self.build_training_step()
        
        # Game state
        self.game_state = game.FlappyBirdGame()
        self.saver = tf.train.Saver()
        self.load_model()
        
    def weight_variable(self, shape):
        # Initialize weights with a small random value
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # Initialize biases with a small constant value
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def conv2d_layer(self, x, W, stride):
        # adding a convolutional layer
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_layer(self, x):
        # adding a max pooling layer
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def build_network(self):
        # adding the structure of the Q-network
        W_conv1 = self.weight_variable([8, 8, 4, 32])
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        W_fc1 = self.weight_variable([1600, 512])
        b_fc1 = self.bias_variable([512])
        W_fc2 = self.weight_variable([512, self.action_count])
        b_fc2 = self.bias_variable([self.action_count])

        # Input and hidden layers
        state_input = tf.placeholder("float", [None, 80, 80, 4])
        conv1 = tf.nn.relu(self.conv2d_layer(state_input, W_conv1, 4) + b_conv1)
        pool1 = self.max_pool_layer(conv1)
        conv2 = tf.nn.relu(self.conv2d_layer(pool1, W_conv2, 2) + b_conv2)
        conv3 = tf.nn.relu(self.conv2d_layer(conv2, W_conv3, 1) + b_conv3)
        flattened_conv3 = tf.reshape(conv3, [-1, 1600])
        hidden_fc = tf.nn.relu(tf.matmul(flattened_conv3, W_fc1) + b_fc1)

        # Output layer
        q_values = tf.matmul(hidden_fc, W_fc2) + b_fc2

        return state_input, q_values, hidden_fc

    def build_training_step(self):
        # Defining placeholders and loss for training
        self.action_input = tf.placeholder("float", [None, self.action_count])
        self.target_q = tf.placeholder("float", [None])
        q_action = tf.reduce_sum(tf.multiply(self.q_values, self.action_input), reduction_indices=1)
        loss = tf.reduce_mean(tf.square(self.target_q - q_action))
        return tf.train.AdamOptimizer(1e-6).minimize(loss)

    def load_model(self):
        # Initialize or load model parameters
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("trained_models")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Model loaded from:", checkpoint.model_checkpoint_path)
        else:
            print("No previous model found, initializing new model.")

    def preprocess_frame(self, frame):
        # converting frame to grayscale and resize
        gray_frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, binary_frame = cv2.threshold(gray_frame, 1, 255, cv2.THRESH_BINARY)
        return binary_frame

    def store_transition(self, current_state, action, reward, next_state, terminal):
        # storeing experience in replay memory
        self.replay_memory.append((current_state, action, reward, next_state, terminal))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()

    def train_batch(self):
        # Sampling a batch from replay memory and perform training
        minibatch = random.sample(self.replay_memory, self.batch_size)
        state_batch = [item[0] for item in minibatch]
        action_batch = [item[1] for item in minibatch]
        reward_batch = [item[2] for item in minibatch]
        next_state_batch = [item[3] for item in minibatch]
        target_batch = []

        q_next = self.q_values.eval(feed_dict={self.state_input: next_state_batch})
        for i, item in enumerate(minibatch):
            terminal = item[4]
            if terminal:
                target_batch.append(reward_batch[i])
            else:
                target_batch.append(reward_batch[i] + self.gamma * np.max(q_next[i]))

        self.optimizer.run(feed_dict={
            self.target_q: target_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def select_action(self, state, timestep):
        # selecting action based on epsilon-greedy policy
        action = np.zeros(self.action_count)
        action_index = 0
        if timestep % self.frame_per_action == 0:
            if random.random() <= self.epsilon:
                print("Performing random action.")
                action_index = random.randrange(self.action_count)
                action[random.randrange(self.action_count)] = 1
            else:
                q_values = self.q_values.eval(feed_dict={self.state_input: [state]})[0]
                action_index = np.argmax(q_values)
                action[action_index] = 1
        else:
            action[0] = 1  # No action

        if self.epsilon > self.final_epsilon and timestep > self.observe_phase:
            self.epsilon -= (self.initial_epsilon - self.final_epsilon) / self.explore_phase

        return action

    def train_agent(self):
        initial_action = np.zeros(self.action_count)
        initial_action[0] = 1
        frame, _, terminal = self.game_state.update_frame(initial_action)
        state = np.stack((self.preprocess_frame(frame),) * 4, axis=2)
        timestep = 0

        while True:
            action = self.select_action(state, timestep)
            next_frame, reward, terminal = self.game_state.update_frame(action)
            next_frame_processed = self.preprocess_frame(next_frame)
            # Ensure next_frame_processed has shape (80, 80, 1)
            next_frame_processed = np.reshape(next_frame_processed, (80, 80, 1))
            # Append along axis 2 to create the next state
            next_state = np.append(next_frame_processed, state[:, :, :3], axis=2)
            next_state = np.append(next_frame_processed, state[:, :, :3], axis=2)

            self.store_transition(state, action, reward, next_state, terminal)

            if timestep > self.observe_phase:
                self.train_batch()

            state = next_state
            timestep += 1

            if timestep % 10000 == 0:
                self.saver.save(self.session, f'trained_models/{self.game_name}-dqn', global_step=timestep)

            print("TIMESTEP", timestep, "/ STATE", "observe" if timestep <= self.observe_phase else "train", "/ EPSILON", self.epsilon, "/ ACTION", np.argmax(action), "/ REWARD", reward)

def main():
    agent = DQNAgent()
    agent.train_agent()

if __name__ == "__main__":
    main()
