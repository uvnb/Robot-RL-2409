import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks import ActorNetwork, CriticNetwork

## DDPG agent class with enhanced stability for drawing tasks
class DDPGAgent:
    def __init__(self, env, input_dims, alpha=1e-4, beta=1e-3, gamma=0.98, 
                 tau=0.001, max_size=int(1e6), noise_factor=0.1, batch_size=128):
        
        # Configurable parameters for better tuning
        self.alpha = alpha  # Actor learning rate
        self.beta = beta    # Critic learning rate
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update rate
        self.batch_size = batch_size
        self.noise_factor = noise_factor
        self.initial_noise_factor = noise_factor
        
        # Enhanced stability parameters
        self.gradient_clip_norm = 1.0
        self.target_update_freq = 2  # Update targets every 2 steps
        self.update_counter = 0
        
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)

        self._initialize_networks(self.n_actions)
        self.update_parameters(tau=1)

    # Choose action based on actor network with improved exploration for drawing tasks
    def choose_action(self, state, add_noise=True):
        """Choose action with improved exploration for drawing tasks"""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)[0]
        
        if add_noise:
            # Use Ornstein-Uhlenbeck-like noise for smoother exploration
            noise = np.random.normal(0, self.noise_factor, size=self.n_actions)
            # Add correlation to make movements smoother (important for drawing)
            if hasattr(self, 'prev_noise'):
                noise = 0.7 * self.prev_noise + 0.3 * noise
            self.prev_noise = noise
            
            actions = actions + noise
        
        # Clip actions to valid range
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        
        return actions.numpy()
    
    def reset_noise(self):
        """Reset noise correlation for new episode"""
        if hasattr(self, 'prev_noise'):
            self.prev_noise = np.zeros(self.n_actions)
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    # Main DDPG algorithms learning process
    def learn(self):
        if self.memory.counter < self.batch_size:
            return None, None

        # Sample batch size of experiences from replay buffer
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        # Calculate critic network loss with enhanced stability
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(new_states)
            # Add small noise to target actions for regularization
            target_noise = tf.random.normal(tf.shape(target_actions), stddev=0.2)
            target_noise = tf.clip_by_value(target_noise, -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions + target_noise, 
                                            self.min_action, self.max_action)
            
            new_critic_value = tf.squeeze(self.target_critic(new_states, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma * new_critic_value * (1 - dones)
            critic_loss = tf.keras.losses.MSE(target, critic_value)
            critic_loss_value = critic_loss.numpy()

        # Apply gradient descent with clipping for stability
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_network_gradient, _ = tf.clip_by_global_norm(critic_network_gradient, 
                                                           self.gradient_clip_norm)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables 
        ))

        # Calculate actor network loss
        actor_loss_value = None
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            actor_loss = -self.critic(states, new_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
            actor_loss_value = actor_loss.numpy()
        
        # Apply gradient descent with clipping for stability
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_network_gradient, _ = tf.clip_by_global_norm(actor_network_gradient, 
                                                          self.gradient_clip_norm)
        self.actor.optimizer.apply_gradients(zip(
                actor_network_gradient, self.actor.trainable_variables 
            ))
        
        # Update target networks with reduced frequency for stability
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_parameters()
        return actor_loss_value, critic_loss_value

    # Update actor/critic target networks parameters with soft update rule
    def update_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_critic.set_weights(weights)

    def save_models(self):
        print("---- saving models ----")
        self.actor.save_weights(self.actor.checkpoints_file)
        self.critic.save_weights(self.critic.checkpoints_file)
        self.target_actor.save_weights(self.target_actor.checkpoints_file)
        self.target_critic.save_weights(self.target_critic.checkpoints_file)

    def load_models(self):
        print("---- loading models ----")
        self.actor.load_weights(self.actor.checkpoints_file)
        self.critic.load_weights(self.critic.checkpoints_file)
        self.target_actor.load_weights(self.target_actor.checkpoints_file)
        self.target_critic.load_weights(self.target_critic.checkpoints_file)

    def _initialize_networks(self, n_actions):
        model = "ddpg"
        self.actor = ActorNetwork(n_actions, name="actor", model=model)
        self.critic = CriticNetwork(name="critic", model=model)
        self.target_actor = ActorNetwork(n_actions, name="target_actor", model=model)
        self.target_critic = CriticNetwork(name="target_critic", model=model)

        self.actor.compile(keras.optimizers.Adam(learning_rate=self.alpha))
        self.critic.compile(keras.optimizers.Adam(learning_rate=self.beta))
        self.target_actor.compile(keras.optimizers.Adam(learning_rate=self.alpha))
        self.target_critic.compile(keras.optimizers.Adam(learning_rate=self.beta))

    



