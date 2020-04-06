import numpy as np 
import gym

import tensorflow as tf 
from tensorflow.keras.layers import Dense
from gym import wrappers
import os
import wandb

wandb.init(project="bipedalwalker_1")


## initialize actor, taget_actor
## initialize critic, target_critic

## initialize reply buffer
class Actor(tf.keras.Model):
	def __init__(self, state_dim, action_dim, max_action, units=[400, 300], name="Actor"):
		super().__init__(name=name)

		self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim, ))
		self.l1 = Dense(units[0], name="L1")
		self.l2 = Dense(units[1], name="L2")
		self.l3 = Dense(action_dim, name="L3")

		self.max_action = max_action

        # with tf.device("/cpu:0"):
        #     self(tf.constant(np.zeros(shape=(1,)+state_shape, dtype=np.float32)))

	def call(self, inputs):

		inputs_ = self.input_layer(inputs)
		features = tf.nn.relu(self.l1(inputs_))
		features = tf.nn.relu(self.l2(features))
		features = self.l3(features)
		action = self.max_action * tf.nn.tanh(features)
		return action

## Critic network

class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, units=[400, 300], name="Critic"):
        super().__init__(name=name)

        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim+action_dim, ))
        self.l1 = Dense(units[0], name="L1")
        self.l2 = Dense(units[1], name="L2")
        self.l3 = Dense(1, name="L3")

        # dummy_state = tf.constant(
        #     np.zeros(shape=(1,)+state_dim, dtype=np.float32))
        dummy_state = tf.constant(
            np.zeros(shape=[1,state_dim], dtype=np.float32))
        dummy_action = tf.constant(
            np.zeros(shape=[1, action_dim], dtype=np.float32))
        
        # with tf.device("/cpu:0"):
        #     self([dummy_state, dummy_action])

    def call(self, inputs):

        states, actions = inputs
        print(type(states), type(actions))
        features = tf.concat([states, actions], axis=1)
        features = self.input_layer(features)
        features = tf.nn.relu(self.l1(features))
        features = tf.nn.relu(self.l2(features))
        features = self.l3(features)
        return features



class DDPG():

	def __init__(self, state_dim, action_dim, max_action, env, actor_units=[128, 64], critic_units =[128, 64], buffer_size=1000000, 
						start_learning = 10000, update_per_step =1, update_target_per_step = 50, save_per_step = 50, batch_size = 100,
						lr_actor = 0.001, lr_critic = 0.001, discount = 0.99, num_epochs = 100, num_steps = 4000):


		self.state_dim = state_dim
		self.action_dim = action_dim

		self.actor = Actor(self.state_dim, self.action_dim, max_action, actor_units)
		self.target_actor = Actor(self.state_dim, self.action_dim, max_action, actor_units)
		self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
		self.actor.compile(optimizer = self.actor_optimizer, loss = 'huber_loss')

		self.update_target_variables(self.actor, self.target_actor, eta = 1.)


        
		self.critic = Critic(self.state_dim, self.action_dim, critic_units)
		self.target_critic = Critic(self.state_dim, self.action_dim, critic_units)
		self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_critic)
		self.critic.compile(optimizer = self.critic_optimizer, loss = 'huber_loss')

		self.update_target_variables(self.critic, self.target_critic, eta=1.)


		self.buffer_size = buffer_size
		self.env = env


		self.obs = np.empty((self.buffer_size,)+ self.env.reset().shape)
		self.actions = np.empty((self.buffer_size,) + self.env.action_space.sample().shape)
		self.rewards = np.empty((self.buffer_size), dtype = np.float32)
		self.dones = np.empty((self.buffer_size), dtype = np.bool)
		self.next_states = np.empty((self.buffer_size, )+self.env.reset().shape)


		self.next_idx = 0
		self.start_learning = start_learning
		self.update_per_step = update_per_step
		self.update_target_per_step = update_target_per_step
		self.batch_size = batch_size
		self.num_in_buffer = 0
		self.discount = discount
		self.num_epochs = num_epochs
		self.num_steps = num_steps
		self.save_per_step = save_per_step
		# self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
		# self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate = lr_critic)



	def train(self):

		step = 0
		obs = self.env.reset()

		for epoch in range(self.num_epochs):

			epoch_reward = 0.0
			

			for _ in range(self.num_steps):

				
				action = self._get_action(obs, 1)
				next_obs, reward, done, info = self.env.step(action)
				
				epoch_reward += reward

				self.store_transition(obs, action, reward, next_obs, done)
				self.num_in_buffer = min(self.num_in_buffer+1, self.buffer_size)

				if done:
					obs = self.env.reset()

				else:
					obs = next_obs

				step += 1
				if step > self.start_learning and  not step % self.update_per_step:

					a_loss, c_loss = self.train_step()

					wandb.log({'actor loss': a_loss})
					wandb.log({'critic loss': c_loss})

					# print('epoch: {}, actor loss: {}, critic loss:{}'.format(epoch, a_loss, c_loss ))

				if step > self.start_learning and not step % self.update_target_per_step:

					self.update_target_variables(self.actor, self.target_actor)
					self.update_target_variables(self.critic, self.target_critic)

				if step > self.start_learning and not step % self.save_per_step:

					self.save_model(self.actor, 'ddpg/models/actor_checkpoint')
					self.save_model(self.critic, 'ddpg/models/critic_checkpoint')

			print("epoch_reward  == ", epoch_reward)
			wandb.log({'episode_total_reward': epoch_reward})

		# np.save('weights', weights)
		# wandb.save('weights.npy')


	def train_step(self):

		idxes = self.sample(self.batch_size)
		s_batch = self.obs[idxes]
		a_batch = self.actions[idxes]
		r_batch = self.rewards[idxes]
		ns_batch = self.next_states[idxes]
		done_batch = self.dones[idxes]

		critic_loss = self.update_critic(s_batch, a_batch, r_batch, ns_batch, done_batch)
		actor_loss = self.update_actor(s_batch, a_batch, r_batch, ns_batch, done_batch)
		
		return actor_loss, critic_loss


	def update_critic(self, s_batch, a_batch, r_batch, ns_batch, done_batch):
		
		not_dones = 1. - done_batch
		target_Q = self.target_critic([ns_batch, self.target_actor(ns_batch)])
		target_Q = r_batch + (not_dones * self.discount * target_Q)

		target_Q = tf.stop_gradient(target_Q)
		# current_Q = self.critic([s_batch, a_batch])
		# print("@@@@@@s_batch: ", type(s_batch), s_batch.shape)
		# print('##### a_batch: ', type(a_batch), a_batch.shape)
		
		critic_loss = self.critic.train_on_batch([s_batch, a_batch], target_Q.numpy())
		return critic_loss


	def update_actor(self, s_batch, a_batch, r_batch, ns_batch, done_batch):
		
		with tf.GradientTape() as tape:

			next_action = self.actor(s_batch)
			actor_loss = -tf.reduce_mean(self.critic([s_batch, next_action]))

		actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))




	def sample(self, n):

		assert n<self.num_in_buffer
		return np.random.choice(self.num_in_buffer, self.batch_size, replace = False)


	def store_transition(self, obs, action, reward, next_state, done):

		n_idx = self.next_idx % self.buffer_size
		self.obs[n_idx] = obs
		self.actions[n_idx] = action
		self.rewards[n_idx] = reward
		self.next_states[n_idx] =  next_state
		self.dones[n_idx] =  done 
		self.next_idx  = (self.next_idx+1)%self.buffer_size



	def _get_action(self, state, max_action, sigma=1.):

		action = self.actor(state.reshape(1, self.state_dim))
		action += tf.random.normal(shape=action.shape, mean=0., stddev=sigma, dtype=tf.float32)
		action = action.numpy()
		# print("$$$$$$$$$$$ action: ", action)
		return np.clip(action, -max_action, max_action)[0]
		# return tf.clip_by_value(action, -max_action, max_action)[0]

	def update_target_variables(self, model, target_model, eta = 0.095):

		target_model.set_weights((1-eta)*np.array(target_model.get_weights()) + eta*np.array(model.get_weights()))

	
	def save_model(self, model, model_path_dir):


		model.save_weights(model_path_dir)


if __name__ == '__main__':

	env = gym.make("BipedalWalker-v3")
	env = wrappers.Monitor(env, os.path.join(os.getcwd(), 'ddpg/videos/bipedalWalker/bpw'), force = True)

	state_shape = env.reset().shape[0]
	action_dim = env.action_space.sample().shape[0]

	ddpg = DDPG(state_dim=state_shape, action_dim=action_dim, max_action =1, env=env)
	ddpg.train()














