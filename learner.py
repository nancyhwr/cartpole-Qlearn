import numpy as np 
import gym
import matplotlib.pyplot as plt 
from history_plot import EpisodeHistory as HistoryTracker



class Learner():


	def __init__(self, exploration_likelihood, decay, learning_rate, discount, num_epoch, num_time):

		self.exploration_likelihood = exploration_likelihood 
		self.decay = decay
		self._num_actions = 2
		self.learning_rate = learning_rate
		self.discount = discount
		self.num_epoch = num_epoch
		self.num_time  = num_time
		self.temp = 100
		self.temp_decay = 0.99

		self.num_discretization_bins  = 10
		self._state_bins = [
            # Cart position.
			self._discretize_range(-2.4, 2.4, self.num_discretization_bins),
            # Cart velocity.
			self._discretize_range(-3.0, 3.0, self.num_discretization_bins),
            # Pole angle.
			self._discretize_range(-0.5, 0.5, self.num_discretization_bins),
            # Tip velocity.
			self._discretize_range(-2.0, 2.0, self.num_discretization_bins)
		]
		self._max_bins = max(len(bin) for bin in self._state_bins)
		self._num_states = np.prod([len(self._state_bins[i]) for i in range(len(self._state_bins))])
		self.q = {}
		for s1 in np.arange(0, len(self._state_bins[0])+1, 1):
			for s2 in np.arange(0, len(self._state_bins[1])+1, 1):
				for s3 in np.arange(0, len(self._state_bins[2])+1, 1):
					for s4 in np.arange(0, len(self._state_bins[3])+1, 1):

						self.q[tuple([s1, s2, s3, s4]), 0] = 0
						self.q[tuple([s1, s2, s3, s4]), 1] = 0


	## discretize the states
	@staticmethod
	def _discretize_range(lower_bound, upper_bound, num_bins):
		return np.linspace(lower_bound, upper_bound, num_bins)

	@staticmethod
	def _discretize_value(value, bins):
		return np.digitize(x=value, bins=bins)

	def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single integer.
		state = sum(
			self._discretize_value(feature, self._state_bins[i]) * ((self._max_bins + 1) ** i)
			for i, feature in enumerate(observation)
		)
		return state

	
	def get_state(self, observation):

		state = tuple([self._discretize_value(feature, self._state_bins[i]).item(0) for i, feature in enumerate(observation)])
		return state


	def boltzaman(self, next_state):

	
		q_values = [self.q[next_state, i] for i in range(self._num_actions)]
		exp_q = [np.exp(v/self.temp) for v in q_values]
		p_action = [v/sum(exp_q) for v in exp_q]
		next_action = np.argmax([p_action]) 
		return next_action

	# epsilon exploration policy
	def take_action(self, next_state):
	
		enable_exploration = np.random.uniform(0, 1) < self.exploration_likelihood
		if enable_exploration:
			next_action = np.random.randint(0, self._num_actions)
		else:
			next_action = np.argmax([self.q[next_state, i] for i in range(self._num_actions)])

		return next_action


	def state_hash(self, state):

		return sum([state[i] for i in range(len(state))])

	def run_agent(self, env):

		num_redos  = []
		# epoch_history = HistoryTracker(self.num_epoch, 100) # num_epoch number of EPOCHS and update every 100 step
		# epoch_history.create_plot()
		
		for epoch_idx in range(self.num_epoch):

			observation = env.reset()
			temp_state = self.get_state(observation)
			temp_action = np.random.choice([0, 1])
			fail_count = 0

			for time_idx in range(self.num_time):
				# env.render()
				observation, reward, done, info = env.step(temp_action)
				next_state = self.get_state(observation)
				
				if done and time_idx < self.num_time-1:
					fail_count = time_idx
					reward = - self.num_time
					# epoch_history[epoch_idx] = time_idx+1
					# epoch_history.update_plot(epoch_idx)
					env.reset()
					break
					
				self.q[temp_state, temp_action] += self.learning_rate*(reward + self.discount * \
					max([self.q[next_state,i] for i in range(self._num_actions)]) - self.q[temp_state, temp_action])
				
				temp_state = next_state
				temp_action = self.take_action(temp_state)
				# temp_action = self.boltzaman(temp_state)


			num_redos.append(fail_count)
			self.exploration_likelihood = self.exploration_likelihood * self.decay
			self.temp = self.temp * self.temp_decay

		return num_redos


if __name__ == "__main__":

	random_state = 10
	agent = Learner(exploration_likelihood=0.5, decay = 0.99, learning_rate = 0.5, discount = 1.0, num_epoch = 1000, num_time = 500)
	env  = gym.make('CartPole-v0')
	env.seed(random_state)
	np.random.seed(random_state)
	
	num_redo = agent.run_agent(env)

	smooth_num_fails = [np.average(num_redo[:i]) for i in np.arange(1, len(num_redo), 1)]
	x = [i for i in range(len(num_redo))]

	plt.plot(x, num_redo)
	plt.plot([j for j in range(len(smooth_num_fails))], smooth_num_fails)
	plt.show()










