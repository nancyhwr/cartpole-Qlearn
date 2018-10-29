import gym
import numpy as np
import matplotlib.pyplot as plt


class Train():


	def __init__(self):

		self.step = 200
		self.episode = 200
		

	def run_episode(self, env, parameters):

		# env.reset()
		observation = env.reset()
		# env.render()
		totalreward = 0

		for i in range(self.step):
			action = 0 if np.matmul(parameters,observation) < 0 else 1
			observation, reward, done, info = env.step(action)
			totalreward += reward

		return totalreward


	def train(self):

		env = gym.make('CartPole-v0')
		bestreward = 0
		bestparams = None
		
		for i in range(self.episode):
			# print('[episode] == ', i)
			env.render()
				# env.monitor.start('cartpole-experiments/', force= True)
			parameters = np.random.rand(4) * 2 - 1
			reward = self.run_episode(env,parameters)
			print('[Episode] = ', i, '[Reward] = ', reward)

			if reward > bestreward:
				bestreward = reward
				bestparams = parameters

			if reward == 300:
				break
				
		return bestreward, bestparams
		
		

# agent1 = Train()
env = gym.make('CartPole-v0')
observation = env.reset()

for i, feature in enumerate(observation):
	print(i)
	print('[feature] = ', feature)
# best_reward, best_param = agent1.train()
# print('[best_reward] = ', best_reward)







