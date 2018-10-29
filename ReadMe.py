import gym


env = gym.make('CartPole-v0')
observation  = env.reset()   # return the state
for i, feature in enumerate(observation):
	print(i, feature) # i = [0, 1, 2, 3] features: each index's value

observation, reward, done, info = env.step(0)   # make an action and returnt those info