import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

class EpisodeHistory:

	def __init__(self, capacity = 5000, num_epoch=5000, plot_episode_count= 50, num_time_step = 200):
		
		self.lengths = np.zeros(capacity, dtype=int) # define the length of the x-axis
		self.capacity = capacity
		self.num_epoch = num_epoch # define the number of epochs
		self.plot_episode_count = plot_episode_count # define the frequency to update the plot; here is every 50 steps
		self.num_time_step = num_time_step # define the maximum of time steps in one epoch 
		

	def __getitem__(self, episode_index):
		return self.lengths[episode_index]

	def __setitem__(self, episode_index, episode_length):
		self.lengths[episode_index] = episode_length
       
       
	def create_plot(self):
		self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
		self.fig.canvas.set_window_title("Episode Length History")

		self.ax.set_xlim(0, self.num_epoch + 5)
		self.ax.set_ylim(0, self.num_time_step + 5)
		self.ax.yaxis.grid(True)

		self.ax.set_title("Episode Length History")
		self.ax.set_xlabel("Episode #")
		self.ax.set_ylabel("Length, timesteps")

		self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
		self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")

	def update_plot(self, episode_index):
        
		plot_right_edge = episode_index
		plot_left_edge = max(0, plot_right_edge-self.plot_episode_count)
		
		x = range(plot_left_edge, plot_right_edge)
		y = self.lengths[plot_left_edge:plot_right_edge]

		self.point_plot.set_xdata(x)
		self.point_plot.set_ydata(y)
		self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

		# update the mean value line

		mean_kernel_size = 100
		rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index])) 
		rolling_means = pd.rolling_mean(
			rolling_mean_data,
			window=mean_kernel_size,
			min_periods=0
			)[mean_kernel_size:]
		self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
		self.mean_plot.set_ydata(rolling_means)

		plt.draw()
		plt.pause(0.0001)
   