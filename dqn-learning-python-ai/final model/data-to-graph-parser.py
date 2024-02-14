import math
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Specify the path to your text file
txt_file_path_dqn = '5-csv.txt'
txt_file_path_a_star = 'a-star-csv.txt'

episode_number = []

average_reward_number_dqn = []
average_length_number_dqn = []

average_reward_number_a_star = []
average_length_number_a_star = []

# Open the text file
with open(txt_file_path_dqn, 'r') as file_dqn:
    # Iterate over each line in the file
    for line in file_dqn:
        # Split the line into values using a comma as the delimiter
        values = line.strip().split(',')
        # Unpack values from the list into variables
        episode, average_reward, average_length, best_reward, best_length, score, length, epsilon = map(float, values)
        episode_number.append(episode)
        average_reward_number_dqn.append(average_reward)
        average_length_number_dqn.append(average_length)

with open(txt_file_path_a_star, 'r') as file_a_star:
    # Iterate over each line in the file
    for line in file_a_star:
        # Split the line into values using a comma as the delimiter
        values = line.strip().split(',')
        # Unpack values from the list into variables
        episode, average_reward, average_length, best_reward, best_length, score, length, epsilon = map(float, values)
        average_reward_number_a_star.append(average_reward)
        average_length_number_a_star.append(average_length)

plt.plot(episode_number, average_reward_number_dqn, label='Average DQN Reward')
plt.plot(episode_number, average_length_number_dqn, label='Average DQN Length')
plt.plot(episode_number, average_reward_number_a_star, label='Average A* Reward')
plt.plot(episode_number, average_length_number_a_star, label='Average A* Length')
plt.xlabel('Episode Number')
plt.ylabel('Value')
plt.legend()
plt.show()
