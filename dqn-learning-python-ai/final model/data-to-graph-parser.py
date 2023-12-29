import math
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Specify the path to your text file
txt_file_path = '5-csv.txt'

episode_number = []
average_reward_number = []
average_length_number = []

# Open the text file
with open(txt_file_path, 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Split the line into values using a comma as the delimiter
        values = line.strip().split(',')

        # Unpack values from the list into variables
        episode, average_reward, average_length, best_reward, best_length, score, length, epsilon = map(float, values)

        episode_number.append(episode)
        average_reward_number.append(average_reward)
        average_length_number.append(average_length)

plt.plot(episode_number, average_reward_number, label='Average Reward')
plt.plot(episode_number, average_length_number, label='Average Length')
plt.xlabel('Episode Number')
plt.ylabel('Value')
plt.legend()
plt.show()
