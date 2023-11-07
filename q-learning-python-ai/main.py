import gymnasium as gym
import random
import numpy as np
import time

env = gym.make('FrozenLake-v1')

action_space_size = env.action_space.n
print(action_space_size)
state_space_size = env.observation_space.n
print(state_space_size)
print("----------")

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

# Q-learning algorithm 
for episode in range(num_episodes):
    state, info = env.reset()

    done = False
    rewards_current_episode = 0

    for Step in range(max_steps_per_episode):

        # Exploration-exploitation trade-off
        exploitation_rate_threshold = random.uniform(0, 1)
        if exploitation_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()
        
        new_state, reward, done, truncated, info = env.step(action)

        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print ("----------Average reward per thousand episodes----------")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("----------Q-table----------")
print(q_table)

env = gym.make('FrozenLake-v1', render_mode="human")
for episode in range(3):
    state, info = env.reset()
    done = False
    print("----------Episode ", episode+1, "----------")
    time.sleep(1)

    for Step in range(max_steps_per_episode):
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, info = env.step(action)

        if done:
            if reward == 1:
                print("----------You reached the goal!----------")
                time.sleep(1)
            else:
                print("----------You fell through a hole!----------")
                time.sleep(1)
            break
        
        state = new_state

env.close()

#episodes = 10
#for episode in range(1, episodes+1):
#    state = env.reset()
#    done = False
#    score = 0
#
#    while not done:
#        action = random.choice([0, 1])
#        _, reward, done, _, _ = env.step(action)
#        score += reward
#        env.render()
#    
#    print(f"Episdoe: {episode}, Score: {score}")
#
#env.close()
