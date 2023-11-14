import random
from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=4)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
print(observation_space)
print(action_space)

episodes = 1
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render("ascii")
        action = random.choice([0])
        rewards = {}
        observation, rewards, done, info = env.step([action, action, action, action])

        food = observation[:, :, 0]
        s1 = observation[:, :, 1]
        s2 = observation[:, :, 2]
        s3 = observation[:, :, 3]
        s4 = observation[:, :, 4]

        print("-----food-----")
        print(food)
        print("-----s1-----")
        print(s1)
        print("-----s2-----")
        print(s2)
        print("-----s3-----")
        print(s3)
        print("-----s4-----")
        print(s4)
        print("----------")

        done = done[0]
        #score += reward
    
    print(f"Episode: {episode}, Score: {score}")

env.close()

env.close()
