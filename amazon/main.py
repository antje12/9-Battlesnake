import random
from battlesnake_gym.snake_gym import BattlesnakeGym
from battlesnake_gym.snake import Snake

env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=1)
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
print(observation_space)
print(action_space)

episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 3])
        _, reward, done, _ = env.step([action])
        #score += reward
        env.render()
    
    print(f"Episdoe: {episode}, Score: {score}")

env.close()

env.close()
