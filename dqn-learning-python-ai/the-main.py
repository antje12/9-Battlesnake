import math  
import random  
  
import numpy as np  
import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
  
from battlesnake_gym.snake_gym import BattlesnakeGym  
from battlesnake_gym.snake import Snake  
 
# taken from https://andrew-gordienko.medium.com/reinforcement-learning-dqn-w-pytorch-7c6faad3d1e 
 
# 11*11 (height * width) sized map with 3 values in each cell
features = 3
height = 11
width = 11
env = BattlesnakeGym(map_size=(height, width), number_of_snakes=4) 
observation_space = (features, height, width) 
print("Observation space:", observation_space) 
action_space = 4 
print("Action space:", action_space) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Device:", DEVICE) 
print("--------------------") 

HIDDEN_LAYER1_DIMS = 64
HIDDEN_LAYER2_DIMS = 128
HIDDEN_LAYER3_DIMS = 512
HIDDEN_LAYER4_DIMS = 256
LEARNING_RATE = 0.0005

MEM_SIZE = 20000
BATCH_SIZE = 100

EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.001
GAMMA = 0.85

EPISODES = 3000

class Network(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        input_channels = features  # Number of channels in the input (e.g., features for each cell)
        
        # 2D array processing (spatial local 5*5 info processing)
        self.conv1 = nn.Conv2d(input_channels, HIDDEN_LAYER1_DIMS, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER1_DIMS, HIDDEN_LAYER2_DIMS, kernel_size=5, stride=1, padding=2)
        
        # 1D array processing (global info processing)
        self.fc3 = nn.Linear(HIDDEN_LAYER2_DIMS * height * width, HIDDEN_LAYER3_DIMS) 
        self.fc4 = nn.Linear(HIDDEN_LAYER3_DIMS, HIDDEN_LAYER4_DIMS) 

        self.out = nn.Linear(HIDDEN_LAYER4_DIMS, action_space) 
        
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE) 
        self.loss = nn.MSELoss() 
        self.to(DEVICE) 
    
    def forward(self, x): 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, HIDDEN_LAYER2_DIMS * height * width)  # Flatten the output of the convolutional layers
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        x = self.out(x) 
        return x 
 
class ReplayBuffer: 
    def __init__(self): 
        self.mem_count = 0 
        
        self.states = np.zeros((MEM_SIZE, *observation_space),dtype=np.int64) 
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64) 
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32) 
        self.states_ = np.zeros((MEM_SIZE, *observation_space),dtype=np.int64) 
        self.dones = np.zeros(MEM_SIZE, dtype=bool) 
     
    def add(self, state, action, reward, state_, done): 
        mem_index = self.mem_count % MEM_SIZE 
         
        self.states[mem_index]  = state
        self.actions[mem_index] = action 
        self.rewards[mem_index] = reward 
        self.states_[mem_index] = state_ 
        self.dones[mem_index] =  1 - done 
 
        self.mem_count += 1 
     
    def sample(self): 
        MEM_MAX = min(self.mem_count, MEM_SIZE) 
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True) 
         
        states  = self.states[batch_indices] 
        actions = self.actions[batch_indices] 
        rewards = self.rewards[batch_indices] 
        states_ = self.states_[batch_indices] 
        dones   = self.dones[batch_indices] 
 
        return states, actions, rewards, states_, dones 
 
class DQN_Solver: 
    def __init__(self, model=None): 
        self.memory = ReplayBuffer() 
        self.exploration_rate = EXPLORATION_MAX 
 
        if model is not None: 
            self.network = model 
        else: 
            self.network = Network() 
 
    def choose_action(self, observation): 
        if random.random() < self.exploration_rate: 
            return env.action_space[0].sample() 
         
        state = torch.tensor(observation).float().detach() 
        state = state.to(DEVICE) 
        state = state.unsqueeze(0) 
        q_values = self.network(state) 
        return torch.argmax(q_values).item() 
     
    def learn(self): 
        if self.memory.mem_count < BATCH_SIZE: 
            return 
         
        states, actions, rewards, states_, dones = self.memory.sample() 
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE) 
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE) 
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE) 
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE) 
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE) 
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64) 
 
        q_values = self.network(states) 
        next_q_values = self.network(states_) 
 
        predicted_value_of_now = q_values[batch_indices, actions] 
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0] 
         
        q_target = rewards + GAMMA * predicted_value_of_future * dones 
 
        loss = self.network.loss(q_target, predicted_value_of_now) 
        self.network.optimizer.zero_grad() 
        loss.backward() 
        self.network.optimizer.step() 
 
        self.exploration_rate *= EXPLORATION_DECAY 
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate) 
 
    def returning_epsilon(self): 
        return self.exploration_rate 
 
def save_model(model, filepath): 
    torch.save(model.state_dict(), filepath) 
    print(f"Model saved to {filepath}") 
 
def load_model(filepath):  
    model = Network()  
    model.load_state_dict(torch.load(filepath))  
    print(f"Model loaded from {filepath}")  
    return model  
 
def extract_state(me, food, enemy1, enemy2, enemy3):
    map_array = np.zeros((11, 11, 3), dtype=int)  # 2D array with 3 possible values pr. cell
    for x in range(11):
        for y in range(11):
            cell_me = me[x][y]
            if cell_me == 5:  # my head
                map_array[x, y, 0] = 1  # Assigning value 3 for the player's head
            elif cell_me > 0:  # my body
                map_array[x, y, 1] = 1  # Assigning value 1 for the player's body
            cell_enemy1 = enemy1[x][y]
            cell_enemy2 = enemy2[x][y]
            cell_enemy3 = enemy3[x][y]
            if cell_enemy1 > 0 or cell_enemy2 > 0 or cell_enemy3 > 0:  # enemy
                map_array[x, y, 1] = 1  # Assigning value 1 for the enemy's body
            cell_food = food[x][y]
            if cell_food > 0:  # food
                map_array[x, y, 2] = 1  # Assigning value 2 for food
    # Add batch dimension
    map_array = map_array.reshape(features, height, width)
    return map_array
 
agent = DQN_Solver(load_model("final_model.pth"))
  
best_reward = 0 
best_length = 0 
average_reward = 0 
average_length = 0  
episode_number = [] 
average_reward_number = [] 
average_length_number = []

SAVE_FREQUENCY = 100

for i in range(1, EPISODES+1): 
    state, reward, done, info = env.reset() 
    food = state[:, :, 0] 
    snake = state[:, :, 1] 
    enemy1 = state[:, :, 2] 
    enemy2 = state[:, :, 3] 
    enemy3 = state[:, :, 4] 

    state1 = extract_state(snake, food, enemy1, enemy2, enemy3)
    state2 = extract_state(enemy1, food, snake, enemy2, enemy3)
    state3 = extract_state(enemy2, food, enemy1, snake, enemy3)
    state4 = extract_state(enemy3, food, enemy1, enemy2, snake)

    score1 = 0 
    score2 = 0 
    score3 = 0 
    score4 = 0 
 
    backup_snake = snake 
    backup_enemy1 = enemy1 
    backup_enemy2 = enemy2 
    backup_enemy3 = enemy3 
 
    while True: 
        #env.render("ascii") 
        action1 = agent.choose_action(state1) 
        action2 = agent.choose_action(state2) 
        action3 = agent.choose_action(state3) 
        action4 = agent.choose_action(state4) 

        state_, reward, done, info = env.step([action1, action2, action3, action4]) 
 
        food = state_[:, :, 0] 
        snake = state_[:, :, 1] 
        enemy1 = state_[:, :, 2] 
        enemy2 = state_[:, :, 3] 
        enemy3 = state_[:, :, 4] 
        
        state_1 = extract_state(snake, food, enemy1, enemy2, enemy3)
        state_2 = extract_state(enemy1, food, snake, enemy2, enemy3)
        state_3 = extract_state(enemy2, food, enemy1, snake, enemy3)
        state_4 = extract_state(enemy3, food, enemy1, enemy2, snake)

        agent.memory.add(state1, action1, reward[0], state_1, done[0]) 
        agent.memory.add(state2, action2, reward[1], state_2, done[1]) 
        agent.memory.add(state3, action3, reward[2], state_3, done[2]) 
        agent.memory.add(state4, action4, reward[3], state_4, done[3]) 

        agent.learn()

        state1 = state_1 
        state2 = state_2
        state3 = state_3
        state4 = state_4
        
        score1 += reward[0] 
        score2 += reward[1] 
        score3 += reward[2] 
        score4 += reward[3]
 
        if done[0] and done[1] and done[2] and done[3]: 
            score = max(score1, score2, score3, score4)
            if score > best_reward: 
                best_reward = score 
            average_reward += score
            
            snake_length = np.sum(np.array(backup_snake))-4 
            enemy1_length = np.sum(np.array(backup_enemy1))-4 
            enemy2_length = np.sum(np.array(backup_enemy2))-4 
            enemy3_length = np.sum(np.array(backup_enemy3))-4 
            length = max(snake_length, enemy1_length, enemy2_length, enemy3_length)
            if length > best_length: 
                best_length = length 
            average_length += length 
            
            print("Episode {} Avg Reward {} Avg Length {} Top Reward {} Top Length {} Last Reward {} Last Length {} Explore {}"
                  .format(i, round(average_reward/i, 3), round(average_length/i, 3), best_reward, best_length, score, length, agent.returning_epsilon())) 
            break 
 
        backup_snake = snake 
        backup_enemy1 = enemy1 
        backup_enemy2 = enemy2 
        backup_enemy3 = enemy3 
 
        episode_number.append(i) 
        average_reward_number.append(average_reward/i) 
        average_length_number.append(average_length/i) 

    if i % SAVE_FREQUENCY == 0:
        # Save the model every 500 episodes
        save_model(agent.network, f"wip/wip_model_{i}.pth")

# Save the final model after training 
save_model(agent.network, "final/final_model.pth") 
 
#plt.plot(episode_number, average_reward_number, label='Average Reward') 
#plt.plot(episode_number, average_length_number, label='Average Length') 
#plt.xlabel('Episode Number')
#plt.ylabel('Value')
#plt.legend()
#plt.show()