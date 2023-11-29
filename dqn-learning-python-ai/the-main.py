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
 
# 11*11 (height * width) sized map with 1 value in each cell
features = 1
height = 11
width = 11
env = BattlesnakeGym(map_size=(height, width), number_of_snakes=1) 
observation_space = (features, height, width) 
print("Observation space:", observation_space) 
action_space = 4 
print("Action space:", action_space) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Device:", DEVICE) 
print("--------------------") 

EPISODES = 1000
LEARNING_RATE = 0.001

MEM_SIZE = 2000
BATCH_SIZE = 32
GAMMA = 0.95

EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01

HIDDEN_LAYER1_DIMS = 16
HIDDEN_LAYER2_DIMS = 32

HIDDEN_LAYER3_DIMS = 256
HIDDEN_LAYER4_DIMS = 128
 
best_reward = 0 
average_reward = 0 
episode_number = [] 
average_reward_number = [] 
average_length_number = [] 
 
class Network(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        input_channels = features  # Number of channels in the input (e.g., features for each cell)
        
        # 2D array processing (spatial local 3*3 info processing)
        self.conv1 = nn.Conv2d(input_channels, HIDDEN_LAYER1_DIMS, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(HIDDEN_LAYER1_DIMS, HIDDEN_LAYER2_DIMS, kernel_size=3, stride=1, padding=1)
        
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
        
        self.states = np.zeros((MEM_SIZE, *observation_space),dtype=np.float32) 
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64) 
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32) 
        self.states_ = np.zeros((MEM_SIZE, *observation_space),dtype=np.float32) 
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
 
def extract_state(me, food):
    map_array = np.zeros((11, 11), dtype=int)  # 2D array
    for x in range(11):
        for y in range(11):
            cell_me = me[x][y]
            if cell_me == 5:  # my head
                map_array[x, y] = 3 / 3  # Assigning value 3 for the player's head
            elif cell_me > 0:  # my body
                map_array[x, y] = 1 / 3  # Assigning value 1 for the player's body
            cell_food = food[x][y]
            if cell_food > 0:  # food
                map_array[x, y] = 2 / 3  # Assigning value 2 for food
    # Add batch dimension
    map_array = map_array.reshape(1, 11, 11)
    return map_array
 
agent = DQN_Solver() 
  
len_sum = 0  
 
for i in range(1, EPISODES+1): 
    state, reward, done, info = env.reset() 
    food = state[:, :, 0] 
    snake = state[:, :, 1] 
    state = extract_state(snake, food)
    score = 0 
 
    backup_snake = snake 
 
    while True: 
        #env.render("ascii") 
        action = agent.choose_action(state) 
        state_, reward, done, info = env.step([action]) 
 
        food = state_[:, :, 0] 
        snake = state_[:, :, 1] 
        #max_number = np.sum(snake)-5 
        #print("Max score:", max_number) 
        state_ = extract_state(snake, food)
        agent.memory.add(state, action, reward[0], state_, done[0]) 
        agent.learn() 
        state = state_ 
        score += reward[0] 
 
        if done[0]: 
            if score > best_reward: 
                best_reward = score 
            average_reward += score 
            #print ("Score: {}".format(max)) 
            np_snake = np.array(backup_snake) 
            snake_length = np.sum(np_snake)-4 
            len_sum += snake_length 
            #print("Avg. length: ", len_sum/(i+1)) 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon())) 
            break 
 
        backup_snake = snake 
 
        episode_number.append(i) 
        average_reward_number.append(average_reward/i) 
        average_length_number.append(len_sum/i) 
 
print("Avg. length: ", len_sum/EPISODES) 
# Save the final model after training 
save_model(agent.network, "final_model.pth") 
 
plt.plot(episode_number, average_reward_number, label='Average Reward') 
plt.plot(episode_number, average_length_number, label='Average Length') 
plt.xlabel('Episode Number')
plt.ylabel('Value')
plt.legend()
plt.show()