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
 
# inspired by https://andrew-gordienko.medium.com/reinforcement-learning-dqn-w-pytorch-7c6faad3d1e 
 
env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=4) 
observation_space = env.observation_space.shape 
print("Observation space:") 
print(observation_space) 
action_space = env.action_space[0].n 
print("Action space:") 
print(action_space) 
print("--------------------") 
 
EPISODES = 1000 
LEARNING_RATE = 0.001 
 
MEM_SIZE = 1000 
BATCH_SIZE = 200 
GAMMA = 0.90 
 
EXPLORATION_MAX = 1.00 
EXPLORATION_DECAY = 0.995 
EXPLORATION_MIN = 0.001 
 
HIDDEN_LAYER1_DIMS = 64 
HIDDEN_LAYER2_DIMS = 200 
HIDDEN_LAYER3_DIMS = 64 
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
best_reward = 0 
average_reward = 0 
episode_number = [] 
average_reward_number = [] 
 
class Network(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
        input_size = 11*11*3 #np.prod(observation_space) 
        self.fc1 = nn.Linear(input_size, HIDDEN_LAYER1_DIMS) 
        self.fc2 = nn.Linear(HIDDEN_LAYER1_DIMS, HIDDEN_LAYER2_DIMS) 
        self.fc3 = nn.Linear(HIDDEN_LAYER2_DIMS, HIDDEN_LAYER3_DIMS) 
        self.fc4 = nn.Linear(HIDDEN_LAYER3_DIMS, action_space) 
 
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE) 
        self.loss = nn.MSELoss() 
        self.to(DEVICE) 
     
    def forward(self, x): 
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x)) 
        x = F.relu(self.fc3(x)) 
        x = self.fc4(x) 
 
        return x 
 
class ReplayBuffer: 
    def __init__(self): 
        self.mem_count = 0 
         
        input_size = 11*11*3 #np.prod(observation_space) 
        self.states = np.zeros((MEM_SIZE, input_size),dtype=np.float32) 
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64) 
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32) 
        self.states_ = np.zeros((MEM_SIZE, input_size),dtype=np.float32) 
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
            #return random.randint(0, 0) 
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
        danger = np.logical_or(np.logical_or(enemy1, enemy2), enemy3).astype(int) 
        my_state = np.concatenate([ 
            me.flatten(), 
            food.flatten(), 
            danger.flatten()]) 
        return my_state 
 
agent = DQN_Solver() 
evil_agent1 = DQN_Solver() 
evil_agent2 = DQN_Solver() 
evil_agent3 = DQN_Solver() 
 
len_sum = 0 
 
for i in range(1, EPISODES+1): 
    state, reward, done, info = env.reset() 
    food = state[:, :, 0] # 1 = food 
     
    snake1 = state[:, :, 1] # 5 = head, 1 = body, is zeroed out if dead 
    snake2 = state[:, :, 2] 
    snake3 = state[:, :, 3] 
    snake4 = state[:, :, 4] 
 
    state1 = extract_state(snake1, food, snake2, snake3, snake4) 
    state2 = extract_state(snake2, food, snake3, snake4, snake1) 
    state3 = extract_state(snake3, food, snake4, snake1, snake2) 
    state4 = extract_state(snake4, food, snake1, snake2, snake3) 
 
    score1 = 0 
    score2 = 0 
    score3 = 0 
    score4 = 0 
 
    while True: 
        #env.render("ascii") 
 
        action1 = agent.choose_action(state1) 
        action2 = evil_agent1.choose_action(state2) 
        action3 = evil_agent2.choose_action(state3) 
        action4 = evil_agent3.choose_action(state4) 
 
        state_, rewards, dones, info = env.step([action1, action2, action3, action4]) 
        food = state_[:, :, 0] 
 
        snake1 = state_[:, :, 1] 
        snake2 = state_[:, :, 2] 
        snake3 = state_[:, :, 3] 
        snake4 = state_[:, :, 4] 
 
        #max_number = np.sum(snake)-5 
        #print("Max score:", max_number) 
        state_1 = extract_state(snake1, food, snake2, snake3, snake4) 
        state_2 = extract_state(snake2, food, snake3, snake4, snake1) 
        state_3 = extract_state(snake3, food, snake4, snake1, snake2) 
        state_4 = extract_state(snake4, food, snake1, snake2, snake3) 
         
        agent.memory.add(state1, action1, rewards[0], state_1, dones[0]) 
        evil_agent1.memory.add(state2, action2, rewards[1], state_2, dones[1]) 
        evil_agent2.memory.add(state3, action3, rewards[2], state_3, dones[2]) 
        evil_agent3.memory.add(state4, action4, rewards[3], state_4, dones[3]) 
 
        agent.learn() 
        state = state_ 
        score1 += rewards[0] 
        score2 += rewards[1] 
        score3 += rewards[2] 
        score4 += rewards[3] 
 
        if dones[0]: 
            score = score1 
            if score > best_reward: 
                best_reward = score 
            average_reward += score 
            #print ("Score: {}".format(max)) 
            lengths = info["snake_max_len"] 
            len_sum += lengths[0] 
            #print("Avg. length: ", len_sum/(i+1)) 
            print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}" 
                  .format(i, average_reward/i, best_reward, score, agent.returning_epsilon())) 
            break 
             
        episode_number.append(i) 
        average_reward_number.append(average_reward/i) 
 
print("Avg. length: ", len_sum/EPISODES) 
# Save the final model after training 
save_model(agent.network, "final_model.pth") 
 
plt.plot(episode_number, average_reward_number) 
plt.show()