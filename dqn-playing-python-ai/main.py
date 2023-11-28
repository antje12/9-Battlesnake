# Welcome to
# __________         __    __  .__                               __
# \______   \_____ _/  |__/  |_|  |   ____   ______ ____ _____  |  | __ ____
#  |    |  _/\__  \\   __\   __\  | _/ __ \ /  ___//    \\__  \ |  |/ // __ \
#  |    |   \ / __ \|  |  |  | |  |_\  ___/ \___ \|   |  \/ __ \|    <\  ___/
#  |________/(______/__|  |__| |____/\_____>______>___|__(______/__|__\\_____>
#
# This file can be a nice home for your Battlesnake logic and helper functions.
#
# To get you started we've included code to prevent your Battlesnake from moving backwards.
# For more info see docs.battlesnake.com

import random
import typing
from math import sqrt

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

env = BattlesnakeGym(map_size=(11, 11), number_of_snakes=1)
observation_space = env.observation_space.shape
action_space = env.action_space[0].n
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPISODES = 1000
LEARNING_RATE = 0.0005

MEM_SIZE = 20000
BATCH_SIZE = 128
GAMMA = 0.98

EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.85
EXPLORATION_MIN = 0.001

HIDDEN_LAYER1_DIMS = 256
HIDDEN_LAYER2_DIMS = 128

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_size = np.prod(observation_space)
        self.fc1 = nn.Linear(input_size, HIDDEN_LAYER1_DIMS)
        self.fc2 = nn.Linear(HIDDEN_LAYER1_DIMS, HIDDEN_LAYER2_DIMS)
        self.fc3 = nn.Linear(HIDDEN_LAYER2_DIMS, action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        input_size = np.prod(observation_space)
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
    def __init__(self, model):
        self.memory = ReplayBuffer()
        self.exploration_rate = 0
        self.network = model

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

agent = DQN_Solver(load_model("single_261.pth"))

# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
# TIP: If you open your Battlesnake URL in a browser you should see this data
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "antje12", # TODO: Your Battlesnake Username
        "color": "#A4D389", # TODO: Choose color
        "head": "evil", # TODO: Choose head
        "tail": "bolt", # TODO: Choose tail
    }

# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")

# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    moves = ["up", "down", "left", "right"]
    next_move = random.choice(moves)

    board = game_state['board']
    my_head = game_state["you"]["body"][0] # Coordinates of your head
    foods = game_state['board']['food']
    hazards = game_state['board']['hazards']
    opponents = game_state['board']['snakes']

    map = np.zeros((11, 11), dtype=int)
    for oponent in opponents:
        for part in oponent["body"]:
            x, y = part["x"], part["y"]
            map[x, y] = 1
    for hazard in hazards:
        x, y = hazard["x"], hazard["y"]
        map[x, y] = 1
    
    for food in foods:
        x, y = food["x"], food["y"]
        map[x, y] = 2
    
    x, y = my_head["x"], my_head["y"]
    map[x, y] = 3

    # 0 to 3 corresponding to Snake.UP, Snake.DOWN, Snake.LEFT, Snake.RIGHT 
    action = agent.choose_action(state)

    if action == 0: next_move = "up"
    elif action == 1: next_move = "down"
    elif action == 2: next_move = "left"
    elif action == 3: next_move = "right"

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
