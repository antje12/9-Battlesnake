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

import numpy as np  
import matplotlib  
matplotlib.use('TkAgg')  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  

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
    print("GAME OVER")

# DQN code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Game board
features = 3
height = 11
width = 11
observation_space = (features, height, width) 
action_space = 4 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# Hyperparameter
LEARNING_RATE = 0.0005 
HIDDEN_LAYER1_DIMS = 32
HIDDEN_LAYER2_DIMS = 64
HIDDEN_LAYER3_DIMS = 256
HIDDEN_LAYER4_DIMS = 128

# Agent Code
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

class DQN_Solver: 
    def __init__(self, model): 
        self.network = model
 
    def choose_action(self, observation):          
        state = torch.tensor(observation).float().detach() 
        state = state.to(DEVICE) 
        state = state.unsqueeze(0) 
        q_values = self.network(state) 
        return torch.argmax(q_values).item() 

def load_model(filepath):  
    model = Network()
    model.load_state_dict(torch.load(filepath))  
    print(f"Model loaded from {filepath}")  
    return model

agent = DQN_Solver(load_model("final_model.pth"))
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DQN code

# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
# See https://docs.battlesnake.com/api/example-move for available data
def move(game_state: typing.Dict) -> typing.Dict:

    moves = ["up", "down", "left", "right"]
    next_move = random.choice(moves)

    #board = game_state['board']
    my_head = game_state["you"]["body"][0] # Coordinates of your head
    foods = game_state['board']['food']
    hazards = game_state['board']['hazards']
    opponents = game_state['board']['snakes']

    state = extract_state(my_head, foods, hazards, opponents)

    # 0 to 3 corresponding to Snake.UP, Snake.DOWN, Snake.LEFT, Snake.RIGHT 
    action = agent.choose_action(state)

    if action == 0: next_move = "up"
    elif action == 1: next_move = "down"
    elif action == 2: next_move = "left"
    elif action == 3: next_move = "right"

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def extract_state(my_head, foods, hazards, opponents):
    map_array = np.zeros((11, 11, 3), dtype=int)

    # Danger cells
    for oponent in opponents: # Includes me
        for part in oponent["body"]:
            x, y = part["x"], part["y"]
            map_array[x, y, 1] = 1
    for hazard in hazards:
        x, y = hazard["x"], hazard["y"]
        map_array[x, y, 1] = 1

    # Food cells
    for  food in foods:
        x, y = food["x"], food["y"]
        map_array[x, y, 2] = 1

    # Player cell
    x, y = my_head["x"], my_head["y"]
    map_array[x, y, 0] = 1 # Set head
    map_array[x, y, 1] = 0 # Clear dangers
    map_array[x, y, 2] = 0 # Clear food
            
    # Add batch dimension
    map_array = map_array.reshape(features, height, width)
    return map_array

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
    run_server({"info": info, "start": start, "move": move, "end": end})
