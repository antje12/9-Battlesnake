import math  
import random  
import typing
from math import sqrt
  
import numpy as np  
import matplotlib  
matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt
  
from battlesnake_gym.snake_gym import BattlesnakeGym  
from battlesnake_gym.snake import Snake  
 
# taken from https://andrew-gordienko.medium.com/reinforcement-learning-dqn-w-pytorch-7c6faad3d1e 
 
# 11*11 (height * width) sized map with 3 values in each cell
features = 3
height = 11
width = 11
env = BattlesnakeGym(map_size=(height, width), number_of_snakes=4, food_spawn_locations=[(1,2)]) 
observation_space = (features, height, width) 
print("Observation space:", observation_space) 
action_space = 4 
print("Action space:", action_space) 

EPISODES = 1

# A* code start ====================================================
class Node:
    def __init__(self, x, y, parent=None, depth=0, cost=0, f=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.depth = depth
        self.cost = cost
        self.f = f

    def key(self):
        return f"{self.x}:{self.y}"

    def path(self):
        current_node = self
        path = [self]
        while current_node.parent:
            current_node = current_node.parent
            path.append(current_node)
        return path

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

def aStarSearch(board, me, goal, dangers):
    fringe = {}
    visited = {}

    initial_node = Node(me["x"], me["y"])
    goal_node = Node(goal["x"], goal["y"])

    fringe[initial_node.key()] = initial_node

    while len(fringe) > 0:
        node = getCheapestNode(fringe)
        fringe.pop(node.key())
        visited[node.key()] = node

        if node == goal_node:
            path_list = node.path()
            if len(path_list) >= 2:
                return path_list[-2]
            else:
                return path_list[-1]

        children = expandNode(board, node, goal_node, dangers, fringe, visited)
        for child in children:
            fringe[child.key()] = child
    
    # no path to food found!
    return survivalSearch(board, initial_node, dangers)

def expandNode(board, node, goal, dangers, fringe, visited):
    successors = []
    children = getChildren(board, node, dangers)

    for child in children:
        if child.key() in visited:
            continue

        child.parent = node
        child.depth = node.depth + 1
        child.cost = node.cost + 1
        child.f = f(child, goal)

        if child.key() in fringe and child.cost > fringe[child.key()].cost:
            continue

        successors.append(child)
    return successors

def getChildren(board, node, dangers): # Lookup list of successor states
    children = []
    addIfValid(board, children, Node(node.x+1, node.y), dangers)
    addIfValid(board, children, Node(node.x-1, node.y), dangers)
    addIfValid(board, children, Node(node.x, node.y+1), dangers)
    addIfValid(board, children, Node(node.x, node.y-1), dangers)
    return children 

def addIfValid(board, children, node, dangers):
    if node.x < 0 or board['width'] - 1 < node.x:
        return
    elif node.y < 0 or board['height'] - 1 < node.y:
        return
    elif node.key() in dangers:
        return
    else:
        children.append(node)

def getCheapestNode(fringe):
    cheapest = None
    for key in fringe:
        if cheapest is None or fringe[key].f < fringe[cheapest].f:
            cheapest = key
    return fringe[cheapest]

def f(n, goal):
    return g(n) + 5 * h(n.x, n.y, goal.x, goal.y)

def g(n):
    return n.cost # travel cost

def h(nodeX, nodeY, goalX, goalY): # heuristic cost
    dx = nodeX - goalX # x distance to target
    dy = nodeY - goalY # y distance to target
    distance = sqrt(dx * dx + dy * dy) # a^2 + b^2 = c^2 # c = sqrt(a^2 + b^2)
    return distance

# survivalSearch
def survivalSearch(board, me, dangers):
    children = getChildren(board, me, dangers)
    if len(children) > 0:
        return random.choice(children)
    return me
# A* code end   ====================================================

def getClosestGoal(me, goals):
    best = None
    bestDistance = None
    for goal in goals:
        distance = h(me["x"], me["y"], goal["x"], goal["y"])
        if best is None or distance < bestDistance:
            best = goal
            bestDistance = distance
    return best

def extract_move(me, food, enemy1, enemy2, enemy3):
    my_head = {"x": 0, "y": 0}
    goals = [ ]
    dangers = set()
    for x in range(11):
        for y in range(11):
            cell_me = me[x][y]
            if cell_me == 5:  # my head
                my_head["x"] = x
                my_head["y"] = y
            elif cell_me > 0:  # my body
                dangers.add(f"{x}:{y}")
            cell_enemy1 = enemy1[x][y]
            cell_enemy2 = enemy2[x][y]
            cell_enemy3 = enemy3[x][y]
            if cell_enemy1 > 0 or cell_enemy2 > 0 or cell_enemy3 > 0:  # enemy
                dangers.add(f"{x}:{y}")
            cell_food = food[x][y]
            if cell_food > 0:  # food
                goals.append({"x": x, "y": y})
    if goals:
        goal = getClosestGoal(my_head, goals)
        result = aStarSearch(board, my_head, goal, dangers)
    else:
        initial_node = Node(my_head["x"], my_head["y"])
        result = survivalSearch(board, initial_node, dangers)
    moves = ["up", "down", "left", "right"]
    if result.x < my_head["x"]: return 0
    elif result.x > my_head["x"]: return 1
    elif result.y < my_head["y"]: return 2
    elif result.y > my_head["y"]: return 3
    return 0
  
best_reward = 0 
best_length = 0 
average_reward = 0 
average_length = 0  
episode_number = [] 
average_reward_number = [] 
average_length_number = []

board = {
  "height": 11,
  "width": 11
}

for i in range(1, EPISODES+1): 
    state, reward, done, info = env.reset() 
    food = state[:, :, 0] 
    snake = state[:, :, 1] 
    enemy1 = state[:, :, 2] 
    enemy2 = state[:, :, 3] 
    enemy3 = state[:, :, 4] 

    score1 = 0 
    score2 = 0 
    score3 = 0 
    score4 = 0 
 
    backup_snake = snake 
    backup_enemy1 = enemy1 
    backup_enemy2 = enemy2 
    backup_enemy3 = enemy3 
 
    while True: 
        env.render("ascii") 
        action1 = extract_move(snake, food, enemy1, enemy2, enemy3)
        action2 = extract_move(enemy1, food, snake, enemy2, enemy3)
        action3 = extract_move(enemy2, food, enemy1, snake, enemy3)
        action4 = extract_move(enemy3, food, enemy1, enemy2, snake)

        state_, reward, done, info = env.step([action1, action2, action3, action4]) 
 
        food = state_[:, :, 0] 
        snake = state_[:, :, 1] 
        enemy1 = state_[:, :, 2] 
        enemy2 = state_[:, :, 3] 
        enemy3 = state_[:, :, 4] 
        
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
            
            print("Episode {} Avg Reward {} Avg Length {} Top Reward {} Top Length {} Last Reward {} Last Length {}"
                  .format(i, round(average_reward/i, 3), round(average_length/i, 3), best_reward, best_length, score, length)) 
            break 
 
        backup_snake = snake 
        backup_enemy1 = enemy1 
        backup_enemy2 = enemy2 
        backup_enemy3 = enemy3 
 
        episode_number.append(i) 
        average_reward_number.append(average_reward/i) 
        average_length_number.append(average_length/i) 
 
#plt.plot(episode_number, average_reward_number, label='Average Reward') 
#plt.plot(episode_number, average_length_number, label='Average Length') 
#plt.xlabel('Episode Number')
#plt.ylabel('Value')
#plt.legend()
#plt.show()