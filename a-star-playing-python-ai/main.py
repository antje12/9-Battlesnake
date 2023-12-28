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
    food = game_state['board']['food']
    hazards = game_state['board']['hazards']
    opponents = game_state['board']['snakes']

    goal = getClosestGoal(my_head, food)

    dangers = set()
    for oponent in opponents:
        for part in oponent["body"]:
            dangers.add(f"{part['x']}:{part['y']}")
    for hazard in hazards:
        dangers.add(f"{hazard['x']}:{hazard['y']}")

    result = aStarSearch(board, my_head, goal, dangers)
    if result.x < my_head["x"]: next_move = "left"
    elif result.x > my_head["x"]: next_move = "right"
    elif result.y < my_head["y"]: next_move = "down"
    elif result.y > my_head["y"]: next_move = "up"

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def getClosestGoal(me, goals):
    best = None
    bestDistance = None
    for goal in goals:
        distance = h(me["x"], me["y"], goal["x"], goal["y"])
        if best is None or distance < bestDistance:
            best = goal
            bestDistance = distance
    return best

# A* code
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
            return node.path()[-2]

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

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
