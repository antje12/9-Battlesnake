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
        "author": "antje12",  # TODO: Your Battlesnake Username
        "color": "#A4D389",  # TODO: Choose color
        "head": "evil",  # TODO: Choose head
        "tail": "bolt",  # TODO: Choose tail
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

    is_move_safe = {"up": True, "down": True, "left": True, "right": True}

    # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

    if my_neck["x"] < my_head["x"]: is_move_safe["left"] = False
    elif my_neck["x"] > my_head["x"]: is_move_safe["right"] = False
    elif my_neck["y"] < my_head["y"]: is_move_safe["down"] = False
    elif my_neck["y"] > my_head["y"]: is_move_safe["up"] = False

    # TODO: Step 1 - Prevent your Battlesnake from moving out of bounds
    board_width = game_state['board']['width'] - 1
    board_height = game_state['board']['height'] - 1

    if (my_head["x"] == 0): is_move_safe["left"] = False
    if (my_head["x"] == board_width): is_move_safe["right"] = False
    if (my_head["y"] == 0): is_move_safe["down"] = False
    if (my_head["y"] == board_height): is_move_safe["up"] = False

    # TODO: Step 2 - Prevent your Battlesnake from colliding with itself
    my_body = game_state['you']['body']
    for n in my_body:
        if (n["x"]+1 == my_head["x"] and n["y"] == my_head["y"]): is_move_safe["left"] = False
        if (n["x"]-1 == my_head["x"] and n["y"] == my_head["y"]): is_move_safe["right"] = False
        if (n["y"]+1 == my_head["y"] and n["x"] == my_head["x"]): is_move_safe["down"] = False
        if (n["y"]-1 == my_head["y"] and n["x"] == my_head["x"]): is_move_safe["up"] = False

    # TODO: Step 3 - Prevent your Battlesnake from colliding with other Battlesnakes
    opponents = game_state['board']['snakes']

    for m in opponents:
        their_body = m["body"]
        for n in their_body:
            if (n["x"]+1 == my_head["x"] and n["y"] == my_head["y"]): is_move_safe["left"] = False
            if (n["x"]-1 == my_head["x"] and n["y"] == my_head["y"]): is_move_safe["right"] = False
            if (n["y"]+1 == my_head["y"] and n["x"] == my_head["x"]): is_move_safe["down"] = False
            if (n["y"]-1 == my_head["y"] and n["x"] == my_head["x"]): is_move_safe["up"] = False

    # Are there any safe moves left?
    safe_moves = []
    for move, isSafe in is_move_safe.items():
        if isSafe:
            safe_moves.append(move)

    if len(safe_moves) == 0:
        print(f"MOVE {game_state['turn']}: No safe moves detected! Moving down")
        return {"move": "down"}

    # Choose a random move from the safe ones
    next_move = random.choice(safe_moves)

    # TODO: Step 4 - Move towards food instead of random, to regain health and survive longer
    food = game_state['board']['food']
    hazards = game_state['board']['hazards']

    goal = getClosestGoal(my_head, food)

    dangers = set()
    for oponent in opponents:
        for part in oponent["body"]:
            dangers.add(f"{part['x']}:{part['y']}")
    for hazard in hazards:
        dangers.add(f"{hazard['x']}:{hazard['y']}")

    result = aStarSearch(my_head, goal, dangers)
    print("Go to --------------------------------")
    print(result.key())
    print("--------------------------------------")

    if result.x < my_head["x"]: next_move = "left"
    elif result.x > my_head["x"]: next_move = "right"
    elif result.y < my_head["y"]: next_move = "down"
    elif result.y > my_head["y"]: next_move = "up"

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}

def getClosestGoal(me, goals):
    best = None
    bestScore = None
    for goal in goals:
        if best is None or score(me, goal) < bestScore:
            best = goal
            bestScore = score(me, goal)
    return best

def score(n, goal): # heuristic cost
    dx = n["x"] - goal["x"] # x distance to target
    dy = n["y"] - goal["y"] # y distance to target
    distance = sqrt(dx * dx + dy * dy) # a^2 + b^2 = c^2 # c = sqrt(a^2 + b^2)
    return distance

'''
A* code
'''
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

def aStarSearch(me, goal, dangers):
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

        children = expandNode(node, goal_node, dangers, fringe, visited)
        for child in children:
            fringe[child.key()] = child

def expandNode(node, goal, dangers, fringe, visited):
    successors = []
    children = getChildren(node, dangers)

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

def getChildren(node, dangers):  # Lookup list of successor states
    children = []
    addIfValid(children, Node(node.x+1, node.y, node), dangers)
    #children.append(Node(node.x+1, node.y, node))
    addIfValid(children, Node(node.x-1, node.y, node), dangers)
    addIfValid(children, Node(node.x, node.y+1, node), dangers)
    addIfValid(children, Node(node.x, node.y-1, node), dangers)
    print("Children-----------")
    for child in children:
        print(child.key())
    print("-------------------")
    return children  # successor_fn( 'C' ) returns ['F', 'G']

def addIfValid(children, node, dangers):
    if node.x < 0:
        return
    elif node.x > 10:
        return
    elif node.y < 0:
        return
    elif node.y > 10:
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
    return g(n) + 5 * h(n, goal)

def g(n):
    return n.cost # travel cost

def h(n, goal): # heuristic cost
    dx = n.x - goal.x # x distance to target
    dy = n.y - goal.y # y distance to target
    distance = sqrt(dx * dx + dy * dy) # a^2 + b^2 = c^2 # c = sqrt(a^2 + b^2)
    return distance

# Start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
