class Node:  # Node has only PARENT_NODE, STATE, DEPTH
    def __init__(self, x, y, parent=None, depth=0, cost=0, f=0):
        self.X = x
        self.Y = y

        self.PARENT_NODE = parent

        self.DEPTH = depth
        self.COST = cost
        self.F = f

    def key(self):  # Create a list of nodes from the root to this node.
        return "" + self.x + ":" + self.y

    def path(self):  # Create a list of nodes from the root to this node.
        current_node = self
        path = [self]
        while current_node.PARENT_NODE:  # while current node has parent
            current_node = current_node.PARENT_NODE  # make parent the current node
            path.append(current_node)   # add current node to path
        return path
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

'''
Search the tree for the goal state and return path from initial state to goal state
'''
def A_STAR_SEARCH(xF, yF, xT, yT):
    fringe = []
    visited = []

    initial_node = Node(xF, yF)
    goal_node = Node(xT, yT)

    fringe = INSERT(initial_node, fringe)
    while fringe is not None:
        node = REMOVE_CHEAPEST_NODE(fringe)
        visited = INSERT(node, fringe)

        if node == goal_node:
            return node.path()
        
        children = EXPAND(node, goal_node, fringe, visited)
        fringe = INSERT_ALL(children, fringe)

'''
Expands node and gets the successors (children) of that node.
Return list of the successor nodes.
'''
def EXPAND(node, goal, fringe, visited):
    successors = []
    children = getChildren(node)

    for child in children:

        if child in visited: 
            continue

        child.PARENT_NODE = node
        child.DEPTH = node.DEPTH + 1
        child.COST = node.COST + 1
        child.F = F(child, goal)

        if child in visited and child.COST > fringe.get(s.key()).cost): 
            continue

        successors = INSERT(child, successors)
    return successors

'''
Successor function, mapping the nodes to its successors
'''
def getChildren(node):  # Lookup list of successor states
    children = []
    children = INSERT(Node(node.x+1, node.y, node), children)
    children = INSERT(Node(node.x-1, node.y, node), children)
    children = INSERT(Node(node.x, node.y+1, node), children)
    children = INSERT(Node(node.x, node.y-1, node), children)
    return children  # successor_fn( 'C' ) returns ['F', 'G']

'''
Insert node in to the queue (fringe).
'''
def INSERT(node, queue):
    queue.append(node)
    return queue

'''
Insert list of nodes into the fringe
'''
def INSERT_ALL(list, queue):
    queue.extend(list)
    return queue

'''
Removes and returns the cheapest element from fringe
'''
def REMOVE_CHEAPEST_NODE(queue):
    # Find the cheapest node! f(n) = g(n) + h(n)
    cheapest = None
    for n in queue:
        if cheapest == None:
            cheapest = n
        elif (f(n) < f(cheapest)):
            cheapest = n
    queue.remove(cheapest)
    return cheapest

def f(n):
    return g(n) + h(n)

def g(n):
    # travel cost
    return n.COST

def h(n):
    # heuristic cost
    return H_COST[n.STATE]

GOAL_STATES = {('A', 'Clean', 'Clean'), ('B', 'Clean', 'Clean')}
# The heuristic function is the amount of dirt left
H_COST = {('A', 'Dirty', 'Dirty'): 3,
          ('A', 'Clean', 'Dirty'): 2,
          ('A', 'Dirty', 'Clean'): 1,
          ('A', 'Clean', 'Clean'): 0,
          ('B', 'Dirty', 'Dirty'): 3,
          ('B', 'Dirty', 'Clean'): 2,
          ('B', 'Clean', 'Dirty'): 1,
          ('B', 'Clean', 'Clean'): 0, }
# You can only travel 1 space, at the cost of 1
TRAVEL_COST = 1
