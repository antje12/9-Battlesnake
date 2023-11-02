
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
def aStarSearch(xF, yF, xT, yT):
    fringe = {}
    visited = {}

    initial_node = Node(xF, yF)
    goal_node = Node(xT, yT)

    fringe[initial_node.key] = initial_node

    while len(fringe) > 0:
        node = getCheapestNode(fringe)
        fringe.pop(node.key)
        visited[node.key] = node

        if node == goal_node:
            return node.path()
        
        children = expandNode(node, goal_node, fringe, visited)
        for child in children:
            fringe[child.key] = child

'''
Expands node and gets the successors (children) of that node.
Return list of the successor nodes.
'''
def expandNode(node, goal, fringe, visited):
    successors = []
    children = getChildren(node)

    for child in children:

        if child.key in visited: 
            continue

        child.PARENT_NODE = node
        child.DEPTH = node.DEPTH + 1
        child.COST = node.COST + 1
        child.F = f(child, goal)

        if child.key in fringe and child.COST > fringe[child.key()].cost: 
            continue

        successors.append(child, successors)
    return successors

'''
Successor function, mapping the nodes to its successors
'''
def getChildren(node):  # Lookup list of successor states
    children = []
    children.append(Node(node.x+1, node.y, node), children)
    children.append(Node(node.x-1, node.y, node), children)
    children.append(Node(node.x, node.y+1, node), children)
    children.append(Node(node.x, node.y-1, node), children)
    return children  # successor_fn( 'C' ) returns ['F', 'G']

'''
Removes and returns the cheapest element from fringe
'''
def getCheapestNode(fringe):
    # Find the cheapest node! f(n) = g(n) + h(n)
    cheapest = None
    for n in fringe:
        if cheapest == None:
            cheapest = n
        elif (n.F < cheapest.F):
            cheapest = n
    return cheapest

def f(n, goal):
    return g(n) + 0 + h(n, goal)

def g(n):
    # travel cost
    return n.COST

def h(n, goal):
    # heuristic cost
    # x distance to target
    dx = n.x - goal.x
    # y distance to target
    dy = n.y - goal.y

    # a^2 + b^2 = c^2
    # c = sqrt(a^2 + b^2)
    distance = sqrt(dx * dx + dy * dy)
    return distance
