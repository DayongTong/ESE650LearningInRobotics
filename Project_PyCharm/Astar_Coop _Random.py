import random
import numpy as np
import matplotlib.pyplot as plt
from maps import map3

class Node():
    """A node class for A* Pathfinding"""
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

class Agent():
    def __init__(self, current=None, path=[]):
        self.current = current
        self.replan = True
        self.path = path

    def moveOneStep(self):
        self.current = self.path.pop(0)

        if self.path == []:
            self.replan = True

def generateRandomGoal(maze, current):
    while True:
        x = random.randint(1,maze.shape[0]-1)
        y = random.randint(3,maze.shape[1]-3)

        if maze[x,y]==1 and (x,y) != current:
            return (x,y)

def getFreeNeighbour(maze, current, agents):
    left = (current[0], current[1]-1)
    right = (current[0], current[1]+1)
    up = (current[0]-1, current[1])
    down = (current[0]+1, current[1])

    for pos in [left, right, up, down]:
        if pos[0]>=0 and pos[0]<maze.shape[0] and pos[1]>=0 and pos[1]<maze.shape[1]:
            valid = True
            for agent in agents:
                if agent.path != [] and agent.path[0] == pos:
                    valid = False
            if valid:
                return pos

    return current

def validataPath(current_node, next_position, agents):
    for agent in agents:
        path = agent.path

        if current_node.g+1 < len(path):
            if path[current_node.g+1] == next_position:
                return False
            if path[current_node.g] == next_position and path[current_node.g+1] == current_node.position:
                return False

    return True

def astar(maze, start, end, agents):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = []
    open_pos = []
    closed_pos = []

    # Add the start node
    open_list.append(start_node)
    open_pos.append(start)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        open_pos.pop(current_index)
        closed_pos.append(current_node.position)

        # Found the goal
        if current_node == end_node:
            path = []

            current = current_node
            while current is not None:
                path.append(current.position)                
                current = current.parent

            return path[::-1] # Return reversed path

        # # Generate children
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares
            
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > maze.shape[0]-1 or node_position[0] < 0 or node_position[1] > maze.shape[1]-1 or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] == 0:
                continue

            if not validataPath(current_node, node_position, agents):
                continue

           # Create new node
            child = Node(current_node, node_position)

            if node_position not in closed_pos:
                child = Node(current_node, node_position)

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                if node_position in open_pos:
                    index = open_pos.index(node_position)
                    if open_list[index].g > child.g:
                        open_list[index] = child

                # Add the child to the open list
                else:
                    open_list.append(child)
                    open_pos.append(node_position)

    return None


def main():
    random.seed(42)

    maze = map3.maze
    numAgents = map3.numAgents

    posx, posy = map3.posx, map3.posy
    goalx, goaly = posx.copy(), posy.copy()

    plt.ion()
    plt.imshow(maze, cmap='hot', interpolation='nearest')

    goals, = plt.plot(goaly, goalx, '*', markersize = 10, color = 'brown')
    position,  = plt.plot(posy, posx, 'o', markersize = 8, color = 'blue')

    plt.draw()
    plt.pause(2)

    agents = []

    for i in range(numAgents):
        agent = Agent(current=(posx[i],posy[i]))
        agents.append(agent)

    for step in range(20000):
        for index, agent in enumerate(agents):
            if agent.replan:

                goal = generateRandomGoal(maze, agent.current)
                agent.current = getFreeNeighbour(maze, agent.current, agents)
                path = astar(maze, agent.current, goal, agents)

                if path == None:
                    agent.path = [agent.current]
                else:
                    agent.path = path

                goalx[index] = goal[0]
                goaly[index] = goal[1]

                agent.replan = False

        for index, agent in enumerate(agents):
            agent.moveOneStep()
            
            posx[index] = agent.current[0]
            posy[index] = agent.current[1]

        position.set_data(posy,posx)
        goals.set_data(goaly,goalx)

        plt.draw()
        plt.pause(0.1)

if __name__ == '__main__':
    main()