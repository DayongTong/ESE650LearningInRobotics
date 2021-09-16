import random
import numpy as np
import matplotlib.pyplot as plt
from maps import map1
from time import sleep

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
        self.grab = True
        self.path = path
        self.totalSteps = 0

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

def getMinDistDropOff(current, locations):
    minDist = abs(current[0]-locations[0][0]) + abs(current[1]-locations[0][1])
    dropOff = locations[0]

    for loc in locations:
        dist = abs(current[0]-loc[0]) + abs(current[1]-loc[1])
        if dist < minDist:
            minDist = dist
            dropOff = loc
    
    return dropOff

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

    maze = map1.maze
    numAgents = map1.numAgents
    locations = map1.locations

    pos_x, pos_y = map1.posx, map1.posy
    drop_x, drop_y = map1.dropx, map1.dropy
    pick_x, pick_y = map1.pickx, map1.picky   

    plt.ion()
    plt.imshow(maze, cmap='hot', interpolation='nearest')

    dropOff,  = plt.plot(drop_y, drop_x, '*', markersize = 10, color = 'red')
    pickUp, = plt.plot(pick_y, pick_x, '*', markersize = 10, color = 'brown')
    position,  = plt.plot(pos_y, pos_x, 'o', markersize = 8, color = 'blue')

    plt.draw()
    plt.pause(2)

    agents = []

    for loc in locations:
        agent = Agent(current=loc)
        agents.append(agent)

    reward = 0
    reward_list = np.zeros((20000,))

    for step in range(20000):

        reward_list[step] = (reward/numAgents)

        if step%100 == 0:
            print(step)

        for index, agent in enumerate(agents):
            if agent.replan:
                if agent.grab:
                    goal = generateRandomGoal(maze, agent.current)
                    agent.current = getFreeNeighbour(maze, agent.current, agents)
                    path = astar(maze, agent.current, goal, agents)

                    if path == None:
                        agent.path = [agent.current]
                        agent.grab = True
                    else:
                        reward += 1
                        agent.path = path
                        agent.grab = False
                else:
                    goal = getMinDistDropOff(agent.current, locations)
                    agent.current = getFreeNeighbour(maze, agent.current, agents)
                    path = astar(maze, agent.current, goal, agents)

                    if path == None:
                        agent.path = [agent.current]
                        agent.grab = False
                    else:
                        agent.path = path
                        agent.grab = True

                pick_x[index] = goal[0]
                pick_y[index] = goal[1]

                agent.replan = False

        for index, agent in enumerate(agents):
            agent.moveOneStep()
            
            pos_x[index] = agent.current[0]
            pos_y[index] = agent.current[1]

        position.set_data(pos_y,pos_x)
        pickUp.set_data(pick_y,pick_x)

        plt.draw()
        plt.pause(0.1)

    np.save('map1_4.npy', reward_list)

    plt.plot(reward_list)
    plt.show()

if __name__ == '__main__':
    main()