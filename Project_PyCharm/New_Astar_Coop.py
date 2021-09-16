import oneshotmaps
import random
import numpy as np
import matplotlib.pyplot as plt

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
        self.path = path
        self.replan = True
        self.grab = True
    
    def moveOneStep(self):
        self.current = self.path.pop(0)

        if self.path == []:
            self.replan = True

            # if it is at the left most column (unloading)
            # then reset states to grabbing
            if self.current[1] == 0:
                self.grab = True
            else:   # it should be delivering if that was false
                self.grab = False

def generateRandomGoal(maze, occupied_goal):
    while True:
        x = random.randint(1,maze.shape[0]-1)
        y = random.randint(3,maze.shape[1]-3)

        # maze = 1, then it is not occupied by obstacle
        if maze[x,y]==1 and (x,y) not in occupied_goal:
            return (x,y)

def getMinDistDropOff(current, locations):
    minDist = abs(current[0]-locations[0][0]) + abs(current[1]-locations[0][1])
    dropOff = locations[0]

    # locations is list all of the delivery drop point
    for loc in locations:
        dist = abs(current[0]-loc[0]) + abs(current[1]-loc[1])
        if dist < minDist:
            minDist = dist
            dropOff = loc
    
    return dropOff

def validataPath(current_node, next_position, agents):
    for agent in agents:
        path = agent.path

        if len(path) == 1 and path[0] == next_position:
            return False
            
        if current_node.g+1 < len(path):
            if path[current_node.g+1] == next_position:
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
            pos = current_node.position
            path = [pos,pos,pos,pos,pos]

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

    return [start]

def main():
    random.seed(42)
    maze = oneshotmaps.map0
    
    plt.ion()
    plt.imshow(maze, cmap='hot', interpolation='nearest')

    posx = [0,0,0,0,0,0,0,0,0,0,0,0]
    posy = [0,0,0,0,0,0,0,0,0,0,0,0]

    drop_x = [4,5,8,9,12,13,16,17,20,21,24,25]
    drop_y = [0,0,0,0,0,0,0,0,0,0,0,0]

    pick_x = [0,0,0,0,0,0,0,0,0,0,0,0]
    pick_y = [0,0,0,0,0,0,0,0,0,0,0,0]

    dropOff,  = plt.plot(drop_y, drop_x, '*', markersize = 10, color = 'red')
    pickUp, = plt.plot(pick_y, pick_x, '*', markersize = 10, color = 'brown')
    position,  = plt.plot(posy, posx, 'o', markersize = 8, color = 'blue')

    locations = [(4,0),(5,0),(8,0),(9,0),(12,0),(13,0),(16,0),(17,0),(20,0),(21,0),(24,0),(25,0)]

    agents = []
    free_dropoff = []
    occupied_goal = []

    for loc in locations:
        agent = Agent(current=loc)
        agents.append(agent)

    iteration = 0
    collision = 0

    while True:
        iteration += 1
        
        for index, agent in enumerate(agents):
            if agent.replan:
                if agent.grab:
                    goal = generateRandomGoal(maze, occupied_goal)
                    free_dropoff.append(agent.current)
                    occupied_goal.append(goal)
                else:
                    goal = getMinDistDropOff(agent.current, free_dropoff)
                    free_dropoff.remove(goal)   # this is list of dropoff that no agent is going
                    occupied_goal.remove(agent.current) # this is list of white cells that cannot be used as goals

                pick_x[index] = goal[0]
                pick_y[index] = goal[1]
                
                path = astar(maze, agent.current, goal, agents)
                agent.path = path
                agent.replan = False

        for index, agent in enumerate(agents):
            agent.moveOneStep()
            
            posx[index] = agent.current[0]
            posy[index] = agent.current[1]

        for i in range(12):
            for j in range(i+1,12):
                if agents[i].current == agents[j].current and i!=j:
                    collision += 1
                    print(iteration, collision)

        position.set_data(posy, posx)
        pickUp.set_data(pick_y,pick_x)

        plt.draw()
        plt.pause(0.2)

if __name__ == '__main__':
    main()