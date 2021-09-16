import copy

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


def astar(maze, start, end, solution):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

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
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            thiscost = copy.deepcopy(current_node.g)
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1], thiscost # Return reversed path, shape is (n, 2)

        # Generate children
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > maze.shape[0]-1 or node_position[0] < 0 or node_position[1] > maze.shape[1]-1 or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] == 0:
                continue

            valid = True
            for path in solution:
                if current_node.g + 1 < len(path):
                    if path[current_node.g + 1] == node_position:
                        valid = False
                    if path[current_node.g+1] == current_node.position and node_position == path[current_node.g]:
                        valid = False

            if not valid:
                continue

            # Create new node
            child = Node(current_node, node_position)

            add_child = True

            for closed_child in closed_list:
                if child == closed_child:
                    add_child = False

            if add_child:

                # Create the f, g, and h values
                child.g = current_node.g + 1
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                child.f = child.g + child.h

                # Child is already in the open list
                in_list = False
                for open_node in open_list:
                    if child == open_node and child.g < open_node.g:
                        open_node = child
                        in_list = True
                        break

                # Add the child to the open list
                if not in_list:
                    open_list.append(child)


def main():

    maze = np.full((30,51), 1)

    for row in range(30):
        for col in range(50):
            if (row%4 == 2 or row%4 == 3) and (col%12 != 0 and col%12 != 1):
            # if (row%4 == 2 or row%4 == 3 or row%4 == 1) and (col%12 != 0 and col%12 != 1):
                maze[row, col] = 0

    plt.ion()
    plt.imshow(maze, cmap='hot', interpolation='nearest')

    start_agents = [(0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),
                    (9,0),(10,0),(11,0),(12,0),(13,0),(14,0),(15,0),(16,0),
                    (0,49),(1,49),(2,49),(3,49),(4,49),(5,49),(6,49),(7,49),(8,49),
                    (9,49),(10,49),(11,49),(12,49),(13,49),(14,49),(15,49),(16,49),(17,49),(18,49)]

    end_agents = [(0,49),(1,49),(2,49),(3,49),(4,49),(5,49),(6,49),(7,49),(8,49),
                  (9,49),(10,49),(11,49),(12,49),(13,49),(14,49),(15,49),(16,49),(17,49),(18,49),
                  (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),
                  (9,0),(10,0),(11,0),(12,0),(13,0),(14,0),(15,0),(16,0)]

    start_agents = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                    (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0),
                    (0, 50), (1, 50), (2, 50), (3, 50), (4, 50), (5, 50), (6, 50), (7, 50), (8, 50),
                    (9, 50), (10, 50), (11, 50), (12, 50), (13, 50), (14, 50), (15, 50), (16, 50), (17, 50), (18, 50)]

    end_agents = [(0, 50), (1, 50), (2, 50), (3, 50), (4, 50), (5, 50), (6, 50), (7, 50), (8, 50),
                  (9, 50), (10, 50), (11, 50), (12, 50), (13, 50), (14, 50), (15, 50), (16, 50), (17, 50), (18, 50),
                  (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                  (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0)]

    solution = []

    total_cost = 0
    for num in range(len(start_agents)):
        start = start_agents[num]
        end = end_agents[num]
        path, tempcost = astar(maze, start, end, solution)
        solution.append(path)
        total_cost += tempcost

        for step in range (len(path)-1):
            current = path[step]
            nxt = path[step+1]

            x_dist = (nxt[0] - current[0]) * 0.3
            y_dist = (nxt[1] - current[1]) * 0.3

            # plt.arrow(current[1], current[0], y_dist, x_dist, head_width=0.6, head_length=0.3, fc='b', ec='b')
            plt.arrow(current[1], current[0], y_dist, x_dist, head_width=0.3, head_length=0.2, fc='b', ec='b')

    pos_x = []
    pos_y = []

    for num in range(len(start_agents)):
        pos_x.append(start_agents[num][0])
        pos_y.append(start_agents[num][1])

    position,  = plt.plot(pos_y, pos_x, '*', markersize = 12, color = 'red')

    print("this is total cost", total_cost)

    t = 0
    while True:

        if t == 0:
            plt.pause(2)

        for num in range(len(solution)):
            path = solution[num]

            if t<len(path):
                node = path[t]
                pos_x[num] = node[0]
                pos_y[num] = node[1]

        position.set_data(pos_y, pos_x)
        plt.draw()
        plt.pause(0.5)

        t += 1

    plt.show()

if __name__ == '__main__':
    main()