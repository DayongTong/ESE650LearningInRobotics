import copy

import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop


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


def astar(maze, start, end, solution, constraint):
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
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1],  current_node.g  # Return reversed path, shape is (n, 2)

        # Generate children
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > maze.shape[0] - 1 or node_position[0] < 0 or node_position[1] > maze.shape[1] - 1 or \
                    node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] == 0:
                continue

            valid = True

            # print(node_position)
            # if constraint != []:
            #     if node_position == constraint[0] and (current_node.g + 1) == constraint[1]:
            #         # if path[current_node.g + 1] == constraint[1]:
            #         valid = False
            #         # print("yay")
            for k in range(len(constraint)):
                if node_position == constraint[k][0] and (current_node.g + 1) == constraint[k][1]:
                    continue

            ### TODO CHANGE THIS PLZZZZ to use constraint properly
            if solution != []:
                for path in solution:
                    if current_node.g + 1 < len(path):
                        if path[current_node.g + 1] == node_position:
                            valid = False
                        if path[current_node.g + 1] == current_node.position and node_position == path[current_node.g]:
                            valid = False

                        # if current_node.g + 1 == constraint[2]:
                        #     # print(constraint)
                        #     if path[current_node.g + 1] == constraint[1]:
                        #         valid = False

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
                child.h = ((child.position[0] - end_node.position[0]) ** 2) + (
                            (child.position[1] - end_node.position[1]) ** 2)
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

    return [start], np.Inf



class Conflict():
    """Conflict node for Conflict-based search"""

    def __init__(self, constraint, solution, cost):
        # self.position = position    # list of positions of the conflict, it is a list of tuples, each tuple a position
        # self.time = time
        # constraint format is (agent, position, time) meaning this agent cannot appear at this location at this time
        self.constraint = constraint
        # list of time corresponding to position of conflict,
        # each time the number of steps it takes from the beginning to that conflict
        self.solution = solution    # all agents' path
        # cost is a list of individual costs, need to np.sum(cost) to get real SIC cost
        self.cost = cost  # SIC: sum of individual costs heuristic, essentially sum of A* costs for all agents

    def __lt__(self, other):
        return np.sum(self.cost) < np.sum(other.cost)


def validate_path(solution):

    len_vec = []    # length of each agent's path
    for agent_path in solution:
        len_vec.append(np.shape(agent_path)[0])
    max_path_length = np.max(len_vec)
    max_path_agent = np.argmax(len_vec)
    num_of_agent = len(solution)

    for i in range(max_path_length):    # iterate through each timestep
        agent_positions = []
        for j in range(num_of_agent):   # iterate through each agent
            if i < len_vec[j]:      # there is still more path for this agent
                curr_pos = solution[j][i]
                if curr_pos in agent_positions and curr_pos[1] != 0 and curr_pos[1] != 50:
                    agent1 = agent_positions.index(curr_pos)
                    agent2 = j
                    return False, (agent1, agent2, curr_pos, i)
                else:
                    agent_positions.append(curr_pos)
            else:
                # just to fill agent_position with something, to keep  agent_pos indexing same as solution
                agent_positions.append(solution[j][-1])
                # agent_positions.append((np.Inf, np.Inf))

    return True, 0


# Conflict-Based Search
def cbs(maze, start_agents, end_agents):
    solution = []
    sic_cost = []
    constraint = {}
    for num in range(len(start_agents)):
        start = start_agents[num]
        end = end_agents[num]
        path, this_cost = astar(maze, start, end, [], [])
        solution.append(path)
        sic_cost.append(this_cost)
        constraint[num] = []

    # constraint should be dict of tuples (conflict location, conflict time)
    # keys to dict is agent number, location is another tuple inside this constraint tuple
    # each node only need one set of constraint, because previous nodes have taken care of those conflicts
    # in while loop, for each conflict cbs only need to adjust one path for that agent
    root = Conflict(constraint, solution, sic_cost)
    open_conflict = []
    heappush(open_conflict, root)

    count = 0
    while open_conflict != [] and count <= 10000:
        curr_conflict = heappop(open_conflict)
        validated, new_conflict = validate_path(curr_conflict.solution)
        if validated:
            print("this is total cost of CBS", np.sum(curr_conflict.cost))
            return curr_conflict.solution

        conflict_agents = [new_conflict[0], new_conflict[1]]
        conflict_position = new_conflict[2]
        # print(conflict_agents)
        for agent_num in conflict_agents:
            new_constraint = copy.deepcopy(curr_conflict.constraint)
            new_constraint[agent_num].append((conflict_position, new_conflict[3]))

            new_sic_cost = copy.deepcopy(curr_conflict.cost)
            new_solution = copy.deepcopy(curr_conflict.solution)

            new_path, new_cost = astar(maze, start_agents[agent_num],
                                       end_agents[agent_num], curr_conflict.solution,
                                       new_constraint[agent_num])
            new_sic_cost[agent_num] = new_cost
            new_solution[agent_num] = new_path


            # for agent_and_constraint in new_constraint.items():
            #     new_path, new_cost = astar(maze, start_agents[agent_and_constraint[0]],
            #                                end_agents[agent_and_constraint[0]], curr_conflict.solution,
            #                                agent_and_constraint[1])
            #     new_sic_cost[agent_and_constraint[0]] = new_cost
            #     new_solution[agent_and_constraint[0]] = new_path

            # new_path, new_cost = astar(maze, start_agents[agent_num], end_agents[agent_num],
            #                            curr_conflict.solution, new_constraint[agent_num])
            # new_sic_cost[agent_num] = new_cost
            # new_solution[agent_num] = new_path
            if new_cost < np.Inf:
                new_conflict_node = Conflict(new_constraint, new_solution, new_sic_cost)
                heappush(open_conflict, new_conflict_node)
            # print(new_constraint[agent_num])





        count += 1
        print(count)

    return curr_conflict.solution


def main():
    maze = np.full((30, 51), 1)

    for row in range(30):
        for col in range(50):
            if (row%4 == 2 or row%4 == 3) and (col%12 != 0 and col%12 != 1):
            # if (row % 4 == 2) and (col % 12 != 0 and col % 12 != 1 and col % 12 != 2 and col % 12 != 3 and col % 12 != 4 and col % 12 != 5):
            # if (row % 4 == 2 or row % 4 == 3 or row % 4 == 1) and (col % 12 != 0 and col % 12 != 1):
                maze[row, col] = 0

    plt.ion()
    plt.imshow(maze, cmap='hot', interpolation='nearest')

    start_agents = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                    (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0),
                    (0, 50), (1, 50), (2, 50), (3, 50), (4, 50), (5, 50), (6, 50), (7, 50), (8, 50),
                    (9, 50), (10, 50), (11, 50), (12, 50), (13, 50), (14, 50), (15, 50), (16, 50), (17, 50), (18, 50)]

    end_agents = [(0, 50), (1, 50), (2, 50), (3, 50), (4, 50), (5, 50), (6, 50), (7, 50), (8, 50),
                  (9, 50), (10, 50), (11, 50), (12, 50), (13, 50), (14, 50), (15, 50), (16, 50), (17, 50), (18, 50),
                  (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
                  (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 0)]

    # start_agents = [(0, 0), (1, 0),  (4, 0), (5, 0), (6, 0), (7, 0), (8, 0),
    #                 (9, 0), (10, 0),  (13, 0), (14, 0), (15, 0), (16, 0),
    #                 (0, 50), (1, 50), (2, 50), (3, 50), (4, 50), (5, 50),  (8, 50),
    #                 (9, 50), (10, 50),  (13, 50), (14, 50), (15, 50), (16, 50)]
    #
    # end_agents = [(0, 50), (1, 50),  (4, 50), (5, 50), (6, 50), (7, 50), (8, 50),
    #               (9, 50), (10, 50),  (13, 50), (14, 50), (15, 50), (16, 50),
    #               (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0),  (8, 0),
    #               (9, 0), (10, 0),  (13, 0), (14, 0), (15, 0), (16, 0)]

    solution = []

    # for num in range(len(start_agents)):
    #     start = start_agents[num]
    #     end = end_agents[num]
    #     path = astar(maze, start, end, solution)
    #     solution.append(path)
    #
    #     for step in range(len(path) - 1):
    #         current = path[step]
    #         nxt = path[step + 1]
    #
    #         x_dist = (nxt[0] - current[0]) * 0.5
    #         y_dist = (nxt[1] - current[1]) * 0.5
    #
    #         plt.arrow(current[1], current[0], y_dist, x_dist, head_width=0.1, head_length=0.2, fc='b', ec='b')

    solution = cbs(maze, start_agents, end_agents)

    for num in range(len(start_agents)):
        path = solution[num]
        for step in range(len(path) - 1):
            current = path[step]
            nxt = path[step + 1]

            x_dist = (nxt[0] - current[0]) * 0.3
            y_dist = (nxt[1] - current[1]) * 0.3

            # plt.arrow(current[1], current[0], y_dist, x_dist, head_width=0.6, head_length=0.3, fc='b', ec='b')
            plt.arrow(current[1], current[0], y_dist, x_dist, head_width=0.3, head_length=0.2, fc='b', ec='b')

    pos_x = []
    pos_y = []

    for num in range(len(start_agents)):
        pos_x.append(start_agents[num][0])
        pos_y.append(start_agents[num][1])

    position, = plt.plot(pos_y, pos_x, '*', markersize=12, color='red')

    t = 0
    while True:

        if t == 0:
            plt.pause(2)

        for num in range(len(solution)):
            path = solution[num]

            if t < len(path):
                node = path[t]
                pos_x[num] = node[0]
                pos_y[num] = node[1]

        check = np.array([pos_x, pos_y])
        if np.shape(np.unique(check, axis=1))[1] < np.shape(check)[1]:
            print("collision")

        position.set_data(pos_y, pos_x)
        plt.draw()
        plt.pause(0.5)

        t += 1

    # plt.show()


if __name__ == '__main__':
    main()