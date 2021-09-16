import random
import numpy as np
import matplotlib.pyplot as plt

###########################maze##################################
maze = np.full((30,40), 1)
random.seed(42)

numAgents = 20
agent_pos = []

posx = []
posy = []

for i in range(120):
    while True:
        row = random.randint(0,29)
        col = random.randint(0,39)

        if maze[row,col] == 1:
            maze[row,col] = 0
            break

for i in range(numAgents):
    while True:
        row = random.randint(0,29)
        col = random.randint(0,39)

        if maze[row,col] == 1 and ((row,col) not in agent_pos):
            posx.append(row)
            posy.append(col)
            agent_pos.append((row,col))
            break

# plt.imshow(maze, cmap='hot', interpolation='nearest'); 
# plt.savefig('map3.png')
# plt.show()