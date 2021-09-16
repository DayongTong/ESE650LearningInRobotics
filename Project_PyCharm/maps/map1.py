import random
import numpy as np
import matplotlib.pyplot as plt
###########################MAP1##################################
maze = np.full((22,50), 1)

for row in range(22):
    for col in range(50):
        if (row%3 == 1 or row%3 == 2) and ((col-1)%12 != 0):
            maze[row, col] = 0

maze[:,0] = 1
numAgents = 20

posx = list(range(1,21))
dropx = list(range(1,21))
pickx = list(range(1,21))

posy = [0] * numAgents
dropy = [0] * numAgents
picky = [0] * numAgents

locations = [(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),
             (9,0),(10,0),(11,0),(12,0),(13,0),(14,0),
             (15,0),(16,0),(17,0),(18,0),(19,0),(20,0)]

# plt.imshow(maze, cmap='hot', interpolation='nearest'); 
# plt.savefig('map1.png')
# plt.show()

