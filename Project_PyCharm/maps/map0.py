import numpy as np
import matplotlib.pyplot as plt

###########################MAP0##################################
maze = np.full((30,50), 1)

for row in range(30):
    for col in range(50):
        if (row%4 == 2 or row%4 == 3) and (col%12 != 0 and col%12 != 1):
            maze[row, col] = 0

numAgents = 4

# posx = list(range(1,29))
# dropx = list(range(1,29))
# pickx = list(range(1,29))

posx = [0] * numAgents
dropx = [0] * numAgents
pickx = [0] * numAgents

posy = [0] * numAgents
dropy = [0] * numAgents
picky = [0] * numAgents

locations = [(1,0),(10,0),(19,0),(28,0)]

# plt.imshow(maze, cmap='hot', interpolation='nearest'); 
# plt.savefig('map0.png')
# plt.show()

