import numpy as np
import matplotlib.pyplot as plt

# map 0
map0 = np.full((30,50), 1)

for row in range(30):
    for col in range(50):
        if (row%4 == 2 or row%4 == 3) and (col%12 != 0 and col%12 != 1):
            map0[row, col] = 0

# map 1
map1 = np.full((22,50), 1)

for row in range(22):
    for col in range(50):
        if (row%3 == 1 or row%3 == 2) and (col%12 != 0 and col%12 != 1):
            map1[row, col] = 0

# plt.imshow(map0, cmap='hot', interpolation='nearest')
# plt.show()

