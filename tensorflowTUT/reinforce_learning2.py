# http://www.cnblogs.com/LHWorldBlog/p/9249011.html

import numpy as np
GAMMA = 0.80
num_steps = 100
SIZE = 6
R = np.asarray([[-1, -1, -1, -1,  0, -1],
                [-1, -1, -1, 0, -1, 100],
                [-1, -1, -1, 0, -1, -1],
                [-1, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, -1, 100],
                [-1, 0, -1, -1, 0, 100]])
Q = np.zeros([SIZE, SIZE], np.float32)


def getMaxQ(statey):
    state = []
    for i in range(SIZE):
        if(Q[statey, i]!=-1):
            state.append(Q[statey, i])
    return max(state[:])


def QLearning():
    for statex in range(SIZE):
        for statey in range(SIZE):
            if (R[statex, statey]!=-1):
                Q[statex, statey] = R[statex, statey]+GAMMA * getMaxQ(statey)

count = 0
while count < num_steps:
    QLearning()
    count += 1

print(Q)