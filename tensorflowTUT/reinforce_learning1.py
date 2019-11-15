# https://www.jianshu.com/p/1c0d5e83b066

import numpy as np
GAMMA = 0.80
ALPHA = 0.010
num_steps = 1000
SIZE = 5
R = np.asarray([[-10, 10, -10, 0, 0],
                [0, 0, 0, 0, 10],
                [0, 0, 0, 0, 0],
                [10, 0, 0, 0, 0],
                [0, 10, 0, 0, 0]])
Q = np.zeros([SIZE, SIZE], np.float32)


def getMaxQ(statex, statey):
    state = []
    if statex > 0:
        state.append(Q[statex-1, statey])
    if statey > 0:
        state.append(Q[statex, statey-1])
    if statex < SIZE-1:
        state.append(Q[statex+1, statey])
    if statey < SIZE-1:
        state.append(Q[statex, statey+1])
    return max(state[:])


def QLearning():
    for statex in range(SIZE):
        for statey in range(SIZE):
            # Q[statex, statey] = (1-ALPHA)*Q[statex, statey] + ALPHA* (R[statex, statey]+GAMMA * getMaxQ(statex, statey))
            Q[statex, statey] = R[statex, statey] + GAMMA * getMaxQ(statex, statey)

count = 0
while count < num_steps:
    QLearning()
    count += 1

print(Q)