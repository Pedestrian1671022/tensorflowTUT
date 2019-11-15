# https://baijiahao.baidu.com/s?id=1597978859962737001&wfr=spider&for=pc   走到中间位置

import numpy as np
GAMMA = 0.80
num_steps = 10000
STATES = 25
ACTIONS = 4
R = np.asarray([[0, 0, 1, 1],
                [-1, 0, 1, 1],
                [-1, 0, -1, 1],
                [1, 0, -1, 1],
                [1, 0, 0, 1],
                [0, -1, 1, 1],
                [-1, -1, 1, 1],
                [-1, -1, -1, 2],
                [1, -1, -1, 1],
                [1, -1, 0, 1],
                [0, -1, 1, -1],
                [-1, -1, 2, -1],
                [0, 0, 0, 0],
                [2, -1, -1, -1],
                [1, -1, 0, -1],
                [0, 1, 1, -1],
                [-1, 1, 1, -1],
                [-1, 2, -1, -1],
                [1, 1, -1, -1],
                [1, 1, 0, -1],
                [0, 1, 1, 0],
                [-1, 1, 1, 0],
                [-1, 1, -1, 0],
                [1, 1, -1, 0],
                [1, 1, 0, 0]])
Q = np.zeros([STATES, ACTIONS], np.float32)


def getMaxQ(state, action):
    states = []
    if (action == 0):
        states.extend(Q[state-1, :])
    if (action == 1):
        states.extend(Q[state-5, :])
    if (action == 2):
        states.extend(Q[state+1, :])
    if (action == 3):
        states.extend(Q[state+5, :])
    return max(states[:])


def QLearning():
    for state in range(STATES):
        for action in range(ACTIONS):
            if (R[state, action]!=0):
                Q[state, action] = R[state, action]+GAMMA * getMaxQ(state, action)

count = 0
while count < num_steps:
    QLearning()
    count += 1

print(Q)