from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import pickle
from os import path

import numpy as np
# Fetch data
filename = path.join(path.dirname(__file__), "../log",
                     'ml_NORMAL_3_2020-07-27_18-48-38.pickle')
log = pickle.load((open(filename, 'rb')))

Frames = []
Balls = []
PlatformPos = []
sceneInfos = log['scene_info']
commands = []
print(sceneInfos[0])
print(log['command'][0])

for sceneInfo in sceneInfos:
    # print(sceneInfo)
    Frames.append(sceneInfo['frame'])
    Balls.append([sceneInfo['ball'][0], sceneInfo['ball'][1]])
    PlatformPos.append(sceneInfo['platform'])

for command in log['command']:
    if command == 'RIGHT':
        commands.append('RIGHT')
    elif command == 'LEFT':
        commands.append('LEFT')
    else:
        commands.append('NONE')
# print(Balls)
