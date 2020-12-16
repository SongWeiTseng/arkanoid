import pickle
import numpy as np
import os


def get_feature_and_target(data):
    feature = np.zeros((1, 5))
    target = []
    scene_infos = data['scene_info']
    commands = data['command']
    # feature
    for i, scene_info in enumerate(scene_infos):
        if i > 0:
            row = get_row(scene_info, scene_infos[i-1]['ball'])
        else:
            row = get_row(scene_info, (0, 0))
        feature = np.vstack((feature, row))
    feature = feature[1:]
    # print("feature shape", feature.shape)

    # target

    for command in commands:
        if command == "NONE":
            target.append(0)
        elif command == "MOVE_LEFT":
            target.append(-1)
        elif command == "MOVE_RIGHT":
            target.append(1)
        else:
            target.append(0)
    target = np.array(target)
    return (feature, target)


def get_row(scene_info, previous_ball):
    ball_x = scene_info["ball"][0]
    ball_y = scene_info["ball"][1]
    platform_x = scene_info['platform'][0]
    vel_x = previous_ball[0]-ball_x
    vel_y = previous_ball[1]-ball_y

    row = np.array(
        [ball_x, ball_y, platform_x, vel_x, vel_y]
    )
    # print(row)
    return row


def combine_multiple_data(data_set):
    X = np.array([0, 0, 0, 0, 0])
    y = np.array([0])
    for data in data_set:
        feature, target = get_feature_and_target(data)
        X = np.vstack((X, feature))
        y = np.hstack((y, target))
        print("feature shape", feature.shape)
        print("target shape", target.shape)
    X = X[1::]
    y = y[1::]
    return X, y


def get_dataset():
    path = os.path.join(os.path.dirname(__file__), "..", "log")
    allFile = os.listdir(path)
    data_set = []
    for file in allFile:
        with open(os.path.join(path, file), "rb") as f:
            data_set.append(pickle.load(f))
    return data_set


if __name__ == "__main__":

    file = os.path.join(
        os.path.dirname(__file__), "..", "log",
        "ml_NORMAL_3_2020-07-27_18-48-38.pickle")
    with open(file, "rb") as f:
        data = pickle.load(f)
    print(get_row(data['scene_info'][250], (182, 381)))
    feature, target = get_feature_and_target(data)
    print(feature)
    print("feature shape", feature.shape)
    print("target shape", target.shape)
    pass
