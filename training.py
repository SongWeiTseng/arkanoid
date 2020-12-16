from preprocessing import *
# from save_model import save_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle
# I put this program in the same folder as MLGame/games/arkaonid/ml
# you can edit path to get log folder
if __name__ == "__main__":
    # preprocessing
    data_set = get_dataset()
    X, y = combine_multiple_data(data_set)

    # %% training
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    param_grid = {'weights': ('uniform', 'distance'),
                  'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),
                  #   'gamma': [0.1, 1, 10],
                  #   'epsilon': [0.01, 0.05, 0.1, 0.5, 1.0]
                  }
    knn = KNeighborsClassifier(n_neighbors=3)

    gclf = GridSearchCV(knn, param_grid, cv=5)
    gclf.fit(x_train, y_train)
    y_predict = gclf.predict(x_test)

    # extract the best parameters
    bestModel = gclf.best_estimator_
    best_score = gclf.best_score_
    print("Best Model:", bestModel)
    print("Training score:", best_score)
    print("Test score", accuracy_score(y_predict, y_test))
    # %% save the model
    with open("model.pickle", 'wb') as f:
        pickle.dump(bestModel, f)
    # save_model(bestModel, "model.pickle")
