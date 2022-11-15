# We want to find the hyperparameters that produce the best result with scikit-learn's Random Forest Classifier on our a9a training set
# The hyper paramaters we are modifying are:
# n_estimators Default: 100
# max_depth Default: None
# lambda Default: None
# learning_rate Default: None
# missing Default: None
# objective Default: 'binary:logistic'

from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Just for testing the function within the file
data_train = load_svmlight_file("a9a.txt")
data_test = load_svmlight_file("a9a.t")
X_train = data_train[0]
y_train = data_train[1]
X_test = data_test[0]
y_test = data_test[1]


def determine_xgboost_hp(X_train, y_train, X_test, y_test, default_n_estimators=100, default_max_depth=None, default_lambda=None, default_learning_rate=None, default_missing=np.nan, default_objective='binary:logistic'):

    bdt_default = XGBClassifier(n_estimators=default_n_estimators, max_depth=default_max_depth, reg_lambda=default_lambda, learning_rate=default_learning_rate, missing=default_missing, objective=default_objective)
    bdt_default.fit(X_train, y_train)
    bdt_default_prediction = bdt_default.predict(X_test)
    default_score = metrics.accuracy_score(y_test, bdt_default_prediction)
    # print("Default score: " + str(default_score))

    param_grid = {
        'n_estimators': [x for x in range(50, 1000, 50)],
        'max_depth': [None],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'learning_rate': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'missing': [np.nan, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        # 'objective': ['binary:logistic'], Not modifying objective
    }

    # Largely just used for testing purposes
    alt_param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'reg_lambda': [3.0],
        'learning_rate': [None],
        'missing': [np.nan],
        # 'objective': ['binary:logistic'], Not modifying objective
    }

    bdt = XGBClassifier()
    # Can increase the number of iterations of our search to increase the probability of getting a good combination of parameters while sacrificing time, the extreme end would just be a grid search
    random_search = RandomizedSearchCV(estimator = bdt, param_distributions=param_grid, cv = 5, n_jobs = -1, verbose = 2, return_train_score=True, n_iter=10)
    random_search.fit(X_train, y_train)
    bdt_best_params = random_search.best_params_

    bdt_best_params_classifier = XGBClassifier(n_estimators=bdt_best_params['n_estimators'], max_depth=bdt_best_params['max_depth'], reg_lambda=bdt_best_params['reg_lambda'], learning_rate=bdt_best_params['learning_rate'], missing=bdt_best_params['missing'])
    bdt_best_params_classifier.fit(X_train, y_train)
    bdt_best_params_prediction = bdt_best_params_classifier.predict(X_test)
    best_params_score = metrics.accuracy_score(y_test, bdt_best_params_prediction)
    # print("Best parameters score: " + str(best_params_score))

    return (default_score, bdt_best_params, best_params_score, random_search.cv_results_)
#1e-5
def try_xgboost_combination(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None, reg_lambda=None, learning_rate=None, missing=np.nan):
    bdt = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, reg_lambda=reg_lambda, learning_rate=learning_rate, missing=missing)
    bdt.fit(X_train, y_train)
    bdt_prediction = bdt.predict(X_test)
    score = metrics.accuracy_score(y_test, bdt_prediction)
    return score

# print(try_xgboost_combination(X_train, y_train, X_test, y_test, reg_lambda=3.0))
    

# bdt_out = determine_xgboost_hp(X_train, y_train, X_test, y_test)
# print(bdt_out[0])
# print(bdt_out[1])
# print(bdt_out[2])
# print(bdt_out[3])
