# We want to find the hyperparameters that produce the best result with scikit-learn's Random Forest Classifier on our a9a training set
# The hyper paramaters we are modifying are:
# n_estimators Default: 100
# max_depth Default: None
# lambda Default: 1
# learning_rate Default: 1.0
# missing Default: None
# objective Default: 'binary:logistic'

from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# Just for testing the function within the file
data_train = load_svmlight_file("a9a.txt")
data_test = load_svmlight_file("a9a.t")
X_train = data_train[0]
y_train = data_train[1]
X_test = data_test[0]
y_test = data_test[1]


def determine_xgboost_hp(X_train, y_train, X_test, y_test, default_n_estimators, default_max_depth, default_lambda, default_learning_rate, default_missing, default_objective):

    rf_default = XGBClassifier(n_estimators=default_n_estimators, max_depth=default_max_depth, reg_lambda=default_lambda, learning_rate=default_learning_rate, missing=default_missing, objective=default_objective)
    rf_default.fit(X_train, y_train)
    rf_default_prediction = rf_default.predict(X_test)
    default_score = metrics.accuracy_score(y_test, rf_default_prediction)
    # print("Default score: " + str(default_score))

    param_grid = {
        'n_estimators': [x for x in range(50, 1000, 50)],
        'max_depth': [None, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'learning_rate': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'missing': [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        # 'objective': ['binary:logistic'], Not modifying objective
    }

    rf = XGBClassifier()
    random_search = RandomizedSearchCV(estimator = rf, param_distributions=param_grid, cv = 5, n_jobs = -1, verbose = 2, return_train_score=True, n_iter=10)
    random_search.fit(X_train, y_train)
    rf_best_params = random_search.best_params_

    rf_best_params_classifier = XGBClassifier(n_estimators=rf_best_params['n_estimators'], max_depth=rf_best_params['max_depth'], reg_lambda=rf_best_params['reg_lambda'], learning_rate=rf_best_params['learning_rate'], missing=rf_best_params['missing'])
    rf_best_params_classifier.fit(X_train, y_train)
    rf_best_params_prediction = rf_best_params_classifier.predict(X_test)
    best_params_score = metrics.accuracy_score(y_test, rf_best_params_prediction)
    # print("Best parameters score: " + str(best_params_score))

    return (default_score, rf_best_params, best_params_score, random_search.cv_results_)
    


# rf_out = determine_xgboost_hp(X_train, y_train, X_test, y_test, 100, None, 1, 1.0, 0, None)
# print(rf_out[0])
# print(rf_out[1])
# print(rf_out[2])
# print(rf_out[3])
