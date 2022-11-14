# We want to find the hyperparameters that produce the best result with scikit-learn's Random Forest Classifier on our a9a training set
# The hyper paramaters we are modifying are:
# n_estimators Default: 100
# bootstrap Default: True
# max_depth Default: None
# min_impurity_decrease Default: 0.0
# min_samples_leaf Default: 1

from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Just for testing the function within the file
data_train = load_svmlight_file("a9a.txt")
data_test = load_svmlight_file("a9a.t")
X_train = data_train[0]
y_train = data_train[1]
X_test = data_test[0]
y_test = data_test[1]


def determine_random_forest_hp(X_train, y_train, X_test, y_test, default_n_estimators, default_bootstrap, default_max_depth, default_min_impurity_decrease, default_min_samples_leaf):
    
    rf_default = RandomForestClassifier(n_estimators=default_n_estimators, bootstrap=default_bootstrap, max_depth=default_max_depth, min_impurity_decrease=default_min_impurity_decrease, min_samples_leaf=default_min_samples_leaf)
    rf_default.fit(X_train, y_train)
    rf_default_prediction = rf_default.predict(X_test)
    default_score = metrics.accuracy_score(y_test, rf_default_prediction)
    # print("Default score: " + str(default_score))

    param_grid = {
        'n_estimators': [x for x in range(100, 1000, 50)],
        'bootstrap': [True, False],
        'max_depth': [None, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        'min_impurity_decrease': [x*0.5 for x in range(20)],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    rf = RandomForestClassifier()
    random_search = RandomizedSearchCV(estimator = rf, param_distributions=param_grid, cv = 5, n_jobs = -1, verbose = 2, return_train_score=True, n_iter=10)
    random_search.fit(X_train, y_train)
    rf_best_params = random_search.best_params_

    rf_best_params_classifier = RandomForestClassifier(n_estimators=rf_best_params['n_estimators'], bootstrap=rf_best_params['bootstrap'], max_depth=rf_best_params['max_depth'], min_impurity_decrease=rf_best_params['min_impurity_decrease'], min_samples_leaf=rf_best_params['min_samples_leaf'])
    rf_best_params_classifier.fit(X_train, y_train)
    rf_best_params_prediction = rf_best_params_classifier.predict(X_test)
    best_params_score = metrics.accuracy_score(y_test, rf_best_params_prediction)
    # print("Best parameters score: " + str(best_params_score))

    return (default_score, rf_best_params, best_params_score, random_search.cv_results_)

def try_random_forest_combination(X_train, y_train, X_test, y_test, n_estimators=100, bootstrap=True, max_depth=None, min_impurity_decrease=0.0, min_samples_leaf=1):
    rf = RandomForestClassifier(n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth, min_impurity_decrease=min_impurity_decrease, min_samples_leaf=min_samples_leaf)
    rf.fit(X_train, y_train)
    rf_prediction = rf.predict(X_test)
    score = metrics.accuracy_score(y_test, rf_prediction)
    return score
    
print(try_random_forest_combination(X_train, y_train, X_test, y_test))


# rf_out = determine_random_forest_hp(X_train, y_train, X_test, y_test, 100, True, None, 0, 1)
# print(rf_out[0])
# print(rf_out[1])
# print(rf_out[2])
# print(rf_out[3])
