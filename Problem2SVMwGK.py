# We want to find the hyperparameters that produce the best result with scikit-learn's Random Forest Classifier on our a9a training set
# The hyper paramaters we are modifying are:
# kernel_type Default: 'rbf'
# gamma Default: 'scale'
# c Default: '1.0'


from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

# Just for testing the function within the file
data_train = load_svmlight_file("a9a.txt")
data_test = load_svmlight_file("a9a.t")
X_train = data_train[0]
y_train = data_train[1]
X_test = data_test[0]
y_test = data_test[1]


def determine_SVM_hp(X_train, y_train, X_test, y_test, default_kernel_type, default_gamma, default_c):
    
    rf_default = SVC(kernel=default_kernel_type, gamma=default_gamma, C=default_c)
    rf_default.fit(X_train, y_train)
    rf_default_prediction = rf_default.predict(X_test)
    default_score = metrics.accuracy_score(y_test, rf_default_prediction)
    print("Default score: " + str(default_score))

    # param_grid = {
    #     'n_estimators': [x for x in range(100, 1000, 50)],
    #     'bootstrap': [True, False],
    #     'max_depth': [None, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
    # }

    # rf = RandomForestClassifier()
    # random_search = RandomizedSearchCV(estimator = rf, param_distributions=param_grid, cv = 5, n_jobs = -1, verbose = 2, return_train_score=True, n_iter=20)
    # random_search.fit(X_train, y_train)
    # rf_best_params = random_search.best_params_

    # rf_best_params_classifier = RandomForestClassifier(n_estimators=rf_best_params['n_estimators'], bootstrap=rf_best_params['bootstrap'], max_depth=rf_best_params['max_depth'], min_impurity_decrease=rf_best_params['min_impurity_decrease'], min_samples_leaf=rf_best_params['min_samples_leaf'])
    # rf_best_params_classifier.fit(X_train, y_train)
    # rf_best_params_prediction = rf_best_params_classifier.predict(X_test)
    # best_params_score = metrics.accuracy_score(y_test, rf_best_params_prediction)
    # print("Best parameters score: " + str(best_params_score))

    # return (default_score, rf_best_params, best_params_score, random_search.cv_results_)
    


rf_out = determine_SVM_hp(X_train, y_train, X_test, y_test, 'rbf', 'scale', 1.0)
# print(rf_out[0])
# print(rf_out[1])
# print(rf_out[2])
# print(rf_out[3])
