# We want to find the hyperparameters that produce the best result with scikit-learn's Random Forest Classifier on our a9a training set
# The hyper paramaters we are modifying are:
# kernel_type Default: 'rbf'
# gamma Default: 'scale'
# c Default: 1.0


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
    # print("Default score: " + str(default_score))

    param_grid = {
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'C': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
    }

    rf = SVC()
    random_search = RandomizedSearchCV(estimator = rf, param_distributions=param_grid, cv = 5, n_jobs = -1, verbose = 2, return_train_score=True, n_iter=10)
    random_search.fit(X_train, y_train)
    rf_best_params = random_search.best_params_

    rf_best_params_classifier = rf_default = SVC(kernel=rf_best_params['kernel'], gamma=rf_best_params['gamma'], C=rf_best_params['C'])
    rf_best_params_classifier.fit(X_train, y_train)
    rf_best_params_prediction = rf_best_params_classifier.predict(X_test)
    best_params_score = metrics.accuracy_score(y_test, rf_best_params_prediction)
    # print("Best parameters score: " + str(best_params_score))

    return (default_score, rf_best_params, best_params_score, random_search.cv_results_)
    


# rf_out = determine_SVM_hp(X_train, y_train, X_test, y_test, 'rbf', 'scale', 1.0)
# print(rf_out[0])
# print(rf_out[1])
# print(rf_out[2])
# print(rf_out[3])
