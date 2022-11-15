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


def determine_SVM_hp(X_train, y_train, X_test, y_test, default_kernel_type='rbf', default_gamma='scale', default_c=1.0):
    
    svm_default = SVC(kernel=default_kernel_type, gamma=default_gamma, C=default_c)
    svm_default.fit(X_train, y_train)
    svm_default_prediction = svm_default.predict(X_test)
    default_score = metrics.accuracy_score(y_test, svm_default_prediction)
    # print("Default score: " + str(default_score))

    param_grid = {
        'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'C': [x*0.5 for x in range(1, 20)],
    }

    # Largely just used for testing purposes
    alt_param_grid = {
        'kernel': ['rbf'],
        'gamma': ['scale'],
        'C': [2.0],
    }

    svm = SVC()
    # Can increase the number of iterations of our search to increase the probability of getting a good combination of parameters while sacrificing time, the extreme end would just be a grid search
    random_search = RandomizedSearchCV(estimator = svm, param_distributions=alt_param_grid, cv = 5, n_jobs = -1, verbose = 2, return_train_score=True, n_iter=20)
    random_search.fit(X_train, y_train)
    svm_best_params = random_search.best_params_

    svm_best_params_classifier = svm_default = SVC(kernel=svm_best_params['kernel'], gamma=svm_best_params['gamma'], C=svm_best_params['C'])
    svm_best_params_classifier.fit(X_train, y_train)
    svm_best_params_prediction = svm_best_params_classifier.predict(X_test)
    best_params_score = metrics.accuracy_score(y_test, svm_best_params_prediction)
    # print("Best parameters score: " + str(best_params_score))

    return (default_score, svm_best_params, best_params_score, random_search.cv_results_)

def try_SVM_combination(X_train, y_train, X_test, y_test, kernel_type='rbf', gamma='scale', c=1.0):
    svm = SVC(kernel=kernel_type, gamma=gamma, C=c)
    svm.fit(X_train, y_train)
    svm_prediction = svm.predict(X_test)
    score = metrics.accuracy_score(y_test, svm_prediction)
    return score
    
# print(try_SVM_combination(X_train, y_train, X_test, y_test, gamma='auto'))

svm_out = determine_SVM_hp(X_train, y_train, X_test, y_test)
print(svm_out[0])
print(svm_out[1])
print(svm_out[2])
print(svm_out[3])
