from sklearn.svm import SVC
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import linear_kernel, poly_kernel, matern_kernel
from bayes_opt import BayesianOptimization


# defining the kernels
K_exp  = Kinterface(data=X, kernel=rbf_kernel,  kernel_args={"sigma": 0.0003}) # RBF kernel 
K_poly = Kinterface(data=X, kernel=poly_kernel, kernel_args={"b": 3})      # polynomial kernel with degree=3
K_lin  = Kinterface(data=X, kernel=linear_kernel)                          # linear kernel
K_mat  = Kinterface(data=X, kernel=matern_kernel)

# np.random.seed(42)
kf = KFold(n_splits=3, shuffle=True)
print(kf)  

# KFolde training helper function
def KFold_train(X,Y_train,kf,clf, metrics, print_report = False):
    kf.get_n_splits(X)
    n, d = kf.n_splits, len(metrics)
    score = np.zeros((n, d))
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        
        clf.fit(X_train,y_train)
        pred_y = clf.predict(X_test)
        for metric, j in zip(metrics, range(d)):
            score[i,j] = metric(y_test, pred_y)
            
        if print_report:
            print(classification_report(y_test, pred_y))
            print(score[i,:])
        i+=1
    
    return np.mean(score, axis=0)

def KFold_train_score(X, Y_train,kf,clf, metrics, print_report = False):
    kf.get_n_splits(X)
    n, d = kf.n_splits, len(metrics)
    score = np.zeros((n, d))
    i = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y_train[train_index], Y_train[test_index]
        print(X_train.shape)
        print(X_test.shape)
        clf.fit(X_train,y_train)
        pred_y = svm_clf.decision_function(X_test)
        for metric, j in zip(metrics, range(d)):
            score[i,j] = metric(y_test, pred_y)
            
        if print_report:
            print(classification_report(y_test, pred_y))
            print(score[i,:])
        i+=1
    
    return np.mean(score, axis=0)

# Roubost KFolds
def roubst_KCV(n_rand, X,Y_train,kf,clf, metrics, print_report = False):
    d = len(metrics)
    roubst_score = np.zeros((n_rand, d))
    for n in range(n_rand):
        roubst_score[n,:] = KFold_train(X,Y_train,kf,clf, metrics, print_report = False)
    return np.mean(roubst_score, axis=0), np.std(roubst_score, axis=0)

def roubst_KCV_score(n_rand, X,Y_train,kf,clf, metrics, print_report = False):
    d = len(metrics)
    roubst_score = np.zeros((n_rand, d))
    for n in range(n_rand):
        roubst_score[n,:] = KFold_train_score(X,Y_train,kf,clf, metrics, print_report = False)
    return np.mean(roubst_score, axis=0), np.std(roubst_score, axis=0)

# defining black box function for BO
def black_box_function(alph, beta, epsolon, psi):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    combined_kernel = lambda x, y: \
    alph * K_exp(x, y) + beta * K_lin(x, y) + epsolon * K_poly(x, y) + psi * K_mat(x, y)
    
    svm_clf = SVC(kernel=combined_kernel)
    #np.random.seed(42)
    m, std = roubst_KCV(5,X,y,kf, svm_clf,[accuracy_score, precision_score, recall_score])
#     m, std = KFold_train_score(X,Y_train, kf, svm_clf,[roc_auc_score])
    return m[0] + std[0]/2