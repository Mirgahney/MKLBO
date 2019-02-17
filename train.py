from sklearn.svm import SVC
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import linear_kernel, poly_kernel, matern_kernel
from bayes_opt import BayesianOptimization
import pandas as pd
import argparse

# reading data
X = pd.read_csv('data/train_X.csv')
y = pd.read_csv('data/train_y.csv')

# defining the kernels
K_exp  = Kinterface(data=X, kernel=rbf_kernel,  kernel_args={"sigma": 0.0003}) # RBF kernel 
K_poly = Kinterface(data=X, kernel=poly_kernel, kernel_args={"b": 3})      # polynomial kernel with degree=3
K_lin  = Kinterface(data=X, kernel=linear_kernel)                          # linear kernel
K_mat  = Kinterface(data=X, kernel=matern_kernel)

# redaing routine
def read_data(path):
	X = pd.read_csv(path+'train_X.csv')
	y = pd.read_csv(path+'train_y.csv')

	return X,y

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

# training routine 
def train(X,y, alph_bound = (0,5), beta_bound= (0,5), epsolon_bound= (0,5), psi_bound= (0,5)):

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

	# Bounded region of parameter space
	pbounds = {'alph': (0, 5), 'beta': (0, 5),'epsolon':(0,5), 'psi' : (0,5)}

	# BO Optimizer
	optimizer = BayesianOptimization(
	    f=black_box_function,
	    pbounds=pbounds,
	    random_state=1,
	)

	# preform the optimization
	optimizer.maximize(
	    init_points=4,
	   n_iter=50,
	)

	# printing the final result
	print(optimizer.max['params'])
	combined_kernel = lambda x, y: \
	    optimizer.max['params']['alph'] * K_exp(x, y) + optimizer.max['params']['beta'] * K_lin(x, y) + optimizer.max['params']['epsolon'] * K_poly(x, y) + \
	    optimizer.max['params']['psi'] * K_mat(x, y)
	
	svm_clf = SVC(kernel=combined_kernel)

	return svm_clf

# printing the final result
print(optimizer.max['params'])
combined_kernel = lambda x, y: \
    optimizer.max['params']['alph'] * K_exp(x, y) + optimizer.max['params']['beta'] * K_lin(x, y) + optimizer.max['params']['epsolon'] * K_poly(x, y) + \
    optimizer.max['params']['psi'] * K_mat(x, y)
# np.random.seed(42)
svm_clf = SVC(kernel=combined_kernel)
print(roubst_KCV(5,X,y,kf, svm_clf,[accuracy_score, precision_score, recall_score, f1_score]))
print(roubst_KCV_score(5,X,Y_train, kf, svm_clf,[roc_auc_score]))

# wriring the result to a file
with open('result/result.txt','w') as f:
	f.write('MKLBO with 4 kerels\n')
	f.write('Accuracy score  Precision score  Recall score  F1 score\n')
	f.write(str(roubst_KCV(5,X,y,kf, svm_clf,[accuracy_score, precision_score, recall_score, f1_score]) + '\n'))
	f.write('AUC ROC\n')
	f.write(str(roubst_KCV_score(5,X,Y_train, kf, svm_clf,[roc_auc_score]), + '\n'))

def main():

	# reading data
	print('Reading data ----------------- \n')
	X,y = read_data(path)

	
if __name__ == 'main':
	main()