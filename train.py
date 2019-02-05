from sklearn.svm import SVC
from mklaren.kernel.kinterface import Kinterface
from mklaren.kernel.kernel import linear_kernel, poly_kernel, matern_kernel
from bayes_opt import BayesianOptimization


# defining the kernels
K_exp  = Kinterface(data=X, kernel=rbf_kernel,  kernel_args={"sigma": 0.0003}) # RBF kernel 
K_poly = Kinterface(data=X, kernel=poly_kernel, kernel_args={"b": 3})      # polynomial kernel with degree=3
K_lin  = Kinterface(data=X, kernel=linear_kernel)                          # linear kernel
K_mat  = Kinterface(data=X, kernel=matern_kernel)