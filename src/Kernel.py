"""
Ridge Regression implementation with Kernels.
Exponential and Polynomial Kernels are available for use.
In addition Least Squares Regression implemented with Fourier Basis Functions.

"""

import numpy as np
import matplotlib.pyplot as plt

default_path = "C:\\solution\\data\\"

def retrieve_data(path, is_training):
    """
    Retrieves data from a specified *path* on the local filesystem.
    Data has to be split into two files named:: "Train.txt" and "Test.txt"
    accordingly and placed under the path which will be specified during function call.

    Two import options are available::

        retrieve_data(path, is_training: True)

        retrieve_data(path, is_training: False)

    In the first case the file "regTrain.txt" is read by the :fun:'np.loadtxt'.

    In the second case the file "regTest.txt" is read by the :fun:'np.loadtxt'.

    """
    if is_training:
        return np.loadtxt(path + "regTrain.txt").T
    else:
        return np.loadtxt(path + "regTest.txt").T


def fourier_basis(index, data):
    if index == 0:
        return np.ones(len(data))
    elif index % 2 == 1:
        return np.cos(np.dot(2 * np.pi * (index + 1) / 2, data)) / (index + 1) * 2
    else:
        return np.sin(np.dot(2 * np.pi * index / 2, data)) / index * 2


def get_phi_matrix(data, k):
    return np.array([fourier_basis(index, data) for index in range(k)]).T


def get_weights_matrix(phi, y, lamda):
    return np.linalg.inv(phi.T.dot(phi) + lamda * np.identity(len(phi[0]))).dot(phi.T).dot(y)


def predict_with_kernel(kernel, y, lamda):
    return np.linalg.inv(kernel + lamda * np.identity(len(kernel))).dot(y)


def get_exponential_kernel(x, y, l):
    exponential_kernel = np.ndarray(shape=(len(x), len(y)))
    for i in range(0, len(x), 1):
        for j in range(0, len(y), 1):
            exponential_kernel[i][j] = np.exp(-0.5 * (((x[i] - y[j]) / l) ** 2))
    return exponential_kernel


def get_polynomial_kernel(x, y, d):
    polynomial_kernel = np.ndarray(shape=(len(x), len(y)))
    for i in range(len(x)):
        for j in range(0, len(y), 1):
            polynomial_kernel[i][j] = (x[i] * y[j] + 1) ** d
    return polynomial_kernel


def predict(phi_kernel, w_c, k=-1):
    return phi_kernel[:, :k].dot(w_c[:k])


def mean(training_data, prediction_data):
    return np.sqrt(np.mean((training_data - prediction_data) ** 2) * 2)


def plot(x, ys):
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(ys) + 1)))
    for y in ys:
        plt.plot(x, y, c=next(color))


class KernelRidgeRegression(object):


    def __init__(self, k=17, path=default_path):
        self.x_train, self.y_train = retrieve_data(path=path, is_training=True)
        self.x_test, self.y_test = retrieve_data(path=path, is_training=False)
        self.phi_train = get_phi_matrix(self.x_train, k=k)
        self.phi_test = get_phi_matrix(self.x_test, k=k)

    def print_data_and_phi(self):
        plot(self.x_train, self.phi_train.T)
        plot(self.x_train, (self.y_train,))
        plt.show()

    def print_error_test_train(self):
        weights = [get_weights_matrix(self.phi_train, self.y_train, i / 4) for i in range(0, 1, 1)]
        err_test_w = [[mean(predict(self.phi_test, w, k), self.y_test) for k in range(1, 18, 2)] for w in weights]
        err_train_w = [[mean(predict(self.phi_train, w, k), self.y_train) for k in range(1, 18, 2)] for w in weights]

        plot(range(len(err_test_w)), err_test_w)
        plot(range(len(err_train_w)), err_train_w)
        plt.show()

    def compare_kernels(self):
        exp_ker_train = get_exponential_kernel(self.x_train, self.x_train, 0.04)
        exp_ker_test = get_exponential_kernel(self.x_test, self.x_train, 0.04)
        pol_ker_train = get_exponential_kernel(self.x_train, self.x_train, 5)
        pol_ker_test = get_exponential_kernel(self.x_test, self.x_train, 5)

        predict_exp = [predict_with_kernel(exp_ker_train, self.y_train, lamda=i / 5) for i in range(8, 27, 1)]
        predict_pol = [predict_with_kernel(pol_ker_train, self.y_train, lamda=i / 5) for i in range(8, 27, 1)]

        err_test_c_exp = [mean(predict(exp_ker_test, c), self.y_test) for c in predict_exp]
        err_train_c_exp = [mean(predict(exp_ker_train, c), self.y_train) for c in predict_exp]
        err_test_c_pol = [mean(predict(pol_ker_test, c), self.y_test) for c in predict_pol]
        err_train_c_pol = [mean(predict(pol_ker_train, c), self.y_train) for c in predict_pol]

        plot(range(len(predict_exp)), (err_test_c_exp,))
        plot(range(len(predict_exp)), (err_train_c_exp,))
        plot(range(len(predict_pol)), (err_test_c_pol,))
        plot(range(len(predict_pol)), (err_train_c_pol,))
        plt.show()

    def variation_d (self):
        pol_kernels_d = [get_polynomial_kernel(self.x_train, self.x_train, 10 * 2 ** d) for d in range(-5, 4, 1)]

        cs_pol_d = [predict_with_kernel(ker, self.y_train, 5) for ker in pol_kernels_d]

        err_test_c_pol_d = [mean(predict(pol_kernels_d, c), self.y_test) for c in cs_pol_d]
        err_train_c_pol_d = [mean(predict(pol_kernels_d, c), self.y_train) for c in cs_pol_d]

        plot(range(len(pol_kernels_d)), (err_test_c_pol_d,))
        plot(range(len(pol_kernels_d)), (err_train_c_pol_d,))

        plt.show()

    def main(self, path=default_path):

        # point 3a:
        # self.print_data_and_phi()
        #
        # # point 3b and 3c:
        # self.print_error_test_train()

        # point 4:
        self.compare_kernels()
        self.variation_d()


if __name__ == "__main__":
    Kernel = KernelRidgeRegression()
    Kernel.main()
