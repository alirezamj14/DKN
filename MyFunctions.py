
import numpy as np
from numpy.linalg import norm
import math

def SSFN_train(X_train, X_test, T_train, T_test, SSFN_hparameters):
    """[Implements SSFN]

    Args:
        X_train ([float]): [The matrix of training data. Each column contains one sample.]
        X_test ([float]): [The matrix of testing data. Each column contains one sample.]
        T_train ([float]): [The matrix of training target. Each column contains one sample.]
        T_test ([float]): [The matrix of testing target. Each column contains one sample.]
        SSFN_hparameters ([dic]): [The dictionary of hyperparameters of SSFN.]

    Returns:
        [float]: [Training and testing error in dB.]
    """
    data = SSFN_hparameters["data"]
    lam = SSFN_hparameters["lam"]
    mu = SSFN_hparameters["mu"]
    kMax = SSFN_hparameters["kMax"]
    ni = SSFN_hparameters["NodeNum"]
    L = SSFN_hparameters["LayerNum"]

    P=X_train.shape[0]
    Q=T_train.shape[0]
    VQ=np.concatenate([np.eye(Q), (-1) * np.eye(Q)], axis=0)
    eps_o = 2 * np.sqrt(2*Q);

    train_error=[]
    test_error=[]
    test_accuracy=[]
    train_accuracy=[]

    O_ls = LS(X_train, T_train, lam)
    t_hat = np.dot(O_ls, X_train)
    t_hat_test = np.dot(O_ls, X_test)

    train_error.append(compute_nme(T_train,t_hat))
    test_error.append(compute_nme(T_test,t_hat_test))
    train_accuracy.append(calculate_accuracy(T_train,t_hat))
    test_accuracy.append(calculate_accuracy(T_test,t_hat_test))

    #   Initializing the algorithm for the first time
    Yi=X_train;
    Pi=P;
    Yi_test=X_test;

    for layer in range(1, L+1):
        # _logger.info("Begin to optimize layer {}".format(layer))
        Ri = 2 * np.random.rand(ni, Pi) - 1

        Zi_part1=np.dot(VQ, t_hat)
        Zi_part2=np.dot(Ri,Yi)
        Zi_part2 = Zi_part2 / np.linalg.norm(Zi_part2, axis=0)
        Zi=np.concatenate([Zi_part1, Zi_part2], axis=0)
        Yi_temp=activation(Zi)
        
        Oi=LS_ADMM(Yi_temp, T_train, eps_o, mu, kMax)    #   The ADMM solver for constrained least square
        t_hat=np.dot(Oi,Yi_temp)

        ##########  Test
        #  Following the same procedure for test data
        Zi_part1_test = np.dot(VQ, t_hat_test)
        Zi_part2_test = np.dot(Ri,Yi_test)
        Zi_part2_test = Zi_part2_test / np.linalg.norm(Zi_part2_test, axis=0)
        Zi_test=np.concatenate([Zi_part1_test, Zi_part2_test], axis=0)
        Yi_test_temp=activation(Zi_test)
        t_hat_test=np.dot(Oi,Yi_test_temp)

        train_error.append(compute_nme(T_train,t_hat))
        test_error.append(compute_nme(T_test,t_hat_test))
        train_accuracy.append(calculate_accuracy(T_train,t_hat))
        test_accuracy.append(calculate_accuracy(T_test,t_hat_test))

        train_listsP = [ '%.2f' % elem for elem in train_error ]
        test_listsP = [ '%.2f' % elem for elem in test_error ]


        Yi = Yi_temp
        Yi_test=Yi_test_temp
        Pi=Yi.shape[0]

    return calculate_accuracy(t_hat,T_train), calculate_accuracy(t_hat_test,T_test)


def calculate_accuracy(S, T):
    # S: predicted
    # T: given
    Y = np.argmax(S, axis=0)
    T = np.argmax(T, axis=0)
    accuracy = np.sum([Y == T]) / Y.shape[0]
    return accuracy

def activation(Z):
    Y = relu(Z)
    return Y

def relu(x):
    return np.maximum(0, x)

def compute_nme(S, T):
    """
    compute NME value 

    Parameters
    ----------
    S : np.ndarray
    predicted matrix
    T : np.ndarray
    given matrix

    Returns
    ----------
    nme : int
    NME value
    """
    numerator = norm((S - T), 'fro')
    denominator = norm(T, 'fro')
    nme = 20 * np.log10(numerator / denominator)
    return nme

def LS(X_train, T_train, lam):
    """[Solve the optimization problem as regularized least-squares]
        Solves the following minimization:
        O = argmin_{O} ||T - OX||_F + \lambda ||O||_F
    Returns:
        [float]: [The optimized linear mapping.]
    """
    P = X_train.shape[0]
    m = X_train.shape[1]
    
    if P < m:
        Ols = np.dot(np.dot(T_train,X_train.T), np.linalg.inv(np.dot(X_train,X_train.T) + lam * np.eye(P))).astype(np.float32)
    else:
        Ols = np.dot(T_train,np.linalg.inv(np.dot(X_train.T, X_train) + lam * np.eye(m))).dot(X_train.T)
    
    return Ols

def LS_ADMM(Y, T, eps_o, mu, kMax):
    """Optimize O by ADMM method"""
    p=Y.shape[0]
    q=T.shape[0]
    Z, Lam = np.zeros((q, p)), np.zeros((q, p))
    MyTemp = np.linalg.inv(np.dot(Y, Y.T) + 1 / mu * np.eye(p))
    TYT=np.dot(T, Y.T)
    for _ in range(kMax):
        O = np.dot(TYT + 1 / mu * (Z + Lam), MyTemp)
        Z = project_function(O, Lam, eps_o)
        Lam = Lam + Z - O
        
    return O

def project_function(O, Lam, epsilon):
    """Projection for ADMM"""
    Z = O - Lam
    frobenius_norm = math.sqrt(np.sum(Z**2))
    if frobenius_norm > epsilon:
        value = Z * (epsilon/frobenius_norm)
    else:
        value = Z
    
    return value