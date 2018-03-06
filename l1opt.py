'''
STAT 640 HW1 (640 only pb1)
Coordinate Descent Solver for LASSO Regression
Language: Python 3.0
'''

import numpy as np
from sklearn import linear_model

# eps: numerical tolerence
eps   = 1e-7

def get_obj(X,Y,lam,betal):
    '''
    get value of objective function
    the objective function is given by
    f = (1/2*n)*||Y- X*beta||_2^2 + lambda * ||beta||_1
    '''
    n = len(Y)
    res = Y - np.dot(X, betal)
    regular = np.dot(res.transpose(),res)/(2*n)
    penalty = lam*sum(abs(betal))
    return (regular + penalty)[0,0]

def soft_threshold(lam, b):
    '''
    soft threshold function
    it is used to get the optimal value for 1D Lasso
    '''
    if (b>lam):
        return b - lam
    elif (b<-lam):
        return b + lam
    else:
        return 0

def fit_Lasso(X,Y,lam):
    '''
    fit Lasso regression
    input: X, Y, lam
    output: solution beta
    '''
    # get problem size
    (n,p) = np.shape(X)
    assert(np.shape(Y)==(n,1))
    # use Ridge solution as an initial guess
    invXtX = np.linalg.inv(np.dot(X.transpose(),X) + lam*np.eye(p))
    betal = np.dot(invXtX, np.dot(X.transpose(),Y))
    obj = get_obj(X,Y,lam,betal)
    # set maximum iteration
    maxIter = 1000
    # apply coordinate descent method
    for it in range(0,maxIter):
        # each iteration, minimize one variable
        for i in range(0,p):
            Ai = X[:,i]
            Ai = Ai.reshape(n,1)
            sumcol = np.zeros(n)
            for j in range(0,p):
                if (j != i):
                    Aj = X[:,j]
                    sumcol += betal[j]*Aj
            sumcol = sumcol.reshape(n,1)
            sumcol = Y - sumcol
            AtA = np.dot(Ai.transpose(), Ai)[0,0]
            AtAy = np.dot(Ai.transpose(), sumcol)[0,0]
            b = AtAy/AtA
            betal[i] = soft_threshold(lam*n/AtA, b)
        # evaluate new objective function and use this as
        # stopping criteria
        obj_old = obj
        obj = get_obj(X,Y,lam,betal)
        if (obj_old - obj < eps):
            return betal
    return betal

def main():
    # take n=50, p = 10 for example
    n = 50
    p = 5
    # generate random vectors
    X = np.random.rand(n,p)
    Y = np.random.rand(n,1)
    lam = 0.05
    # use packages in sklearn
    clf = linear_model.Lasso (alpha = lam, fit_intercept=False)
    clf.fit (X, Y)
    betal_sklearn = clf.coef_
    betal_sklearn = betal_sklearn.reshape((p,1))
    obj_sklearn = get_obj(X,Y,lam,betal_sklearn)
    # use our own solver to solve Lasso
    betal_ours = fit_Lasso(X,Y,lam)
    obj_ours = get_obj(X,Y,lam,betal_ours)
    diff = np.linalg.norm(betal_sklearn - betal_ours)/np.linalg.norm(betal_sklearn)
    print('='*10 + ' [Solutions] ' + '='*27)
    print('Sulotion from sklearn:')
    print(betal_ours)
    print('-'*50)
    print('Sulotion from our solver:')
    print(betal_sklearn)
    print('='*10 + ' [Summary] ' + '='*29)
    print('Objective function from sklearn   : %f'%obj_sklearn)
    print('Objective function from our solver: %f'%obj_ours)
    print('Difference between the solutions  : %f'%diff)
    print('='*50)

       
if __name__=='__main__':
    main()
