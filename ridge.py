import numpy as np
from scipy import stats
from sklearn import linear_model

######################################
# Load in Data
######################################
def read_data(filename):
    print('Reading %s'%filename)
    f = open(filename)
    X = []
    sk = 1
    while True:
        line = f.readline()
        if (not line):
            break
        sline = line.strip().split(',')
        row = [float(selem) for selem in sline]
        X.append(row[:])
        if (sk>5000):
            break
        sk += 1
    Xarr = np.array(X)
    f.close()
    return Xarr

def load_data():
    Xtr = read_data('../dat/train_X_ecog.csv')
    Ytr = read_data('../dat/train_Y_ecog.csv')
    Xts = read_data('../dat/test_X_ecog.csv')
    return Xtr, Ytr, Xts

def ridge(Xtr, Ytr, Xts, lam):
    Yts = np.zeros((len(Xts),32))
    X0 = stats.zscore(Xtr)
    X = np.append(X0, np.sin(X0/2), axis=1)
    X = np.append(X, np.cos(2*X0), axis=1)
    X = np.append(X, np.sin(X0), axis=1)
    X = np.append(X, np.cos(X0), axis=1)
    X = np.append(X, np.sin(1/(X0+1e-3)), axis=1)
    Xs0 = stats.zscore(Xts)
    Xs = np.append(Xs0, np.sin(Xs0/2), axis=1)
    Xs = np.append(Xs, np.cos(2*Xs0), axis=1)
    Xs = np.append(Xs, np.sin(Xs0), axis=1)
    Xs = np.append(Xs, np.cos(Xs0), axis=1)
    Xs = np.append(Xs, np.sin(1/(Xs0+1e-3)), axis=1)
    Y = Ytr
    clf = linear_model.Ridge (alpha = lam*len(Ytr), fit_intercept=True)
    clf.fit (X, Y) 
    betar = clf.coef_.transpose()
    mu = clf.intercept_
    Yts = mu + np.dot(Xs, betar)    
    return Yts

def cross_validation(Xtr, Ytr, lam=1):
    K = 5
    n = len(Xtr)
    n_fold = n//K
    RMSE = []
    RMSE_class = np.zeros(32)
    for k in range(0,K):
        start = k*n_fold
        if (k==K-1):
            end = n
        else:
            end = (k+1)*n_fold
        Xtr_cv = np.append(Xtr[0:start],Xtr[end:n],axis=0)
        Ytr_cv = np.append(Ytr[0:start],Ytr[end:n],axis=0)
        Xts_cv = Xtr[start:end]
        Yts_cv = Ytr[start:end]
        Y_pre = ridge(Xtr_cv, Ytr_cv, Xts_cv, lam)
        diff_cv = Yts_cv - Y_pre
        for i in range(0,32):            
            RMSE_class[i] += np.sum(diff_cv[:,i]**2)
            RMSE0 = np.sqrt(np.sum(diff_cv[:,i]**2)/diff_cv[:,i].size)
            #print('RMSE %d = %f'%(i,RMSE0))
        RMSE.append(np.sqrt(np.sum(diff_cv**2)/diff_cv.size))
        print('RMSE(%d) = %f'%(k,RMSE[k]))
    print('RMSE = %f'%(np.mean(RMSE)))
    
#    for i in range(0,32):
#        RMSE1 = np.sqrt(RMSE_class[i]/len(Ytr))
#        print('RMSE(class %d) = %f'%(i,RMSE1))

def gen_output(Xtr, Ytr, Xts, lam=1):
    Yts = ridge(Xtr, Ytr, Xts, lam)
    Yts = Yts.transpose().reshape(Yts.size)
    f = open('ridge_predicion.csv','w')
    f.write('Id,Prediction\n')
    for i in range(len(Yts)):
        f.write("%d,%.15f\n"%(i+1,Yts[i]))
    f.close()

Xtr, Ytr, Xts = load_data()
lam = 2.3
cross_validation(Xtr, Ytr, lam)
#gen_output(Xtr, Ytr, Xts, lam)
