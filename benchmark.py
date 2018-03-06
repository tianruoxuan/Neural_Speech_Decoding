import numpy as np
from scipy import stats
import statistics

######################################
# Load in Data
######################################
def read_data(filename):
    print('Reading %s'%filename)
    f = open(filename)
    X = []
    while True:
        line = f.readline()
        if (not line):
            break
        sline = line.strip().split(',')
        row = [float(selem) for selem in sline]
        X.append(row[:])
    Xarr = np.array(X)
    f.close()
    return Xarr

Xtr = read_data('../dat/train_X_ecog.csv')
Ytr = read_data('../dat/train_Y_ecog.csv')
Xts = read_data('../dat/test_X_ecog.csv')


######################################
# Standardize and Center Data
######################################
X = stats.zscore(Xtr)
Xs = stats.zscore(Xts)
mu = Ytr.mean(axis=0)
Y = Ytr - mu

######################################
# Ridge Estimate & Predictions
######################################
lam = 1.0
XtXinv = np.linalg.inv(np.dot(X.transpose(),X)/len(Ytr) + lam*np.eye(70*6))
betar = np.dot(XtXinv, np.dot(X.transpose(),Y)/len(Ytr))
Yhtr = mu + np.dot(X,betar);
Yhts = mu + np.dot(Xs,betar)
