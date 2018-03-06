'''
STAT 640 2016 Fall: Kaggle Competition
Team: LuckyCat
Members: Ruoxuan Tian, Ting Qi
'''

#################################################################
# Libraries
#################################################################
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import datetime


#################################################################
# Read data from given file
#################################################################
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


#################################################################
# Read breakpoints from given file
#################################################################
def read_breakpoints(filename):
    print('Reading %s'%filename)
    f = open(filename)
    bp = [0]
    while True:
        line = f.readline()
        if (not line):
            break
        bp.append(int(line))
    bp_arr = np.array(bp)
    f.close()
    return bp_arr


#################################################################
# Load all of the data
#################################################################
def load_data():
    Xtr = read_data('train_X_ecog.csv')
    Ytr = read_data('train_Y_ecog.csv')
    Xts = read_data('test_X_ecog.csv')
    bptr = read_breakpoints('train_breakpoints.txt')
    bpts = read_breakpoints('test_breakpoints.txt')
    return Xtr, Ytr, Xts, bptr, bpts


#################################################################
# Train data and return regression result
# input:
#    Xtr: training X
#    Ytr: training Y
#    Xts: testing X
#    bptr: training breakpoints
#    bpts: testing breakpoints
# output:
#    Yts: testing Y
#################################################################
def train_data(Xtr, Ytr, Xts, bptr, bpts):

    # project X to 1-dimensional data
    def projectX(X):
        return X[:,221] + X[:,151]

    # get Pearson similarity for two vectors
    def get_similarity(X1, X2):
        n1 = len(X1)
        n2 = len(X2)
        a_avg = np.mean(X1)
        b_avg = np.mean(X2)
        ab = a2 = b2 = 0
        for i in range(n1):
            j = int(float(i)/n1*n2)
            ab += (X1[i]-a_avg)*(X2[j]-b_avg)
            a2 += (X1[i]-a_avg)**2
            b2 += (X2[j]-b_avg)**2
        sim = ab/(sqrt(a2)*sqrt(b2)) + 1
        return sim

    Wtr = projectX(Xtr)
    Wts = projectX(Xts)

    ntr = len(bptr)-1
    nts = len(bpts)-1

    # pre-compute weight table
    weightTable = np.zeros((nts,ntr))
    for ts_sentence in range(nts):
        ts_start = bpts[ts_sentence]
        ts_end = bpts[ts_sentence+1]
        X1Dts = Wts[ts_start:ts_end]
        for tr_sentence in range(ntr):
            tr_start = bptr[tr_sentence]
            tr_end = bptr[tr_sentence+1]
            X1Dtr = Wtr[tr_start:tr_end]
            sim = get_similarity(X1Dts, X1Dtr)
            weightTable[ts_sentence,tr_sentence] = sim
        weightTable[ts_sentence,:] /= sum(weightTable[ts_sentence,:])

    # take weighted average
    Nstencil = 2
    Yts = np.zeros((len(Xts),32))
    for ts_sentence in range(nts):
        ts_start = bpts[ts_sentence]
        ts_end = bpts[ts_sentence+1]
        for i in range(ts_start, ts_end):
            pos = float(i-ts_start)/(ts_end-ts_start)
            for tr_sentence in range(ntr):
                tr_start = bptr[tr_sentence]
                tr_end = bptr[tr_sentence+1]
                j = int(pos*(tr_end-tr_start)) + tr_start
                temp = np.zeros(32)
                for k in range(j-Nstencil, j+Nstencil+1):
                    kk = min(max(k,tr_start),tr_end-1)
                    temp += Ytr[kk,:]
                Yts[i,:] += temp*weightTable[ts_sentence,tr_sentence]/(2*Nstencil+1)
    
    return Yts


#################################################################
# 5-set cross validation
# input:
#    Xall: training X
#    Yall: training Y
#    bpall: training breakpoints
#################################################################
def cross_validation(Xall, Yall, bpall):

    print('Running cross validation')
    
    K = 5
    n_data = len(Xall)
    n_sentence = len(bpall) - 1
    n_fold = n_sentence//K
    RMSE = []
    RMSE_class = np.zeros(32)
    w_diff = [0]*32
    for k in range(0,K):
        start_sentence = k*n_fold
        if (k==K-1):
            end_sentence = n_sentence
        else:
            end_sentence = (k+1)*n_fold
        start_data = bpall[start_sentence]
        end_data = bpall[end_sentence]

        Xtr = np.append(Xall[:start_data],Xall[end_data:],axis=0)
        Ytr = np.append(Yall[:start_data],Yall[end_data:],axis=0)
        Xts = Xall[start_data:end_data]
        Yts = Yall[start_data:end_data]
        bptr = [0]
        bpts = [0]
        for i in range(0,start_sentence):
            bptr += [bptr[-1] + bpall[i+1] - bpall[i]]
        for i in range(start_sentence, end_sentence):
            bpts += [bpts[-1] + bpall[i+1] - bpall[i]]
        for i in range(end_sentence,n_sentence):
            bptr += [bptr[-1] + bpall[i+1] - bpall[i]]

        Ypre = train_data(Xtr, Ytr, Xts, bptr, bpts)

        # function to visualize result
        def plotRes(sid, bp, Y0, Y1):
            U, D, V = np.linalg.svd(Y0, full_matrices= False)
            S = np.diag(D)
            W0 = np.dot(U, S)
            W1 = Y1.dot(np.linalg.inv(V))
            plt.figure()
            t = np.arange(bp[sid],bp[sid+1])
            plt.plot(t, W0[t,0], 'b-o', label='True Data')
            plt.plot(t, W1[t,0], 'r-o', label='Fitted Data')
            plt.legend(loc=4)
            plt.show()
        
        #plotRes(0,bpts,Yts,Ypre)
        
        # difference
        Ydiff = Yts - Ypre

        # RMSE
        RMSE.append(np.sqrt(np.sum(Ydiff**2)/Ydiff.size))
        print('RMSE(Fold %d) = %f'%(k,RMSE[k]))

    print('RMSE(Average) = %f'%(np.mean(RMSE)))

    return np.mean(RMSE)


#################################################################
# generate output to be uploaded in Kaggle
#################################################################
def gen_output(Xtr, Ytr, Xts, bptr, bpts):

    print('Generating output for Kaggle')
    
    Yts = train_data(Xtr, Ytr, Xts, bptr, bpts)

    # write predicted data
    Yts = Yts.transpose().reshape(Yts.size)
    time = datetime.datetime.utcnow()
    f = open('weighted_average_predicion_%02d%02d%02d%02d.csv'%(time.month,time.day,time.hour,time.minute),'w')
    f.write('Id,Prediction\n')
    for i in range(len(Yts)):
        f.write("%d,%.15f\n"%(i+1,Yts[i]))
    f.close()


#################################################################
# main function
# load data, cross validation or generate output
#################################################################
Xtr, Ytr, Xts, bptr, bpts = load_data()

cross_validation(Xtr, Ytr, bptr)

gen_output(Xtr, Ytr, Xts, bptr, bpts)
