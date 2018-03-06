import numpy as np
from scipy import stats
from sklearn import linear_model
import datetime

#################################################################
# Parameters
# all parameters are used as global variables
# 1. use_svd:
#    Use svd decomposition on Y, so Y=U*S*V. Instead of fitting
#    Y, we now fit U
# 2. alpha_set:
#    Alpha parameter in elastic net. Since there are total 32
#    species in the fitting data (32 columns in Y), we tune
#    parameters for each species
# 3. l1_ratio_set:
#    L1_raito parameter in elastic net. Similarly, we tune
#    parameters for all 32 species.
# 4. use_rbf:
#    For each species, if we want to use radial basis functions
#    to add non-linearity.
# 5. use_tri:
#    For each species, if we want to use trigonometric functions
#    to add non-linearity.
# 6. use_qua:
#    For each species, if we want to use quadratic polynomials
#    to add non-linearity.
# 7. partition:
#    Patition the data according to their position in time line
#################################################################
# Ture or False
use_svd = True

bps = [80, 379, 692, 1013, 1310, 1598, 1902, 2180, 2490, 2822,\
       3084, 3370, 3715, 4058, 4366, 4670, 4946, 5215, 5525, 5784, \
       6213, 6402, 6715, 6988, 7237, 7541, 7878, 8210, 8523, 8831,\
       9176, 9417, 9717, 9995, 10300, 10586, 10933, 11301, 11654, 11925,\
       12256, 12555, 12855, 13152, 13398, 13712, 14009, 14293, 14526, 14804, \
       15048, 15363, 15655, 15972, 16238, 16491, 16782, 17033, 17350, 17703, \
       17970, 18278, 18555, 18800, 19084, 19329, 19617, 19952, 20247, 20551, \
       20843, 21110, 21421, 21700, 22029, 22288, 22628, 22919, 23246, 23515, \
       23861, 24148, 24403, 24642, 24960, 25229, 25512, 25806, 26062, 26326, \
       26653, 26894, 27232, 27503, 27760, 28144, 28405, 28700, 28968, 29259, \
       29531, 29774, 30096, 30411, 30674, 30972, 31213, 31512, 31752, 32071, \
       32394, 32695, 32962, 33261, 33550, 33820, 34097, 34361, 34697, 35003, \
       35293, 35582, 35867, 36163, 36434, 36707, 36999, 37286, 37633, 37966, \
       38286, 38579, 38862, 39115, 39399, 39711, 39981, 40362, 40627, 40947]

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
    Xtr = read_data('../dat/train_X_ecog.csv')
    Ytr = read_data('../dat/train_Y_ecog.csv')
    Xts = read_data('../dat/test_X_ecog.csv')
    bptr = read_breakpoints('../dat/train_breakpoints.txt')
    bpts = read_breakpoints('../dat/test_breakpoints.txt')
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
#    Wts: testing W (Yts=Wts*Vtr)
#################################################################
def train_data(Xtr, Ytr, Xts, bptr, bpts):
    global use_svd

    # use svd to fit W instead of Y
    if (use_svd):
        Utr, Dtr, Vtr = np.linalg.svd(Ytr, full_matrices= False)
        Str = np.diag(Dtr)
        Wtr = np.dot(Utr, Str)
    else:
        Vtr = np.eye(32)
        Wtr = Ytr
        
    # fit each column and each pattern of W
    Wts = np.zeros((len(Xts),32))
    ntr = len(bptr)-1
    nts = len(bpts)-1
    for s_sentence in range(nts):
        s_start = bpts[s_sentence]
        s_end = bpts[s_sentence+1]
        for i in range(s_start, s_end):
            pos = float(i-s_start)/(s_end-s_start)
            for t_sentence in range(ntr):
                t_start = bptr[t_sentence]
                t_end = bptr[t_sentence+1]
                j = int(pos*(t_end-t_start)) + t_start                
                Wts[i,:] += Wtr[j,:]

    Wts /= ntr

    # recover Y by Y=W*V
    Yts = np.dot(Wts, Vtr)
    
    return Yts, Wts


#################################################################
# 5-set cross validation
# input:
#    Xall: training X
#    Yall: training Y
#    bpall: training breakpoints
#################################################################
def cross_validation(Xall, Yall, bpall):
    
    global use_svd
    
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

        Ypre, Wpre = train_data(Xtr, Ytr, Xts, bptr, bpts)

        # W=Y*inv(V)
        if (use_svd):
            _, _, Vtr = np.linalg.svd(Ytr, full_matrices= False)
        else:
            Vtr = np.eye(32)
        '''
        Wts = np.dot(Yts, np.linalg.inv(Vtr))
        Wdiff = Wts - Wpre
        for i in range(0,32):
            w_diff[i] += np.sum(Wdiff[:,i]**2)

        # difference
        Ydiff = Yts - Ypre
        '''
        
        # difference
        Ydiff = Yts - Ypre

        # RMSE
        RMSE.append(np.sqrt(np.sum(Ydiff**2)/Ydiff.size))
        print('RMSE(%d) = %f'%(k,RMSE[k]))

    #argw = np.argsort(w_diff)
    #argw = argw[::-1]
    #sumw = np.sum(w_diff)
    print('RMSE = %f'%(np.mean(RMSE)))
    #print('Top 3 Mismatch: (%d,%.1f%%), (%d,%.1f%%), (%d,%.1f%%)'%(argw[0], w_diff[argw[0]]/sumw*100, argw[1], w_diff[argw[1]]/sumw*100, argw[2], w_diff[argw[2]]/sumw*100))

    return np.mean(RMSE)


#################################################################
# generate output to be uploaded in Kaggle
#################################################################
def gen_output(Xtr, Ytr, Xts, bptr, bpts):

    Yts,_ = train_data(Xtr, Ytr, Xts, bptr, bpts)

    # write predicted data
    Yts = Yts.transpose().reshape(Yts.size)
    time = datetime.datetime.utcnow()
    f = open('average_y_predicion_%02d%02d%02d%02d.csv'%(time.month,time.day,time.hour,time.minute),'w')
    f.write('Id,Prediction\n')
    for i in range(len(Yts)):
        f.write("%d,%.15f\n"%(i+1,Yts[i]))
    f.close()


#################################################################
# main function
# load data, cross validation or generate output
#################################################################
Xtr, Ytr, Xts, bptr, bpts = load_data()

#cross_validation(Xtr, Ytr, bptr)
gen_output(Xtr, Ytr, Xts, bptr, bpts)
