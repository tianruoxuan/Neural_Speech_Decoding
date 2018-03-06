import numpy as np
from scipy import stats
from sklearn import linear_model
import matplotlib.pyplot as plt
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

# positive number
alpha_set = [1.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
             1.0, 1.0]

# number between 0 and 1
l1_ratio_set = [0.8, 0.4, 0.6, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, \
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, \
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, \
                0.5, 0.5]
# True or False
use_rbf = [True, True, True, False, False, False, False, False, False, False, \
           False, False, False, False, False, False, False, False, False, False, \
           False, False, False, False, False, False, False, False, False, False, \
           False, False]

# True or False
use_tri = [True, True, True, False, False, False, False, False, False, False, \
           False, False, False, False, False, False, False, False, False, False, \
           False, False, False, False, False, False, False, False, False, False, \
           False, False]

# True or False
use_qua = [True, True, True, False, False, False, False, False, False, False, \
           False, False, False, False, False, False, False, False, False, False, \
           False, False, False, False, False, False, False, False, False, False, \
           False, False]

# Integers between 0 and 240
important_var = [2, 142, 212, 83] #2, 142, 212, 83

# True or False
use_partition = [True, True, True, False, False, False, False, False, False, False, \
                 False, False, False, False, False, False, False, False, False, False, \
                 False, False, False, False, False, False, False, False, False, False, \
                 False, False]

# partition of [0,1] interval
partition = [ [[0,0.13],[0.13,0.94],[0.94,1]], \
              [[0,0.14],[0.14,0.96],[0.96,1]], \
              [[0,0.17],[0.17,1]] ]


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
# Process X before regression
# input:
#    X_raw: raw data
# output:
#    X : processed data
# if use_rbf is turned on, radial basis functions will be added
# if use_tri is turned on, trigonometric functions will be added
#################################################################
def process_x(X_raw, kind):
    global use_rbf
    global use_tri
    global use_qua
    global important_var

    # centralize and scale X
    X0 = stats.zscore(X_raw)

    # dimension of data
    n = X0.shape[0]
    p = X0.shape[1]

    # linear part
    X = X0

    # use_rbf for this species
    if (use_rbf[kind]):
        width = 0.4
        xi_set = np.arange(-2,2.1,width)
        n_points = len(xi_set)
        rbf_shape = (n, p*n_points)
        Xrbf = np.zeros(rbf_shape)
        for i in range(0,n_points):
            xi = xi_set[i]
            Xrbf[:,i*p:(i+1)*p] = np.exp(-(X0-xi)**2/(2*width**2))
        X = np.append(X, Xrbf, axis=1)

    # use_tri for this species
    if (use_tri[kind]):
        n_phase = 2
        period = 4.0
        tri_shape = (n, p*n_phase*2)
        Xtri = np.zeros(tri_shape)
        for i in range(0,n_phase):
            Xtri[:,2*i*p:(2*i+1)*p] = np.sin(X0*(i+1)*np.pi/period)
            Xtri[:,(2*i+1)*p:(2*i+2)*p] = np.cos(X0*(i+1)*np.pi/period)
        X = np.append(X, Xtri, axis=1)

    # use_qua for this species
    if (use_qua[kind]):
        n_var = len(important_var)
        for i in range(0, n_var-1):
            indi = important_var[i]
            j = i+1
            indj = important_var[j]
            xij = X0[:,indi]*X0[:,indj]
            X = np.append(X, xij.reshape(-1,1), axis=1)

    return X


#################################################################
# Elastic net regression
# input:
#    Xtr: training X
#    Ytr: training Y
#    Xts: testing X
#    alpha_in: parameter alpha in elastic net
#    l1_ratio_in: parameter l1_ratio in elastic net
#    kind: species to be fit
# output:
#    Yts: fitted testing Y
#################################################################
def elastic_net(Xtr, Ytr, Xts, alpha_in, l1_ratio_in, kind):

    X  = process_x(Xtr,kind)
    Xs = process_x(Xts,kind)
    Y  = Ytr
    clf = linear_model.ElasticNet(alpha=alpha_in,l1_ratio=l1_ratio_in,fit_intercept=True)
    clf.fit (X, Y) 
    Yts = clf.predict(Xs)
    return Yts


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
    global alpha_set
    global l1_ratio_set
    global partition
    global use_partition

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
    for kind in range(0,32):
        if (use_partition[kind]):
            for pattern in partition[kind]:
                # decompose tranning data into patterns
                for i in range(0,len(bptr)-1):
                    sp = int(bptr[i] + (bptr[i+1]-bptr[i])*pattern[0])
                    ep = int(bptr[i] + (bptr[i+1]-bptr[i])*pattern[1])
                    if (i==0):
                        Xtrp = Xtr[sp:ep]
                        Wtrp = Wtr[sp:ep]
                    else:
                        Xtrp = np.append(Xtrp, Xtr[sp:ep], axis=0)
                        Wtrp = np.append(Wtrp, Wtr[sp:ep], axis=0)
                # decompose testing data into patterns
                for i in range(0,len(bpts)-1):
                    sp = int(bpts[i] + (bpts[i+1]-bpts[i])*pattern[0])
                    ep = int(bpts[i] + (bpts[i+1]-bpts[i])*pattern[1])
                    if (i==0):
                        Xtsp = Xts[sp:ep]
                    else:
                        Xtsp = np.append(Xtsp, Xts[sp:ep], axis=0)
                Wtsp = elastic_net(Xtrp, Wtrp[:,kind], Xtsp, alpha_set[kind], l1_ratio_set[kind], kind)
                pos = 0
                for i in range(0,len(bpts)-1):
                    sp = int(bpts[i] + (bpts[i+1]-bpts[i])*pattern[0])
                    ep = int(bpts[i] + (bpts[i+1]-bpts[i])*pattern[1])
                    lp = ep - sp
                    Wts[sp:ep,kind] = Wtsp[pos:pos+lp]
                    pos += lp
            
        else:
            Wts[:,kind] = elastic_net(Xtr, Wtr[:,kind], Xts, alpha_set[kind], l1_ratio_set[kind], kind)
    
    # take care of the initial jump of species 0
    initJump = 0
    for i in bptr[:-1]:
        initJump += (Wtr[i+1,0] - Wtr[i,0])
    initJump /= len(bptr)-1
    for i in bpts[:-1]:
        Wts[i,0] = Wts[i+1,0] - initJump

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

    K = 5
    n_data = len(Xall)
    n_sentence = len(bpall) - 1
    n_fold = n_sentence//K
    RMSE = []
    RMSE_class = np.zeros(32)
    w_diff = [0]*32
    for k in range(0,1):
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
        
        plotRes(0,bpts,Yts,Ypre)

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
    f = open('elastic_net_predicion_%02d%02d%02d%02d.csv'%(time.month,time.day,time.hour,time.minute),'w')
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
#gen_output(Xtr, Ytr, Xts, bptr, bpts)
