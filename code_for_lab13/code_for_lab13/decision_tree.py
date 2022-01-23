import pdb
import numpy as np
import operator
import copy
import csv

# Binary (+1, -1) prediction.  This assumes that the data points are
# the ROWS of the matrix!!  The last column of the data matrix are the
# labels.

class DTN:
    N_THRESHOLD = 3 # don't split if node has fewer examples than this
    H_THRESHOLD = .04 # don't split if node has entropy less than this
    H_REDUCTION_THRESHOLD = .01 # don't split if it doesn't reduce H by this
    index = 0
    def __init__(self, data=None, config = None):
        self.config = config
        if config != None:
            self.N_THRESHOLD = config[0]
            self.H_THRESHOLD = config[1]
            self.H_REDUCTION_THRESHOLD = config[2]
        DTN.index += 1
        self.index = DTN.index          # unique number
        self.data = data                # store the data
        self.p = None                   # prob of positive - look at set_h
        if data is not None:
            self.n = float(data.shape[0]) # number of data points
            self.indices = range(data.shape[1]-1) # feature indices
            self.set_h()                           # compute entropy
        self.splits = {}
        # The test is data[:,fi] < th
        self.fi = None                  # feature index
        self.th = None                  # threshold
        self.lc = None                  # left child
        self.rc = None                  # right child
        self.parent = None              # parent

    # Create split on feat i at value th
    def split(self, i, th):
        self.fi = i
        self.th = th
        self.lc = DTN(d[d[:, i] < th], config = self.config)
        self.rc = DTN(d[d[:, i] >= th], config = self.config)
        self.splits[i].remove(th)

    # Evalute candidate split by weighted average entropy
    def split_eval(self, i, th):
        lc = DTN(self.data[self.data[:, i] < th], config = self.config)
        rc = DTN(self.data[self.data[:, i] >= th], config = self.config)
        pl = lc.n / self.n
        pr = 1.0 - pl
        avgH = pl * lc.h + pr * rc.h
        return avgH, lc, rc

    # Entropy of class labels in this node, assumes 1, -1
    def set_h(self):
        b = .001
        npos = np.sum(self.data[:,-1] == 1) # count labels = 1
        p = (npos + b) / (self.n + b + b)
        self.p = p
        self.h = - (p * np.log(p) + (1 - p) * np.log(1 - p))

    def build_tree(self):
        if self.h < self.H_THRESHOLD or self.n <= self.N_THRESHOLD:
            return
        # Find the best split
        (i, th, (h, lc, rc)) = argmax([(i, th, self.split_eval(i, th)) \
                                                   for i in self.indices \
                                                   for th in self.get_splits(i)],
                                      lambda x: -x[2][0]) # x=(a,b,(h,c,d))
         
        if (self.h - h) < self.H_REDUCTION_THRESHOLD:
            return
        # Curse again!
        self.fi = i
        self.th = th
        self.lc = lc
        self.rc = rc
        self.lc.parent = self
        self.rc.parent = self
        self.lc.build_tree()
        self.rc.build_tree()

    # The candidate splits between the values in the training set.
    def get_splits(self, i):
        if i not in self.splits:
            d = np.sort(self.data[:,i], axis = None)
            d1 = d[:-1]
            d2 = d[1:]
            self.splits[i] = (d1 + d2) / 2.0
        return self.splits[i]

    # Classify a data point
    def classify(self, x):
        if self.fi == None:             # lead
            return self.p               # prob of posistive
        elif x[self.fi] < self.th:
            return self.lc.classify(x)  # satisfies test, left branch
        else:
            return self.rc.classify(x)  # fails test, right branch

    def display(self, depth=0, max_depth=3):
        if depth > max_depth:
            print(depth*'  ', 'Depth >', max_depth)
            return
        if self.fi is None:
            print(depth*'  ', '=>', "%.2f"%self.p, '[', 'n=', self.n, ']')
            return
        print(depth*'  ', 'Feat', self.fi, '<', self.th, '[', 'n=', self.n, ']')
        self.lc.display(depth+1, max_depth)
        self.rc.display(depth+1, max_depth)

def DT(X, Y, config = None):
    D = np.hstack([X, Y])               # points are rows of X
    root = DTN(D, config = config)
    root.build_tree()
    return root

def classification_error_DT(dt, X, Y):
    pred = np.array([np.apply_along_axis(dt.classify,1,X)]).T - 0.5 # predicts +,-
    return np.mean(np.sign(Y * pred) > 0.0)

# Evaluate on a train/test split
def dt_eval(trainingX, trainingY, testX, testY, max_depth=5, verbose=True, config = None):
    dt = DT(trainingX, trainingY, config = config)
    acc = classification_error_DT(dt, testX, testY)
    if verbose:
        dt.display(max_depth=max_depth)
        print('Test accuracy', acc)
    return acc    

def argmax(l, f):
    """
    @param l: C{List} of items
    @param f: C{Procedure} that maps an item into a numeric score
    @returns: the element of C{l} that has the highest score
    """
    vals = [f(x) for x in l]
    return l[vals.index(max(vals))]

######################################################################
# For auto dataset (same as in HW 2, except returns data in rows)
######################################################################

def load_auto_data(path_data='auto-mpg.tsv'):
    """
    Returns a list of dict with keys:
    """
    numeric_fields = {'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                      'acceleration', 'model_year', 'origin'}
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            for field in list(datum.keys()):
                if field in numeric_fields and datum[field]:
                    datum[field] = float(datum[field])
            data.append(datum)
    return data

# Feature transformations
def std_vals(data, f):
    vals = [entry[f] for entry in data]
    avg = sum(vals)/len(vals)
    dev = [(entry[f] - avg)**2 for entry in data]
    sd = (sum(dev)/len(vals))**0.5
    return (avg, sd)

def standard(v, std):
    return [(v-std[0])/std[1]]

def raw(x):
    return [x]

def one_hot(v, entries):
    vec = len(entries)*[0]
    vec[entries.index(v)] = 1
    return vec

# The class (mpg) added to the front of features (points are rows)
def auto_data_and_labels(auto_data, features):
    features = [('mpg', raw)] + features
    std = {f:std_vals(auto_data, f) for (f, phi) in features if phi==standard}
    entries = {f:list(set([entry[f] for entry in auto_data])) \
               for (f, phi) in features if phi==one_hot}
    print('avg and std', std)
    print('entries in one_hot field', entries)
    findex = 0
    # Print the meaning of the features
    for (f, phi) in features[1:]:
        if phi == standard:
            print(findex, f, 'std')
            findex += 1
        elif phi == one_hot:
            for entry in entries[f]:
                print(findex, f, entry, 'one_hot')
                findex += 1
        else:
            print(findex, f, 'raw')
            findex += 1
    vals = []
    for entry in auto_data:
        phis = []
        for (f, phi) in features:
            if phi == standard:
                phis.extend(phi(entry[f], std[f]))
            elif phi == one_hot:
                phis.extend(phi(entry[f], entries[f]))
            else:
                phis.extend(phi(entry[f]))
        vals.append(np.array([phis]))
    data_labels = np.vstack(vals)
    np.random.seed(0)
    np.random.shuffle(data_labels)
    return data_labels[:, 1:], data_labels[:, 0:1]

# USE FOR QUESTION 1.1 AND PARAMETER TUNING
# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are standard and one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', one_hot),
            ('displacement', standard),
            ('horsepower', standard),
            ('weight', standard),
            ('acceleration', standard),
            ## Drop model_year by default
            ## ('model_year', raw),
            ('origin', one_hot)]

'''
# USE FOR QUESTION 1.2
# A small feature set
features = [('weight', standard),
            ('origin', one_hot)]
'''

'''
# USE FOR QUESTION 1.3
# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are standard and one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('weight', raw),
            ('origin', raw)]
'''

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = load_auto_data('auto-mpg.tsv')

# Construct the standard data and label arrays
auto_data, auto_labels = auto_data_and_labels(auto_data_all, features)

######################################################################
# Apply the decision tree to the auto data
######################################################################

# Run a single train/test split
def auto_test(pct=0.25):                # pct is for test
    X = auto_data
    Y = auto_labels
    (n, d) = X.shape
    indices = np.random.permutation(n)  # randomize the data set
    tx = int((1-pct)*n)                 # size of training split
    training_idx, test_idx = indices[:tx], indices[tx:]
    trainingX, testX = X[training_idx,:], X[test_idx,:]
    trainingY, testY = Y[training_idx,:], Y[test_idx,:]
    return dt_eval(trainingX, trainingY, testX, testY)

# Cross validate with k folds
def auto_xval(k=10, verbose = False, config = None):
    indices = np.random.permutation(auto_data.shape[0])
    X = auto_data[indices,:]
    Y = auto_labels[indices,:]
    s_data = np.array_split(X, k, axis=0)
    s_labels = np.array_split(Y, k, axis=0)
    score_sum = 0
    for i in range(k):
        data_train = np.concatenate(s_data[:i] + s_data[i+1:], axis=0)
        labels_train = np.concatenate(s_labels[:i] + s_labels[i+1:], axis=0)
        data_test = np.array(s_data[i])
        labels_test = np.array(s_labels[i])
        score_sum += dt_eval(data_train, labels_train, data_test, labels_test, verbose = verbose, config = config)
    print('Xval accuraracy', score_sum/k)
    return score_sum/k

print('Loaded decision_tree.py')
print(auto_xval())
