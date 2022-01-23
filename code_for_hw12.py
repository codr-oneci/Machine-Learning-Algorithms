import pdb
import numpy as np
import time

genres = ['Western', 'Comedy', 'Children', 'Crime', 'Musical', 'Adventure', 'Drama', 'Horror', 'War', 'Documentary', 'Romance', 'Animation', 'Film-Noir', 'Sci-Fi', 'Mystery', 'Fantasy', 'IMAX', 'Action', 'Thriller']

# Data is a list of (i, j, r) triples
ratings_small = \
[(0, 0, 5), (0, 1, 3), (0, 3, 1),
 (1, 0, 4), (1, 3, 1), 
 (2, 0, 1), (2, 1, 1), (2, 3, 5), 
 (3, 0, 1), (3, 3, 4), 
 (4, 1, 1), (4, 2, 5), (4, 3, 4)]

def pred(data, x):
    (a, i, r) = data
    (u, b_u, v, b_v) = x
    return np.dot(u[a].T,v[i]) + b_u[a] + b_v[i]

# Utilities
import pickle
# After retrieving the output x from mf_als, you can use this function to save the output so
# you don't have to re-train your model
def save_model(x):
    pickle.dump(x, open("ALSmodel", "wb"))

# After training and saving your model once, you can use this function to retrieve the previous model
def load_model():
    x = pickle.load(open("ALSmodel", "rb"))
    return x

# Compute the root mean square error
def rmse(data, x):
    error = 0.
    for datum in data:
        error += (datum[-1] - pred(datum, x))**2
    return np.sqrt(error/len(data))

# Counts of users and movies, used to calibrate lambda
def counts(data, index):
    item_count = {}
    for datum in data:
        j = datum[index]
        if j in item_count:
            item_count[j] += 1
        else:
            item_count[j] = 1
    c = np.ones(max(item_count.keys())+1)
    for i,v in item_count.items(): c[i]=v
    return c

# The ALS outer loop
def mf_als(data_train, data_validate, k=2, lam=0.02, max_iter=100, verbose=False):
    # size of the problem
    n = max(d[0] for d in data_train)+1 # users
    m = max(d[1] for d in data_train)+1 # items
    # which entries are set in each row and column
    us_from_v = [[] for i in range(m)]  # II (i-index-set)
    vs_from_u = [[] for a in range(n)]  # AI (a-index set)
    for (a, i, r) in data_train:
        us_from_v[i].append((a, r))
        vs_from_u[a].append((i, r))
    # Initial guess at u, b_u, v, b_v
    # Note that u and v are lists of column vectors (columns of U, V).
    x = ([np.random.normal(1/k, size=(k,1)) for a in range(n)],
          np.zeros(n),
          [np.random.normal(1/k, size=(k,1)) for i in range(m)],
          np.zeros(m))
    # Alternation, modifies the contents of x
    start_time = time.time()
    for i in range(max_iter):
        update_U(data_train, vs_from_u, x, k, lam)
        update_V(data_train, us_from_v, x, k, lam)
        if verbose:
            print('train rmse', rmse(data_train, x), 'validate rmse', data_validate and rmse(data_validate, x))
        if data_validate == None: # code is slower, print out progress
            print("Iteration {} finished. Total Elapsed Time: {:.2f}".format(i + 1, time.time() - start_time))
    # The root mean square errors measured on validate set
    if data_validate != None:
        print('validate rmse=', rmse(data_validate, x))
    return x

# X : n x k
# Y : n
def ridge_analytic(X, Y, lam):
    (n, k) = X.shape
    xm = np.mean(X, axis = 0, keepdims = True)   # 1 x n
    ym = np.mean(Y)                              # 1 x 1
    Z = X - xm                                   # d x n
    T = Y - ym                                   # 1 x n
    th = np.linalg.solve(np.dot(Z.T, Z) + lam * np.identity(k), np.dot(Z.T, T))
    # th_0 account for the centering
    th_0 = (ym - np.dot(xm, th))                 # 1 x 1
    return th.reshape((k,1)), float(th_0)

# Example from lab handout
Z = np.array([[1], [1], [5], [1], [5], [5], [1]])
b_v = np.array([[3], [3], [3], [3], [3], [5], [1]])
B = np.array([[1, 10], [1, 10], [10, 1], [1, 10], [10, 1], [5, 5], [5, 5]])
# Solution with offsets, using ridge_analytic provided in code file
u_a, b_u_a = ridge_analytic(B, (Z - b_v), 1)
print('With offsets', u_a, b_u_a)
# Solution using previous model, with no offsets
u_a_no_b = np.dot(np.linalg.inv(np.dot(B.T, B) + 1 * np.identity(2)), np.dot(B.T, Z))
print('With no offsets', u_a_no_b)

def update_U(data, vs_from_u, x, k, lam):
    (u, b_u, v, b_v) = x
    for a in range(len(u)):
        if not vs_from_u[a]: continue
        V = np.hstack([v[i] for (i, _) in vs_from_u[a]]).T
        y = np.array([r-b_v[i] for (i, r) in vs_from_u[a]])
        u[a], b_u[a] = ridge_analytic(V, y, lam)
    return x

def update_V(data, us_from_v, x, k, lam):
    (u, b_u, v, b_v) = x
    for a in range(len(v)):
        if not us_from_v[a]: continue
        V = np.hstack([u[i] for (i, _) in us_from_v[a]]).T
        y = np.array([r-b_u[i] for (i, r) in us_from_v[a]])
        v[a], b_v[a] = ridge_analytic(V, y, lam)
    return x

# Simple test case
# mf_als(ratings_small, ratings_small,lam=0.01, max_iter=10, k=2)

# The SGD outer loop
def mf_sgd(data_train, data_validate, step_size_fn, k=2, lam=0.02, max_iter=100, verbose=False):
    # size of the problem
    ndata = len(data_train)
    n = max(d[0] for d in data_train)+1
    m = max(d[1] for d in data_train)+1
    # Distribute the lambda among the users and items
    lam_uv = lam/counts(data_train,0), lam/counts(data_train,1)
    # Initial guess at u, b_u, v, b_v (also b)
    x = ([np.random.normal(1/k, size=(k,1)) for j in range(n)],
         np.zeros(n),
         [np.random.normal(1/k, size=(k,1)) for j in range(m)],
         np.zeros(m))
    di = int(max_iter/10.)
    for i in range(max_iter):
        if i%di == 0 and verbose:
            print('i=', i, 'train rmse=', rmse(data_train, x),
                  'validate rmse', data_validate and rmse(data_validate, x))
        step = step_size_fn(i)
        j = np.random.randint(ndata)            # pick data item
        sgd_step(data_train[j], x, lam_uv, step) # modify x
    print('k=', k, 'rmse', rmse(data_validate, x))
    return x

def sgd_step(data, x, lam, step):
    (a, i, r) = data
    (u, b_u, v, b_v) = x
    (lam_u, lam_v) = lam
    # predicted rating
    pred = np.dot(u[a].T,v[i]) + b_u[a] + b_v[i]
    # prediction error
    e = r - pred
    temp_u = u[a]           # so we don't update v with updated u
    u[a] = u[a] - step*(-e*v[i] + lam_u[a]*u[a])
    b_u[a] = b_u[a] - step*(-e)
    v[i] = v[i] - step*(-e*temp_u + lam_v[i]*v[i])
    b_v[i] = b_v[i] - step*(-e)
    return x

# Simple validate case
print("SGD")
mf_sgd(ratings_small, ratings_small, step_size_fn=lambda i: 0.1,
       lam=0.01, max_iter=1000, k=2)

def load_ratings_data_small(path_data='ratings.csv'):
    """
    Returns two lists of triples (i, j, r) (training, validate)
    """
    # we want to "randomly" sample but make it deterministic
    def user_hash(uid):
        return 71 * uid % 401
    def user_movie_hash(uid, iid):
        return (17 * uid + 43 * iid) % 61
    
    data_train = []
    data_validate = []
    with open(path_data) as f_data:
        for line in f_data:
            (uid, iid, rating, timestamp) = line.strip().split(",")
            h1 = user_hash(int(uid))
            if h1 <= 40:
                h2 = user_movie_hash(int(uid), int(iid))
                if h2 <= 12:
                    data_validate.append([int(uid), int(iid), float(rating)])
                else:
                    data_train.append([int(uid), int(iid), float(rating)])
                                     
    print('Loading from', path_data, 
          'users_train', len(set(x[0] for x in data_train)), 
          'items_train', len(set(x[1] for x in data_train)),
          'users_validate', len(set(x[0] for x in data_validate)), 
          'items_validate', len(set(x[1] for x in data_validate)))
    return data_train, data_validate

def load_ratings_data(path_data='ratings.csv'):
    """
    Returns a list of triples (i, j, r)
    """
    data = []
    with open(path_data) as f_data:
        for line in f_data:
            (uid, iid, rating, timestamp) = line.strip().split(",")
            data.append([int(uid), int(iid), float(rating)])

    print('Loading from', path_data, 
          'users', len(set(x[0] for x in data)), 
          'items', len(set(x[1] for x in data)))
    return data

def load_movies(path_movies='movies.csv'):
    """
    Returns a dictionary mapping item_id to item_name and another dictionary
    mapping item_id to a list of genres
    """
    data = {}
    genreMap = {}
    with open(path_movies, encoding = "utf8") as f_data:
        for line in f_data:
            parts = line.strip().split(",")
            item_id = int(parts[0])
            item_name = ",".join(parts[1:-1]) # file is poorly formatted
            item_genres = parts[-1].split("|")
            
            data[item_id] = item_name
            genreMap[item_id] = item_genres
    return data, genreMap

def baseline(train, validate):
    item_sum = {}
    item_count = {}
    total = 0
    for (i, j, r) in train:
        total += r
        if j in item_sum:
            item_sum[j] += 3
            item_count[j] += 1
        else:
            item_sum[j] = r
            item_count[j] = 1
    error = 0
    avg = total/len(train)
    for (i, j, r) in validate:
        pred = item_sum[j]/item_count[j] if j in item_count else avg
        error += (r - pred)**2
    return np.sqrt(error/len(validate))

# Load the movie data
# Below is code for the smaller dataset, used in section 3 of the HW
def tuning_als(max_iter_als=20, verbose=True):
    b1, v1 = load_ratings_data_small() 
    print('Baseline rmse (predict item average)', baseline(b1, v1))
    print('Running on the MovieLens data')
    lams = [0.01, 0.1,1,10,100]
    ks = [1,2,3]
    for k in ks:
        for lam in lams:
            print('ALS, k=', k, 'lam', lam)
            mf_als(b1, v1, lam = lam, max_iter=max_iter_als, k=k, verbose=verbose)

def compute_and_save_large_model():
    data = load_ratings_data()
    print('Running ALS on the MovieLens data for 20 iterations.')
    x = mf_als(data, None, lam = 1.0, max_iter = 20, k = 10)
    print('Saving the model')
    save_model(x)

#data = load_ratings_data()
#movies, genres = load_movies()
#model = mf_als(data, None, k = 10, lam=1, max_iter=20)
#save_model(model)

model = load_model()
triple=load_ratings_data(path_data='ratings.csv')
titledict,genredict=load_movies(path_movies='movies.csv')
(u, b_u, v, b_v) = model
print(np.shape(b_v))
#code for computing best genre
#indici={}
#for t in triple:
#    if t[0]==270894:
#        if t[1] not in indici.keys():
#            indici[t[1]]=t[2]
#g_points={}
#for i in indici.keys():
#    for x in genredict[i]:
#        if x not in g_points.keys():
#            g_points[x]=indici[i]
#        else:
#            g_points[x] += indici[i]
#print(g_points)

#score={}
#for t in triple:
#    i=t[1]
#    score[i]=np.dot(u[270894],v[i].T)+b_u[270894]+b_v[i]
#    print(np.shape(score[i]))
#print(sorted(score, key=score.get, reverse=True)[:5])


#id=2628
#similarities={}
#for i in range(np.shape(b_v)[0]):
#    if i==id:
#        continue
#    similarities[i]=np.dot(v[id].T,v[i])/(np.sqrt(np.dot(v[id].T,v[id])*np.dot(v[i].T,v[i])))
#d=similarities.copy()
#for i in similarities.keys():
#    if similarities[i]<0:
#        del d[i]
#most_similar=sorted(d, key=d.get, reverse=True)[:np.shape(b_v)[0]]
#final=[]
#print(most_similar[:10])
#k=0
#for i in most_similar:
#    if i in genredict.keys():
#        final.append(i)
#        ok+=1
#        if ok==10:
#            break
#print(final)

similarities={}
print(genres)
for gen in genres:
    suma=0;n=0;
    for j in range(np.shape(b_v)[0]):
        if j in genredict.keys() and gen in genredict[j] and gen!='Comedy':
            for i in range(np.shape(b_v)[0]): # i must be comedy
                if i==j or i not in genredict.keys() or 'Comedy' not in genredict[i]:  # i don't count the similarity with oneself
                    continue   #don't care about similarity in the same genre
                suma+=np.dot(v[j].T,v[i])/(np.sqrt(np.dot(v[j].T,v[j])*np.dot(v[i].T,v[i])))
                n+=1
    if gen!='Comedy':
        similarities[gen]=suma/n
print(sorted(similarities, key=similarities.get, reverse=True))
print(similarities)

#for i in range(np.shape(film_scores+1)[0]):
   # scores[i]=film_scores[i]
#best50=sorted(scores, key=scores.get, reverse=True)[:1000]
#print(best50)
#a=0
#ok=0
#for film in best50:
    #if ok==50:
        #break
    #if film in genredict.keys():
        #ok+=1
        #if 'Animation' in genredict[film]:
            #a+=1
#print(a)