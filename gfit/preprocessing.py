import numpy as np


def base(y,k=5,itr=20):
    # find a base line by k-means method (with k=k+1).
    ymin,ymax = np.min(y),np.max(y) 
    dy = (ymax-ymin) / (k - 1)
    centroid = [ymin+dy*i for i in range(k)]
    for _ in range(itr):
        # === E step ===
        r_mat = np.zeros([k,len(y)])
        d_mat = np.zeros([k,len(y)])
        for i in range(k):
            d_mat[i,:] = np.abs(y-centroid[i])
        r = np.argmin(d_mat,axis=0)
        r_mat[r,np.arange(len(y))] = 1

        # === M step ===
        nk = np.sum(r_mat,axis=1)
        centroid = np.dot(r_mat,y)/nk
    return centroid[0]
                                            
                                            
def mid_pooling(y,p=3):
    # The midpoint value within p-th neighbors (2*p+1 candidates) is employed.
    # Up to p-consecutive outliers are removed.
    m = 2 * p + 1  # window size
    y_pad = y.copy()
    for i in range(p):
        y_pad = np.insert(y_pad,0,y[0])
        y_pad = np.append(y_pad,y[-1])
    n = len(y_pad)
    y_mat = np.zeros([m,len(y)])
    for i in range(m):
        y_mat[i,:] = y_pad[i:n-2*p+i]
    argsort_mat = np.argsort(y_mat,axis=0)     
    return y_mat[argsort_mat[p],np.arange(len(y))]


def smoothing(y,p=3):
    m = 2 * p + 1  # window size
    y_exp = y.copy()
    for i in range(p):
        y_exp = np.insert(y_exp,0,y[0])
        y_exp = np.append(y_exp,y[-1])
    n = len(y_exp)
    y_mat = np.zeros([m,len(y)])
    for i in range(m):
        y_mat[i,:] = y_exp[i:n-2*p+i]
    return np.mean(y_mat,axis=0)


def above(y,b=None):
    if b is None: b = base(y)
    mask = (y <= b)
    y_above = y.copy()
    y_above[mask] = b
    return y_above


def noise(y,d=5):
    y_dif = np.abs(y[:-d] - y[d:])
    return np.sum(y_dif)/(len(y)-d)
    
    
def peak_extraction(y,mu,h): 
    ny = noise(y)
    n  = sum(1 for i in h if i > ny/2)
    h_argsort = np.argsort(h)
    mu_sort   = mu[h_argsort]
    peak_pos  = mu_sort[-n:]
    return np.sort(peak_pos)

    
    