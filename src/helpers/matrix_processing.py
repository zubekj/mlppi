import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.ndimage.measurements import center_of_mass
from collections import Counter

def to_vector(mat, *args):
    """
    Apply each function in args to mat
    Return vector of resulting values

    Assumes each of the functions returns float or sequence of floats
    """
    l = []
    for func in args:
        tmp = func(mat)
        if isinstance(tmp, list):
            l += tmp
        else:
            l.append(tmp)
    return np.array(l, dtype=float)

def largest_connected(mat, num=3, num_connected=3, min_strength=0):
    """
    Return size of the 'num' largest connected component in 'mat'
    where 'mat' is a matrix representing bipartite undirected graph.
    Only 'num_connected' strongest edges of each vertex are taken
    into account (this makes the graph directed).
    """
    mat[mat < min_strength] = 0

    # Include only strongest connections
    fmat = np.zeros(mat.shape)
    ind = mat.argsort()[:,-num_connected:]
    for i in xrange(ind.shape[0]):
        fmat[i, ind[i,:]] = mat[i, ind[i, :]]
    ind = mat.argsort(axis=0)[:,-num_connected:]
    for i in xrange(ind.shape[1]):
        fmat[ind[:,i], i] = mat[ind[:, i], i]
    fmat = fmat > 0

    # Construct 'traditional' adjacency matrix from mat
    m_zeros = np.zeros((fmat.shape[0], fmat.shape[0]))
    n_zeros = np.zeros((fmat.shape[1], fmat.shape[1]))
    adj_mat = np.vstack([np.hstack([m_zeros, fmat]), np.hstack([np.transpose(fmat), n_zeros])])
    nodes = connected_components(adj_mat, directed=True)[1]

    # WIP: extracting scores
    most_common = Counter(nodes).most_common(num)

    ind1 = []
    ind2 = []
    for i in np.where(nodes == most_common[0][0])[0]:
        if i < mat.shape[0]:
            ind1.append(i)
        else:
            ind2.append(i-mat.shape[0])
    mean_score = np.mean(mat[ind1, :][:, ind2])
    if np.isnan(mean_score):
        mean_score = 0
    ###

    counts = [float(count)/(mat.shape[0]+mat.shape[1])
                for label, count in most_common]
    if len(counts) < num:
        counts += [0] * (num-len(counts))

    return counts

def best_interactions(mat, num=20):
    """Returns best interactions scores from matrix"""
    scores = mat.flatten()
    scores.sort()
    return list(scores[-num:])

def score_distribution(mat, bins=10):
    """Histogram of scores"""
    return [float(c)/mat.size for c in np.histogram(mat, bins, range=(0.0, 0.5))[0]]

def count_larger(mat, threshold):
    """Count elements of mat larger than given threshold"""
    return np.count_nonzero(mat > threshold)

def count_intersections(mat, num=5):
    """
    """
    sums1 = np.sum(mat, axis=0)
    sums2 = np.sum(mat, axis=1)
    ind1 = sums1.argsort()
    ind2 = sums2.argsort()

    intersections = np.sum([mat[j,i] for i in ind1[-num:] for j in ind2[-num:]])

    ind1 = np.array(ind1[-num:], dtype=float)/len(ind1)
    ind2 = np.array(ind2[-num:], dtype=float)/len(ind2)

    ind1 = list(ind1)
    if len(ind1) < num:
        ind1 += [-1] * (num-len(ind1))

    ind2 = list(ind2)
    if len(ind2) < num:
        ind2 += [-1] * (num-len(ind2))

    #return [np.var(ind1[-num:]), np.var(ind2[-num:])]
    #return ind1 + ind2
    return intersections

def count_along_axis(mat, axis=0, num=20):
    """
    """
    sums = np.sum(mat, axis=axis)
    sums.sort()

    s = list(sums[-num:])
    if len(s) < num:
        s += [0.0] * (num-len(s))

    return s
    #return np.mean(sums[-num:])

def count_diagonals(mat, num=10):
    """
    """
    n = max(mat.shape)

    sums = [np.diag(mat, i).sum() for i in range(-n+1, n)]
    sums.sort()

    s = list(sums[-num:])
    if len(s) < num:
        s += [0.0] * (num-len(s))

    return s

def count_squares(mat, dim=5, num=20):
    """
    """
    squares = []
    for i in xrange(mat.shape[0]):
        for j in xrange(mat.shape[1]):
            xs, ys = max(0, i-dim), max(0, j-dim)
            xe, ye = i+dim, j+dim
            squares.append(np.sum(mat[xs:xe,:][:,ys:ye]))

    squares.sort()

    s = list(squares[-num:])
    if len(s) < num:
        s += [0.0] * (num-len(s))
    return s


def mass_center(mat):
    mass = center_of_mass(mat)
    return [float(mass[0])/mat.shape[0], float(mass[1])/mat.shape[1]]

def rotate_mass(mat):
    """
    Return matrix mat (2D) rotated so that its center of mass is in its upper left quarter
    """
    center = center_of_mass(mat)
    if center[0] <= mat.shape[0]/2:
        if center[1] <= mat.shape[1]/2:
            return mat
        else:
            return np.rot90(mat)
    else:
        if center[1] <= mat.shape[1]/2:
            return np.rot90(mat, k=3)
        else:
            return np.rot90(mat, k=2)
