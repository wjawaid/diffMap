import numpy as np
from scipy.sparse.linalg import eigsh


def applyGaussianKernel(d2, rsigmas, csigmas=None):
    """Applies Gaussian Kernel given sigmas

    Given sigmas this function will return a Gaussian kernel.

    Keyword arguments:
    rsigmas -- refers to row sigmas and is symmetric when calculating
    the diffusion map.
    csigmas -- is optional and only required when projecting on a 
    diffusion map.
    """   
    if len(rsigmas) != d2.shape[0]:
      raise ValueError('Length of rsigmas does not match d2 dimensions')
    if csigmas is None:
      sigmaMat = np.matrix(rsigmas)
      sigmaMat = sigmaMat.transpose() * sigmaMat
      sigmaSqd = np.power(rsigmas, 2)
      sigmaSum = np.matrix(sigmaSqd)
      sigmaSum = sigmaSum + sigmaSum.transpose()
    else:
      if len(csigmas) != d2.shape[1]:
        raise ValueError('Length of csigmas does not match d2 dimensions')
      sigmaMat = np.matrix(rsigmas).transpose() * np.matrix(csigmas)
      rsigmaSqd = np.power(rsigmas, 2)
      csigmaSqd = np.power(csigmas, 2)
      sigmaSum = np.matrix(rsigmaSqd).transpose() + np.matrix(csigmaSqd)
    W = np.multiply(np.sqrt(2 * sigmaMat / sigmaSum), np.exp(-d2 / sigmaSum))  
    return(W)


def diffuse(data, ndims=20, nsig=5, sigmas=0):
    """Diffusion map function

    Calculates diffusion map dimensionality reduction as described by
    Laleh Haghverdi et al.

    data --  matrix of values with cell in columns and genes in rows
    ndims -- number of dimensions to return (default 20)
    nsig --  number of neighbours when calculating sigmas (default 5)
    sigmas -- Give your own sigma(s). If 0 they will be automatically 
              calculated (default 0)

    Returns a list of: eigenvalues, eigenvectors, sigmas, gaussian and
                       redundant first vector
    """
    # d2 = sp.spatial.distance.pdist(data.transpose(), 'cosine')
    # d2 = sp.spatial.distance.squareform(d2)
    d2 = np.power(1 - np.corrcoef(data.T), 2)
    np.fill_diagonal(d2, 0)
    if sigmas == 0:
      sigmas = [np.sqrt(np.sort(d2[i, :])[nsig-1])/2 for i in xrange(0, d2.shape[0])]
      W = applyGaussianKernel(d2, sigmas)
    elif (isinstance(sigmas, float) or isinstance(sigmas, int)):
      W = np.exp(-d2 / (2 * sigmas**2))
    np.fill_diagonal(W, 0)
    D = np.matrix(sum(W))
    markov = W / D
    q = D.transpose() * D
    H = W / q
    dH = sum(H)
    rdH = np.matrix(1. / np.sqrt(dH))
    Hp = np.multiply(H, (rdH.transpose() * rdH))
    n = d2.shape[0]
    evals, evecs = eigsh(Hp, ndims + 1, which='LM')
    # evals, evecs = eigh(Hp)
    srtOrd = np.argsort(evals)[::-1]
    evals = evals[srtOrd]
    evecs = evecs[:, srtOrd]
    evecs = np.array(rdH) * np.array(evecs).transpose()
    evals = evals[1:]
    evec0 = evecs[0,:]
    evecs = evecs.transpose()[:, 1:]
    return([evals, evecs, sigmas, W, evec0])


def c2c(A,B):
    """Calculate pearson correlation coefficient

    Calculates pearson correlation between rows of two arrays.
    Keyword values:
    A -- M x T shaped array
    B -- N x T shaped array

    returns an M x N shaped array

    """
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))


def diffProj(x, data, evals, evecs, sigmas, gaussian, vec0, nsig):
    """Project on diffusion map

    Projects new data on a previously produced diffusion map.
    Keyword values:
    x    -- new data in the form of a G x N array
    data -- original data in the form of a G x M array
    evals -- eigenvalues from base diffusion map
    evecs -- eigenvectors from base diffusion map
    sigmas -- sigmas used for base diffusion map
    gaussian - gaussian generated while calulating base diffusion map
    vec0 -- redundant first eigenvector, usually removed
    nsig -- number of nearest neighbours for calulating new sigmas use
            same as for original diffusion map

    """
    d2 = np.power(1 - c2c(x.T, data.T), 2)
    nsigmas = [np.sqrt(np.sort(d2[i, :])[nsig-1])/2
               for i in xrange(0, d2.shape[0])]
    W = applyGaussianKernel(d2, nsigmas, sigmas)
    W[abs(W-1) < 1e-15] = 0
    H = W / np.sum(W, 1)
    H = H / np.sum(gaussian, 1).T
    Hp = np.matrix(H / np.sum(H, 1))
    predMat = Hp * \
              np.concatenate((np.matrix(vec0).T, np.matrix(evecs)), axis=1) * \
              np.diag(np.append([1],1/evals))
    predMat = np.array(predMat[:,1:])
    return(predMat)
