
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def gradient(img):
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient

def div(grad):
    '''
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    #res = res.flatten()
    return res


def psi(x, mu):
    '''
    Huber function needed to compute tv_smoothed
    '''
    res = np.abs(x)
    m = res < mu
    res[m] = x[m]**2/(2*mu) + mu/2
    return res


def huber(x,mu):
    return np.sum(psi(x,mu))

def derivative_normterm(x,mu):
    res = x
    m = np.abs(res)<mu
    n = np.abs(res)>=mu
    res[m] = 1/mu
    res[n] = np.sign(x)
    return res


def tv_smoothed(x, mu):
    '''
    Moreau-Yosida approximation of Total Variation
    see Weiss, Blanc-Féraud, Aubert, "Efficient schemes for total variation minimization under constraints in image processing"
    '''
    g = gradient(x)
    g = np.sqrt(g[0]**2 + g[1]**2)
    return np.sum(psi(g, mu))


#since the adjoint of the l1-norm, the resolvent operator reduces to 
# pointwise euclidean projectors onto l2-balls

def proj_l1_grad(g, Lambda):
    '''
    proximity operator of l1
    '''
    res = np.copy(g)
    L = Lambda*np.ones_like(g[0])
    n = np.maximum(np.sqrt(g[0]**2+g[1]**2),L)
    res[0] = res[0]/n
    res[1]= res[1]/n #g/max(alpa, |g|)
    res = Lambda*res
    return res

def proj_l1(g, beta=1.0):
    '''
    proximity operator of l1
    '''
    res = np.copy(g)
    B = beta*np.ones_like(g)
    n = np.maximum(np.abs(g), beta)
    res = res/n
    #res = np.concatenate((res1,res2), axis = 0)
    return res

def proj_unitintervall(g):
    '''proximity operator of indicator function'''
    g[g>1] = 1
    g[g<0] = 0
    return g



def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel()) #ravel transforms matrix to long vector

def norm1(mat):
    return np.sum(np.abs(mat))

#skalarprdoukt
def mydot(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())

mu = 0.0001
def Fid1(img1,img2):
   '''
   Computes M1(u) u*(f-((u,f)/|u|)**2)
   '''
   if img1.dtype != np.dtype('float64'): img1 = img1.astype('float64')
   if img2.dtype != np.dtype('float64'): img2 = img2.astype('float64')
   (n,m) = img1.shape
   if (n,m) != img2.shape:
      print('The images do not have the same size')
      return -1
   return img1*(img2-(np.ones_like(img2)*(np.dot(img1.ravel(),img2.ravel())/huber(img1,mu))))**2


def Fid2(img1,img2):
   '''
   Computes M1(u)
   '''
   if img1.dtype != np.dtype('float64'): img1 = img1.astype('float64')
   if img2.dtype != np.dtype('float64'): img2 = img2.astype('float64')
   (n,m) = img1.shape
   if (n,m) != img2.shape:
      print('The images do not have the same size')
      return -1
   return ((np.ones_like(img1)-img1)*(img2-np.ones_like(img2)*
                 np.dot((np.ones_like(img1)-img1).ravel(),img2.ravel())/(huber(np.ones_like(img1)-img1,mu)))**2)



def mymulti(a,b,h):
    '''a = M^T,b = factor2'''
    prod = np.zeros_like(h)
    for i in range(np.shape(a)[0]):
        prod1 = a[i]*b
        prod[i] = mydot(prod1,h)
    return prod    

def adjoint_der_Fid1(img1,img2,h):
    '''Computes adjoint of derivative and applies it on h T^2*h + 2*M^T*U*T*h'''
    h = h.flatten()
    h = h.astype("float32")
    T_f = (img2.ravel() - np.ones_like(img2.ravel())*(mydot(img1.ravel(),img2.ravel())/huber(img1,0.0001)))
    S_1 = T_f**2*h.transpose()   
    '''reshape to column'''
    S_1 = np.reshape(S_1,(np.shape(img1.ravel())[0],1)) 
    M_T = (-img2.ravel()*huber(img1,0.001)*np.ones_like(img2.ravel()) + np.sign(img1.ravel())*mydot(img1.ravel(),img2.ravel()))/huber(img1,0.0001)**2
    '''computation of U*T pointwise'''
    factor2 = img1.ravel()*T_f.ravel() 
    S_2 = 2*mymulti(M_T,factor2, h) 
    S_gesamt = np.reshape(S_1,(np.shape(S_2))) + S_2
    S_gesamt = np.reshape(S_gesamt, (np.shape(img1)))
    return S_gesamt
    

def adjoint_der_Fid2(img1,img2,h):
    '''Computes adjoint of derivative'''
   # h= np.reshape(h,(np.shape(img1.ravel())[0],1))
    h = h.flatten()
    T_f = (img2.ravel() - np.ones_like(img2.ravel())*(mydot((np.ones_like(img1)-img1).ravel(),img2.ravel())/huber(np.ones_like(img1)-img1,0.0001)))
    S_1 = -T_f**2*h.transpose()
    S_1 = np.reshape(S_1,(np.shape(img1.ravel())[0],1))  
    M_T = (img2.ravel()*(huber(1-img1.ravel(),0.001)*np.ones_like(img2.ravel())) - np.sign(np.ones_like(img1.ravel())-img1.ravel())*mydot(((np.ones_like(img1.ravel()))-img1.ravel()),img2.ravel()))/huber((np.ones_like(img1)-img1),0.0001)**2
    factor2 = (np.ones_like(img1.ravel())-img1.ravel())*T_f.ravel() 
    S_2 = 2*mymulti(M_T,factor2, h)
    S_gesamt = np.reshape(S_1,(np.shape(S_2))) + S_2
    S_gesamt = np.reshape(S_gesamt, (np.shape(img1)))
    return S_gesamt
    
def indi(img1, eps):
    '''Computes indicator function on convex set'''
    if np.max(img1)<=1+eps and np.min(img1)>=-0.01:
        return 0
    else:
        return float('inf')
    
    
# #projection operator for nonoverlaüping channels   , das ist aber für sum |u|^2 ACHTUNG
def proj_unitball(g): 
#     '''
#     proximity operator of l1
#     '''
     g = np.asarray(g)
     res = np.copy(g)
     n = np.maximum(np.sum(np.abs(g), 0), 1.0)
     res = res/n
     #res = np.concatenate((res1,res2), axis = 0)
     return res





'''Projection onto the unit simplex'''

def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() <= s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho+1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


#application of function along channel axis

# def euclidean_proj_l1ball(v, s=1):
#     """ Compute the Euclidean projection on a L1-ball
#     Solves the optimisation problem (using the algorithm from [1]):
#         min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s
#     Parameters
#     ----------
#     v: (n,) numpy array,
#        n-dimensional vector to project
#     s: int, optional, default: 1,
#        radius of the L1-ball
#     Returns
#     -------
#     w: (n,) numpy array,
#        Euclidean projection of v on the L1-ball of radius s
#     Notes
#     -----
#     Solves the problem by a reduction to the positive simplex case
#     See also
#     --------
#     euclidean_proj_simplex
#     """
#     assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
#     n, = v.shape  # will raise ValueError if v is not 1-D
#     # compute the vector of absolute values
#     u = np.abs(v)
#     # check if v is already a solution
#     if u.sum() <= s:
#         # L1-norm is <= s
#         return v
#     # v is not already a solution: optimum lies on the boundary (norm == s)
#     # project *u* on the simplex
#     w = euclidean_proj_simplex(u, s=s)
#     # compute the solution to the original problem on v
#     w *= np.sign(v)
#     return w