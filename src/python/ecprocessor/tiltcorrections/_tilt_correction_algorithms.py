'''Anemometer Tilt Correction Algorithms'''

import numpy as np
from scipy import optimize

from ._helper import *

def get_double_rotation_angles(U, V, W):
    '''
    compute the double rotation wind direction and tilt angle (Wilczak et al, 2001).
    
    Parameters
    ----------
    U, V, W: np.ndarray
        u, v, w components of windspeed.
    
    Returns
    -------
    theta, phi : np.ndarray
        wind direction and tilt angle to use for each time block.
    '''
    
    # compute wind direction in radians. 0 = aligned with U
    theta = (np.arctan2(V, U)%(2*np.pi))
    phi = np.arctan2(W, U*np.cos(theta) + V*np.sin(theta))
    return theta, phi

def get_triple_rotation_angles(U, V, W):
    '''
    compute the triple rotation wind direction and tilt angle (Wilczak et al, 2001).
    
    Parameters
    ----------
    U, V, W: np.ndarray
        u, v, w components of windspeed.
    
    Returns
    -------
    theta, phi, psi : np.ndarray
        wind direction, tilt angle, and second tilt angle to use for each time block.
    ''' 
    U, V, W = np.atleast_1d(U), np.atleast_1d(V), np.atleast_1d(W)

    theta, phi = get_double_rotation_angles(U, V, W)
    uvw_rot = double_rotation_fit_from_angles(U, V, W, theta, phi)
    u_rot, v_rot, w_rot = uvw_rot[:, 0], uvw_rot[:, 1], uvw_rot[:, 2]
    y = 2*(v_rot*w_rot).mean(0)
    x = (v_rot**2).mean() - w_rot**2
    psi = (np.arctan2(y, x))/2

    return theta, phi, psi

def triple_rotation_fit_from_angles(U, V, W, theta, phi, psi):
    '''
    Given an direction and tilt angle, transform wind coordinates U, V, W using the triple-rotation method. (Wilczak et al, 2001).
    
    Scalar inputs are converted to 1-dimensional arrays.

    Parameters
    ----------
    U, V, W : np.ndarray 
        Wind coordinates
    theta, phi, psi : np.ndarray
        Rotation angles
    
    Returns
    -------
    uvw_rot: np.ndarray
        shape (..., 3): U, V, W are indexed by the last dimension.
    '''

    U, V, W = np.atleast_1d(U), np.atleast_1d(V), np.atleast_1d(W)
    
    uvw_rot = double_rotation_fit_from_angles(U, V, W, theta, phi)
    u_rot, v_rot, w_rot = u_rot, v_rot, w_rot = uvw_rot[:, 0], uvw_rot[:, 1], uvw_rot[:, 2]
    
    u_rot_2 = u_rot
    v_rot_2 = v_rot*np.cos(psi) + w_rot*np.sin(psi)
    w_rot_2 = -v_rot*np.sin(psi) + w_rot*np.sin(psi)

    uvw_rot_2 = np.stack((u_rot_2, v_rot_2, w_rot_2), axis=-1)

    return uvw_rot_2

def triple_rotation_fit_from_uvw(U, V, W):
    '''
    Transform wind coordinates U, V, W using the triple-rotation method. (Wilczak et al, 2001).
    
    Scalar inputs are converted to 1-dimensional arrays.

    Parameters
    ----------
    U, V, W : np.ndarray 
        Wind coordinates
    
    Returns
    -------
    uvw_rot: np.ndarray
        shape (..., 3): U, V, W are indexed by the last dimension.
    '''

    U, V, W = np.atleast_1d(U), np.atleast_1d(V), np.atleast_1d(W)
    
    theta, phi, psi = get_triple_rotation_angles(U, V, W, theta, phi)
    uvw_rot_2 = triple_rotation_fit_from_angles(U, V, W, theta, phi, psi)

    return uvw_rot_2

def get_continuous_planar_fit_angles(theta, phi, N):
    '''fit a fourier series approximation to a random variable phi ~ theta using N terms
    
    \phi(\theta) ~ \sum_{k=0}^{N} a_k \cos(k \theta) + b_k \sin(k \theta)
    
    scipy.optimize.curve_fit() requires that 2*(N + 1) >= theta.shape[0]
    
    Parameters
    ----------
    theta : 1d array
        predictor variable
    phi : 1d array
        response variable
    N : int
        number of fourier terms to compute (0 = mean only, 1 = standard planar fit). 
    
    Returns
    -------
    theta : np.ndarray
        original theta that was provided as input
    phi_approx : function
        a function to compute the fourier approximation of phi(theta). Takes one argument, theta, an np.ndarray
    dphi_approx_dtheta: function
        a function to compute the derivative of the fourier approximation of phi(theta). Takes one argument, theta, an np.ndarray.
    
    Examples
    --------
    See continuous_planar_fit_from_angles() and continuous_planar_fit_from_uvw() documentation
    '''
    
    N_max = N + 1
    p0 = np.zeros(2*N_max)
    p0[0] = phi.mean()
    p, _ = optimize.curve_fit(
        f = fourier_func,
        xdata=theta,
        ydata=phi,
        p0=p0
    )
        
    a, b = p[::2], p[1::2]
    
    # get the derivative and approximate functions
    phi_func = lambda theta: fourier_func(theta, *p)
    phidot_func = lambda theta: dfourier_func_dtheta(theta, *p)
    
    return theta, phi_func, phidot_func
    
def double_rotation_fit_from_angles(U, V, W, theta, phi):
    '''
    Given an direction and tilt angle, transform wind coordinates U, V, W using the double-rotation method. (Wilczak et al, 2001).
    
    Scalar inputs are converted to 1-dimensional arrays.

    Parameters
    ----------
    U, V, W : np.ndarray 
        Wind coordinates
    theta, phi : np.ndarray
        Rotation angles
    
    Returns
    -------
    uvw_rot: np.ndarray
        shape (..., 3): U, V, W are indexed by the last dimension.
        
    Examples
    --------
    # scalars only
    >>> double_rotation_fit_from_angles(U=1, V=2, W=3, theta=np.pi/6, phi=np.pi/12)
    array([[2.57889927, 1.23205081, 2.41481457]])
    
    # timeseries of 4 measurements
    >>> size = 4
    >>> U = np.linspace(0, 3, size)
    >>> V = U + 1
    >>> W = U - V
    >>> theta, phi = get_double_rotation_angles(U, V, W)
    >>> double_rotation_fit_from_angles(U, V, W, theta, phi).round(4)
    array([[ 1.4142,  0.    , -0.    ],
           [ 2.4495,  0.    ,  0.    ],
           [ 3.7417,  0.    ,  0.    ],
           [ 5.099 ,  0.    ,  0.    ]])
           
    # timeseries of 4 measurements from 2 heights, with one rotation angle.
    >>> size = 4
    >>> U = np.linspace([0, 0], [3, 1], size)
    >>> V = U + 1
    >>> W = U - V
    >>> Ubar, Vbar, Wbar = U.mean(), V.mean(), W.mean()
    >>> theta, phi = get_double_rotation_angles(Ubar, Vbar, Wbar)
    >>> double_rotation_fit_from_angles(U, V, W, theta, phi).round(3)
    array([[[ 1.225,  0.447, -0.548],
            [ 1.225,  0.447, -0.548]],

           [[ 2.449, -0.   ,  0.   ],
            [ 1.633,  0.298, -0.365]],

           [[ 3.674, -0.447,  0.548],
            [ 2.041,  0.149, -0.183]],

           [[ 4.899, -0.894,  1.095],
            [ 2.449, -0.   ,  0.   ]]])
    '''
    
    U, V, W = np.atleast_1d(U), np.atleast_1d(V), np.atleast_1d(W)
    
    u_tmp = U*np.cos(theta) + V*np.sin(theta)
    v_tmp = -U*np.sin(theta) + V*np.cos(theta)
    
    u_rot = u_tmp*np.cos(phi) + W*np.sin(phi)
    v_rot = v_tmp
    w_rot = -u_tmp*np.sin(phi) + W*np.cos(phi)
    
    uvw_rot = np.stack((u_rot, v_rot, w_rot), axis=-1)
    
    return uvw_rot

def double_rotation_fit_from_uvw(U, V, W):
    '''
    Transform wind coordinates U, V, W using the double-rotation method. (Wilczak et al, 2001).
    
    Scalar inputs are converted to 1-dimensional arrays.

    Parameters
    ----------
    U, V, W : np.ndarray 
        Wind coordinates
    
    Returns
    -------
    uvw_rot: np.ndarray
        shape (..., 3): U, V, W are indexed by the last dimension.
        
    Examples
    --------
    # scalars only
    >>> double_rotation_fit_from_uvw(U=1, V=2, W=3)
    array([[2.57889927, 1.23205081, 2.41481457]])
    
    # timeseries of 4 measurements
    >>> size = 4
    >>> U = np.linspace(0, 3, size)
    >>> V = U + 1
    >>> W = U - V
    >>> double_rotation_fit_from_uvw(U, V, W).round(4)
    array([[ 1.4142,  0.    , -0.    ],
           [ 2.4495,  0.    ,  0.    ],
           [ 3.7417,  0.    ,  0.    ],
           [ 5.099 ,  0.    ,  0.    ]])
           
    # timeseries of 4 measurements from 2 heights, with one rotation angle.
    >>> size = 4
    >>> U = np.linspace([0, 0], [3, 1], size)
    >>> V = U + 1
    >>> W = U - V
    >>> Ubar, Vbar, Wbar = U.mean(), V.mean(), W.mean()
    >>> double_rotation_fit_from_uvw(U, V, W, theta, phi).round(3)
    array([[[ 1.225,  0.447, -0.548],
            [ 1.225,  0.447, -0.548]],

           [[ 2.449, -0.   ,  0.   ],
            [ 1.633,  0.298, -0.365]],

           [[ 3.674, -0.447,  0.548],
            [ 2.041,  0.149, -0.183]],

           [[ 4.899, -0.894,  1.095],
            [ 2.449, -0.   ,  0.   ]]])
    '''
    
    theta, phi = get_double_rotation_angles(U, V, W)
    uvw_rot = double_rotation_fit_from_angles(U, V, W, theta, phi)
    return uvw_rot
   
def continuous_planar_fit_from_angles(U, V, W, theta, phi, phidot):
    '''
    Given an direction and tilt angle, transform wind coordinates U, V, W using the continuous planar fit method. (Ross et al, 2015).
    
    Scalar inputs are converted to 1-dimensional arrays.
    
    All coordinates must be broadcastable. 
    
    Parameters
    ----------
    U, V, W : np.ndarray
        Wind coordinates.
    theta, phi, phidot : np.ndarray
        Rotation angles (and rotation angle derivative).
    
    Returns
    -------
    uvw_rot: np.ndarray
        an array with dimension (..., 3). Dimension -1 gives the rotated u, v, or w coordinate.
       
    Examples
    --------
    First-order fit over 1000 timepoints. Analagous to standard planar fit
    >>> size = 1000
    >>> U = np.linspace(0, 3, size)
    >>> V = np.sqrt(U)
    >>> W = 0.1*(U - V)
    >>> # compute exact theta and phi using the double rotation method
    >>> theta, phi = get_double_rotation_angles(U, V, W)
    >>> # functions to approximate phi(theta) and dphi/dtheta(theta)
    >>> phi_func, phidot_func = get_continuous_planar_fit_angles(theta, phi, N=1)
    >>> # evaluate phi_func, phidot_func at theta
    >>> phi_sim, phidot_sim = phi_func(theta), phidot_func(theta)
    >>> # compute CPF
    >>> continuous_planar_fit_from_angles(U, V, W, theta, phi_sim, phidot_sim).round(2)
    array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           [ 5.51251622e-02, -3.45179331e-05,  2.55481492e-04],
           [ 7.80584541e-02, -3.64756002e-05,  2.57256374e-04],
           ...,
           [ 3.46033948e+00, -1.13228381e-03,  3.50453595e-03],
           [ 3.46337943e+00, -1.13977483e-03,  3.52759909e-03],
           [ 3.46641934e+00, -1.14728442e-03,  3.55071841e-03]])
       
    Second-order fit for 1000 timepoints across 4 heights
    >>> size = 1000
    >>> U = np.linspace([0, 0, 0, 0], [1, 2, 3, 4], size)
    >>> V = np.sqrt(U)
    >>> W = 0.1*(U - V)
    >>> theta, phi = get_double_rotation_angles(U, V, W)
    >>> # get_continuous_planar_fit_angles can only accept 1d inputs
    >>> theta_phi = np.stack((theta.T, phi.T), axis=2)  # theta_phi.shape: (4, 1000, 2)
    >>> # map across first axis, retrieve functions to compute theta, phi_sim, phidot_sim
    >>> funcs = list(map(
    >>>     lambda x: get_continuous_planar_fit_angles(x[:, 0], x[:, 1], N=2),
    >>>     theta_phi
    >>> ))
    >>> # compute the angles using the functions returned by get_continuous_planar_fit_angles()
    >>> phi_sim = np.array([func[0](t) for func, t in zip(funcs, theta.T)]).T
    >>> phidot_sim = np.array([func[1](t) for func, t in zip(funcs, theta.T)]).T
    >>> # fit 
    >>> continuous_planar_fit_from_angles(U, V, W, theta, phi_sim, phidot_sim)
    # output ommitted 
    
    '''
    
    U, V, W = np.atleast_1d(U), np.atleast_1d(V), np.atleast_1d(W)
    
    # construct orthonormal basis from phi and theta angles
    # ihat, jhat, khat are shape (3, ...)
    ihat = np.array([
        np.cos(theta)*np.cos(phi), 
        np.sin(theta)*np.cos(phi), 
        np.sin(phi)
    ])

    jhat = (
        1/np.sqrt(np.cos(phi)**2 + phidot**2) 
        * np.array([
            -np.sin(theta)*np.cos(phi) - np.cos(theta)*np.sin(phi)*phidot,
            np.cos(theta)*np.cos(phi) - np.sin(theta)*np.sin(phi)*phidot,
            np.cos(phi)*phidot
        ])
    )

    khat = np.cross(ihat, jhat, axisa=0, axisb=0, axisc=0)
    
    # construct the change-of-basis tensor.
    # Dimensions 0 and 1 encode the ihat, jhat, khat as row vectors.
    A = np.stack((ihat, jhat, khat), axis=0)
    # Construct the velocity vector
    # Axis 0 envodes the velocity at a column vector.
    UVW = np.stack((U, V, W), axis=0) 
    
    # now we have 
    # * UVW_v... = old wind element (u, v, or w) v at some time/whatever coordinates summarized by the "..." ellipsis.
    # * A_µv... = coordinate v, of new basis vector (ihat, jhat, or khat) µ at some time/whatever coordinates summarized by the "..." ellipsis.
    # Perform the multiplication A_µv UVW_v at each time/whatever coordinate '...'
    uvw_rot = np.einsum('mn...,n...->...m', A, UVW)
    return uvw_rot

def continuous_planar_fit_from_uvw(U, V, W, N):
    '''
    given wind coordinates u, v, w, rotate them using an N-th order continuous planar fit (Ross et al, 2015)
    
    Parameters
    ----------
    U, V, W : np.ndarray
        Wind coordinates.
    N : int
        order of the fourier fit (N=0 fits only the mean, N=1 gives standard planar fit)
    
    Returns
    -------
    uvw_rot: np.ndarray
        an array with dimension (..., 3). Dimension -1 gives the rotated u, v, or w coordinate.
      
    Examples
    --------
    First-order fit over 1000 timepoints. Analagous to standard planar fit
    >>> size = 1000
    >>> U = np.linspace(0, 3, size)
    >>> V = np.sqrt(U)
    >>> W = 0.1*(U - V)
    >>> continuous_planar_fit_from_wind(U, V, W, 1).round(2)
    array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           [ 5.51251622e-02, -3.45179331e-05,  2.55481492e-04],
           [ 7.80584541e-02, -3.64756002e-05,  2.57256374e-04],
           ...,
           [ 3.46033948e+00, -1.13228381e-03,  3.50453595e-03],
           [ 3.46337943e+00, -1.13977483e-03,  3.52759909e-03],
           [ 3.46641934e+00, -1.14728442e-03,  3.55071841e-03]])
    '''
    
    theta, phi = get_double_rotation_angles(U, V, W)
    theta, phi_approx, phidot_approx = get_continuous_planar_fit_angles(theta, phi, N)
    phi_sim, phidot_sim = phi_approx(theta), phidot_approx(theta)
    
    return continuous_planar_fit_from_angles(U, V, W, theta, phi_sim, phidot_sim)