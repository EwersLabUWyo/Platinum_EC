"""
Algorithms for computing planar fit methods.
(Wilczak et al 2001, Ross et al 2015)

Author: Alex Fox
Created 2023-01-30
"""

import numpy as np
from scipy import optimize

from tilt_correction_methods.double_rotation_methods import get_double_rotation_angles

__all__ = [
    'fourier_func',
    'dfourier_func_dtheta',
    'get_continuous_planar_fit_angles',
    'continuous_planar_fit_from_angles',
    'continuous_planar_fit_from_uvw'    
]

def fourier_func(theta, *p):
    '''compute the value of a fourier series
    
    Parameters
    ----------
    theta : np.ndarray
        independent variable
    p : 1d iterable
        fourier coefficients [a_0, b_0, ..., a_N-1, b_N-1], where a are the cosine coefficients and b are the sine coefficients
    
    Returns
    -------
    out -  np.ndarray
        the value of the fourier series at theta.
    '''
    # a = p[::2]
    # b = p[1::2]
    p = np.array(p)
    c = p[::2] + 1j*p[1::2]
    N = len(c)
    terms = np.array([c[n]*np.exp(1j*n*theta) for n in range(N)])
    out = terms.real.sum(0) + terms.imag.sum(0)
    return out

def dfourier_func_dtheta(theta, *p):
    '''compute the value of the derivative of a fourier series
    
    Parameters
    ----------
    theta : np.ndarray
        independent variable
    p : iterable
        fourier coefficients [a_0, b_0, ..., a_N-1, b_N-1], where a are the cosine coefficients and b are the sine coefficients
    
    Returns
    -------
    out -  np.ndarray
        the value of the derivative of the fourier series at theta.
    '''
    
    # a = p[::2]
    # b = p[1::2]
    p = np.array(p)
    c = p[::2] + 1j*p[1::2]
    N = len(c)
    terms = np.array([1j*n*c[n]*np.exp(1j*n*theta) for n in range(N)])
    out = terms.real.sum(0) + terms.imag.sum(0)
    return out

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
    
    return phi_func, phidot_func

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
    phi_approx, phidot_approx = get_continuous_planar_fit_angles(theta, phi, N)
    phi_sim, phidot_sim = phi_approx(theta), phidot_approx(theta)
    uvw_rot = continuous_planar_fit_from_angles(U, V, W, theta, phi_sim, phidot_sim)
    return uvw_rot