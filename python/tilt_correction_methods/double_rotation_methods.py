"""
Algorithms for computing the double rotation tilt correction.
(Wilczak et al, 2001)

Author: Alex Fox
Created 2023-01-30
"""


import numpy as np

__all__ = [
    'get_double_rotation_angles',
    'double_rotation_fit_from_angles',
    'double_rotation_fit_from_uvw'
]

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