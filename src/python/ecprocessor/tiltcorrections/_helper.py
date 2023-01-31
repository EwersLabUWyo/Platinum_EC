import numpy as np

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