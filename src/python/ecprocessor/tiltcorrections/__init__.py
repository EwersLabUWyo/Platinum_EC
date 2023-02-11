'''
===========================
Anemometer Tilt Corrections
===========================

Compute Rotation Angles and Coordinate Transformations
=======================
get_double_rotation_angles - compute theta, phi using double rotations
get_triple_rotation_angles - compute theta, phi, psi using triple rotations
get_continuous_planar_fit_angles - compute theta, phi, dphi/dtheta using planar/continuous fits
double_rotation_fit_from_angles - compute rotated u, v, w if given theta and phi
triple_rotation_fit_from_angle - compute rotated u, v, w if given theta and phi
continuous_planar_fit_from_angles - compute rotated u, v, w if given theta and phi
double_rotation_fit_from_uvw - compute rotated u, v, w
triple_rotation_fit_from_uvw - compute rotated u, v, w
continuous_planar_fit_from_uvw - compute rotated u, v, w

Helper Functions
================
fourier_func - value of a finite fourier series
dfourier_func_dtheta - value of the derivative of a finite fourier series
'''

from ._tilt_correction_algorithms import *

__all__ = [
    'get_double_rotation_angles', 
    'get_triple_rotation_angles',
    'get_continuous_planar_fit_angles', 
    'double_rotation_fit_from_angles', 
    'triple_rotation_fit_from_angles',
    'continuous_planar_fit_from_angles', 
    'double_rotation_fit_from_uvw',
    'triple_rotation_fit_from_uvw',
    'continuous_planar_fit_from_uvw'
    'fourier_func', 
    'dfourier_func_dtheta'
    ]