"""
Contains methods for analysis, detection and location of the bifurcation points
"""

from __future__ import division
import scipy as sp
import scipy.linalg as ln
import scipy.optimize as opt


def tau_hopf(eig_vals, eps = 1e-13):
    """
    :param eig_vals: contains all the eigne values of the system
    :return: tau_hopf (value of the test function
    """

    # Pick out only the complex eigen values
    # eps = 1e-15  # it seems 1e-15 is too low. -> due to numerical errors values larger than 1e-15 can appear
    # eps = 1e-13  # it seems 1e-13 is fine.
    i_cmp = sp.where(sp.absolute(eig_vals.imag) > eps)
    vals_complex = eig_vals[i_cmp]

    # Calculate tau_hopf
    if vals_complex.size == 0:
        tau_h = 0
    else:
        tau_h = 1
        for i, val in enumerate(vals_complex):
            i_cc = sp.where(
                (sp.absolute(val.real - vals_complex.real) < eps) & (sp.absolute(val.imag + vals_complex.imag) < eps))[
                0][0]
            tau_h *= sp.sqrt(val.real + vals_complex[i_cc].real)

    return sp.real(tau_h)


def tau_pitchfork(eig_vals):
    eps = 1e-10

    # This is the implementation according to the book:
    # =================================================
    i_real = sp.where(sp.absolute(eig_vals.imag) < eps)
    vals_real = eig_vals[i_real].real

    # This is another option, where all the eig values are considered (comment above if you want ot use this)
    # vals_real = eig_vals.real

    if vals_real.size == 0:
        tau_p = 0
    else:
        tau_p = 1
        for val in vals_real:
            tau_p *= val

    return tau_p


# def R1_hopf_modified(eig_vals):
#     # According to the book, the function should return the absolute value (see below), however the derivative of
#     # absolute function arouund 0 is not defined. Hence I chose to simply return the value of the smallest eig value.
#
#     # return sp.absolute(val_cmp_min.real)
#     # ----------------------------------------------------------------------------------------------------------------
#
#     eps = 1e-15
#
#     i_cmp = sp.where(sp.absolute(eig_vals.imag) > eps)
#     vals_cmp = eig_vals[i_cmp]
#
#     i_cmp_min = sp.absolute(vals_cmp.real).argmin()
#     val_cmp_min = vals_cmp[i_cmp_min]
#
#     return val_cmp_min.real


# def R1_pitchfork_modified(eig_vals):
#     # According to the book, the function should return the absolute value (see below), however the derivative of
#     # absolute function around 0 is not defined. Hence I chose to simply return the value of the smallest eig value.
#
#     # return sp.absolute(val_real_min)
#     # ----------------------------------------------------------------------------------------------------------------
#
#     eps = 1e-15
#
#     i_real = sp.where(sp.absolute(eig_vals.imag) < eps)
#     vals_real = eig_vals[i_real].real
#
#     i_real_min = sp.absolute(vals_real).argmin()
#     val_real_min = vals_real[i_real_min]
#
#     return val_real_min


def R1_hopf_modified(Q, dQdU):
    """
    :param Q: system matrix of the linearised system
    :param dQdU: sensitivity of the system matrix of the linearised system wrt. velocity (U)
    :return:
        R1: value of the modified test function
        dR1dU: sensitivity of the modified test function wrt. velocity (U)

    According to the book, the function should return the absolute value (see below), however the derivative of
    absolute function arouund 0 is not defined. Hence I chose to simply return the value of the smallest eig value.

    return sp.absolute(val_cmp_min.real)
    """
    eps = 1e-15

    # calculate eigenvalues and eigenvectors of Q
    vals, vecs = ln.eig(Q)

    # determine R1:
    i_cmp = sp.where(sp.absolute(vals.imag) > eps)[0]
    vals_cmp = vals[i_cmp]

    i_cmp_min = i_cmp[sp.absolute(vals_cmp.real).argmin()]
    R1 = vals[i_cmp_min].real

    # determine dR1dU
    dWdU = ln.inv(vecs).dot(dQdU.dot(vecs))
    dR1dU = dWdU[i_cmp_min, i_cmp_min].real

    return R1, dR1dU


def R1_pitchfork_modified(Q, dQdU):
    """
    :param Q: system matrix of the linearised system
    :param dQdU: sensitivity of the system matrix of the linearised system wrt. velocity (U)
    :return:
        R1: value of the modified test function
        dR1dU: sensitivity of the modified test function wrt. velocity (U)

    According to the book, the function should return the absolute value (see below), however the derivative of
    absolute function arouund 0 is not defined. Hence I chose to simply return the value of the smallest eig value.

    return sp.absolute(val_cmp_min.real)
    """

    eps = 1e-15

    # calculate eigenvalues and eigenvectors of Q
    vals, vecs = ln.eig(Q)

    # Determine R1:
    i_real = sp.where(sp.absolute(vals.imag) < eps)[0]
    vals_real = vals[i_real].real

    i_real_min = i_real[sp.absolute(vals_real).argmin()]
    R1 = vals[i_real_min].real

    # determine dR1dU
    dWdU = ln.inv(vecs).dot(dQdU.dot(vecs))
    dR1dU = dWdU[i_real_min, i_real_min].real

    return R1, dR1dU


def locatate_bifurcation(func, U_min, U_max, type):
    """
    Determine the exact location and parameters of the bifurcation point
    :param func: function that returns the eigenvalues of the system
    :param U_min: lower bound of the velocity range [m/s]
    :param U_max: upper bound of the velocity range [m/s]
    :param type: type of bifuctaion to look for: 'hopf', 'pitchfork'
    :return: bifurcation parameters
    """

    xtol = 1e-10

    if type=='hopf':
        tau = lambda U: tau_hopf(ln.eigvals(func(U)))
    elif type == 'pitchfork':
        tau = lambda U: tau_pitchfork(ln.eigvals(func(U)))
    else:
        return -1

    # Find the bifurcation point:
    U_bf = opt.brentq(tau, U_min, U_max, xtol=xtol)

    # Calculate bifurcation parameters (frequency, damping ratio):
    Q = func(U_bf)
    evals = ln.eigvals(Q)

    eval_bf = evals[sp.absolute(evals.real).argmin()]

    if type == 'hopf':
        omega_bf = sp.absolute(eval_bf)
        zeta_bf = -eval_bf.real/sp.absolute(eval_bf)
    elif type == 'pitchfork':
        omega_bf = 0
        zeta_bf = 0

    return U_bf, omega_bf, zeta_bf
