"""
This file is obsolete! Use aerodynamics_typical_section.py instead!
"""

from __future__ import division
import scipy as sp

def get_approximation_coefficients():
    """
    Wagner function approximation:
    phi(t) = 1 - psi1*e**(-eps1*U*t/b) - psi2*e**(-eps2*U*t/b)


    :return: coefficients required for the calculation of the approximation of the Wagner function
    """
    coefW ={
        'psi1': 0.165,
        'psi2': 0.335,
        'eps1': 0.0455,
        'eps2': 0.3
    }

    return coefW


def approx_wagner_func(t, U, constant):
    """
    calculates the Wagner function approximation for the given time and velocity
    :param t: time [s]
    :param U: velocity [m/s]
    :param constant: system properties
    :return: phi
    """

    b = constant['b']

    coefW = get_approximation_coefficients()
    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    phi = 1. - psi1*sp.exp(-eps1*U*t/b) - psi2*sp.exp(-eps2*U*t/b)

    return phi


def approx_wagner_func_dot(t, U, constant):
    """
    calculates the Wagner function approximation for the given time and velocity
    :param t: time [s]
    :param U: velocity [m/s]
    :param constant: system properties
    :return: phi
    """

    b = constant['b']

    coefW = get_approximation_coefficients()
    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    phidot = (psi1*eps1*U/b)*sp.exp(-eps1*U*t/b) + (psi2*eps2*U/b)*sp.exp(-eps2*U*t/b)

    return phidot
