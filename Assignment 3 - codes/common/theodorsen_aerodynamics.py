"""
This file is obsolete! Use aerodynamics_typical_section.py instead!
"""

from __future__ import division
import scipy as sp


def theodorsen_coefficients(constant):
    a = constant['a']
    ch = constant['ch']

    mu = sp.arccos(ch)

    T1 = -(1./3.)*sp.sqrt(1 - ch**2)*(2 + ch**2) + ch*mu
    T2 = ch*(1 - ch**2) - sp.sqrt(1 - ch**2)*(1 + ch**2)*mu + ch*mu**2
    T3 = -(1./8. + ch**2)*mu**2 + (1./4.)*ch*sp.sqrt(1 - ch**2)*mu*(7 + 2*ch**2) - (1./8.)*(1 - ch**2)*(5*ch**2 + 4)
    T4 = -mu + ch*sp.sqrt(1 - ch**2)
    T5 = -(1 - ch**2) - mu**2 + 2*ch*sp.sqrt(1 - ch**2)*mu
    T6 = T2
    T7 = -(1./8. + ch**2)*mu + (1./8.)*ch*sp.sqrt(1 - ch**2)*(7 + 2*ch**2)
    T8 = -(1./3.)*sp.sqrt(1 - ch**2)*(2*ch**2 + 1) + ch*mu
    T9 = (1./2.)*((1./3.)*(sp.sqrt(1 - ch**2)**3) + a*T4)
    T10 = sp.sqrt(1 - ch**2) + mu
    T11 = mu*(1 - 2*ch) + sp.sqrt(1 - ch**2)*(2 - ch)
    T12 = sp.sqrt(1 - ch**2)*(2 + ch) - mu*(2*ch + 1)
    T13 = (1./2.)*(-T7 - (ch - a)*T1)
    T14 = (1./16.) + (1./2.)*a*ch

    # TODO: download paper: A note on typographical errors defining Theodorsen's coefficients for aeroelastic analysis
    # TODO: check coeffcient T9: is it really sqrt()**3?? -> THeodorsen paper is not conclusive

    T = {
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'T4': T4,
        'T5': T5,
        'T6': T6,
        'T7': T7,
        'T8': T8,
        'T9': T9,
        'T10': T10,
        'T11': T11,
        'T12': T12,
        'T13': T13,
        'T14': T14
    }

    return T


def theodorsen_function(k):
    """
    Calculates the value of the Theodorsen function at given reduced frequency, k. Theodorsen function is calculated
    using an approximation.

    :param k: reduced frequency, omega*b/U, [-]
    :return: value of the Theodorsen function
    """
    return 1 - 0.165/(1 - 0.0455j/k) - 0.334/(1 - 0.3j/k)
