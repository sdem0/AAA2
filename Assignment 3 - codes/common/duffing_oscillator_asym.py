from __future__ import division
import scipy as sp
from numpy import pi as pi


# Duffing oscillator parameters:
def duffing_param(U, constant):
    rho = constant['rho']
    c = constant['c']
    xf = constant['xf']
    Ialfa = constant['Ialfa']
    kalfa1 = constant['kalfa1']
    kalfa3 = constant['kalfa3']

    b = c/2.
    a = xf/b - 1.

    m = Ialfa + rho*pi*b**4*(1./8. + a**2)
    d = 2*rho*U*pi*b**3*a*(a - 0.5)
    k1 = kalfa1 - 2*pi*rho*U**2*b**2*(a + 0.5)
    k3 = kalfa3

    omegan = sp.sqrt(abs(k1/m))

    mbar = 1.
    dbar = d/(m*omegan)
    k1bar = k1/(m*omegan**2)
    k3bar = k3/(m*omegan**2)

    constant['a'] = a
    constant['b'] = b
    constant['m'] = m
    constant['d'] = d
    constant['k1'] = k1
    constant['k3'] = k3

    constant['mbar'] = mbar
    constant['dbar'] = dbar
    constant['k1bar'] = k1bar
    constant['k3bar'] = k3bar

    return constant


def duffing_oscillator_nondim(x, constant):
    x1 = x[0]
    x2 = x[1]

    alfaf = constant['alfaf']

    dbar = constant['dbar']
    k1bar = constant['k1bar']
    k3bar = constant['k3bar']

    xprime = (-dbar*x1 - (k1bar+3*k3bar*alfaf**2)*x2 - k3bar*x2**3 - 3*k3bar*alfaf*x2**2, x1)

    return xprime


# Linearised system matrix about Fixed Point 1:
def lin_sys_mat1(U, constant):
    constant = duffing_param(U, constant)

    m = constant['m']
    d = constant['d']
    k1 = constant['k1']
    k3 = constant['k3']
    alfaf = constant['alfaf']

    A = sp.array([[-d/m, -(k1 + 3*k3*alfaf**2)/m], [1., 0.]])

    return A


# Linearised system matrix about Fixed Point 2:
def lin_sys_mat2(Xf, U, constant):
    constant = duffing_param(U, constant)

    m = constant['m']
    d = constant['d']
    k1 = constant['k1']
    k3 = constant['k3']
    alfaf = constant['alfaf']

    A = sp.array([[-d/m, -(k1 + 3*k3*alfaf**2 + 6*k3*alfaf*Xf[1] + 3*k3*Xf[1]**2)/m], [1., 0.]])

    return A


# Calculate the fixed points of the system
def fixed_points(U, constant):
    constant = duffing_param(U, constant)
    k1 = constant['k1']
    k3 = constant['k3']
    alfaf = constant['alfaf']

    Xf1 = (0, 0)

    U_fold = speed_fold(constant)

    if U <= U_fold:
        Xf2 = (sp.nan, sp.nan)
        Xf3 = (sp.nan, sp.nan)
    else:
        Xf2 = (0, (-3*alfaf + sp.sqrt(9.*alfaf**2 - 4*(3*alfaf**2 + k1/k3)))/2)
        Xf3 = (0, (-3*alfaf - sp.sqrt(9.*alfaf**2 - 4*(3*alfaf**2 + k1/k3)))/2)

    return Xf1, Xf2, Xf3


def fixed_points_nondim(U, constant):
    constant = duffing_param(U, constant)

    alfaf = constant['alfaf']
    k1bar = constant['k1bar']
    k3bar = constant['k3bar']

    Xf1 = (0, 0)

    U_fold = speed_fold(constant)

    if U <= U_fold:
        Xf2 = (sp.nan, sp.nan)
        Xf3 = (sp.nan, sp.nan)
    else:
        Xf2 = (0, (-3*alfaf + sp.sqrt(9.*alfaf**2 - 4*(3*alfaf**2 + k1bar/k3bar)))/2)
        Xf3 = (0, (-3*alfaf - sp.sqrt(9.*alfaf**2 - 4*(3*alfaf**2 + k1bar/k3bar)))/2)

    return Xf1, Xf2, Xf3


# Critical speed (for existance of X_F2,3):
def speed_fold(constant):
    # Calculate the fold velocity of the duffing oscillator
    rho = constant['rho']
    c = constant['c']
    xf = constant['xf']
    Ialfa = constant['Ialfa']
    kalfa1 = constant['kalfa1']
    kalfa3 = constant['kalfa3']
    alfaf = constant['alfaf']

    b = c/2.
    a = xf/b - 1.

    U_fold = sp.sqrt((4*kalfa1 + 3*kalfa3*alfaf**2)/(8.*pi*rho*b**2*(a + 0.5)))

    return U_fold
