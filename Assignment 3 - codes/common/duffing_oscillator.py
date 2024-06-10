from __future__ import division
import scipy as sp
from numpy import pi as pi


# Typical parameters used for the Duffing osciallator:
def get_constant():
    """
    creates and returns the standard set of parameters of the Duffing oscillator used in the book:
    :return: constant dictionary
    """
    constant = {
        'rho': 1.225,  # air density
        'c': 0.25,  # airfoil chord length
        'xf': 0.1,  # distance of the elastic axis from the leading edge
        'Ialfa': 0.08,  # moment of inertia
        'kalfa1': 50.,  # torsional stiffness - linear part
        'kalfa3': 50.*100,  # torsional stiffness - cubic part
    }

    return constant


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

    omegan = sp.sqrt(abs(k1/m)) # natural frequency (no damping)

    zeta = d/(2*omegan*m)
    omegad = omegan*sp.sqrt(1-zeta**2) # natural frequency including damping

    # non-dimensional parameters are introduced in at the top of page 35 in Example 2.6
    mbar = 1.
    dbar = d/(m*omegan)
    k1bar = k1/(m*omegan**2)
    k3bar = k3/(m*omegan**2)

    constant['m'] = m
    constant['d'] = d
    constant['k1'] = k1
    constant['k3'] = k3
    constant['omegan'] = omegan
    constant['omegad'] = omegad
    constant['zeta'] = zeta

    constant['mbar'] = mbar
    constant['dbar'] = dbar
    constant['k1bar'] = k1bar
    constant['k3bar'] = k3bar

    constant['a'] = a
    constant['b'] = b

    return constant


# non-linear system function governing the behaviour of the Duffing oscillator
def duffing_oscillator(x, constant):
    x1 = x[0]
    x2 = x[1]
    m = constant['m']
    d = constant['d']
    k1 = constant['k1']
    k3 = constant['k3']

    xprime = ((-d*x1 - k1*x2 - k3*x2**3)/m, x1)

    return xprime


# non-linear system function governing the behaviour of the Duffing oscillator in NONDIMENSIONAL FORM
def duffing_oscillator_nondim(x, constant):
    x1 = x[0]
    x2 = x[1]

    dbar = constant['dbar']
    k1bar = constant['k1bar']
    k3bar = constant['k3bar']

    xprime = (-dbar*x1 - k1bar*x2 - k3bar*x2**3, x1)

    return xprime


# NONDIMENSIONAL FORM of the duffing oscillator with ADDED COULOMB FRICTION
def duffing_oscillator_friction_nondim(x, constant):
    x1 = x[0]
    x2 = x[1]

    dbar = constant['dbar']
    k1bar = constant['k1bar']
    k3bar = constant['k3bar']
    Fbar = constant['F']/(constant['m']*constant['omegan']**2)  # Coulomb damping!

    xprime = (-dbar*x1 - k1bar*x2 - k3bar*x2**3 - Fbar*sp.sign(x1), x1)

    return xprime


def jacobian_duffing(x, constant):
    """
    :param x: value of the DOFs at which to evaluate the Jacobian
    :param constant: parameters of the Duffing oscillator
    :return: A, the jacobian matrix of the Duffing oscillator
    """

    x1 = x[0]
    x2 = x[1]

    dbar = constant['dbar']
    k1bar = constant['k1bar']
    k3bar = constant['k3bar']

    jacobian = [[-dbar, -k1bar - 3*k3bar*x2**2], [1, 0]]

    return jacobian


# Linearised system matrix about Fixed Point 1:
def lin_sys_mat1(U, constant):
    constant = duffing_param(U, constant)

    m = constant['m']
    d = constant['d']
    k1 = constant['k1']

    A = sp.array([[-d/m, -k1/m], [1., 0.]])

    return A


# Linearised system matrix about Fixed Point 2:
def lin_sys_mat2(U, constant):
    constant = duffing_param(U, constant)

    m = constant['m']
    d = constant['d']
    k1 = constant['k1']
    k3 = constant['k3']

    A = sp.array([[-d/m, 2*k1/m], [1., 0.]])

    return A


# Calculate the fixed points of the system
def fixed_points(U, constant):
    constant = duffing_param(U, constant)
    k1 = constant['k1']
    k3 = constant['k3']

    Xf1 = (0, 0)

    U_D = speed_critical(constant)

    if U <= U_D:
        Xf2 = (0, 0)
        Xf3 = (0, 0)
    else:
        Xf2 = (0, sp.sqrt(-k1/k3))
        Xf3 = (0, -1.*sp.sqrt(-k1/k3))

    return Xf1, Xf2, Xf3


# Calculate the fixed points of the system in non-dimensional form
def fixed_points_nondim(U, constant):
    constant = duffing_param(U, constant)
    k1bar = constant['k1bar']
    k3bar = constant['k3bar']

    Xf1 = (0, 0)

    U_D = speed_critical(constant)

    if U <= U_D:
        Xf2 = (0, 0)
        Xf3 = (0, 0)
    else:
        Xf2 = (0, sp.sqrt(-k1bar/k3bar))
        Xf3 = (0, -1.*sp.sqrt(-k1bar/k3bar))

    return Xf1, Xf2, Xf3


# Critical speed (for existance of X_F2,3):
def speed_critical(constant):
    rho = constant['rho']
    c = constant['c']
    xf = constant['xf']
    Ialfa = constant['Ialfa']
    kalfa1 = constant['kalfa1']
    kalfa3 = constant['kalfa3']

    b = c/2.
    a = xf/b - 1.

    U_D = sp.sqrt(kalfa1/(2.*pi*rho*b**2*(a + 0.5)))

    return U_D
