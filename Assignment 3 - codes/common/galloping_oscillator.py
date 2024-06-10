from __future__ import division
import scipy as sp
import scipy.linalg as ln
from numpy import pi as pi


def get_constant():
    """
    creates and returns the standard set of parameters used in the book:
    :return: constant dictionary
    """
    # supercritical bifurcation:
    constant = {
        'beta': 1.96e-3,
        'n': 4.3e-4,
        'A': 2.69,
        'B': 168,
        'C': 6270,
        'D': 59900
    }

    return constant


def galloping_param(V, constant):
    beta = constant['beta']
    n = constant['n']
    A, B, C, D = constant['A'], constant['B'], constant['C'], constant['D']

    C1 = -(n*A*V - 2.*beta)
    C3 = (n*B/V)
    C5 = -(n*C/V**3)
    C7 = (n*D/V**5)

    constant['C1'] = C1
    constant['C3'] = C3
    constant['C5'] = C5
    constant['C7'] = C7

    return constant


def galloping_oscillator(x, constant):
    x1 = x[0]
    x2 = x[1]

    C1 = constant['C1']
    C3 = constant['C3']
    C5 = constant['C5']
    C7 = constant['C7']

    xprime = sp.array([-(C1*x1 + C3*x1**3 + C5*x1**5 + C7*x1**7) - x2, x1])

    return xprime


def velocity_crit(constant):
    beta = constant['beta']
    n = constant['n']
    A = constant['A']

    return 2*beta/(n*A)


def galloping_oscillator_linearised(V, constant):
    beta = constant['beta']
    n = constant['n']
    A = constant['A']

    Q = sp.zeros((2, 2))

    Q[0, 0] = (n*A*V - 2*beta)
    Q[0, 1] = -1.
    Q[1, 0] = 1.

    return Q


def lco_radius_estimate(V, constant):
    beta = constant['beta']
    n = constant['n']
    A, B, C, D = constant['A'], constant['B'], constant['C'], constant['D']

    C1 = (n*A*V - 2*beta)*pi
    C3 = -3*n*B*pi/(4*V)
    C5 = 5*n*C*pi/(8*V**3)
    C7 = -35*n*D*pi/(64*V**5)

    roots = sp.empty((len(V), 7), dtype='complex')

    for i in range(len(V)):
        p = [C7[i], 0, C5[i], 0, C3[i], 0, C1[i], 0]
        roots[i, :] = sp.roots(p)

    # keep only real roots (complex roots are discarded):
    roots[roots.imag != 0] = sp.nan
    return sp.real(roots)


def lco_stability_estimate(r, V, constant):
    beta = constant['beta']
    n = constant['n']
    A, B, C, D = constant['A'], constant['B'], constant['C'], constant['D']

    return r*((n*A*V - 2*beta)*pi - (3*n*B*pi/(4*V))*r**2 + (
            5*n*C*pi/(8*V**3))*r**4 - (35*n*D*pi/(64*V**5))*r**6)


def lco_fold(constant):
    beta = constant['beta']
    n = constant['n']
    A, B, C, D = constant['A'], constant['B'], constant['C'], constant['D']

    # Find ratio w = r/V as roots of the poly. characteristic equation:
    C1 = -3.*n*B*pi/2.
    C3 = 5.*n*C*pi/2.
    C5 = -105*n*D*pi/32.
    p = [C5, 0, C3, 0, C1, 0]

    w = sp.roots(p)

    # filter out Complex values:
    w = sp.real(w[w.imag == 0])

    # Find the fold speed V (V = non-dimensional velocity):
    V_fold = 2.*beta/(n*A - 3.*n*B*w**2/4. + 5.*n*C*w**4/8. - 35.*n*D*w**6/64.)

    r = w*V_fold

    return V_fold, r, w


def poly_damping(x1, V, constant):
    beta = constant['beta']
    n = constant['n']
    A, B, C, D = constant['A'], constant['B'], constant['C'], constant['D']

    return -(n*A*V - 2*beta)*x1 + (n*B/V)*x1**3 - (n*C/V**3)*x1**5 + (
            n*D/V**5)*x1**7


def rk45(dt, x, constant, delta=1e-3):
    x = sp.array(x)

    eps = 1.

    while eps > delta:
        # @ t_i:
        k0 = dt*galloping_oscillator(x, constant)
        x0 = x + k0/4.

        # @ t_i + dt/4:
        k1 = dt*galloping_oscillator(x0, constant)
        x1 = x + 3.*k0/32. + 9.*k1/32.

        # @ t_i + 3*dt/8:
        k2 = dt*galloping_oscillator(x1, constant)
        x2 = x + (1932.*k0 - 7200.*k1 + 7296.*k2)/2197.

        # @ t_i + 12*dt/13:
        k3 = dt*galloping_oscillator(x2, constant)
        x3 = x + 439.*k0/216. - 8.*k1 + 3680.*k2/513. - 845.*k3/4104.

        # @ t_i + dt:
        k4 = dt*galloping_oscillator(x3, constant)
        x4 = x - 8.*k0/27. + 2.*k1 - 3544.*k2/2565. + 1859.*k3/4104. - 11.*k4/40.

        # @ t_i + dt/2:
        k5 = dt*galloping_oscillator(x4, constant)
        x5 = x + 16.*k0/135. + 6656.*k2/12825. + 28561.*k3/56430. - 9.*k4/50. + 2.*k5/55.

        eps = ln.norm(x5 - x4)

        if eps > delta:
            dt = 0.5*dt

    return dt, x5


# region Floquet analysis:
def jacobian_floquet(tau, r, constant):
    """
    Calculates the jacobian matrix used in the analysis of the Floquet's parameters necessary for the assessment of the
    stability of the galloping oscillator.

    NOTE: constant HAS to be updated to the velocity of interest using "galloping_param" method. Otherwise the constants
    C1...C7 will have the wrong value.

    :param tau: non-dimensional time (tau = t*omega) [-]
    :param r: amplitude (radius) of the LCO
    :param constant: properties of the galloping oscillator.
    :return: jacobian matrix
    """

    C1 = constant['C1']
    C3 = constant['C3']
    C5 = constant['C5']
    C7 = constant['C7']

    A = sp.zeros((2, 2))
    A[0, 0] = -C1 - C3*3*(r**2)*sp.cos(tau)**2 - C5*5*(r**4)*sp.cos(tau)**4 - C7*7*(r**6)*sp.cos(
        tau)**6  # the signs infront of the individual terms do not match the book because of the implementationf of C1...C7
    A[0,1] = -1
    A[1,0] = 1

    return A

# endregion
