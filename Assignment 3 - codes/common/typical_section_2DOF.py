from __future__ import division
import scipy as sp
import scipy.linalg as ln
from numpy import pi as pi


def get_constant():
    """
    Typical properties of the 2DOF system used in the book.
    NOTE: the nonlinear stiffnesses are set to 0!!!
    :return: a dictionary containing all the system properties
    """
    constant = {
        'm': 3.3843,  # section mass [kg]
        'S': 0.0859,  # section static moment [kgm]
        'Ith': 0.0135,  # section moment of inertia about pitch axis [kgm**2]
        'c': 0.2540,  # chord [m]
        'b': 0.1270,  # semi-chord (chord/2) [m]
        'a': -0.5,  # (xf/b -1) [-]
        'xf': 0.0635,  # distnace to the pitch axis (measured from the LE) [m]
        'rho': 1.225,  # air density [kg/m**3]
        'Kh': 2818.8,  # heave (plunge) stiffness [N/m]
        'Kh3': 0.0,  # heave (plunge) stiffness [N/m**3]
        'Kth': 37.3,  # pitch stiffness [Nm/rad]
        'Kth3': 0.0,  # cubic pitch stiffness [Nm/rad**3]
        'Ch': 0.0,  # structural damping in heave DOF
        'Cth': 0.0,  # structural damping in pitch DOF
        'event': 'none'  # event to detect: 'none', 'pitch', 'plunge'
    }

    return constant


def matrix_mass_str(constant):
    m = constant['m']  # mass of the section
    S = constant['S']  # static moment of the section about the pitch axis
    Ith = constant['Ith']  # moment of inertia of the section about the pitch axis

    A = [[m, S], [S, Ith]]

    return sp.array(A)


def matrix_mass_qsaero(constant):
    a = constant['a']
    b = constant['b']

    B = b**2*sp.array([[pi, -pi*a*b], [-pi*a*b, pi*b**2*(1./8. + a**2)]])

    return B


def matrix_damping_str(constant):
    Ch = constant['Ch']
    Cth = constant['Cth']

    C = [[Ch, 0.], [0., Cth]]

    return sp.array(C)


def matrix_damping_qsaero(constant):
    a = constant['a']
    b = constant['b']

    D1 = b**2*sp.array([[0, pi], [0, pi*(0.5 - a)*b]])
    D2 = sp.array([[2*pi*b, 2*pi*b**2*(0.5 - a)], [-2*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a)]])

    D = D1 + D2

    return D


def matrix_stiffness_str(constant):
    Kh = constant['Kh']
    Kth = constant['Kth']

    E = [[Kh, 0.], [0., Kth]]

    return sp.array(E)


def matrix_stiffness_qsaero(constant):
    a = constant['a']
    b = constant['b']

    F = sp.array([[0, 2*pi*b], [0, -2*pi*b**2*(a + 0.5)]])

    return F


def typical_section_2DOF_lin(x, U, constant):
    rho = constant['rho']  # air density

    A = matrix_mass_str(constant)
    B = matrix_mass_qsaero(constant)
    C = matrix_damping_str(constant)
    D = matrix_damping_qsaero(constant)
    E = matrix_stiffness_str(constant)
    F = matrix_stiffness_qsaero(constant)

    M_inv = ln.inv((A + rho*B))

    Q = sp.zeros((4, 4))
    Q[:2, :2] = (-1.)*M_inv.dot(C + rho*U*D)
    Q[:2, -2:] = (-1.)*M_inv.dot(E + rho*U**2*F)
    Q[-2:, :2] = sp.identity(2)

    xprime = Q.dot(x)

    return xprime


def typical_section_2DOF_nonlin(x, U, constant):
    rho = constant['rho']  # air density
    Kth3 = constant['Kth3']
    Kh3 = constant['Kh3']  # this might be the source of error in E3_8 -> since it is added only in E4_7

    # Linear part:
    xprime_lin = typical_section_2DOF_lin(x, U, constant)

    # nonlinear
    A = matrix_mass_str(constant)
    B = matrix_mass_qsaero(constant)

    M_inv = ln.inv((A + rho*B))

    qn_plunge = sp.zeros((4,))
    qn_plunge[:2] = -M_inv.dot([Kh3, 0])

    qn_pitch = sp.zeros((4,))
    qn_pitch[:2] = -M_inv.dot([0., Kth3])

    xprime_nlin = qn_plunge*x[2]**3 + qn_pitch*x[3]**3

    return xprime_lin + xprime_nlin


def sensitivity_dQdU(U, constant):
    """
    returns the sensitivity of the linear part of the system (Q) wrt. velocity (U)
    :param U: velocity
    :param constant: system properties
    :return: sensitivity dQdU
    """

    rho = constant['rho']  # air density

    A = matrix_mass_str(constant)
    B = matrix_mass_qsaero(constant)
    C = matrix_damping_str(constant)
    D = matrix_damping_qsaero(constant)
    E = matrix_stiffness_str(constant)
    F = matrix_stiffness_qsaero(constant)

    M_inv = ln.inv((A + rho*B))

    dQdU = sp.zeros((4, 4))
    dQdU[:2, :2] = (-1.)*M_inv.dot(rho*D)
    dQdU[:2, -2:] = (-1.)*M_inv.dot(2*rho*U*F)

    return dQdU


def rk45(dt, x, U, constant, delta=1e-3):
    x = sp.array(x)

    eps = 1.

    while eps > delta:
        # @ t_i:
        k0 = dt*typical_section_2DOF_nonlin(x, U, constant)
        x0 = x + k0/4.

        # @ t_i + dt/4:
        k1 = dt*typical_section_2DOF_nonlin(x0, U, constant)
        x1 = x + 3.*k0/32. + 9.*k1/32.

        # @ t_i + 3*dt/8:
        k2 = dt*typical_section_2DOF_nonlin(x1, U, constant)
        x2 = x + (1932.*k0 - 7200.*k1 + 7296.*k2)/2197.

        # @ t_i + 12*dt/13:
        k3 = dt*typical_section_2DOF_nonlin(x2, U, constant)
        x3 = x + 439.*k0/216. - 8.*k1 + 3680.*k2/513. - 845.*k3/4104.

        # @ t_i + dt:
        k4 = dt*typical_section_2DOF_nonlin(x3, U, constant)
        x4 = x - 8.*k0/27. + 2.*k1 - 3544.*k2/2565. + 1859.*k3/4104. - 11.*k4/40.

        # @ t_i + dt/2:
        k5 = dt*typical_section_2DOF_nonlin(x4, U, constant)
        x5 = x + 16.*k0/135. + 6656.*k2/12825. + 28561.*k3/56430. - 9.*k4/50. + 2.*k5/55.

        eps = ln.norm(x5 - x4)

        if eps > delta:
            dt = 0.5*dt

    return dt, x5


def rk45_event(dt, x, U, constant, delta=1e-3):
    delta2 = 1e-12  # governing accuracy how close do we need to resolve the local extremum

    # RK 45 method to perform the normal step:
    dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)

    event = False
    isMaximum = False
    if constant['event'] == 'pitch':
        if x_bar5[1]*x[1] < 0:  # change of the direction of angular velocity
            event = True
            while abs(x_bar5[1]) > delta2:
                dt = -dt_bar5*x[1]/(x_bar5[1] - x[1])
                dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)

            x_bar5[1] = 0  # set to 0 in order to prevent the event from firing again
            if x_bar5[3] > x[3]:  # is angular displacement a minimum or a maximum
                isMaximum = True
    elif constant['event'] == 'plunge':
        if x_bar5[0]*x[0] < 0:  # change of the direction of plunge velocity
            event = True
            while abs(x_bar5[0]) > delta2:
                dt = -dt_bar5*x[0]/(x_bar5[0] - x[0])
                dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)

            x_bar5[0] = 0  # set to 0 in order to prevent the event from firing again
            if x_bar5[2] > x[2]:  # is plunge displacement a minimum or a maximum
                isMaximum = True

    return dt_bar5, x_bar5, event, isMaximum
