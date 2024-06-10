from __future__ import division
import scipy as sp
import scipy.linalg as ln
import scipy.interpolate as interpolate
from numpy import pi as pi


def get_constant():
    """
    Typical properties of the 3DOF system used in the book.
    NOTE: the nonlinear stiffnesses are set to 0!!!
    :return: a dictionary containing all the system properties
    """
    constant = {
        'm': 13.5,  # section mass [kg]
        'S': 0.3375,  # section static moment [kgm]
        'Sbt': 0.1055,  # control surface static moment about the its hinge axis [kgm]
        'Ith': 0.0787,  # section moment of inertia about pitch axis [kgm**2]
        'Ith_bt': 0.0136,  # section moment of inertia about pitch axis [kgm**2]
        'Ibt': 0.0044,  # section moment of inertia about pitch axis [kgm**2]
        'c': 0.25,  # chord [m]
        'b': 0.125,  # semi-chord (chord/2) [m]
        'a': -0.2,  # main wing (xf/b -1) [-]
        'ch': 0.5,  # control surface  (xh/b -1) [-]
        'xf': 0.1,  # distance to the pitch axis (measured from the LE) [m]
        'xh': 0.1875,  # distance to the control hinge axis (measured from the LE) [m]
        'rho': 1.225,  # air density [kg/m**3]
        'Kh': 2131.8346,  # heave (plunge) stiffness [N/m]
        'Kh3': 0.0,  # heave (plunge) stiffness [N/m**3]
        'Kth': 198.9712,  # pitch stiffness [Nm/rad]
        'Kth3': 0.0,  # cubic pitch stiffness [Nm/rad**3]
        'Kbt': 17.3489,  # control DOF stiffness [Nm/rad]
        'Kbt3': 0.0,  # cubic control DOF stiffness [Nm/rad**3]
        'Ch': 0.0,  # structural damping in heave DOF
        'Cth': 0.0,  # structural damping in pitch DOF
        'Cbt': 0.0,  # structural damping in control DOF
        'event': 'none'  # event to detect: 'none', 'pitch', 'plunge', 'control'
    }

    return constant


def get_constant_DN():
    """
    Typical properties of the 3DOF system used in the book.
    NOTE: the nonlinear stiffnesses are set to 0!!!
    :return: a dictionary containing all the system properties
    """
    constant = {
        'm': 3.3843,  # section mass [kg]
        'S': 0.08587,  # section static moment [kgm]
        'Sbt': 0.00395,  # control surface static moment about the its hinge axis [kgm]
        'Ith': 0.01347,  # section moment of inertia about pitch axis [kgm**2]
        'Ith_bt': 0.0008281,  # section moment of inertia about pitch axis [kgm**2]
        'Ibt': 0.0003264,  # section moment of inertia about pitch axis [kgm**2]
        'c': 0.25,  # chord [m]
        'b': 0.125,  # semi-chord (chord/2) [m]
        'a': -0.5,  # main wing (xf/b -1) [-]
        'ch': 0.5,  # control surface  (xh/b -1) [-]
        'xf': 0.0625,  # distance to the pitch axis (measured from the LE) [m]
        'xh': 0.1875,  # distance to the control hinge axis (measured from the LE) [m]
        'rho': 1.225,  # air density [kg/m**3]
        'Kh': 3340,  # heave (plunge) stiffness [N/m]
        'Kh3': 0.0,  # heave (plunge) stiffness [N/m**3]
        'Kth': 43.07,  # pitch stiffness [Nm/rad]
        'Kth3': 0.0,  # cubic pitch stiffness [Nm/rad**3]
        'Kbt': 2.90,  # control DOF stiffness [Nm/rad]
        'Kbt3': 0.0,  # cubic control DOF stiffness [Nm/rad**3]
        'Ch': 3340./4000.,  # structural damping in heave DOF
        'Cth': 43.07/4000.,  # structural damping in pitch DOF
        'Cbt': 2.90/4000.,  # structural damping in control DOF
        'event': 'none'  # event to detect: 'none', 'pitch', 'plunge', 'control'
    }

    return constant


def matrix_mass_str(constant):
    m = constant['m']  # mass of the section
    S = constant['S']  # static moment of the section about the pitch axis
    Sbt = constant['Sbt']  # static moment of the control surface about its hinge axis
    Ith = constant['Ith']  # moment of inertia of the section about the pitch axis
    Ith_bt = constant['Ith_bt']  # moment of inertia of the section about the pitch axis
    Ibt = constant['Ibt']  # moment of inertia of the section about the pitch axis

    A = sp.array([[m, S, Sbt], [S, Ith, Ith_bt], [Sbt, Ith_bt, Ibt]])

    return A


def matrix_mass_usaero(constant, T):
    """
    Calculate unsteady aerodynamic mass matrix
    :param constant: dictionary of system properties
    :param T: dictionary of Theodorsen coefficients
    :return: unsteady aerodynamic mass matrix
    """
    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    # Theodorsen coefficients:
    T1 = T['T1']
    T3 = T['T3']
    T7 = T['T7']
    T13 = T['T13']

    B = b**2*sp.array([[pi, -pi*a*b, -T1*b],
                       [-pi*a*b, pi*b**2*(1./8. + a**2), -(T7 + (ch - a)*T1)*b**2],
                       [-T1*b, 2*T13*b**2, -T3*b**2/pi]])

    return sp.array(B)


def matrix_mass_thaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """
    from theodorsen_aerodynamics import theodorsen_coefficients

    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    T = theodorsen_coefficients(constant)
    # Theodorsen coefficients:
    T1 = T['T1']
    T3 = T['T3']
    T7 = T['T7']
    T13 = T['T13']

    B = b**2*sp.array([[pi, -pi*a*b, -T1*b],
                       [-pi*a*b, pi*b**2*(1./8. + a**2), -(T7 + (ch - a)*T1)*b**2],
                       [-T1*b, 2*T13*b**2, -T3*b**2/pi]])

    return sp.array(B)


def matrix_damping_str(constant):
    Ch = constant['Ch']
    Cth = constant['Cth']
    Cbt = constant['Cbt']

    C = [[Ch, 0., 0.], [0., Cth, 0.], [0., 0., Cbt]]

    return sp.array(C)


def matrix_damping_usaero(constant, T, coefW):
    """
    Calculate unsteady aerodynamic damping matrix
    :param constant: dictionary of system properties
    :param T: dictionary of Theodorsen coefficients
    :param coefW: coefficients of the Wagner function approximation
    :return: unsteady aerodynamic damping matrix
    """
    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    T1 = T['T1']
    T4 = T['T4']
    T8 = T['T8']
    T9 = T['T9']
    T11 = T['T11']
    T12 = T['T12']

    psi1 = coefW['psi1']
    psi2 = coefW['psi2']

    D1 = b**2*sp.array([[0, pi, -T4],
                        [0, pi*(0.5 - a)*b, (T1 - T8 - (ch - a)*T4 + T11/2.)*b],
                        [0, (-2.*T9 - T1 + T4*(a - 0.5))*b, -T4*T11*b/(2.*pi)]])

    D2 = sp.array([[2*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    D = D1 + (1 - psi1 - psi2)*D2

    return D


def matrix_damping_thaero(k, constant):
    """
    Calculate unsteady aerodynamic damping matrix
    :param k: reduced frequency, omega*b/U, [-]
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """

    from theodorsen_aerodynamics import theodorsen_coefficients
    from theodorsen_aerodynamics import theodorsen_function

    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    T = theodorsen_coefficients(constant)

    T1 = T['T1']
    T4 = T['T4']
    T8 = T['T8']
    T9 = T['T9']
    T11 = T['T11']
    T12 = T['T12']

    C = theodorsen_function(k)

    D1 = b**2*sp.array([[0, pi, -T4],
                        [0, pi*(0.5 - a)*b, (T1 - T8 - (ch - a)*T4 + T11/2.)*b],
                        [0, (-2.*T9 - T1 + T4*(a - 0.5))*b, -T4*T11*b/(2.*pi)]])

    D2 = sp.array([[2*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    D = D1 + C*D2

    return D


def matrix_stiffness_str(constant):
    Kh = constant['Kh']
    Kth = constant['Kth']
    Kbt = constant['Kbt']

    E = [[Kh, 0., 0.], [0., Kth, 0.], [0., 0., Kbt]]

    return sp.array(E)


def matrix_stiffness_usaero(constant, T, coefW):
    """
    Calculate unsteady aerodynamic stiffness matrix
    :param constant: dictionary of system properties
    :param T: dictionary of Theodorsen coefficients
    :param coefW: coefficients of the Wagner function approximation
    :return: unsteady aerodynamic stiffness matrix
    """
    a = constant['a']
    b = constant['b']

    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T11 = T['T11']
    T12 = T['T12']

    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    F1 = b**2*sp.array([[0., 0., 0.],
                        [0., 0., (T4 + T10)],
                        [0., 0., (T5 - T4*T10)/pi]])

    F2 = sp.array([[0., 2*pi*b, 2*b*T10],
                   [0., -2*pi*b**2*(a + 0.5), -2*b**2*(a + 0.5)*T10],
                   [0., b**2*T12, b**2*T12*T10/pi]])

    F3 = sp.array([[2.*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2.*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    F = F1 + (1 - psi1 - psi2)*F2 + (psi1*eps1/b + psi2*eps2/b)*F3

    return F


def matrix_stiffness_thaero(k, constant):
    """
    Calculate unsteady aerodynamic stiffness matrix
    :param k: reduced frequency, omega*b/U, [-]
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """

    from theodorsen_aerodynamics import theodorsen_coefficients
    from theodorsen_aerodynamics import theodorsen_function

    a = constant['a']
    b = constant['b']

    T = theodorsen_coefficients(constant)

    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T12 = T['T12']

    C = theodorsen_function(k)

    F1 = b**2*sp.array([[0., 0., 0.],
                        [0., 0., (T4 + T10)],
                        [0., 0., (T5 - T4*T10)/pi]])

    F2 = sp.array([[0., 2*pi*b, 2*b*T10],
                   [0., -2*pi*b**2*(a + 0.5), -2*b**2*(a + 0.5)*T10],
                   [0., b**2*T12, b**2*T12*T10/pi]])

    F = F1 + C*F2

    return F


def matrix_stiffness_qsaero(constant, T):
    """
    Calculate unsteady aerodynamic stiffness matrix
    :param constant: dictionary of system properties
    :param T: dictionary of Theodorsen coefficients
    :param coefW: coefficients of the Wagner function approximation
    :return: unsteady aerodynamic stiffness matrix
    """
    a = constant['a']
    b = constant['b']

    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T12 = T['T12']

    F1 = b**2*sp.array([[0., 0., 0.],
                        [0., 0., (T4 + T10)],
                        [0., 0., (T5 - T4*T10)/pi]])

    F2 = sp.array([[0., 2*pi*b, 2*b*T10],
                   [0., -2*pi*b**2*(a + 0.5), -2*b**2*(a + 0.5)*T10],
                   [0., b**2*T12, b**2*T12*T10/pi]])

    F = F1 + F2

    return F


def matrix_aero_influence(constant, T, coefW):
    """
    Calculate the aerodynamic state influence matrix
    :param constant: dictionary of system properties
    :param T: dictionary of Theodorsen coefficients
    :param coefW: coefficients of the Wagner function approximation
    :return: aerodynamic state influence matrix (size: 3x6)
    """

    a = constant['a']
    b = constant['b']

    T10 = T['T10']
    T11 = T['T11']
    T12 = T['T12']

    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    W0 = [-psi1*(eps1/b)**2,
          -psi2*(eps2/b)**2,
          psi1*eps1*(1 - eps1*(0.5 - a))/b,
          psi2*eps2*(1 - eps2*(0.5 - a))/b,
          psi1*eps1*(T10 - eps1*T11/2.)/(pi*b),
          psi2*eps2*(T10 - eps2*T11/2.)/(pi*b)]

    W = sp.array([W0, W0, W0])
    W[0, :] *= 2*pi*b
    W[1, :] *= -2*pi*b**2*(a + 0.5)
    W[2, :] *= b**2*T12

    return W


def matrix_aero_state1():
    W1 = sp.array([[1., 0, 0],
                   [1., 0, 0],
                   [0, 1., 0],
                   [0, 1., 0],
                   [0, 0, 1.],
                   [0, 0, 1.]])

    return W1


def matrix_aero_state2(constant, coefW):
    b = constant['b']

    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    W2 = sp.identity(6)
    W2[0::2, 0::2] *= -eps1/b
    W2[1::2, 1::2] *= -eps2/b

    return W2


def matrix_aero_influence_theodorsen(k, constant):
    """
    this function calculates the aerodynamic influence matrix in frequency domain based on the Theodorsen aerodynamics.

    :param k: reduced frequency, omega*b/U, [-]
    :param constant: system properties
    :return: aerodyamic influene matrix
    """

    b = constant['b']

    # Aerodynamic mass matrix:
    B = matrix_mass_thaero(constant)

    # Aerodunamic damping matrix:
    D = matrix_damping_thaero(k, constant)

    # Aerodynamic stiffness matrix:
    F = matrix_stiffness_thaero(k, constant)

    # Assembled matrix of aerodynamic influence coefficients:
    Q = (-k**2/b**2)*B + (k*1j/b)*D + F

    return Q


def vector_init_condt(x0, constant, T):
    """
    this function returns the (3,) initial condition excitation vector derived from the physical initial condition of
    the system
    :param x0: initial condition vecotr of the physical states (h, th, bt) at t = 0.
    :param constant: system parameters
    :param T: Theodorsen coefficients
    :return: g, the initial condition excitation vector
    """
    h0, th0, bt0 = x0[0], x0[1], x0[2]

    a = constant['a']
    b = constant['b']

    T11 = T['T11']
    T12 = T['T12']

    g = b*(h0 + b*(0.5 - a)*th0 + b*T11*bt0/(2.*pi))*sp.array([2.*pi, -2.*pi*b*(a + 0.5), b*T12])

    return g


def typical_section_3DOF_lin(x, U, constant, T, coefW):
    rho = constant['rho']  # air density

    A = matrix_mass_str(constant)
    B = matrix_mass_usaero(constant, T)
    C = matrix_damping_str(constant)
    D = matrix_damping_usaero(constant, T, coefW)
    E = matrix_stiffness_str(constant)
    F = matrix_stiffness_usaero(constant, T, coefW)
    W = matrix_aero_influence(constant, T, coefW)
    W1 = matrix_aero_state1()
    W2 = matrix_aero_state2(constant, coefW)

    M_inv = ln.inv((A + rho*B))

    Q = sp.zeros((12, 12))
    Q[:3, :3] = (-1.)*M_inv.dot(C + rho*U*D)
    Q[:3, 3:6] = (-1.)*M_inv.dot(E + rho*U**2*F)
    Q[:3, 6:] = (-1.)*rho*U**3*M_inv.dot(W)
    Q[3:6, :3] = sp.identity(3)
    Q[6:, 3:6] = W1
    Q[6:, 6:] = U*W2

    xprime = Q.dot(x)

    return xprime


def typical_section_3DOF_init(phidot, U, x0, constant, T):
    """
    This function returns the initial condition excitation term,

    :param phidot: time derivative of Wagner function
    :param U: velocity [m/s]
    :param x0: initial condition
    :param constant: parameters of the system
    :param T: Theodorsen coefficients
    :return: initial condition excitation term
    """

    rho = constant['rho']  # air density

    A = matrix_mass_str(constant)
    B = matrix_mass_usaero(constant, T)
    g = vector_init_condt(x0, constant, T)

    M_inv = ln.inv((A + rho*B))

    q = sp.zeros((12,))
    q[:3] = rho*U*M_inv.dot(g)

    return q*phidot


def typical_section_3DOF_div_speed(constant, T):
    """
    Calcualte the divergence speed of the typical section.

    :param constant: system parameters
    :param T: Theodorsen coefficients
    :return: divergence speed
    """

    a = constant['a']
    b = constant['b']

    Kth = constant['Kth']
    Kbt = constant['Kbt']

    rho = constant['rho']

    # Theodorsen coefficients:
    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T12 = T['T12']

    A = -(2*(a + 0.5)*(T5 - T4*T10 + T12*T10) + T12*(T4 + T10 - 2*(a + 0.5)*T10))*rho**2*b**4
    B = ((Kth/pi)*(T5 - T4*T10 + T12*T10) - 2*pi*(a + 0.5)*Kbt)*rho*b**2
    C = Kth*Kbt

    roots = sp.roots([A, 0, B, 0, C])

    UD = sp.real(roots[(sp.isreal(roots)) & (roots.real > 0)]).min()
    return UD


def typical_section_3DOF_nonlin(x, U, constant, T, coefW):
    rho = constant['rho']  # air density
    Kh3 = constant['Kh3']
    Kth3 = constant['Kth3']
    Kbt3 = constant['Kbt3']

    # Linear part:
    xprime_lin = typical_section_3DOF_lin(x, U, constant, T, coefW)

    # nonlinear
    A = matrix_mass_str(constant)
    B = matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A + rho*B))

    qn_plunge = sp.zeros((12,))
    qn_plunge[:3] = -M_inv.dot([Kh3, 0., 0.])

    qn_pitch = sp.zeros((12,))
    qn_pitch[:3] = -M_inv.dot([0., Kth3, 0.])

    qn_control = sp.zeros((12,))
    qn_control[:3] = -M_inv.dot([0., 0., Kbt3])

    xprime_nlin = qn_plunge*x[3]**3 + qn_pitch*x[4]**3 + qn_control*x[5]**3

    return xprime_lin + xprime_nlin


def fixed_points_3DOF_nonlin_th(U, constant, T, coefW):
    epsC = 1e-10
    rho = constant['rho']  # air density
    Kth3 = constant['Kth3']

    E = matrix_stiffness_str(constant)
    F = matrix_stiffness_usaero(constant, T, coefW)
    W = matrix_aero_influence(constant, T, coefW)
    W1 = matrix_aero_state1()
    W2 = matrix_aero_state2(constant, coefW)

    Q = sp.zeros((9, 9))
    Q[:3, :3] = E + rho*U**2*F
    Q[:3, 3:] = rho*U**3*W
    Q[3:, :3] = W1
    Q[3:, 3:] = U*W2

    # omit the second row and column to get the matrix of the linear subsystem of equations.
    Ql = sp.delete(Q, 1, 0)
    Ql = sp.delete(Ql, 1, 1)
    Qlinv = ln.inv(Ql)

    # linear contribution of the pitch DOF to the linear subsytem of equations
    ql = Q[:, 1]
    ql = sp.delete(ql, 1, 0)

    # linear contribution of the remaining DOF (plunge, control, aerodynamic states) to the nonlinear subsystem of
    # equtions (equation for th)
    qn = Q[1, :]  # qn.shape  is (9,)
    qn = sp.delete(qn, 1)

    # linear contribution of the pitch DOF to the nonlinear subsystem  of equations (equation for th)
    qnn = Q[1, 1]

    # Calculate Fixed points:
    # =======================

    # One fixed point is always 0:
    qF1 = sp.zeros((9,))

    # The remaining two fixed points:

    # th value:
    thF2 = sp.sqrt((qn.dot(Qlinv.dot(ql)) - qnn)/Kth3)
    thF3 = (-1.)*sp.sqrt((qn.dot(Qlinv.dot(ql)) - qnn)/Kth3)

    # check if thF2 or thF3 are complex if so -> discard -> make it equal to thF1
    if sp.absolute(thF2.imag) > epsC:
        thF2 = qF1[2]

    if sp.absolute(thF3.imag) > epsC:
        thF3 = qF1[2]

    # Remaining DOF vlaues:
    qF2 = (-1.)*Qlinv.dot(ql)*thF2
    qF3 = (-1.)*Qlinv.dot(ql)*thF3

    # insert thF2 and thF3 to get a complete vector of DOFs at fixed point.
    qF2 = sp.insert(qF2, 1, thF2)
    qF3 = sp.insert(qF3, 1, thF3)

    # Expand the reduced vectors of DOFs to a full vector:

    # Expand the reduced vectors of DOFs to a full vector of DOF which contains also the velocities of the structural
    # DOFs.(To maintain compatibility with the previously implemented methods -> e.g. the Jacobian of the nonlinear
    # system)

    mask = sp.zeros((12,), dtype=bool)
    mask[3:] = True

    xF1, xF2, xF3 = sp.zeros((12,)), sp.zeros((12,)), sp.zeros((12,))
    xF2[mask], xF3[mask] = qF2, qF3

    xF = sp.zeros((12, 3))
    xF[:, 0] = xF1
    xF[:, 1] = xF2
    xF[:, 2] = xF3

    # Stability of Fixed Points:
    # ==========================

    isStable = sp.ones((3,), dtype=bool)

    for i in range(3):
        jac = jacobian_nonlin(xF[:, i], U, constant, T, coefW)
        evals = ln.eigvals(jac)
        evals_real = evals[sp.absolute(evals.imag) < epsC]

        # ONLY the REAL eigenvalues (corresponding to aerodynamic states) must be considered because we are looking at
        # the stability wrt. divergence. Flutter instability has already occured (a pair of complex eig. vals with
        # positive real part) but it is stabilised with the LCO.
        if any(evals_real > 0):
            isStable[i] = False

    return xF, isStable


def typical_section_3DOF_eqlin(x, A, U, constant, T, coefW):
    """
    Calculates x_prime using equivalent linearisation in the case of CUBIC hardening stiffness in each DOF

    :param x: vector of DOF at current time step
    :param A: vector of amplitude ([amp_h, amp_th, amp_bt])
    :param U: velocity
    :param constant: properties of the nonlinear system
    :param T: Theodorsen coefficients
    :param coefW: Wagner coefficients
    :return: x_primer
    """
    rho = constant['rho']  # air density
    Kh3 = constant['Kh3']
    Kth3 = constant['Kth3']
    Kbt3 = constant['Kbt3']

    # Linear part:
    # ============
    Q_lin = typical_section_3DOF_lin(sp.identity(12), U, constant, T, coefW)

    # Equivalent stiffness matrix - nonlinear part:
    # =============================================
    A_tmp = matrix_mass_str(constant)
    B_tmp = matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A_tmp + rho*B_tmp))

    E_nlin = sp.zeros((3, 3))
    E_nlin[0, 0] = 3.*Kh3*A[0]**2/4.
    E_nlin[1, 1] = 3.*Kth3*A[1]**2/4.
    E_nlin[2, 2] = 3.*Kbt3*A[2]**2/4.

    Q_eqlin = Q_lin
    Q_eqlin[:3, 3:6] += -M_inv.dot(E_nlin)

    return Q_eqlin.dot(x)


def jacobian_nonlin(x0, U, constant, T, coefW):
    """
    Determine the jacobian of the nonlinear 3DOF system around the point x0.

    :param x0: vector of DOF at which to calculate the Jacobian
    :param U: free-stream velocity
    :param constant: system properties
    :param T: Theodorsen coefficients
    :param coefW: Wagner function coefficients
    :return: Jacobian of the nonlinear system at point x0
    """

    rho = constant['rho']  # air density
    Kh3 = constant['Kh3']
    Kth3 = constant['Kth3']
    Kbt3 = constant['Kbt3']

    # Linear part:
    jac_lin = typical_section_3DOF_lin(sp.identity(12), U, constant, T, coefW)

    # Non-linear part:
    A = matrix_mass_str(constant)
    B = matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A + rho*B))

    qn_plunge = sp.zeros((12,))
    qn_plunge[:3] = -M_inv.dot([Kh3, 0., 0.])

    qn_pitch = sp.zeros((12,))
    qn_pitch[:3] = -M_inv.dot([0., Kth3, 0.])

    qn_control = sp.zeros((12,))
    qn_control[:3] = -M_inv.dot([0., 0., Kbt3])

    jac_nonlin = sp.zeros((12, 12))
    jac_nonlin[:, 3] = 3*qn_plunge*x0[3]**2
    jac_nonlin[:, 4] = 3*qn_pitch*x0[4]**2
    jac_nonlin[:, 5] = 3*qn_control*x0[5]**2

    return jac_lin + jac_nonlin


def sensitivity_dQdU_eqlin(U, constant, T, coefW):
    rho = constant['rho']  # air density

    A = matrix_mass_str(constant)
    B = matrix_mass_usaero(constant, T)
    C = matrix_damping_str(constant)
    D = matrix_damping_usaero(constant, T, coefW)
    E = matrix_stiffness_str(constant)
    F = matrix_stiffness_usaero(constant, T, coefW)
    W = matrix_aero_influence(constant, T, coefW)
    W1 = matrix_aero_state1()
    W2 = matrix_aero_state2(constant, coefW)

    M_inv = ln.inv((A + rho*B))

    dQdU = sp.zeros((12, 12))
    dQdU[:3, :3] = (-1.)*M_inv.dot(rho*D)
    dQdU[:3, 3:6] = (-1.)*M_inv.dot(2*rho*U*F)
    dQdU[:3, 6:] = (-1.)*3*rho*U**2*M_inv.dot(W)
    dQdU[6:, 6:] = W2

    return dQdU


def LCO_branch_eqlin(i_branch, A, Umin, Umax, constant, T, coefW, delta_NR=1e-10):
    """
    The function pinpoints the velocity of the LCO for a given amplitude. The function  is based on newton-Raphson
    nonlinear solver. Therefore a fairly good initial guess of the LCO velocity has to be provided.

    :param i_branch: indicates whcih LCO branch to follow. valid values: 0, 1, 2.
    :param A: LCO amplitude vector (h, th, beta)
    :param U: initial guess for the LCO velocity [m/s]
    :param constant: system properties
    :param T: Theodorsen coefficients
    :param coefW: Wagner coefficients
    :param delta_NR: accuracy of the Newton-Raphson search algorithm
    :return: U, omega of the LCO with amplitude A
    """

    nn = 100
    U = sp.linspace(Umin, Umax, nn)
    w_n = sp.zeros((nn, 3))
    zeta = sp.zeros((nn, 3))

    zeta_min, zeta_max, idx_max = 0, 0, 0
    is_hopf = False

    # Locate the LCO velocity:
    for i, U_i in enumerate(U):

        if i > 3:
            f_w = interpolate.interp1d(U[:i], w_n[:i, :], kind='cubic', axis=0, fill_value='extrapolate')
            f_zeta = interpolate.interp1d(U[:i], zeta[:i, :], kind='cubic', axis=0, fill_value='extrapolate')

            # IMPORTANT:
            # reshape is used to transform the vector from shape (3,) to shape (3,1) -> allows for broadcasting along the
            # last dimension of the reshaped array. This omits the need to copy the array 3 times in order to calculate all
            # the possible differences in the next step
            zeta_interp = f_zeta(U_i)
            w_interp = f_w(U_i)

            if i == 47:
                tmp = 5
        else:
            zeta_interp = sp.nan*sp.ones((3,))
            w_interp = sp.nan*sp.ones((3,))

        w_n[i], zeta[i] = LCO_frequency_damping_eqlin(A, U_i, w_interp, zeta_interp, T, coefW, constant)

        # check for Hopf biffurcation (occurence of flutter point):
        if i > 0:
            if zeta[i - 1, i_branch]*zeta[i, i_branch] < 0:
                Umin = U[i - 1]
                Umax = U[i]

                zeta_min = zeta[i - 1, i_branch]
                zeta_max = zeta[i, i_branch]

                idx_max = i

                is_hopf = True

                break

    # Pinpoint the flutter speed:
    # ===========================

    if is_hopf:
        # NOTE:
        # idx_max +1 is required in ordred to include the last calculated zeta in the interpolation function!
        # Otherwise you keep extrapolating.

        f_w = interpolate.interp1d(U[:idx_max + 1], w_n[:idx_max + 1, :], kind='cubic', axis=0,
                                   fill_value='extrapolate')
        f_zeta = interpolate.interp1d(U[:idx_max + 1], zeta[:idx_max + 1, :], kind='cubic', axis=0,
                                      fill_value='extrapolate')

        # use te Bisection method on the damping coefficient of the selected branch
        R1 = 1
        dU = Umax - Umin

        omega_max = 0

        while (sp.absolute(R1) > delta_NR):
            dU = -dU*zeta_min/(zeta_max - zeta_min)
            Umax = Umin + dU

            w_interp = f_w(Umax)
            zeta_interp = f_zeta(Umax)

            w_n_tmp, zeta_tmp = LCO_frequency_damping_eqlin(A, Umax, w_interp, zeta_interp, T, coefW, constant)
            zeta_max = zeta_tmp[i_branch]
            omega_max = w_n_tmp[i_branch]

            R1 = zeta_max

        return Umax, omega_max

    else:
        return sp.nan, sp.nan


def LCO_frequency_damping_eqlin(A, U_i, w_interp, zeta_interp, T, coefW, constant):
    # Assemble 3DOF system matrix
    x0 = sp.identity(12)  # dummy input to retrieve the 3DOF system matrix
    Q = typical_section_3DOF_eqlin(x0, A, U_i, constant, T, coefW)

    # Calculate eigen values and eigen vectors
    eval, evec = ln.eig(Q)

    # sort out the complex-conjugate eigenvalues and their pertinent eigen vectors:
    eps_fltr = 1e-12
    eval_cc = eval[eval.imag > eps_fltr]
    evec_cc = evec[:, eval.imag > eps_fltr]
    idx = sp.argsort(eval_cc.imag)
    eval_cc = eval_cc[idx]
    if len(eval_cc) > 3:
        print(eval_cc)

    w_n = sp.absolute(eval_cc)
    zeta = -eval_cc.real/sp.absolute(eval_cc)

    # sort based on the exrtrapolted value:
    if not (any(sp.isnan(zeta_interp))):
        # IMPORTANT:
        # reshape is used to transform the vector from shape (3,) to shape (3,1) -> allows for broadcasting along the
        # last dimension of the reshaped array. This omits the need to copy the array 3 times in order to calculate all
        # the possible differences in the next step
        zeta_interp = sp.reshape(zeta_interp, (3, 1))
        w_interp = sp.reshape(w_interp, (3, 1))

        dw = sp.absolute(w_n - w_interp)
        dzeta = sp.absolute(zeta - zeta_interp)

        diff = dw*dzeta + dzeta

        idx_min1 = dw.argmin(axis=0)
        idx_min2 = diff.argmin(axis=0)  # find the minimum element in each row

        w_n = w_n[idx_min2]
        zeta = zeta[idx_min2]

    return w_n, zeta


def stability_LCO_eqlin(A_up, A_down, U, constant, T, coefW):
    Q_up = typical_section_3DOF_eqlin(sp.identity(12), A_up, U, constant, T, coefW)
    evals_up = ln.eigvals(Q_up)

    Q_down = typical_section_3DOF_eqlin(sp.identity(12), A_down, U, constant, T, coefW)
    evals_down = ln.eigvals(Q_down)

    stable_up = True
    if any(evals_up.real > 0):
        stable_up = False

    stable_down = False
    if any(evals_down.real > 0):
        stable_down = True

    return (stable_down and stable_up)


def tvla(dt, x0, U, constant, T, coefW, delta=1e-3):
    """
    tvla stands for Time-Varying Linear Approximation. It calculates the next value of the DOF system
    :param dt:
    :param x0:
    :param U:
    :param constant:
    :param T:
    :param coefW:
    :return: the used time step
    """

    A = jacobian_nonlin(x0, U, constant, T, coefW)
    f = typical_section_3DOF_nonlin(x0, U, constant, T, coefW)

    L, V = ln.eig(A)
    Vinv = ln.inv(V)

    b = Vinv.dot(f)

    # Check convergence of the time step
    while True:
        dx = sp.zeros((12,), dtype='complex')
        for i in range(12):
            dx += -1*V[:, i]*(1/L[i])*(1 - sp.exp(L[i]*dt))*b[i]

        x_bar = x0 + dx.real

        A_bar = jacobian_nonlin(x_bar, U, constant, T, coefW)
        f_bar = typical_section_3DOF_nonlin(x_bar, U, constant, T, coefW)

        L_bar, V_bar = ln.eig(A_bar)
        Vinv_bar = ln.inv(V_bar)

        b_bar = Vinv_bar.dot(f_bar)

        dx_bar = sp.zeros((12,), dtype='complex')
        for i in range(12):
            dx_bar += -1*V_bar[:, i]*(1/L_bar[i])*(1 - sp.exp(-1.*L_bar[i]*dt))*b_bar[i]

        error = dx.real + dx_bar.real
        eps = sp.sqrt(error.dot(error))

        if eps < delta:
            break
        else:
            dt *= 0.5

    x = x0 + dx.real

    return dt, x


def tvla_event(dt, x, U, constant, T, coefW, delta=1e-3):
    """
    tvla_event stands for Time-Varying Linear Approximation with detection of a max/min value of DOF. The
    method is using tvla method as a calculation engine to calculate the next value of the DOF system

    :param dt: initial time step
    :param x: initial condition (known value of DOF)
    :param U: free-stream velocity
    :param constant: system properties
    :param T: Theodorsen coefficients
    :param coefW: Wagner function coefficients
    :param delta: convergence criteria
    :return:
        dt_bat: accepted timestep
        x_bar: value of DOF @ t+ dt_bar
    """
    delta2 = 1e-12

    # tvla method to perform the normal step:
    dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)

    event = False
    if constant['event'] == 'plunge':
        if x_bar[0]*x[0] < 0:  # plunge DOF crossed through 0
            event = True
            while abs(x_bar[0]) > delta2:
                dt = -dt_bar*x[0]/(x_bar[0] - x[0])
                dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)

            x_bar[0] = 0  # set to 0 in order to prevent the event from firing again

    elif constant['event'] == 'pitch':
        if x_bar[1]*x[1] < 0:  # pitch DOF crossed through 0
            event = True
            while abs(x_bar[1]) > delta2:
                dt = -dt_bar*x[1]/(x_bar[1] - x[1])
                dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)

            x_bar[1] = 0  # set to 0 in order to prevent the event from firing again

    elif constant['event'] == 'control':
        if x_bar[2]*x[2] < 0:  # control DOF crossed through 0
            event = True
            while abs(x_bar[2]) > delta2:
                dt = -dt_bar*x[2]/(x_bar[2] - x[2])
                dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)

            x_bar[2] = 0  # set to 0 in order to prevent the event from firing again

    return dt_bar, x_bar, event


def tvla_event_zero(dt, x, U, constant, T, coefW, delta=1e-3):
    """
    tvla_event_zero stands for Time-Varying Linear Approximation with detection of a DOF passing through zero. The
    method is using tvla method as a calculation engine to calculate the next value of the DOF system

    :param dt: initial time step
    :param x: initial condition (known value of DOF)
    :param U: free-stream velocity
    :param constant: system properties
    :param T: Theodorsen coefficients
    :param coefW: Wagner function coefficients
    :param delta: convergence criteria
    :return:
        dt_bat: accepted timestep
        x_bar: value of DOF @ t+ dt_bar
    """
    delta2 = 1e-12

    # tvla method to perform the normal step:
    dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)

    idx_event = -1
    if constant['event'] == 'plunge':
        idx_event = 3

    elif constant['event'] == 'pitch':
        idx_event = 4

    elif constant['event'] == 'control':
        idx_event = 5

    event = False
    if idx_event > 0:
        if x_bar[idx_event]*x[idx_event] < 0:  # pitch DOF crossed through 0
            event = True

            dtA = 0
            dtB = dt
            xA = x
            xB = x_bar

            while abs(x_bar[idx_event]) > delta2:

                # TODO: implement regula falsi method -> the secant method is NOT guaranteed to converge!

                # Regula falsi:
                # =============
                dtC = (dtA*xB[idx_event] - dtB*xA[idx_event])/(xB[idx_event] - xA[idx_event])
                dt_tmp, xC = tvla(dtC, x, U, constant, T, coefW, delta)

                if xC[idx_event]*xA[idx_event] > 0:
                    dtA = dtC
                    xA = xC
                else:
                    dtB = dtC
                    xB = xC

                if abs(xA[idx_event]) < abs(xB[idx_event]):
                    x_bar = xA
                    dt_bar = dtA
                else:
                    x_bar = xB
                    dt_bar = dtB

                # the old method:
                # dt = -dt_bar*x[4]/(x_bar[4] - x[4])
                # dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)

            x_bar[idx_event] = 0  # set to 0 in order to prevent the event from firing again

    return dt_bar, x_bar, event

# def rk45(dt, x, U, constant, delta=1e-3):
#     x = sp.array(x)
#
#     eps = 1.
#
#     while eps > delta:
#         # @ t_i:
#         k0 = dt*typical_section_2DOF_nonlin(x, U, constant)
#         x0 = x + k0/4.
#
#         # @ t_i + dt/4:
#         k1 = dt*typical_section_2DOF_nonlin(x0, U, constant)
#         x1 = x + 3.*k0/32. + 9.*k1/32.
#
#         # @ t_i + 3*dt/8:
#         k2 = dt*typical_section_2DOF_nonlin(x1, U, constant)
#         x2 = x + (1932.*k0 - 7200.*k1 + 7296.*k2)/2197.
#
#         # @ t_i + 12*dt/13:
#         k3 = dt*typical_section_2DOF_nonlin(x2, U, constant)
#         x3 = x + 439.*k0/216. - 8.*k1 + 3680.*k2/513. - 845.*k3/4104.
#
#         # @ t_i + dt:
#         k4 = dt*typical_section_2DOF_nonlin(x3, U, constant)
#         x4 = x - 8.*k0/27. + 2.*k1 - 3544.*k2/2565. + 1859.*k3/4104. - 11.*k4/40.
#
#         # @ t_i + dt/2:
#         k5 = dt*typical_section_2DOF_nonlin(x4, U, constant)
#         x5 = x + 16.*k0/135. + 6656.*k2/12825. + 28561.*k3/56430. - 9.*k4/50. + 2.*k5/55.
#
#         eps = ln.norm(x5 - x4)
#
#         if eps > delta:
#             dt = 0.5*dt
#
#     return dt, x5
#
#
# def rk45_event(dt, x, U, constant, delta=1e-3):
#     delta2 = 1e-12  # governing accuracy how close do we need to resolve the local extremum
#
#     # RK 45 method to perform the normal step:
#     dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)
#
#     event = False
#     isMaximum = False
#     if constant['event'] == 'pitch':
#         if x_bar5[1]*x[1] < 0:  # change of the direction of angular velocity
#             event = True
#             while abs(x_bar5[1]) > delta2:
#                 dt = -dt_bar5*x[1]/(x_bar5[1] - x[1])
#                 dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)
#
#             x_bar5[1] = 0  # set to 0 in order to prevent the event from firing again
#             if x_bar5[3] > x[3]:  # is angular displacement a minimum or a maximum
#                 isMaximum = True
#     elif constant['event'] == 'plunge':
#         if x_bar5[0]*x[0] < 0:  # change of the direction of plunge velocity
#             event = True
#             while abs(x_bar5[0]) > delta2:
#                 dt = -dt_bar5*x[0]/(x_bar5[0] - x[0])
#                 dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)
#
#             x_bar5[0] = 0  # set to 0 in order to prevent the event from firing again
#             if x_bar5[2] > x[2]:  # is plunge displacement a minimum or a maximum
#                 isMaximum = True
#
#     return dt_bar5, x_bar5, event, isMaximum
