# todo: move all the generation of aerodynamic matrices for a typical section to this file -> avoid overlapping the code in multiple files


import numpy as np
from numpy import pi


# region Theodorsen and Wagner functions

def get_theodorsen_coefficients(constant):
    a = constant['a']
    ch = constant['ch']

    mu = np.arccos(ch)

    T1 = -(1./3.)*np.sqrt(1 - ch**2)*(2 + ch**2) + ch*mu
    T2 = ch*(1 - ch**2) - np.sqrt(1 - ch**2)*(1 + ch**2)*mu + ch*mu**2
    T3 = -(1./8. + ch**2)*mu**2 + (1./4.)*ch*np.sqrt(1 - ch**2)*mu*(7 + 2*ch**2) - (1./8.)*(1 - ch**2)*(5*ch**2 + 4)
    T4 = -mu + ch*np.sqrt(1 - ch**2)
    T5 = -(1 - ch**2) - mu**2 + 2*ch*np.sqrt(1 - ch**2)*mu
    T6 = T2
    T7 = -(1./8. + ch**2)*mu + (1./8.)*ch*np.sqrt(1 - ch**2)*(7 + 2*ch**2)
    T8 = -(1./3.)*np.sqrt(1 - ch**2)*(2*ch**2 + 1) + ch*mu
    T9 = (1./2.)*((1./3.)*(np.sqrt(1 - ch**2)**3) + a*T4)
    T10 = np.sqrt(1 - ch**2) + mu
    T11 = mu*(1 - 2*ch) + np.sqrt(1 - ch**2)*(2 - ch)
    T12 = np.sqrt(1 - ch**2)*(2 + ch) - mu*(2*ch + 1)
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


def approx_theodorsen_function(k):
    """
    Calculates the value of the Theodorsen function at given reduced frequency, k. Theodorsen function is calculated
    using an approximation.

    :param k: reduced frequency, omega*b/U, [-]
    :return: value of the Theodorsen function
    """
    return 1 - 0.165/(1 - 0.0455j/k) - 0.334/(1 - 0.3j/k)


def get_wagner_coefficients():
    """
    Wagner function approximation:
    phi(t) = 1 - psi1*e**(-eps1*U*t/b) - psi2*e**(-eps2*U*t/b)


    :return: coefficients required for the calculation of the approximation of the Wagner function
    """
    coefW = {
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

    coefW = get_wagner_coefficients()
    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    phi = 1. - psi1*np.exp(-eps1*U*t/b) - psi2*np.exp(-eps2*U*t/b)

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

    coefW = get_wagner_coefficients()
    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    phidot = (psi1*eps1*U/b)*np.exp(-eps1*U*t/b) + (psi2*eps2*U/b)*np.exp(-eps2*U*t/b)

    return phidot


# endregion

# region 3DOF typical section

def matrix_mass3DOF_usaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """
    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    # Theodorsen coefficients:
    T = get_theodorsen_coefficients(constant)
    T1 = T['T1']
    T3 = T['T3']
    T7 = T['T7']
    T13 = T['T13']

    B = b**2*np.array([[pi, -pi*a*b, -T1*b],
                       [-pi*a*b, pi*b**2*(1./8. + a**2), -(T7 + (ch - a)*T1)*b**2],
                       [-T1*b, 2*T13*b**2, -T3*b**2/pi]])

    return np.array(B)


def matrix_mass3DOF_qsaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix

    the aerodynamic mass matrix in the quasi-aerodynamic approximation is the same as in the case of the unsteady
    aerodynamics

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """

    return matrix_mass3DOF_usaero(constant)


def matrix_mass3DOF_thaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """

    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    # Theodorsen coefficients:
    T = get_theodorsen_coefficients(constant)
    T1 = T['T1']
    T3 = T['T3']
    T7 = T['T7']
    T13 = T['T13']

    B = b**2*np.array([[pi, -pi*a*b, -T1*b],
                       [-pi*a*b, pi*b**2*(1./8. + a**2), -(T7 + (ch - a)*T1)*b**2],
                       [-T1*b, 2*T13*b**2, -T3*b**2/pi]])

    return np.array(B)


def matrix_damping3DOF_usaero(constant):
    """
    Calculate unsteady aerodynamic damping matrix
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """
    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    T = get_theodorsen_coefficients(constant)
    T1 = T['T1']
    T4 = T['T4']
    T8 = T['T8']
    T9 = T['T9']
    T11 = T['T11']
    T12 = T['T12']

    coefW = get_wagner_coefficients()
    psi1 = coefW['psi1']
    psi2 = coefW['psi2']

    D1 = b**2*np.array([[0, pi, -T4],
                        [0, pi*(0.5 - a)*b, (T1 - T8 - (ch - a)*T4 + T11/2.)*b],
                        [0, (-2.*T9 - T1 + T4*(a - 0.5))*b, -T4*T11*b/(2.*pi)]])

    D2 = np.array([[2*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    D = D1 + (1 - psi1 - psi2)*D2

    return D


def matrix_damping3DOF_qsaero(constant):
    """
    Calculate unsteady aerodynamic damping matrix
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """
    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    T = get_theodorsen_coefficients(constant)
    T1 = T['T1']
    T4 = T['T4']
    T8 = T['T8']
    T9 = T['T9']
    T11 = T['T11']
    T12 = T['T12']

    D1 = b**2*np.array([[0, pi, -T4],
                        [0, pi*(0.5 - a)*b, (T1 - T8 - (ch - a)*T4 + T11/2.)*b],
                        [0, (-2.*T9 - T1 + T4*(a - 0.5))*b, -T4*T11*b/(2.*pi)]])

    D2 = np.array([[2*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    D = D1 + D2

    return D


def matrix_damping3DOF_thaero(k, constant):
    """
    Calculate unsteady aerodynamic damping matrix
    :param k: reduced frequency, omega*b/U, [-]
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """

    a = constant['a']
    b = constant['b']
    ch = constant['ch']

    T = get_theodorsen_coefficients(constant)

    T1 = T['T1']
    T4 = T['T4']
    T8 = T['T8']
    T9 = T['T9']
    T11 = T['T11']
    T12 = T['T12']

    C = approx_theodorsen_function(k)

    D1 = b**2*np.array([[0, pi, -T4],
                        [0, pi*(0.5 - a)*b, (T1 - T8 - (ch - a)*T4 + T11/2.)*b],
                        [0, (-2.*T9 - T1 + T4*(a - 0.5))*b, -T4*T11*b/(2.*pi)]])

    D2 = np.array([[2*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    return D1 + C*D2


def matrix_stiffness3DOF_usaero(constant):
    """
    Calculate unsteady aerodynamic stiffness matrix

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """
    a = constant['a']
    b = constant['b']

    T = get_theodorsen_coefficients(constant)
    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T11 = T['T11']
    T12 = T['T12']

    coefW = get_wagner_coefficients()
    psi1 = coefW['psi1']
    psi2 = coefW['psi2']
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    F1 = b**2*np.array([[0., 0., 0.],
                        [0., 0., (T4 + T10)],
                        [0., 0., (T5 - T4*T10)/pi]])

    F2 = np.array([[0., 2*pi*b, 2*b*T10],
                   [0., -2*pi*b**2*(a + 0.5), -2*b**2*(a + 0.5)*T10],
                   [0., b**2*T12, b**2*T12*T10/pi]])

    F3 = np.array([[2.*pi*b, 2*pi*b**2*(0.5 - a), b**2*T11],
                   [-2.*pi*b**2*(a + 0.5), -2*pi*b**3*(a + 0.5)*(0.5 - a), -b**3*(a + 0.5)*T11],
                   [b**2*T12, b**3*T12*(0.5 - a), b**3*T12*T11/(2.*pi)]])

    F = F1 + (1 - psi1 - psi2)*F2 + (psi1*eps1/b + psi2*eps2/b)*F3

    return F


def matrix_stiffness3DOF_thaero(k, constant):
    """
    Calculate unsteady aerodynamic stiffness matrix

    :param k: reduced frequency, omega*b/U, [-]
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """

    a = constant['a']
    b = constant['b']

    T = get_theodorsen_coefficients(constant)

    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T12 = T['T12']

    C = approx_theodorsen_function(k)

    F1 = b**2*np.array([[0., 0., 0.],
                        [0., 0., (T4 + T10)],
                        [0., 0., (T5 - T4*T10)/pi]])

    F2 = np.array([[0., 2*pi*b, 2*b*T10],
                   [0., -2*pi*b**2*(a + 0.5), -2*b**2*(a + 0.5)*T10],
                   [0., b**2*T12, b**2*T12*T10/pi]])

    F = F1 + C*F2

    return F


def matrix_stiffness3DOF_qsaero(constant):
    """
    Calculate unsteady aerodynamic stiffness matrix
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """
    a = constant['a']
    b = constant['b']

    T = get_theodorsen_coefficients(constant)
    T4 = T['T4']
    T5 = T['T5']
    T10 = T['T10']
    T12 = T['T12']

    F1 = b**2*np.array([[0., 0., 0.],
                        [0., 0., (T4 + T10)],
                        [0., 0., (T5 - T4*T10)/pi]])

    F2 = np.array([[0., 2*pi*b, 2*b*T10],
                   [0., -2*pi*b**2*(a + 0.5), -2*b**2*(a + 0.5)*T10],
                   [0., b**2*T12, b**2*T12*T10/pi]])

    F = F1 + F2

    return F


def matrix_aero3DOF_influence(constant):
    """
    Calculate the aerodynamic state influence matrix
    :param constant: dictionary of system properties
    :return: aerodynamic state influence matrix (size: 3x6)
    """

    a = constant['a']
    b = constant['b']

    T = get_theodorsen_coefficients(constant)
    T10 = T['T10']
    T11 = T['T11']
    T12 = T['T12']

    coefW = get_wagner_coefficients()
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

    W = np.array([W0, W0, W0])
    W[0, :] *= 2*pi*b
    W[1, :] *= -2*pi*b**2*(a + 0.5)
    W[2, :] *= b**2*T12

    return W


def matrix_aero3DOF_state1():
    """
    Aerodynamic state equation matrix governing the effect of physical DOFs
    :return: aerodynamic state equation matrix W1
    """
    W1 = np.array([[1., 0, 0],
                   [1., 0, 0],
                   [0, 1., 0],
                   [0, 1., 0],
                   [0, 0, 1.],
                   [0, 0, 1.]])

    return W1


def matrix_aero3DOF_state2(constant):
    """
    Aerodynamic state equation matrix governing the effect of the lag states.
    :return: aerodynamic state equation matrix W2
    """
    b = constant['b']

    coefW = get_wagner_coefficients()
    eps1 = coefW['eps1']
    eps2 = coefW['eps2']

    W2 = np.identity(6)
    W2[0::2, 0::2] *= -eps1/b
    W2[1::2, 1::2] *= -eps2/b

    return W2


def matrix_aero_influence3DOF_theodorsen(k, constant):
    """
    this function calculates the aerodynamic influence matrix in frequency domain based on the Theodorsen aerodynamics.

    :param k: reduced frequency, omega*b/U, [-]
    :param constant: system properties
    :return: aerodynamic influence matrix
    """

    b = constant['b']

    # Aerodynamic mass matrix:
    B = matrix_mass3DOF_thaero(constant)

    # Aerodynamic damping matrix:
    D = matrix_damping3DOF_thaero(k, constant)

    # Aerodynamic stiffness matrix:
    F = matrix_stiffness3DOF_thaero(k, constant)

    # Assembled matrix of aerodynamic influence coefficients:
    Q = (-k**2/b**2)*B + (k*1j/b)*D + F

    return Q


# endregion

# region 2DOF typical section

def matrix_mass2DOF_usaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """
    B = matrix_mass3DOF_usaero(constant)

    return B[:2, :2]


def matrix_mass2DOF_qsaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """
    B = matrix_mass3DOF_qsaero(constant)

    return B[:2, :2]


def matrix_mass2DOF_thaero(constant):
    """
    Calculate unsteady aerodynamic mass matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic mass matrix
    """
    B = matrix_mass3DOF_thaero(constant)

    return B[:2, :2]


def matrix_damping2DOF_usaero(constant):
    """
    Calculate unsteady aerodynamic damping matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """

    D = matrix_damping3DOF_usaero(constant)

    return D[:2, :2]


def matrix_damping2DOF_qsaero(constant):
    """
    Calculate unsteady aerodynamic damping matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """

    D = matrix_damping3DOF_qsaero(constant)

    return D[:2, :2]


def matrix_damping2DOF_thaero(k, constant):
    """
    Calculate unsteady aerodynamic damping matrix for a 2DOF system (pitch-plunge, no control surface)

    :param k: reduced frequency, omega*b/U, [-]
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic damping matrix
    """

    D = matrix_damping3DOF_thaero(k, constant)

    return D[:2, :2]


def matrix_stiffness2DOF_usaero(constant):
    """
    Calculate unsteady aerodynamic stiffness matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """

    F = matrix_stiffness3DOF_usaero(constant)

    return F[:2, :2]


def matrix_stiffness2DOF_qsaero(constant):
    """
    Calculate unsteady aerodynamic stiffness matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """

    F = matrix_stiffness3DOF_qsaero(constant)

    return F[:2, :2]


def matrix_stiffness2DOF_thaero(k, constant):
    """
    Calculate unsteady aerodynamic stiffness matrix for a 2DOF system (pitch-plunge, no control surface)

    :param k: reduced frequency, omega*b/U, [-]
    :param constant: dictionary of system properties
    :return: unsteady aerodynamic stiffness matrix
    """

    F = matrix_stiffness3DOF_thaero(k, constant)

    return F[:2, :2]


def matrix_aero2DOF_influence(constant):
    """
    Calculate the aerodynamic state influence matrix for a 2DOF system (pitch-plunge, no control surface)

    :param constant: dictionary of system properties
    :return: aerodynamic state influence matrix (size: 3x6)
    """

    W = matrix_aero3DOF_influence(constant)

    return W[:2, :4]


def matrix_aero2DOF_state1():
    """
    Aerodynamic state equation matrix governing the effect of physical DOFs for a 2DOF system (pitch-plunge, no control
    surface)

    :return: aerodynamic state equation matrix W1
    """

    W1 = matrix_aero3DOF_state1()

    return W1[:4, :2]


def matrix_aero2DOF_state2(constant):
    """
    Aerodynamic state equation matrix governing the effect of the lag states for a 2DOF system (pitch-plunge, no control
    surface)

    :return: aerodynamic state equation matrix W2
    """
    W2 = matrix_aero3DOF_state2(constant)
    return W2[:4, :4]


def matrix_aero_influence2DOF_theodorsen(k, constant):
    """
    this function calculates the aerodynamic influence matrix in frequency domain based on the Theodorsen aerodynamics
    for a 2DOF system (pitch-plunge, no control surface).

    :param k: reduced frequency, omega*b/U, [-]
    :param constant: system properties
    :return: aerodyamic influene matrix
    """

    Q = matrix_aero_influence3DOF_theodorsen(k, constant)

    return Q[:2, :2]

# endregion
