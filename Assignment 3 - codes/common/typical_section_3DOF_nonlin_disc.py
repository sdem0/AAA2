from __future__ import division
import scipy as sp
import scipy.linalg as ln
import scipy.interpolate as interpolate
from numpy import pi as pi

import typical_section_3DOF as ts3
from theodorsen_aerodynamics import theodorsen_coefficients
from wagner_aerodynamics import get_approximation_coefficients


def get_constant():
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
        'Kh': 0.0,  # heave (plunge) stiffness [N/m]
        'Kth': 0.0,  # pitch stiffness [Nm/rad]
        'Kbt': 0.0,  # control DOF stiffness [Nm/rad]
        'Ch': 3340./4000.,  # structural damping in heave DOF
        'Cth': 43.07/4000.,  # structural damping in pitch DOF
        'Cbt': 2.90/4000.,  # structural damping in control DOF
        'event': 'none',  # event to detect: 'none', 'pitch', 'plunge', 'control'
        'd_h': [],  # heave (plunge) free_play zone
        'd_th': [],  # pitch free_play zone
        'd_bt': [],  # control DOF (plunge) free_play zone
        'Kh_zones': [0.0, 3340],  # heave (plunge) stiffness per zone [N/m]
        'Kth_zones': [0.0, 43.07],  # pitch stiffness per zone [Nm/rad]
        'Kbt_zones': [0.0, 2.90],  # control DOF stiffness per zone [Nm/rad]
    }

    return constant


def typical_section_3DOF_eqlin_3zones(x, U, dA, constant):
    """
    returns the time derivative of the state vector x for the EQUIVALENT LINEARIZED system with free-play. Equivalent
    linearisation is performed over ALL three zones (domains) of the system.

    :param x: state vector
    :param U: free stream velocity [m/s]
    :param dA: a vector of ratios between the free-play zone over the oscillation amplitude for all three DOF [d_h/A_h, d_th/A_th, d_bt/A_bt]
    :param constant: system properties
    :return: time derivative of the state vector x (Q.dot(x))
    """
    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    sigma = sp.arcsin(dA)

    Kh_zones = constant['Kh_zones']
    Kth_zones = constant['Kth_zones']
    Kbt_zones = constant['Kbt_zones']

    Kh_eq = Kh_zones[1] + (1/pi)*(Kh_zones[0] - Kh_zones[1])*(2*sigma[0] + sp.sin(2*sigma[0]))
    Kth_eq = Kth_zones[1] + (1/pi)*(Kth_zones[0] - Kth_zones[1])*(2*sigma[1] + sp.sin(2*sigma[1]))
    Kbt_eq = Kbt_zones[1] + (1/pi)*(Kbt_zones[0] - Kbt_zones[1])*(2*sigma[2] + sp.sin(2*sigma[2]))

    constant['Kh'] = Kh_eq
    constant['Kth'] = Kth_eq
    constant['Kbt'] = Kbt_eq

    xprime = ts3.typical_section_3DOF_lin(x, U, constant, T, coefW)

    return xprime


def LCO_2D_equation_system(r, Kth_eq, U_LCO, constant):
    """
    This function assembles the nonlinear system of equations that must be solved in order to determine all the
    parameters of the 2domain LCO for the eq. lin. system with prescribed region of free-play. The parameters are
    packed in the vector r.

    :param r: vector of LCO parameters (omega, th0, U)
    :param Kth_eq: equivalent stiffness of the linearised system [Nm/rad]
    :param U_LCO: LCU velocity [m/s]
    :param constant: system properties
    :return: residual -> required for the minimisation problem
    """

    # Unpack the parameters:
    Ad_th, th0d = r[0], r[1]

    rho = constant['rho']  # air density
    Kth_zones = constant['Kth_zones']

    # Estimate equivalent stiffness from LCO params
    # ---------------------------------------------
    Kth_eq_estimate = Kth_eq_LCO_2D(Ad_th, th0d, constant)

    # Estimate the LCO centre from the LCO params:
    # ============================================


    # Assemble Q1:
    # ------------
    constant['Kth'] = Kth_zones[0]
    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    x0 = sp.identity(12)
    Q1 = ts3.typical_section_3DOF_lin(x0, U_LCO, constant, T, coefW)
    Q1_inv = ln.inv(Q1)

    # Assemble qn_pitch:
    # ------------------
    A = ts3.matrix_mass_str(constant)
    B = ts3.matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A + rho*B))

    qn_pitch = sp.zeros((12,))
    qn_pitch[:3] = -M_inv.dot([0., 1, 0.])

    # calculate a0:
    # -------------
    # a0d is the ratio of a0-to-free-play zone (a0/d). a0d is introduced to keep the equations in nondimensional form.
    # (normalised wrt. to d)
    a0d = a0_LCO_2D(Ad_th, th0d, constant)

    # calculate the the fixed point (x_f):
    # ------------------------------------
    # xd_f is the ration x_f-to-free-play zone (x_f/d). xd_f is introduced to keep the equations in nondimensional form.

    xd_f = -Q1_inv.dot(qn_pitch*a0d)
    th0d_estimate = xd_f[4]  # xd_f[4] is already normalised with the free-play zone!

    # Calculate residual:
    # ===================
    residual = sp.zeros((2,))
    residual[0] = Kth_eq - Kth_eq_estimate
    residual[1] = th0d - th0d_estimate

    return residual


def LCO_2D_jacobian_equation_system(r, Kth_eq, U_LCO, constant):
    # step in r used to calculate the Jacobian:
    dr = 1e-9*sp.ones((2,))

    jac = sp.zeros((2, 2))

    p0 = LCO_2D_equation_system(r, Kth_eq, U_LCO, constant)

    for i in range(2):
        r_inc = r.copy()
        r_inc[i] += dr[i]
        p_inc = LCO_2D_equation_system(r_inc, Kth_eq, U_LCO, constant)

        jac[:, i] = (p_inc - p0)/dr[i]

    return jac


def LCO_2D_stability(Ad_th, th0d, U_LCO, up_down, constant):

    deltaR = 1e-8  # precision of the solution
    Kth_eq = Kth_eq_LCO_2D(Ad_th, th0d, constant)
    r = sp.array([Ad_th, th0d])
    dr2 = 1  # residual of r2

    while sp.absolute(dr2) > deltaR:
        p2 = LCO_2D_equation_system(r, Kth_eq, U_LCO, constant)[1]
        jac22 = LCO_2D_jacobian_equation_system(r, Kth_eq, U_LCO, constant)[1,1]

        dr2 = -p2/jac22
        r[1] += dr2


    # Assemble the system matrix of the eq. lin. system:
    # ------------------------------------------------

    # update the LCO center offset (due to variation in LCO amplitude):
    th0d = r[1]

    # Update the equivalent stiffness:
    Kth_eq = Kth_eq_LCO_2D(Ad_th, th0d, constant)

    # System matrix of the eq. lin. system:
    constant['Kth'] = Kth_eq
    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    x0 = sp.identity(12)
    Q_eqlin = ts3.typical_section_3DOF_lin(x0, U_LCO, constant, T, coefW)

    evals = ln.eigvals(Q_eqlin)


    if up_down == 'up':
        stable = True
        if any(evals.real > 0):
            stable = False
    else:
        stable = False
        if any(evals.real > 0):
            stable= True

    return stable


def Kth_eq_LCO_2D(Ad_th, th0d, constant):
    """
    This function calculates equivalent stiffness of the pitch DOF for a 2-domain LCO.

    :param Ad_th: LCO amplitude-to-freeplay-zone ratio (Ath/d) [-]
    :param th0d: LCO offset-to-freeplay-zone ratio (th0/d) [-]
    :param constant: system properties
    :return: Kth_eq: equivalent stiffness of the pitch DOF [Nm/rad]
    """

    # Equivalent stiffness for the 2Domain LCO:
    # ---------------------------------------------
    sigma1 = sigma_LCO_2D(Ad_th, th0d)

    Kth_zones = constant['Kth_zones']
    Kth_eq = 0.5*(Kth_zones[0] + Kth_zones[1]) + (Kth_zones[0] - Kth_zones[1])*(2*sigma1 + sp.sin(2*sigma1))/(
                2.*pi)

    return Kth_eq


def a0_LCO_2D(Ad_th, th0d, constant):
    """
    This function calculates the constant term of the equivalent linearisation over for a 2-domain LCO in pitch DOF.

    NOTE: the a0 is normalised with the free-play zone amplitude (delta). To get the actual value of a0, the a0d must
    be multiplied with the free-play zone amplitude (delta)!!

    :param Ad_th: LCO amplitude-to-freeplay-zone ratio (Ath/d) [-]
    :param th0d: LCO offset-to-freeplay-zone ratio (th0/d) [-]
    :param constant: system properties
    :return: a0d: constant term of the eq. linearisation for a 2-domain LCO for pithc DOF [-]
    """

    sigma1 = sigma_LCO_2D(Ad_th, th0d)

    Kth_zones = constant['Kth_zones']
    a0d = 0.5*(Kth_zones[0] + Kth_zones[1])*th0d + 0.5*(Kth_zones[0] - Kth_zones[1]) - (Ad_th/pi)*(
            Kth_zones[0] - Kth_zones[1])*(sigma1*sp.sin(sigma1) + sp.cos(sigma1))

    return a0d


def sigma_LCO_2D(Ad_th, th0d):
    """
    calculates the sigma parameter for the equivalent linearisation in the case of a 2-domain LCO.
    :param Ad_th: LCO amplitude-to-freeplay-zone ratio (Ath/d) [-]
    :param th0d: LCO offset-to-freeplay-zone ratio (th0/d) [-]
    :return: sigma1
    """
    sigma1 = sp.arcsin((1 - th0d)/Ad_th)
    return sigma1