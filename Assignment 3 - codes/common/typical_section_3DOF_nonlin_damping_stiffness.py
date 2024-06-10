from __future__ import division
import scipy as sp
import scipy.linalg as ln
import scipy.interpolate as interpolate
from numpy import pi as pi
import typical_section_3DOF as ts3
import typical_section_3DOF_nonlin_damping as ts3nld
import typical_section_3DOF_nonlin_stiffness as ts3nls
from theodorsen_aerodynamics import theodorsen_coefficients
from wagner_aerodynamics import get_approximation_coefficients


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
        'Ch': 0.0,  # linear structural damping in heave DOF
        'Ch2': 0.0,  # quadratic structural damping in heave DOF
        'Cth': 0.0,  # linear structural damping in pitch DOF
        'Cth2': 0.0,  # quadratic  structural damping in pitch DOF
        'Cbt': 0.0,  # linear structural damping in control DOF
        'Cbt2': 0.0,  # quadratic structural damping in control DOF
        'event': 'none'  # event to detect: 'none', 'pitch', 'plunge', 'control'
    }

    return constant


def ts3DOF_nonlin_damping_stiffness(x, U, constant):
    """
    Calculates x_prime in the case of QUADRATIC DAMPING in each DOF.

    :param x:
    :param U:
    :param constant:
    :return:
    """
    rho = constant['rho']  # air density
    Ch2 = constant['Ch2']
    Cth2 = constant['Cth2']
    Cbt2 = constant['Cbt2']

    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    # Linear part:
    # ============
    xprime_lin = ts3.typical_section_3DOF_lin(x, U, constant, T, coefW)

    # Nonlinear part:
    # ===============

    # Quadratic damping:
    xprime_dmp = ts3nld.quadratic_damping(x, constant)

    # Qubic stiffness:
    xprime_stf = ts3nls.qubic_stiffness(x, constant)

    return xprime_lin + xprime_dmp + xprime_stf


def matrix_damping_stiffness_eqlin(A, A0, omega, U, constant):
    """
        Calculates the main matrix of for the equivalent linearisation in the case of QUADRATIC DAMPING and QUBIC
        STIFFNESS for  each DOF
        :param A: amplitude vector([amp_h, amp_th, amp_bt])
        :param A0: steady-state amplitude vector([amp0_h, amp0_th, amp0_bt]) <- roughly corresponds to non-zero fixed point
        :param omega: omega of the LCO [rad/s]
        :param U: velocity
        :param constant: properties of the nonlinear system
        :return: Q_eqlin
        """
    rho = constant['rho']  # air density
    Ch2 = constant['Ch2']
    Cth2 = constant['Cth2']
    Cbt2 = constant['Cbt2']

    Kh3 = constant['Kh3']
    Kth3 = constant['Kth3']
    Kbt3 = constant['Kbt3']

    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    # Linear part:
    # ============
    Q_lin = ts3.typical_section_3DOF_lin(sp.identity(12), U, constant, T, coefW)

    # Equivalent stiffness matrix - nonlinear part:
    # =============================================
    A_tmp = ts3.matrix_mass_str(constant)
    B_tmp = ts3.matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A_tmp + rho*B_tmp))

    # Equivalent damping:
    # -------------------
    C_nlin = sp.zeros((3, 3))
    C_nlin[0, 0] = 8.*Ch2*A[0]*omega/(3.*pi)
    C_nlin[1, 1] = 8.*Cth2*A[1]*omega/(3.*pi)
    C_nlin[2, 2] = 8.*Cbt2*A[2]*omega/(3.*pi)

    # Equivalent stiffness:
    # ---------------------
    E_nlin = sp.zeros((3, 3))
    E_nlin[0, 0] = Kh3*(3.*A[0]**2/4. + 3.*A0[0]**2)
    E_nlin[1, 1] = Kth3*(3.*A[1]**2/4. + 3.*A0[1]**2)
    E_nlin[2, 2] = Kbt3*(3.*A[2]**2/4. + 3.*A0[2]**2)

    Q_eqlin = Q_lin
    Q_eqlin[:3, :3] += -M_inv.dot(C_nlin)
    Q_eqlin[:3, 3:6] += -M_inv.dot(E_nlin)

    return Q_eqlin


def vector_nonhomo_damping_stiffness_eqlin(A, A0, constant):
    """
        Calculates nonhomogeneous contribution of the static offsets of the LCO to x_prime in the case of QUADRATIC
        DAMPING and QUBIC STIFFNESS in each DOF

        :param x: vector of DOF at current time step
        :param A: amplitude vector([amp_h, amp_th, amp_bt])
        :param A0: steady-state amplitude vector([amp0_h, amp0_th, amp0_bt]) <- roughly corresponds to non-zero fixed point
        :param omega: omega of the LCO [rad/s]
        :param U: velocity
        :param constant: properties of the nonlinear system
        :return: x_primer
        """

    rho = constant['rho']  # air density

    Kh3 = constant['Kh3']
    Kth3 = constant['Kth3']
    Kbt3 = constant['Kbt3']

    T = theodorsen_coefficients(constant)

    # Mass matrix inverse:
    A_tmp = ts3.matrix_mass_str(constant)
    B_tmp = ts3.matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A_tmp + rho*B_tmp))

    # Equivalent stiffness:
    E_nlin = sp.zeros((3, 3))
    E_nlin[0, 0] = Kh3*(3.*A[0]**2/4. + 3.*A0[0]**2)
    E_nlin[1, 1] = Kth3*(3.*A[1]**2/4. + 3.*A0[1]**2)
    E_nlin[2, 2] = Kbt3*(3.*A[2]**2/4. + 3.*A0[2]**2)

    # Non-homogeneous contribution
    qn = sp.zeros((12, 3))
    qn[:3, :3] = -M_inv.dot(sp.identity(3))

    # Stationery coefficient in the Fourier expansion (eq. 5.47, p. 229)
    a0 = sp.zeros((3,))
    a0[0] = Kh3*(3.*A[0]**2*A0[0]/2. + A0[0]**3)
    a0[1] = Kth3*(3.*A[1]**2*A0[1]/2. + A0[1]**3)
    a0[2] = Kbt3*(3.*A[2]**2*A0[2]/2. + A0[2]**3)

    return qn.dot(a0 - E_nlin.dot(A0))


def ts3DOF_damping_stiffness_eqlin(x, A, A0, omega, U, constant):
    """
    Calculates x_prime using equivalent linearisation in the case of QUADRATIC DAMPING in each DOF

    :param x: vector of DOF at current time step
    :param A: amplitude vector([amp_h, amp_th, amp_bt])
    :param A0: steady-state amplitude vector([amp0_h, amp0_th, amp0_bt]) <- roughly corresponds to non-zero fixed point
    :param omega: omega of the LCO [rad/s]
    :param U: velocity
    :param constant: properties of the nonlinear system
    :return: x_primer
    """

    # matrix of the equivalent lenarised system:
    Q_eqlin = matrix_damping_stiffness_eqlin(A, A0, omega, U, constant)

    # nonhomogeneous contribution due to static offset of the LCO:
    q_nonhomo = vector_nonhomo_damping_stiffness_eqlin(A, A0, constant)

    return Q_eqlin.dot(x) + q_nonhomo


def LCO_eq_system_damping_stiffness_eqlin(A, r, constant):
    """
    This function assembles the nonlinear system of equations that must be solved in order to determine all the
    parameters of the LCO for the eq. lin. system with the prescribed amplitude. The parameters are LCO frequency (omega),
    mean amplitude, th0 and the free-stream velocity, U. The parameters are packed in the vector r.

    :param A: amplitude vector ([amp_h, amp_th, amp_bt]) of the LCO around the mean amplitude th0 [rad]
    :param r: vector of LCO parameters (omega, th0, U)
    :param omega: frequency of the LCO [rad/s]
    :param th0: mean amplitude of the LCO [rad]
    :param U: free-stream velocity [m/s]
    :param constant: system properties
    :return: residual -> required for the minimisation problem
    """

    epsC = 1.e-10

    # Unpack the parameters:
    omega, th0, U, = r[0], r[1], r[2]

    # Select the eigenvalue with the smallest real part:
    # ==================================================
    #  this is required for the firs two nonlinear equations Re(lambda) == 0 & omega - |lambda| == 0
    A0 = sp.zeros((3,))
    A0[1] = th0

    Qeqlin = matrix_damping_stiffness_eqlin(A, A0, omega, U, constant)
    evals = ln.eigvals(Qeqlin)

    # IMPORTANT: consider only the COMPLEX eigenvaules -> since these are connected to structural and OSCILATORY states.
    # othewise the algorithm can converege to a different solution (DIVERGENCE at higher velocity)
    evals_cmp = evals[evals.imag > epsC]

    i_min = sp.absolute(evals_cmp.real).argmin()

    eval_min = evals_cmp[i_min]

    # Calculation of the fixed point
    # ===============================
    rho = constant['rho']  # air density
    Kth3 = constant['Kth3']
    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    E = ts3.matrix_stiffness_str(constant)
    F = ts3.matrix_stiffness_usaero(constant, T, coefW)
    W = ts3.matrix_aero_influence(constant, T, coefW)
    W1 = ts3.matrix_aero_state1()
    W2 = ts3.matrix_aero_state2(constant, coefW)

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

    # Calculate residual:
    # ===================

    residual = sp.zeros((3,))
    residual[0] = eval_min.real
    residual[1] = omega - sp.absolute(eval_min)
    residual[2] = Kth3*A0[1]**3 + (3.*Kth3*A[1]**2/2. - qn.dot(Qlinv.dot(ql)) + qnn)*A0[1]

    return residual


def jacobian_damping_stiffness_eqlin(A, r, constant):
    # step in r used to calculate the Jacobian:
    dr = 1e-9*sp.ones((3,))

    jac = sp.zeros((3, 3))

    p0 = LCO_eq_system_damping_stiffness_eqlin(A, r, constant)

    for i in range(3):
        r_inc = r.copy()
        r_inc[i] += dr[i]
        p_inc = LCO_eq_system_damping_stiffness_eqlin(A, r_inc, constant)

        jac[:, i] = (p_inc - p0)/dr[i]

    return jac


def stability_LCO_eqlin(A_up, A_down, A0, omega, U, constant):
    Q_up = matrix_damping_stiffness_eqlin(A_up, A0, omega, U, constant)
    evals_up = ln.eigvals(Q_up)

    Q_down = matrix_damping_stiffness_eqlin(A_down, A0, omega, U, constant)
    evals_down = ln.eigvals(Q_down)

    stable_up = True
    if any(evals_up.real > 0):
        stable_up = False

    stable_down = False
    if any(evals_down.real > 0):
        stable_down = True

    return (stable_down and stable_up)


def rk45(dt, x, U, constant, delta=1e-3):
    x = sp.array(x)

    eps = 1.

    while eps > delta:
        # @ t_i:
        k0 = dt*ts3DOF_nonlin_damping_stiffness(x, U, constant)
        x0 = x + k0/4.

        # @ t_i + dt/4:
        k1 = dt*ts3DOF_nonlin_damping_stiffness(x0, U, constant)
        x1 = x + 3.*k0/32. + 9.*k1/32.

        # @ t_i + 3*dt/8:
        k2 = dt*ts3DOF_nonlin_damping_stiffness(x1, U, constant)
        x2 = x + (1932.*k0 - 7200.*k1 + 7296.*k2)/2197.

        # @ t_i + 12*dt/13:
        k3 = dt*ts3DOF_nonlin_damping_stiffness(x2, U, constant)
        x3 = x + 439.*k0/216. - 8.*k1 + 3680.*k2/513. - 845.*k3/4104.

        # @ t_i + dt:
        k4 = dt*ts3DOF_nonlin_damping_stiffness(x3, U, constant)
        x4 = x - 8.*k0/27. + 2.*k1 - 3544.*k2/2565. + 1859.*k3/4104. - 11.*k4/40.

        # @ t_i + dt/2:
        k5 = dt*ts3DOF_nonlin_damping_stiffness(x4, U, constant)
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
            if x_bar5[4] > x[4]:  # is angular displacement a minimum or a maximum
                isMaximum = True
    elif constant['event'] == 'plunge':
        if x_bar5[0]*x[0] < 0:  # change of the direction of plunge velocity
            event = True
            while abs(x_bar5[0]) > delta2:
                dt = -dt_bar5*x[0]/(x_bar5[0] - x[0])
                dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)

            x_bar5[0] = 0  # set to 0 in order to prevent the event from firing again
            if x_bar5[3] > x[3]:  # is plunge displacement a minimum or a maximum
                isMaximum = True
    elif constant['event'] == 'control':
        if x_bar5[2]*x[2] < 0:  # change of the direction of plunge velocity
            event = True
            while abs(x_bar5[2]) > delta2:
                dt = -dt_bar5*x[2]/(x_bar5[2] - x[2])
                dt_bar5, x_bar5 = rk45(dt, x, U, constant, delta)

            x_bar5[2] = 0  # set to 0 in order to prevent the event from firing again
            if x_bar5[5] > x[5]:  # is plunge displacement a minimum or a maximum
                isMaximum = True

    return dt_bar5, x_bar5, event, isMaximum
