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
        'Kth': 198.9712,  # pitch stiffness [Nm/rad]
        'Kbt': 17.3489,  # control DOF stiffness [Nm/rad]
        'Ch': 0.0,  # linear structural damping in heave DOF
        'Ch2': 0.0,  # quadratic structural damping in heave DOF
        'Cth': 0.0,  # linear structural damping in pitch DOF
        'Cth2': 0.0,  # quadratic  structural damping in pitch DOF
        'Cbt': 0.0,  # linear structural damping in control DOF
        'Cbt2': 0.0,  # quadratic structural damping in control DOF
        'event': 'none'  # event to detect: 'none', 'pitch', 'plunge', 'control'
    }

    return constant


def ts3DOF_nonlin_damping(x, U, constant):
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
    xprime_lin = ts3.typical_section_3DOF_lin(x, U, constant, T, coefW)

    xprime_nlin = quadratic_damping(x, constant)

    return xprime_lin + xprime_nlin


def quadratic_damping(x, constant):
    """
    calculates the nonlinear contribution of the quadratic damping to the EOM in state space format
    :param x: state vector
    :param constant: system properties
    :return: x_dot due to quadratic damping
    """
    rho = constant['rho']  # air density
    Ch2 = constant['Ch2']
    Cth2 = constant['Cth2']
    Cbt2 = constant['Cbt2']

    T = theodorsen_coefficients(constant)

    # nonlinear
    A = ts3.matrix_mass_str(constant)
    B = ts3.matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A + rho*B))

    qn_plunge = sp.zeros((12,))
    qn_plunge[:3] = -M_inv.dot([Ch2, 0., 0.])

    qn_pitch = sp.zeros((12,))
    qn_pitch[:3] = -M_inv.dot([0., Cth2, 0.])

    qn_control = sp.zeros((12,))
    qn_control[:3] = -M_inv.dot([0., 0., Cbt2])

    xprime_nlin = qn_plunge*x[0]*sp.absolute(x[0]) + qn_pitch*x[1]*sp.absolute(x[1]) + qn_control*x[2]*sp.absolute(x[2])

    return xprime_nlin


def ts3DOF_qdrt_damping_eqlin(x, Aw, U, constant):
    """
    Calculates x_prime using equivalent linearisation in the case of QUADRATIC DAMPING in each DOF

    :param x: vector of DOF at current time step
    :param Aw: vector of a product amplitude x omega ([amp_h*omega, amp_th*omega, amp_bt*omega])
    :param U: velocity
    :param constant: properties of the nonlinear system
    :return: x_primer
    """
    rho = constant['rho']  # air density
    Ch2 = constant['Ch2']
    Cth2 = constant['Cth2']
    Cbt2 = constant['Cbt2']

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

    C_nlin = sp.zeros((3, 3))
    C_nlin[0, 0] = 8.*Ch2*Aw[0]/(3.*pi)
    C_nlin[1, 1] = 8.*Cth2*Aw[1]/(3.*pi)
    C_nlin[2, 2] = 8.*Cbt2*Aw[2]/(3.*pi)

    Q_eqlin = Q_lin
    Q_eqlin[:3, :3] += -M_inv.dot(C_nlin)

    return Q_eqlin.dot(x)


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

        w_n[i], zeta[i] = LCO_frequency_damping_eqlin(A, U_i, w_interp, zeta_interp, T)

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


def LCO_frequency_damping_eqlin(Aw, U_i, w_interp, zeta_interp, constant):
    # Assemble 3DOF system matrix
    x0 = sp.identity(12)  # dummy input to retrieve the 3DOF system matrix
    Q = ts3DOF_qdrt_damping_eqlin(x0, Aw, U_i, constant)

    # Calculate eigen values and eigen vectors
    eval = ln.eig(Q)[0]

    # sort out the complex-conjugate eigenvalues and their pertinent eigen vectors:
    eps_fltr = 1e-12
    eval_cc = eval[eval.imag > eps_fltr]
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


def stability_LCO_eqlin(Aw_up, Aw_down, U, constant):
    Q_up = ts3DOF_qdrt_damping_eqlin(sp.identity(12), Aw_up, U, constant)
    evals_up = ln.eigvals(Q_up)

    Q_down = ts3DOF_qdrt_damping_eqlin(sp.identity(12), Aw_down, U, constant)
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
        k0 = dt*ts3DOF_nonlin_damping(x, U, constant)
        x0 = x + k0/4.

        # @ t_i + dt/4:
        k1 = dt*ts3DOF_nonlin_damping(x0, U, constant)
        x1 = x + 3.*k0/32. + 9.*k1/32.

        # @ t_i + 3*dt/8:
        k2 = dt*ts3DOF_nonlin_damping(x1, U, constant)
        x2 = x + (1932.*k0 - 7200.*k1 + 7296.*k2)/2197.

        # @ t_i + 12*dt/13:
        k3 = dt*ts3DOF_nonlin_damping(x2, U, constant)
        x3 = x + 439.*k0/216. - 8.*k1 + 3680.*k2/513. - 845.*k3/4104.

        # @ t_i + dt:
        k4 = dt*ts3DOF_nonlin_damping(x3, U, constant)
        x4 = x - 8.*k0/27. + 2.*k1 - 3544.*k2/2565. + 1859.*k3/4104. - 11.*k4/40.

        # @ t_i + dt/2:
        k5 = dt*ts3DOF_nonlin_damping(x4, U, constant)
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
