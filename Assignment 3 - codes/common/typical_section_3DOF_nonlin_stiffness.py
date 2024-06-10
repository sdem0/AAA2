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


def typical_section_3DOF_nonlin(x, U, constant):
    T = theodorsen_coefficients(constant)
    coefW = get_approximation_coefficients()

    # Linear part:
    xprime_lin = ts3.typical_section_3DOF_lin(x, U, constant, T, coefW)

    # Nonlinear part:
    xprime_nlin = qubic_stiffness(x, constant)

    return xprime_lin + xprime_nlin


def qubic_stiffness(x, constant):
    """
    calculates the nonlinear contribution of the quadratic damping to the EOM in state space format
    :param x:
    :param constant:
    :return: x_dot due to qubic stiffness
    """
    rho = constant['rho']  # air density
    Kh3 = constant['Kh3']
    Kth3 = constant['Kth3']
    Kbt3 = constant['Kbt3']

    T = theodorsen_coefficients(constant)

    A = ts3.matrix_mass_str(constant)
    B = ts3.matrix_mass_usaero(constant, T)

    M_inv = ln.inv((A + rho*B))

    qn_plunge = sp.zeros((12,))
    qn_plunge[:3] = -M_inv.dot([Kh3, 0., 0.])

    qn_pitch = sp.zeros((12,))
    qn_pitch[:3] = -M_inv.dot([0., Kth3, 0.])

    qn_control = sp.zeros((12,))
    qn_control[:3] = -M_inv.dot([0., 0., Kbt3])

    xprime_nlin = qn_plunge*x[3]**3 + qn_pitch*x[4]**3 + qn_control*x[5]**3

    return xprime_nlin


# def fixed_points_3DOF_nonlin_th(U, constant, T, coefW):
#     epsC = 1e-10
#     rho = constant['rho']  # air density
#     Kth3 = constant['Kth3']
#
#     E = ts3.matrix_stiffness_str(constant)
#     F = ts3.matrix_stiffness_usaero(constant, T, coefW)
#     W = ts3.matrix_aero_influence(constant, T, coefW)
#     W1 = ts3.matrix_aero_state1()
#     W2 = ts3.matrix_aero_state2(constant, coefW)
#
#     Q = sp.zeros((9, 9))
#     Q[:3, :3] = E + rho*U**2*F
#     Q[:3, 3:] = rho*U**3*W
#     Q[3:, :3] = W1
#     Q[3:, 3:] = U*W2
#
#     # omit the second row and column to get the matrix of the linear subsystem of equations.
#     Ql = sp.delete(Q, 1, 0)
#     Ql = sp.delete(Ql, 1, 1)
#     Qlinv = ln.inv(Ql)
#
#     # linear contribution of the pitch DOF to the linear subsytem of equations
#     ql = Q[:, 1]
#     ql = sp.delete(ql, 1, 0)
#
#     # linear contribution of the remaining DOF (plunge, control, aerodynamic states) to the nonlinear subsystem of
#     # equtions (equation for th)
#     qn = Q[1, :]  # qn.shape  is (9,)
#     qn = sp.delete(qn, 1)
#
#     # linear contribution of the pitch DOF to the nonlinear subsystem  of equations (equation for th)
#     qnn = Q[1, 1]
#
#     # Calculate Fixed points:
#     # =======================
#
#     # One fixed point is always 0:
#     qF1 = sp.zeros((9,))
#
#     # The remaining two fixed points:
#
#     # th value:
#     thF2 = sp.sqrt((qn.dot(Qlinv.dot(ql)) - qnn)/Kth3)
#     thF3 = (-1.)*sp.sqrt((qn.dot(Qlinv.dot(ql)) - qnn)/Kth3)
#
#     # check if thF2 or thF3 are complex if so -> discard -> make it equal to thF1
#     if sp.absolute(thF2.imag) > epsC:
#         thF2 = qF1[2]
#
#     if sp.absolute(thF3.imag) > epsC:
#         thF3 = qF1[2]
#
#     # Remaining DOF vlaues:
#     qF2 = (-1.)*Qlinv.dot(ql)*thF2
#     qF3 = (-1.)*Qlinv.dot(ql)*thF3
#
#     # insert thF2 and thF3 to get a complete vector of DOFs at fixed point.
#     qF2 = sp.insert(qF2, 1, thF2)
#     qF3 = sp.insert(qF3, 1, thF3)
#
#     # Expand the reduced vectors of DOFs to a full vector:
#
#     # Expand the reduced vectors of DOFs to a full vector of DOF which contains also the velocities of the structural
#     # DOFs.(To maintain compatibility with the previously implemented methods -> e.g. the Jacobian of the nonlinear
#     # system)
#
#     mask = sp.zeros((12,), dtype=bool)
#     mask[3:] = True
#
#     xF1, xF2, xF3 = sp.zeros((12,)), sp.zeros((12,)), sp.zeros((12,))
#     xF2[mask], xF3[mask] = qF2, qF3
#
#     xF = sp.zeros((12, 3))
#     xF[:, 0] = xF1
#     xF[:, 1] = xF2
#     xF[:, 2] = xF3
#
#     # Stability of Fixed Points:
#     # ==========================
#
#     isStable = sp.ones((3,), dtype=bool)
#
#     for i in range(3):
#         jac = jacobian_nonlin(xF[:, i], U, constant, T, coefW)
#         evals = ln.eigvals(jac)
#         evals_real = evals[sp.absolute(evals.imag) < epsC]
#
#         # ONLY the REAL eigenvalues (corresponding to aerodynamic states) must be considered because we are looking at
#         # the stability wrt. divergence. Flutter instability has already occured (a pair of complex eig. vals with
#         # positive real part) but it is stabilised with the LCO.
#         if any(evals_real > 0):
#             isStable[i] = False
#
#     return xF, isStable
#
#
# def typical_section_3DOF_eqlin(x, A, U, constant, T, coefW):
#     """
#     Calculates x_prime using equivalent linearisation in the case of CUBIC hardening stiffness in each DOF
#
#     :param x: vector of DOF at current time step
#     :param A: vector of amplitude ([amp_h, amp_th, amp_bt])
#     :param U: velocity
#     :param constant: properties of the nonlinear system
#     :param T: Theodorsen coefficients
#     :param coefW: Wagner coefficients
#     :return: x_primer
#     """
#     rho = constant['rho']  # air density
#     Kh3 = constant['Kh3']
#     Kth3 = constant['Kth3']
#     Kbt3 = constant['Kbt3']
#
#     # Linear part:
#     # ============
#     Q_lin = typical_section_3DOF_lin(sp.identity(12), U, constant, T, coefW)
#
#     # Equivalent stiffness matrix - nonlinear part:
#     # =============================================
#     A_tmp = matrix_mass_str(constant)
#     B_tmp = matrix_mass_usaero(constant, T)
#
#     M_inv = ln.inv((A_tmp + rho*B_tmp))
#
#     E_nlin = sp.zeros((3, 3))
#     E_nlin[0, 0] = 3.*Kh3*A[0]**2/4.
#     E_nlin[1, 1] = 3.*Kth3*A[1]**2/4.
#     E_nlin[2, 2] = 3.*Kbt3*A[2]**2/4.
#
#     Q_eqlin = Q_lin
#     Q_eqlin[:3, 3:6] += -M_inv.dot(E_nlin)
#
#     return Q_eqlin.dot(x)
#
#
# def jacobian_nonlin(x0, U, constant, T, coefW):
#     """
#     Determine the jacobian of the nonlinear 3DOF system around the point x0.
#
#     :param x0: vector of DOF at which to calculate the Jacobian
#     :param U: free-stream velocity
#     :param constant: system properties
#     :param T: Theodorsen coefficients
#     :param coefW: Wagner function coefficients
#     :return: Jacobian of the nonlinear system at point x0
#     """
#
#     rho = constant['rho']  # air density
#     Kh3 = constant['Kh3']
#     Kth3 = constant['Kth3']
#     Kbt3 = constant['Kbt3']
#
#     # Linear part:
#     jac_lin = typical_section_3DOF_lin(sp.identity(12), U, constant, T, coefW)
#
#     # Non-linear part:
#     A = matrix_mass_str(constant)
#     B = matrix_mass_usaero(constant, T)
#
#     M_inv = ln.inv((A + rho*B))
#
#     qn_plunge = sp.zeros((12,))
#     qn_plunge[:3] = -M_inv.dot([Kh3, 0., 0.])
#
#     qn_pitch = sp.zeros((12,))
#     qn_pitch[:3] = -M_inv.dot([0., Kth3, 0.])
#
#     qn_control = sp.zeros((12,))
#     qn_control[:3] = -M_inv.dot([0., 0., Kbt3])
#
#     jac_nonlin = sp.zeros((12, 12))
#     jac_nonlin[:, 3] = 3*qn_plunge*x0[3]**2
#     jac_nonlin[:, 4] = 3*qn_pitch*x0[4]**2
#     jac_nonlin[:, 5] = 3*qn_control*x0[5]**2
#
#     return jac_lin + jac_nonlin
#
#
# def sensitivity_dQdU_eqlin(U, constant, T, coefW):
#     rho = constant['rho']  # air density
#
#     A = matrix_mass_str(constant)
#     B = matrix_mass_usaero(constant, T)
#     C = matrix_damping_str(constant)
#     D = matrix_damping_usaero(constant, T, coefW)
#     E = matrix_stiffness_str(constant)
#     F = matrix_stiffness_usaero(constant, T, coefW)
#     W = matrix_aero_influence(constant, T, coefW)
#     W1 = matrix_aero_state1()
#     W2 = matrix_aero_state2(constant, coefW)
#
#     M_inv = ln.inv((A + rho*B))
#
#     dQdU = sp.zeros((12, 12))
#     dQdU[:3, :3] = (-1.)*M_inv.dot(rho*D)
#     dQdU[:3, 3:6] = (-1.)*M_inv.dot(2*rho*U*F)
#     dQdU[:3, 6:] = (-1.)*3*rho*U**2*M_inv.dot(W)
#     dQdU[6:, 6:] = W2
#
#     return dQdU
#
#
# def LCO_branch_eqlin(i_branch, A, Umin, Umax, constant, T, coefW, delta_NR=1e-10):
#     """
#     The function pinpoints the velocity of the LCO for a given amplitude. The function  is based on newton-Raphson
#     nonlinear solver. Therefore a fairly good initial guess of the LCO velocity has to be provided.
#
#     :param i_branch: indicates whcih LCO branch to follow. valid values: 0, 1, 2.
#     :param A: LCO amplitude vector (h, th, beta)
#     :param U: initial guess for the LCO velocity [m/s]
#     :param constant: system properties
#     :param T: Theodorsen coefficients
#     :param coefW: Wagner coefficients
#     :param delta_NR: accuracy of the Newton-Raphson search algorithm
#     :return: U, omega of the LCO with amplitude A
#     """
#
#     nn = 100
#     U = sp.linspace(Umin, Umax, nn)
#     w_n = sp.zeros((nn, 3))
#     zeta = sp.zeros((nn, 3))
#
#     zeta_min, zeta_max, idx_max = 0, 0, 0
#     is_hopf = False
#
#     # Locate the LCO velocity:
#     for i, U_i in enumerate(U):
#
#         if i > 3:
#             f_w = interpolate.interp1d(U[:i], w_n[:i, :], kind='cubic', axis=0, fill_value='extrapolate')
#             f_zeta = interpolate.interp1d(U[:i], zeta[:i, :], kind='cubic', axis=0, fill_value='extrapolate')
#
#             # IMPORTANT:
#             # reshape is used to transform the vector from shape (3,) to shape (3,1) -> allows for broadcasting along the
#             # last dimension of the reshaped array. This omits the need to copy the array 3 times in order to calculate all
#             # the possible differences in the next step
#             zeta_interp = f_zeta(U_i)
#             w_interp = f_w(U_i)
#
#             if i == 47:
#                 tmp = 5
#         else:
#             zeta_interp = sp.nan*sp.ones((3,))
#             w_interp = sp.nan*sp.ones((3,))
#
#         w_n[i], zeta[i] = LCO_frequency_damping_eqlin(A, U_i, w_interp, zeta_interp, T, coefW, constant)
#
#         # check for Hopf biffurcation (occurence of flutter point):
#         if i > 0:
#             if zeta[i - 1, i_branch]*zeta[i, i_branch] < 0:
#                 Umin = U[i - 1]
#                 Umax = U[i]
#
#                 zeta_min = zeta[i - 1, i_branch]
#                 zeta_max = zeta[i, i_branch]
#
#                 idx_max = i
#
#                 is_hopf = True
#
#                 break
#
#     # Pinpoint the flutter speed:
#     # ===========================
#
#     if is_hopf:
#         # NOTE:
#         # idx_max +1 is required in ordred to include the last calculated zeta in the interpolation function!
#         # Otherwise you keep extrapolating.
#
#         f_w = interpolate.interp1d(U[:idx_max + 1], w_n[:idx_max + 1, :], kind='cubic', axis=0,
#                                    fill_value='extrapolate')
#         f_zeta = interpolate.interp1d(U[:idx_max + 1], zeta[:idx_max + 1, :], kind='cubic', axis=0,
#                                       fill_value='extrapolate')
#
#         # use te Bisection method on the damping coefficient of the selected branch
#         R1 = 1
#         dU = Umax - Umin
#
#         omega_max = 0
#
#         while (sp.absolute(R1) > delta_NR):
#             dU = -dU*zeta_min/(zeta_max - zeta_min)
#             Umax = Umin + dU
#
#             w_interp = f_w(Umax)
#             zeta_interp = f_zeta(Umax)
#
#             w_n_tmp, zeta_tmp = LCO_frequency_damping_eqlin(A, Umax, w_interp, zeta_interp, T, coefW, constant)
#             zeta_max = zeta_tmp[i_branch]
#             omega_max = w_n_tmp[i_branch]
#
#             R1 = zeta_max
#
#         return Umax, omega_max
#
#     else:
#         return sp.nan, sp.nan
#
#
# def LCO_frequency_damping_eqlin(A, U_i, w_interp, zeta_interp, T, coefW, constant):
#     # Assemble 3DOF system matrix
#     x0 = sp.identity(12)  # dummy input to retrieve the 3DOF system matrix
#     Q = typical_section_3DOF_eqlin(x0, A, U_i, constant, T, coefW)
#
#     # Calculate eigen values and eigen vectors
#     eval, evec = ln.eig(Q)
#
#     # sort out the complex-conjugate eigenvalues and their pertinent eigen vectors:
#     eps_fltr = 1e-12
#     eval_cc = eval[eval.imag > eps_fltr]
#     evec_cc = evec[:, eval.imag > eps_fltr]
#     idx = sp.argsort(eval_cc.imag)
#     eval_cc = eval_cc[idx]
#     if len(eval_cc) > 3:
#         print(eval_cc)
#
#     w_n = sp.absolute(eval_cc)
#     zeta = -eval_cc.real/sp.absolute(eval_cc)
#
#     # sort based on the exrtrapolted value:
#     if not (any(sp.isnan(zeta_interp))):
#         # IMPORTANT:
#         # reshape is used to transform the vector from shape (3,) to shape (3,1) -> allows for broadcasting along the
#         # last dimension of the reshaped array. This omits the need to copy the array 3 times in order to calculate all
#         # the possible differences in the next step
#         zeta_interp = sp.reshape(zeta_interp, (3, 1))
#         w_interp = sp.reshape(w_interp, (3, 1))
#
#         dw = sp.absolute(w_n - w_interp)
#         dzeta = sp.absolute(zeta - zeta_interp)
#
#         diff = dw*dzeta + dzeta
#
#         idx_min1 = dw.argmin(axis=0)
#         idx_min2 = diff.argmin(axis=0)  # find the minimum element in each row
#
#         w_n = w_n[idx_min2]
#         zeta = zeta[idx_min2]
#
#     return w_n, zeta
#
#
# def stability_LCO_eqlin(A_up, A_down, U, constant, T, coefW):
#     Q_up = typical_section_3DOF_eqlin(sp.identity(12), A_up, U, constant, T, coefW)
#     evals_up = ln.eigvals(Q_up)
#
#     Q_down = typical_section_3DOF_eqlin(sp.identity(12), A_down, U, constant, T, coefW)
#     evals_down = ln.eigvals(Q_down)
#
#     stable_up = True
#     if any(evals_up.real > 0):
#         stable_up = False
#
#     stable_down = False
#     if any(evals_down.real > 0):
#         stable_down = True
#
#     return (stable_down and stable_up)
#
#
# def tvla(dt, x0, U, constant, T, coefW, delta=1e-3):
#     """
#     tvla stands for Time-Varying Linear Approximation. It calculates the next value of the DOF system
#     :param dt:
#     :param x0:
#     :param U:
#     :param constant:
#     :param T:
#     :param coefW:
#     :return: the used time step
#     """
#
#     A = jacobian_nonlin(x0, U, constant, T, coefW)
#     f = typical_section_3DOF_nonlin(x0, U, constant, T, coefW)
#
#     L, V = ln.eig(A)
#     Vinv = ln.inv(V)
#
#     b = Vinv.dot(f)
#
#     # Check convergence of the time step
#     while True:
#         dx = sp.zeros((12,), dtype='complex')
#         for i in range(12):
#             dx += -1*V[:, i]*(1/L[i])*(1 - sp.exp(L[i]*dt))*b[i]
#
#         x_bar = x0 + dx.real
#
#         A_bar = jacobian_nonlin(x_bar, U, constant, T, coefW)
#         f_bar = typical_section_3DOF_nonlin(x_bar, U, constant, T, coefW)
#
#         L_bar, V_bar = ln.eig(A_bar)
#         Vinv_bar = ln.inv(V_bar)
#
#         b_bar = Vinv_bar.dot(f_bar)
#
#         dx_bar = sp.zeros((12,), dtype='complex')
#         for i in range(12):
#             dx_bar += -1*V_bar[:, i]*(1/L_bar[i])*(1 - sp.exp(-1.*L_bar[i]*dt))*b_bar[i]
#
#         error = dx.real + dx_bar.real
#         eps = sp.sqrt(error.dot(error))
#
#         if eps < delta:
#             break
#         else:
#             dt *= 0.5
#
#     x = x0 + dx.real
#
#     return dt, x
#
#
# def tvla_event(dt, x, U, constant, T, coefW, delta=1e-3):
#     """
#     tvla_event stands for Time-Varying Linear Approximation with detection of a max/min value of DOF. The
#     method is using tvla method as a calculation engine to calculate the next value of the DOF system
#
#     :param dt: initial time step
#     :param x: initial condition (known value of DOF)
#     :param U: free-stream velocity
#     :param constant: system properties
#     :param T: Theodorsen coefficients
#     :param coefW: Wagner function coefficients
#     :param delta: convergence criteria
#     :return:
#         dt_bat: accepted timestep
#         x_bar: value of DOF @ t+ dt_bar
#     """
#     delta2 = 1e-12
#
#     # tvla method to perform the normal step:
#     dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)
#
#     event = False
#     if constant['event'] == 'plunge':
#         if x_bar[0]*x[0] < 0:  # plunge DOF crossed through 0
#             event = True
#             while abs(x_bar[0]) > delta2:
#                 dt = -dt_bar*x[0]/(x_bar[0] - x[0])
#                 dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)
#
#             x_bar[0] = 0  # set to 0 in order to prevent the event from firing again
#
#     elif constant['event'] == 'pitch':
#         if x_bar[1]*x[1] < 0:  # pitch DOF crossed through 0
#             event = True
#             while abs(x_bar[1]) > delta2:
#                 dt = -dt_bar*x[1]/(x_bar[1] - x[1])
#                 dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)
#
#             x_bar[1] = 0  # set to 0 in order to prevent the event from firing again
#
#     elif constant['event'] == 'control':
#         if x_bar[2]*x[2] < 0:  # control DOF crossed through 0
#             event = True
#             while abs(x_bar[2]) > delta2:
#                 dt = -dt_bar*x[2]/(x_bar[2] - x[2])
#                 dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)
#
#             x_bar[2] = 0  # set to 0 in order to prevent the event from firing again
#
#     return dt_bar, x_bar, event
#
#
# def tvla_event_zero(dt, x, U, constant, T, coefW, delta=1e-3):
#     """
#     tvla_event_zero stands for Time-Varying Linear Approximation with detection of a DOF passing through zero. The
#     method is using tvla method as a calculation engine to calculate the next value of the DOF system
#
#     :param dt: initial time step
#     :param x: initial condition (known value of DOF)
#     :param U: free-stream velocity
#     :param constant: system properties
#     :param T: Theodorsen coefficients
#     :param coefW: Wagner function coefficients
#     :param delta: convergence criteria
#     :return:
#         dt_bat: accepted timestep
#         x_bar: value of DOF @ t+ dt_bar
#     """
#     delta2 = 1e-12
#
#     # tvla method to perform the normal step:
#     dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)
#
#     idx_event = -1
#     if constant['event'] == 'plunge':
#         idx_event = 3
#
#     elif constant['event'] == 'pitch':
#         idx_event = 4
#
#     elif constant['event'] == 'control':
#         idx_event = 5
#
#     event = False
#     if idx_event > 0:
#         if x_bar[idx_event]*x[idx_event] < 0:  # pitch DOF crossed through 0
#             event = True
#
#             dtA = 0
#             dtB = dt
#             xA = x
#             xB = x_bar
#
#             while abs(x_bar[idx_event]) > delta2:
#
#                 # TODO: implement regula falsi method -> the secant method is NOT guaranteed to converge!
#
#                 # Regula falsi:
#                 # =============
#                 dtC = (dtA*xB[idx_event] - dtB*xA[idx_event])/(xB[idx_event] - xA[idx_event])
#                 dt_tmp, xC = tvla(dtC, x, U, constant, T, coefW, delta)
#
#                 if xC[idx_event]*xA[idx_event] > 0:
#                     dtA = dtC
#                     xA = xC
#                 else:
#                     dtB = dtC
#                     xB = xC
#
#                 if abs(xA[idx_event]) < abs(xB[idx_event]):
#                     x_bar = xA
#                     dt_bar = dtA
#                 else:
#                     x_bar = xB
#                     dt_bar = dtB
#
#                 # the old method:
#                 # dt = -dt_bar*x[4]/(x_bar[4] - x[4])
#                 # dt_bar, x_bar = tvla(dt, x, U, constant, T, coefW, delta)
#
#             x_bar[idx_event] = 0  # set to 0 in order to prevent the event from firing again
#
#     return dt_bar, x_bar, event
