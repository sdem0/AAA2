import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln
import json

from numpy import pi

import FEM.fem_linear as fe
import FEM.fem_loads as fl
import FEM.fem_state_space as fss

np.set_printoptions(precision=4, linewidth=400)

if __name__ == "__main__":

    with open('../Configurations/wing_TangDowell.json') as f:
        const = json.load(f)
        U_max = 80


    # with open('../Configurations/wing_Goland.json') as f:
    #     const = json.load(f)
    #     U_max = 300

    fem = fe.initialise_fem(const['fem'])
    dmo = fe.initialise_dmo(const['dmo'], fem)
    tsm = fl.initialise_aero(const['aero'], fem)

    # region Flutter calculation
    n_dof = fem['n_dof']
    y_nd = fem['y_nd']
    b_u = fem['b_u']

    bw_u = tsm['bw_u']
    n_w = tsm['n_w']

    n_dof_red = len(b_u[b_u])
    n_w_red = len(bw_u[bw_u])

    x = np.identity(2*n_dof_red + n_w_red)

    U = np.linspace(0, U_max, 200)

    # Structural DOFs/states:
    arr_omega, arr_damping, arr_phi, arr_evals = [], [], [], []

    # Lag states:
    arr_lag_re, arr_lag_im = [], []

    for U_i in U:
        # Aeroelastic state-space model:
        # ------------------------------
        AA_ae = fss.beam_ae_lin(x, U_i, fem, tsm, dmo)
        evals, evecs_red = ln.eig(AA_ae)

        # Structural states:
        omega = np.abs(evals.imag)
        i_1 = omega.argsort()

        omega = omega[i_1][n_w_red::1]
        damping = evals.real
        damping = damping[i_1][n_w_red::1]

        phi_red = evecs_red.real[:n_dof_red, :]
        phi_red = phi_red[:, i_1][:, n_w_red::1]
        # Reconstruct all the DOFs:
        phi = np.zeros((n_dof, 2*n_dof_red))
        phi[b_u, :] = phi_red

        # Lag states:
        evals = evals[i_1]
        lag_re = evals[:n_w_red].real
        lag_im = evals[:n_w_red].imag

        i_lag = lag_re.argsort()
        lag_re = lag_re[i_lag]
        lag_im = lag_im[i_lag]

        # Save to arrays for plotting
        arr_omega.append(omega)
        arr_damping.append(damping)
        arr_phi.append(phi)

        arr_evals.append(evals)

        arr_lag_re.append(lag_re)
        arr_lag_im.append(lag_im)

    arr_omega = np.array(arr_omega)
    arr_damping = np.array(arr_damping)
    arr_phi = np.array(arr_phi)

    arr_evals = np.array(arr_evals)

    arr_lag_re = np.array(arr_lag_re)
    arr_lag_im = np.array(arr_lag_im)

    # endregion

    # region Analytical solution for static divergence

    # Adopted from eq. 4.55 in Introduction to Structural Dynamics and Aeroelasticity(Hodges and Pierce, 2002)

    # wing parameters
    rho = tsm['rho']
    b = tsm['b_nd'][0]
    c = 2*b
    a = tsm['a_nd'][0]
    e = (0.5 + a)*b
    GJ = fem['GJ'][0]
    y_nd = fem['y_nd']

    qd_div = (GJ/(e*c*2*pi))*(pi/(2*y_nd[-1]))**2
    U_div = np.sqrt(2*qd_div/rho)

    print('Analytic dynamic pressure of static divergence: {} Pa'.format(qd_div))
    print('Analytic velocity of static divergence: {} m/s'.format(U_div))

    # endregion

    # region Plot:

    # Plot the structural modes:
    lbl_y = ['omega, [rad/s]', 'damping, [?]']
    fig, ax = plt.subplots(2, 1, sharex=True, num=100)

    ax[0].axhline(0.0, color='k', linewidth=0.5)
    ax[1].axhline(0.0, color='k', linewidth=0.5)
    # for i in range(n_dof_red):
    for i in range(10):
        ax[0].plot(U, arr_omega[:, i], '.')
        ax[1].plot(U, arr_damping[:, i], '.')

    ax[1].set_xlabel('velocity, [m/s]')
    ax[0].set_ylabel(lbl_y[0])
    ax[1].set_ylabel(lbl_y[1])

    # Plot the lag states:
    lbl_y = ['Re(lag), [?]', 'Im(lag), [?]']
    fig2, ax2 = plt.subplots(2, 1, sharex=True, num=200)
    ax2[0].axhline(0.0, color='k', linewidth=0.5)
    for i in range(n_w_red):
        ax2[0].plot(U, arr_lag_re[:, i], '.')
        ax2[1].plot(U, arr_lag_im[:, i], '.')

    ax2[0].plot(U_div, 0, 'x')

    ax2[1].set_xlabel('velocity, [m/s]')
    ax2[0].set_ylabel(lbl_y[0])
    ax2[1].set_ylabel(lbl_y[1])

    plt.show()

    # endregion
