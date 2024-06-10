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

    # region Flutter calculation:
    n_dof = fem['n_dof']
    y_nd = fem['y_nd']
    b_u = fem['b_u']

    bw_u = tsm['bw_u']
    n_w = tsm['n_w']

    n_dof_red = len(b_u[b_u])

    x = np.identity(2*n_dof_red)

    U = np.linspace(0, U_max, 500)

    # Structural DOFs/states:
    arr_omega, arr_damping, arr_phi, arr_evals = [], [], [], []

    for U_i in U:

        # Aeroelastic state-space model:
        # ------------------------------
        AA_ae = fss.beam_qsae_lin(x, U_i, fem, tsm, dmo)
        evals, evecs_red = ln.eig(AA_ae)

        # SS = fss.ss_fem_qsae_lin(U_i, fem, tsm)
        # AA_ss = SS['A']
        # evals_ss, evecs_ss_red = ln.eig(AA_ss)

        # Structural states:
        omega = np.abs(evals.imag)
        i_1 = omega.argsort()

        omega = omega[i_1][::1]
        damping = evals.real
        damping = damping[i_1][::1]

        phi_red = evecs_red.real[:n_dof_red, :]
        phi_red = phi_red[:, i_1][:, ::1]

        # Reconstruct all the DOFs:
        phi = np.zeros((n_dof, 2*n_dof_red))
        phi[b_u, :] = phi_red

        # Save to arrays for plotting
        arr_omega.append(omega)
        arr_damping.append(damping)
        arr_phi.append(phi)

        arr_evals.append(evals)

    arr_omega = np.array(arr_omega)
    arr_damping = np.array(arr_damping)
    arr_phi = np.array(arr_phi)

    arr_evals = np.array(arr_evals)

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

    # --------------------------
    # Plot the structural modes:
    # --------------------------
    lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']
    arr_fig1, arr_ax1 = [], []

    for i in range(5):
        fig, ax = plt.subplots(3, 1, sharex=True, num=10 + i)
        for j in range(3):
            ax[j].plot(y_nd, arr_phi[0, j::3, i])

            if j == 2:
                ax[j].set_xlabel('span, [m]')

            ax[j].set_ylabel(lbl_y[j])

        fig.suptitle('Eigen freq.: {:.3f} [Hz]'.format(arr_omega[0, i]/(2*pi)))
        arr_fig1.append(fig)
        arr_ax1.append(ax)

    # ---------------
    # Plot v-g plots:
    # ---------------
    lbl_y = ['omega, [rad/s]', 'damping, [?]']
    fig, ax = plt.subplots(2, 1, sharex=True, num=20)

    ax[1].axhline(0.0, color='k', linewidth=0.5)
    ax[1].plot(U_div, 0, 'x')

    # for i in range(n_dof_red):
    for i in range(40):
        ax[0].plot(U, arr_omega[:, i], '.')
        ax[1].plot(U, arr_damping[:, i], '.')

    ax[1].set_xlabel('velocity, [m/s]')
    ax[0].set_ylabel(lbl_y[0])
    ax[1].set_ylabel(lbl_y[1])

    plt.show()

    # endregion
