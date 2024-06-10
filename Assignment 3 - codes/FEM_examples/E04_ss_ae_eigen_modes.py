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

    with open('../Configurations/wing_Goland.json') as f:
        const = json.load(f)

    fem = fe.initialise_fem(const['fem'])
    tsm = fl.initialise_aero(const['aero'], fem)

    n_dof = fem['n_dof']
    y_nd = fem['y_nd']
    b_u = fem['b_u']

    bw_u = tsm['bw_u']
    n_w = tsm['n_w']

    n_dof_red = len(b_u[b_u])
    n_w_red = len(bw_u[bw_u])

    x = np.identity(2*n_dof_red + n_w_red)
    x_qs = np.identity(2*n_dof_red)
    U = 0.0

    # Unsteady Aeroelastic state-space model:
    # ---------------------------------------
    AA_ae = fss.beam_ae_lin(x, U, fem, tsm)
    evals1, evecs1_red = ln.eig(AA_ae)

    omega1 = np.abs(evals1.imag)
    i_1 = omega1.argsort()
    omega1 = omega1[i_1][n_w_red::2]
    phi1_red = evecs1_red.real[:n_dof_red, :]
    phi1_red = phi1_red[:, i_1][:, n_w_red::2]

    # Reconstruct all the DOFs:
    phi1 = np.zeros((n_dof, n_dof_red))
    phi1[b_u, :] = phi1_red

    # Omit aerodynamic modes:
    # omega1 = omega1[n_w_red::2]
    # phi1 = phi1[:, n_w_red::2]


    # Quasi-steady Aeroelastic state-space model:
    # -------------------------------------------
    AA_qs = fss.beam_qsae_lin(x_qs, U, fem, tsm)
    evals3, evecs3_red = ln.eig(AA_qs)

    omega3 = np.abs(evals3.imag)
    i_3 = omega3.argsort()
    omega3 = omega3[i_3][::2]
    phi3_red = evecs3_red.real[:n_dof_red, :]
    phi3_red = phi3_red[:, i_3][:, ::2]

    # Reconstruct all the DOFs:
    phi3 = np.zeros((n_dof, n_dof_red))
    phi3[b_u, :] = phi3_red

    # Quasi-steady Aeroelastic state-space model2:
    # -------------------------------------------
    AA_qs2 = fss.beam_qsae_lin2(x_qs, U, fem, tsm)
    evals4, evecs4_red = ln.eig(AA_qs2)

    omega4 = np.abs(evals4.imag)
    i_4 = omega4.argsort()
    omega4 = omega4[i_4][::2]
    phi4_red = evecs4_red.real[:n_dof_red, :]
    phi4_red = phi4_red[:, i_4][:, ::2]

    # Reconstruct all the DOFs:
    phi4 = np.zeros((n_dof, n_dof_red))
    phi4[b_u, :] = phi4_red

    # Omit aerodynamic modes:
    # omega1 = omega1[n_w_red::2]
    # phi1 = phi1[:, n_w_red::2]

    # Structural-only State-Space model:
    # ----------------------------------
    SS = fss.ss_fem_lin(fem)
    AA = SS['A']

    # Modal analysis:
    evals2, evecs2_red = ln.eig(AA)

    omega2 = np.abs(evals2.imag)
    i_2 = omega2.argsort()
    omega2 = omega2[i_2][::2]
    phi2_red = evecs2_red.real[:n_dof_red, :]
    phi2_red = phi2_red[:, i_2][:, ::2]

    # Reconstruct all the DOFs:
    phi2 = np.zeros((n_dof, n_dof_red))
    phi2[b_u, :] = phi2_red

    print(omega1/(2*pi))
    print('========================')
    print(omega2/(2*pi))
    print('========================')
    print((omega2 - omega1)/omega1)

    # Plot:
    # -----

    lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']
    arr_fig, arr_ax = [], []

    for i in range(15):
        fig, ax = plt.subplots(3, 1, sharex=True, num=100 + i)
        for j in range(3):
            ax[j].plot(y_nd, phi1[j::3, i], y_nd, phi2[j::3, i], '--', y_nd, phi3[j::3, i], ':', y_nd, phi4[j::3, i], '.')

            if j == 2:
                ax[j].set_xlabel('span, [m]')

            ax[j].set_ylabel(lbl_y[j])

        fig.suptitle('2nd Ord: {:.3f} || SS_us: {:.3f} || SS_qs: {:.3f} ||  SS_qs2: {:.3f} [rad/s]'.format(omega1[i], omega2[i], omega3[i], omega4[i]))
        arr_fig.append(fig)
        arr_ax.append(ax)

    plt.show()
