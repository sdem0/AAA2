import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln

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

    n_dof = fem['n_dof']
    y_nd = fem['y_nd']
    b_u = fem['b_u']

    bw_u = tsm['bw_u']
    n_w = tsm['n_w']

    n_dof_red = len(b_u[b_u])

    x = np.identity(2*n_dof_red)

    U = np.linspace(0, 100, 200)

    # Structural DOFs/states:
    arr_omega, arr_damping, arr_phi, arr_evals = [], [], [], []

    for U_i in U:
        # Aeroelastic state-space model:
        # ------------------------------
        AA_ae = fss.beam_qsae_lin(x, U_i, fem, tsm)
        evals, evecs_red = ln.eig(AA_ae)

        # Structural states:
        omega = np.abs(evals.imag)
        i_1 = omega.argsort()

        omega = omega[i_1][::2]
        damping = evals.real
        damping = damping[i_1][::2]

        phi_red = evecs_red.real[:n_dof_red, :]
        phi_red = phi_red[:, i_1][:, ::2]
        # Reconstruct all the DOFs:
        phi = np.zeros((n_dof, n_dof_red))
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

    # Plot:
    # -----

    # Plot the structural modes:
    lbl_y = ['omega, [rad/s]', 'damping, [?]']
    fig, ax = plt.subplots(2, 1, sharex=True, num=100)

    ax[1].axhline(0.0, color='k', linewidth=0.5)
    for i in range(10):
        ax[0].plot(U, arr_omega[:, i])
        ax[1].plot(U, arr_damping[:, i])

    ax[1].set_xlabel('velocity, [m/s]')
    ax[0].set_ylabel(lbl_y[0])
    ax[1].set_ylabel(lbl_y[1])

    plt.show()
