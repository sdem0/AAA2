import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln
import json

import FEM.fem_linear as fe
import FEM.fem_loads as fl
import FEM.fem_state_space as fss
import FEM.fem_validation as val

np.set_printoptions(precision=4, linewidth=400)

if __name__ == "__main__":

    with open('../Configurations/wing_TangDowell.json') as f:
        constant = json.load(f)

    # Decouple torsion from bending:
    constant['fem']['d'] = 0
    constant['fem']['KBT'] = 0

    # chose boundary condition:
    constant['fem']['bc'] = 'CF'

    # region Validation data

    # Calculate the eigen frequencies of the beam in bending:
    if constant['fem']['bc'] == 'FF':
        omega_b, y_b, phi_b = val.beam_FF_modes_bending(constant['fem'])
        omega_t, y_t, phi_t = val.beam_FF_modes_torsion(constant['fem'])
    elif constant['fem']['bc'] == 'CF':
        omega_b, y_b, phi_b = val.beam_CF_modes_bending(constant['fem'])
        omega_t, y_t, phi_t = val.beam_CF_modes_torsion(constant['fem'])

    print('bending eigenfrequencies [rad/s]: {}'.format(omega_b))
    print('torsion eigenfrequencies [rad/s]: {}'.format(omega_t))
    # endregion

    fem = fe.initialise_fem(constant['fem'])

    # region 2nd order system approach:
    MM = fe.mat_mass(fem)
    MMinv = ln.inv(MM)

    KK = fe.mat_stiffness(fem)
    DD = fl.mat_force_dst(fem)

    # Modal analysis:
    # ---------------
    evals1, evecs1_red = ln.eig(-MMinv.dot(KK))

    # add the known DOFs to the eigen vectors
    b_u = fem['b_u']
    n_dof = fem['n_dof']

    evecs1 = np.zeros((n_dof, evecs1_red.shape[1]), dtype=complex)
    evecs1[b_u, :] = evecs1_red

    omega_fem = np.imag(np.sqrt(evals1))

    # Sort the eigen values and vectors:
    # ----------------------------------
    idx_sort1 = omega_fem.argsort()
    omega_fem = omega_fem[idx_sort1]
    evals1 = evals1[idx_sort1]
    evecs1 = evecs1[:, idx_sort1]

    # endregion

    # region State-Space approach:
    SS = fss.ss_fem_lin(fem)
    AA = SS['A']

    # Modal analysis:
    evals2, evecs2_red = ln.eig(AA)

    # Add the known DOFs to the eigenvectors:
    s_u = np.concatenate((b_u, b_u))

    evecs2 = np.zeros((2*n_dof, evecs2_red.shape[1]), dtype=complex)
    evecs2[s_u, :] = evecs2_red

    omega_ss = np.imag(evals2)

    # Sort the eigen values and vectors:
    # ----------------------------------
    n_dof_red = n_dof - np.sum(~b_u)
    idx_sort2 = omega_ss.argsort()
    omega_ss = omega_ss[idx_sort2]
    omega_ss = omega_ss[n_dof_red:]
    evals2 = evals2[idx_sort2]
    evecs2 = evecs2[:, idx_sort2]
    evecs2 = evecs2[:n_dof, n_dof_red:]

    # endregion

    # region Plot
    y_nd = fem['y_nd']

    lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']

    arr_fig, arr_ax = [], []

    # 2nd order approach:
    for i in range(15):
        fig, ax = plt.subplots(3, 1, sharex=True, num=100 + i)
        for j in range(3):
            ax[j].plot(y_nd, evecs1[j::3, i].real, y_nd, evecs2[j::3, i].real, '--')
            ax[j].plot()

            if j == 2:
                ax[j].set_xlabel('span, [m]')

            ax[j].set_ylabel(lbl_y[j])

        fig.suptitle('2nd Order: {:.3f} || State space: {:.3f} [rad/s]'.format(omega_fem[i], omega_ss[i]))
        arr_fig.append(fig)
        arr_ax.append(ax)

    plt.show()

    # endregion
