import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln
import json

import FEM.fem_linear as fe
import FEM.fem_loads as fl
import FEM.fem_validation as val

np.set_printoptions(precision=4, linewidth=400)

if __name__ == "__main__":

    with open('../Configurations/wing_TangDowell.json') as f:
        constant = json.load(f)

    # Decouple torsion from bending:
    constant['fem']['d'] = 0
    constant['fem']['KBT'] = 0

    # Boundary condition:
    constant['fem']['bc'] = 'CF'

    # Initialise FEM properties
    fem = fe.initialise_fem(constant['fem'])
    n_nd = fem['n_nd']
    n_dof = fem['n_dof']
    b_u = fem['b_u']

    # Applied load values at nodes:
    r = 1.0*np.ones((n_nd,))  # torque distribution, [Nm/m]
    f = 1.0*np.ones((n_nd,))  # shear force distribution, [N/m]
    q = 0.0*np.ones((n_nd,))  # bending moment distribution, [Nm/m]

    # region Validation data

    # Calculate the eigen frequencies of the beam in bending:
    y_beam, w, theta = val.beam_CF_constant_distributed_load(1, 1, constant['fem'])

    print('bending maximum deflection [m]: {}'.format(w.max()))
    print('torsion maxium deflection [rad]: {}'.format(theta.max()))
    # endregion

    # region 2nd order system approach:
    KK = fe.mat_stiffness(fem)
    KKinv = ln.inv(KK)

    DD = fl.mat_force_dst(fem)
    u_red = fl.generate_load_vector(fem, r, f, q)

    # Soluiton:
    # ---------------
    resp1_red = KKinv.dot(DD.dot(u_red))

    # add the known DOFs to the eigen vectors
    resp1 = np.zeros((n_dof,))
    resp1[b_u] = resp1_red

    # endregion

    # region State-Space approach:
    # AA, BB = fe.state_space_lin(fem)
    #
    # # Modal analysis:
    # evals2, evecs2_red = ln.eig(AA)
    #
    # # Add the known DOFs to the eigenvectors:
    # s_u = np.concatenate((b_u, b_u))
    #
    # evecs2 = np.zeros((2*n_dof, evecs2_red.shape[1]), dtype=complex)
    # evecs2[s_u, :] = evecs2_red
    #
    # omega_ss = np.imag(evals2)
    #
    # # Sort the eigen values and vectors:
    # # ----------------------------------
    # n_dof_red = n_dof - np.sum(~b_u)
    # idx_sort2 = omega_ss.argsort()
    # omega_ss = omega_ss[idx_sort2]
    # omega_ss = omega_ss[n_dof_red:]
    # evals2 = evals2[idx_sort2]
    # evecs2 = evecs2[:, idx_sort2]
    # evecs2 = evecs2[:n_dof, n_dof_red:]

    # endregion

    # region Plot
    y_nd = fem['y_nd']

    lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']

    fig, ax = plt.subplots(3, 1, sharex=True, num=1)
    for j in range(3):
        if j ==0:
            ax[j].plot(y_beam, theta)
        elif j == 1:
            ax[j].plot(y_beam, w)

        ax[j].plot(y_nd, resp1[j::3])
        ax[j].plot(y_nd, resp1[j::3])

        if j == 2:
            ax[j].set_xlabel('span, [m]')

        ax[j].set_ylabel(lbl_y[j])

    plt.show()

    # endregion
