import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln
import json

import FEM.fem_linear as fe
import FEM.fem_loads as fl
import FEM.fem_validation as val

np.set_printoptions(precision=4, linewidth=400)

if __name__ == "__main__":
    # constant = fe.get_constant_fem()

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
    r = np.zeros((n_nd,))  # Tip torque, [Nm]
    r[-1] = 1.0

    f = np.zeros((n_nd,))  # shear force, [N]
    f[-1] = 1.0

    q = 0.0*np.ones((n_nd,))  # bending moment distribution, [Nm/m]

    # region Validation data

    # Calculate the eigen frequencies of the beam in bending:
    y_beam, w, theta = val.beam_CF_constant_tip_load(1, 1, constant['fem'])

    print('bending maximum deflection [m]: {}'.format(w.max()))
    print('torsion maxium deflection [rad]: {}'.format(theta.max()))
    # endregion

    # region 2nd order system approach:
    KK = fe.mat_stiffness(fem)
    KKinv = ln.inv(KK)

    DD = fl.mat_force_dsc(fem)
    u_red = fl.generate_load_vector(fem, r, f, q)

    # Solution:
    # ---------
    resp1_red = KKinv.dot(DD.dot(u_red))

    # add the known DOFs back to the response vector
    resp1 = np.zeros((n_dof,))
    resp1[b_u] = resp1_red

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
