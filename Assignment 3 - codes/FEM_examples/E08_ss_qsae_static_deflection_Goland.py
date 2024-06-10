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

    with open('../Configurations/wing_TangDowell.json', 'r') as f:
        const = json.load(f)

    fem = fe.initialise_fem(const['fem'])
    tsm = fl.initialise_aero(const['aero'], fem)

    n_nd = fem['n_nd']
    n_dof = fem['n_dof']
    y_nd = fem['y_nd']
    b_u = fem['b_u']

    bss_u = np.append(b_u, b_u)

    n_dof_red = len(b_u[b_u])

    U = 20.0
    aoa0 = 5*pi/180.0

    # region quasi-steady state-space

    SS = fss.ss_fem_qsae_lin(U, fem, tsm)

    AA, BB_aoa0 = SS['A'], SS['B_aoa0']

    AA_inv = ln.inv(AA)

    xi_red = (-1.0)*AA_inv@BB_aoa0*aoa0

    xi = np.zeros(2*n_dof, )
    xi[bss_u] = xi_red

    # endregion

    # region quasi-steady state-space 2

    SS3 = fss.ss_fem_qsae_lin2(U, fem, tsm)

    AA3, BB3_aoa0 = SS3['A'], SS3['B_aoa0']

    AA3_inv = ln.inv(AA3)

    xi3_red = (-1.0)*AA3_inv@BB3_aoa0*aoa0

    xi3 = np.zeros(2*n_dof, )
    xi3[bss_u] = xi3_red

    # endregion


    # region unsteady aero state-space

    SS4 = fss.ss_fem_usae_lin(U, fem, tsm)

    AA4, BB4_aoa0 = SS4['A'], SS4['B_aoa0']

    AA4_inv = ln.inv(AA4)

    xi4_red = (-1.0)*AA4_inv@BB4_aoa0*aoa0

    xi4 = np.zeros(2*n_dof, )
    xi4[bss_u] = xi4_red[:2*n_dof_red]

    # endregion

    # region Compare to deformations due to rigid aerodynamic loads:
    rho = tsm['rho']
    b = tsm['b_nd'][0]
    a = tsm['a_nd'][0]

    # Calculate with and without AoA correction (tip twis of the wing):
    arr_xi2 = []
    for i in range(2):
        # AoA correction:

        daoa = i*xi[-3]  # add tip twist to the aoa to assess the effect of elastic twist on aeroelastic deformation

        # Aerodynamic lift and torque per strip
        l = 2*pi*rho*U**2*b*(aoa0 + daoa)  # [N/m]
        m_xf = 2*pi*rho*U**2*b**2*(0.5 + a)*(aoa0 + daoa)  # [Nm/m]

        # Nodal values of distributed structural torque moment, r, shear force, f, and bending moment, q.
        r = m_xf*np.ones((n_nd,))  # [Nm/m]
        f = l*np.ones((n_nd,))  # [N/m]
        q = np.zeros((n_nd,))  # [Nm/m]

        lds = np.zeros((n_dof,))
        lds[0::3] = r
        lds[1::3] = f
        lds[2::3] = q

        lds_red = lds[b_u]

        # Solve the structural problem directly:
        KK = fe.mat_stiffness(fem)
        KKinv = ln.inv(KK)

        DDdst = fl.mat_force_dst(fem)

        xi2_red = KKinv@DDdst@lds_red

        xi2 = np.zeros((n_dof,))
        xi2[b_u] = xi2_red

        arr_xi2.append(xi2)

    # endregion

    # region Analytical solution

    # Adopted from eq. 4.52 in Introduction to Structural Dynamics and Aeroelasticity(Hodges and Pierce, 2002)

    # wing parameters
    rho = tsm['rho']
    b = tsm['b_nd'][0]
    c = 2*b
    a = tsm['a_nd'][0]
    e = (0.5 + a)*b
    GJ = fem['GJ'][0]
    y_nd = fem['y_nd']
    qd = 0.5*rho*U**2
    lmbd = np.sqrt(qd*c*2*pi*e/GJ)

    # Analytical solution, eq. 4.52:
    th = aoa0*(np.tan(lmbd*y_nd[-1])*np.sin(lmbd*y_nd) + np.cos(lmbd*y_nd) - 1)

    # endregion

    # Plot:
    # -----

    # Plot the structural deformation:
    lbl_y = ['theta, [rad]', 'v, [m]', 'beta, [rad]']

    fig, ax = plt.subplots(3, 1, sharex=True, num=10)

    ax[0].plot(y_nd, th, 'o')
    for j in range(3):
        ax[j].plot(y_nd, xi[j:n_dof:3], '--')
        ax[j].plot(y_nd, xi[n_dof + j::3])
        ax[j].plot(y_nd, xi3[n_dof + j::3], '--')
        ax[j].plot(y_nd, xi4[n_dof + j::3], 'o--')
        ax[j].plot(y_nd, arr_xi2[0][j::3], ':', y_nd, arr_xi2[1][j::3], ':')
        ax[j].plot(y_nd, -arr_xi2[0][j::3], ':', y_nd, -arr_xi2[1][j::3], ':')

        if j == 2:
            ax[j].set_xlabel('span, [m]')

        ax[j].set_ylabel(lbl_y[j])

    fig.suptitle('U = {:.3f} m/s, aoa0 = {:.3f} deg'.format(U, aoa0))

    plt.show()
