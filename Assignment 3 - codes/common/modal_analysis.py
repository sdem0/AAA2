"""
Contains methods for modal analysis
"""

from __future__ import division
import scipy as sp
import scipy.linalg as ln


def get_modes_general(Q):
    """
    Calculate the eigen values, eigen vectors and the general modes of a matrix Q
    :param Q: input matrix
    :return: eigen values, eigen vectors and modes of the input matrix Q
    """

    evals, evecs = ln.eig(Q)

    modes  = evecs.real

    return evals, evecs, modes

def get_modes_structural(Q, MD_sort = 0, n_str=3):
    """
    Calculate the eigen values, eigen vectors and the structural modes of a system matrix Q with n_struct structural DOFs.

    DOFs in the system must be ordered in the following way [[x_struct]_dot, [x_struct], [lag states]].

    The structural modes are sorted in the order of the  MD_sort set of modes. If MD_sort is left at 0, then the modes
    are sorted according to the absolute values of their eigenvalues in ascending order.

    :param Q: input matrix
    :param MD_sort: array of modes used for sorting (e.g. from the previous velocity)
    :param n_str: number of structural DOFs.
    :return: eigen values, eigen vectors and modes of the input matrix Q
    """

    # The eigenvalues and eigenvectors:
    evals, evecs, MD = get_modes_general(Q)


    # 1. pick modes according to their eigen values:
    # ==============================================
    eps_cc = 1e-12
    eval_cc = evals[evals.imag > eps_cc]
    evec_cc = evecs[:, evals.imag > eps_cc]
    MD_cc = MD[:, evals.imag > eps_cc]

    idx = sp.argsort(sp.absolute(eval_cc))

    eval_cc = eval_cc[idx]
    evec_cc = evec_cc[:, idx]
    MD_cc = MD_cc[:, idx]

    if len(idx) == n_str:

        MD_str = MD_cc
        MD_freq = sp.absolute(eval_cc)
        MD_damp = -1.*eval_cc.real/sp.absolute(eval_cc)
    else:
        n_zeros = n_str - len(idx)

        MD_str = sp.zeros((12, n_str))
        MD_str[:,:] = sp.nan

        MD_freq = sp.zeros((n_str,))
        MD_freq[:] = sp.nan

        MD_damp = sp.ones((n_str,))
        MD_damp[:] = sp.nan

        MD_str[:, n_zeros:] = MD_cc
        MD_freq[n_zeros:] = sp.absolute(eval_cc)
        MD_damp[n_zeros:] = -1.*eval_cc.real/sp.absolute(eval_cc)

    return MD_str, MD_freq, MD_damp


# def get_modes_structural2(Q, MD_sort = 0, n_str=3):
#     """
#     2nd try:
#     Calculate the eigen values, eigen vectors and the structural modes of a system matrix Q with n_struct structural DOFs.
#
#     DOFs in the system must be ordered in the following way [[x_struct]_dot, [x_struct], [lag states]].
#
#     The structural modes are sorted in the order of the  MD_sort set of modes. If MD_sort is left at 0, then the modes
#     are sorted according to the absolute values of their eigenvalues in ascending order.
#
#     :param Q: input matrix
#     :param MD_sort: array of modes used for sorting (e.g. from the previous velocity)
#     :param n_str: number of structural DOFs.
#     :return: eigen values, eigen vectors and modes of the input matrix Q
#     """
#
#     # Define template modes:
#     # -----------------------
#     # MD_tmp[:,0] = heave mode
#     # MD_tmp[:,1] = pitch mode
#     # MD_tmp[:,2] = control mode
#
#     MD_tmp = sp.zeros((12,3))
#     MD_tmp[0:3,0:3] = sp.identity(3)
#
#     # The eigenvalues and eigenvectors:
#     # ---------------------------------
#     evals, evecs, MD = get_modes_general(Q)
#
#     # MAC values;
#     mac_MD = mac(MD_tmp, MD)
#
#     idx_srt = sp.argsort(mac_MD, axis=1)
#
#     MD_str = []
#     MD_freq = []
#     MD_damp = []
#
#     for i in range(3):
#         for j in range(12):
#             if mac_MD[i,j]==sp.max(mac_MD[:,idx_srt[j]]):
#                 MD_str.append(MD[:, idx_srt[j]])
#                 MD_freq.append(sp.absolute(evals[idx_srt[j]]))
#                 MD_damp.append(-1.*evals.real[idx_srt[j]]/sp.absolute(evals[idx_srt[j]]))
#                 break
#
#     MD_str = sp.array(MD_str)
#     MD_freq = sp.array(MD_freq)
#     MD_damp = sp.array(MD_damp)
#
#     return MD_str, MD_freq, MD_damp

def mac(MD1, MD2):
    """
    Calcuate the Modal Assurance Criterion (MAC) for the set of input mmodes Q1 and Q2
    :param MD1: first array containing modes shapes as columns (must be of equal dimensions as Q2)
    :param MD2: second array containing modes shapes as columns (must be of equal dimensions as Q1)
    :return: MAC matrix
    """

    # todo: this requires some debugging There is a mistake somewhere tere. Q1.transpose seems to solve the probelm -> why?

    nrm_MD1 = ln.norm(MD1, axis=0)
    nrm_MD2 = ln.norm(MD2, axis=0)

    nrm_MD1 = nrm_MD1.reshape(len(nrm_MD1), 1) # <- reshape to be able to use broadcasting when calculating the matrix of norm products
    nrm_prod = nrm_MD1*nrm_MD2

    MAC = (MD1.transpose().dot(MD2)/nrm_prod)**2

    return MAC

