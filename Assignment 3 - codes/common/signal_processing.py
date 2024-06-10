from __future__ import division
import scipy as sp
import scipy.signal as sg
import scipy.linalg as ln
import scipy.interpolate as interp

import sklearn as skl
import sklearn.cluster as cluster
import sklearn.preprocessing as preprocessing


def get_amp_freq(t, x):
    t = sp.array(t)
    x = sp.array(x)

    # find peaks (Max -> peaks and Min -> valleys):
    idx_max, _ = sg.find_peaks(x)
    idx_min, _ = sg.find_peaks(-1*x)

    t_max = sp.zeros(idx_max.shape)
    amp_max = sp.zeros(idx_max.shape)

    t_min = sp.zeros(idx_min.shape)
    amp_min = sp.zeros(idx_min.shape)

    # find amplitudes by interpolation:
    # Max:
    i = 0
    for idx in idx_max:
        if idx != 0 and idx < len(x) - 1:
            # assmeble the interpolation matrix:
            A = sp.array([[1, t[idx - 1], t[idx - 1]**2],
                          [1, t[idx], t[idx]**2],
                          [1, t[idx + 1], t[idx + 1]**2]])  # modified such that a2 -> a[2], a1 -> a[1], a0 -> a[0]
            a = ln.inv(A).dot([x[idx - 1], x[idx], x[idx + 1]])

            t_max_i = - a[1]/(2.*a[2])  # a2 -> a[2], a1 -> a[1], a0 -> a[0]
            amp_max_i = -a[1]**2/(4.*a[2]) + a[0]
        else:
            t_max_i = t[idx]
            amp_max_i = x[idx]

        t_max[i] = t_max_i
        amp_max[i] = amp_max_i
        i += 1

    # Min:
    i = 0
    for idx in idx_min:
        if idx != 0 and idx < len(x) - 1:
            # assmeble the interpolation matrix:
            A = sp.array([[1, t[idx - 1], t[idx - 1]**2],
                          [1, t[idx], t[idx]**2],
                          [1, t[idx + 1], t[idx + 1]**2]])  # modified such that a2 -> a[2], a1 -> a[1], a0 -> a[0]
            a = ln.inv(A).dot([x[idx - 1], x[idx], x[idx + 1]])

            t_min_i = - a[1]/(2.*a[2])  # a2 -> a[2], a1 -> a[1], a0 -> a[0]
            amp_min_i = -a[1]**2/(4.*a[2]) + a[0]
        else:
            t_min_i = t[idx]
            amp_min_i = x[idx]

        t_min[i] = t_min_i
        amp_min[i] = amp_min_i
        i += 1

    # instantaneous Frequencies:
    t_omega_max = t_max[:-1] + 0.5*sp.diff(t_max)
    omega_max = 2*sp.pi/sp.diff(t_max)

    t_omega_min = t_min[:-1] + 0.5*sp.diff(t_min)
    omega_min = 2*sp.pi/sp.diff(t_min)

    # Postprocess:
    # ============
    # t_amp = sp.append(t_max, t_min)
    # amp = sp.append(amp_max, sp.absolute(amp_min))
    # id_sort = sp.argsort(t_amp)
    # t_amp = t_amp[id_sort]
    # amp = amp[id_sort]

    t_amp = t_max
    amp = sp.zeros(t_max.shape)
    mean = sp.zeros(t_max.shape)

    # Interpolate the min values to the time of the max values -> improve accuracy
    f_min = interp.interp1d(t_min, amp_min, fill_value='extrapolate')

    for i_max, t_max_i in enumerate(t_max):
        amp[i_max] = (amp_max[i_max] - f_min(t_max_i))/2.
        mean[i_max] = (amp_max[i_max] + f_min(t_max_i))/2.

    # t_amp = sp.zeros(t_max.shape)
    # amp = sp.zeros(t_max.shape)
    # mean = sp.zeros(t_max.shape)
    #
    # for i_max, t_max_i in enumerate(t_max):
    #     i_min = sp.absolute(t_min - t_max_i).argmin()
    #     amp[i_max] = (amp_max[i_max] - amp_min[i_min])/2.
    #     mean[i_max] = (amp_max[i_max] + amp_min[i_min])/2.
    #     t_amp[i_max] = (t_max_i + t_min[i_min])/2.

    t_omega = sp.append(t_omega_max, t_omega_min)
    omega = sp.append(omega_max, omega_min)
    id_sort = sp.argsort(t_omega)
    t_omega = t_omega[id_sort]
    omega = omega[id_sort]

    # interpolate to original times:
    f_amp = interp.interp1d(t_amp, amp, kind='cubic', fill_value='extrapolate')
    amp = f_amp(t)

    f_mean = interp.interp1d(t_amp, mean, kind='cubic', fill_value='extrapolate')
    mean = f_mean(t)

    f_omega = interp.interp1d(t_omega, omega, kind='cubic', fill_value='extrapolate')
    omega = f_omega(t)

    return amp, mean, omega


def get_amp_freq_minmax(t, t_min, x_min, t_max, x_max):
    # Instantaneous Frequencies:
    t_omega_max = t_max[:-1] + 0.5*sp.diff(t_max)
    omega_max = 2*sp.pi/sp.diff(t_max)

    t_omega_min = t_min[:-1] + 0.5*sp.diff(t_min)
    omega_min = 2*sp.pi/sp.diff(t_min)

    t_omega = sp.append(t_omega_max, t_omega_min)
    omega = sp.append(omega_max, omega_min)
    id_sort = sp.argsort(t_omega)
    t_omega = t_omega[id_sort]
    omega = omega[id_sort]

    # Instantaneous Mean and Amplitude:
    amp = sp.zeros(t_max.shape)
    mean = sp.zeros(t_max.shape)

    # Interpolate the min values to the time of the max values -> improve accuracy
    f_min = interp.interp1d(t_min, x_min, kind='cubic', fill_value='extrapolate')

    for i_max, t_max_i in enumerate(t_max):
        amp[i_max] = (x_max[i_max] - f_min(t_max_i))/2.
        mean[i_max] = (x_max[i_max] + f_min(t_max_i))/2.

    # interpolate to original times:
    f_amp = interp.interp1d(t_max, amp, kind='cubic', fill_value='extrapolate')
    amp = f_amp(t)

    f_mean = interp.interp1d(t_max, mean, kind='cubic', fill_value='extrapolate')
    mean = f_mean(t)

    f_omega = interp.interp1d(t_omega, omega, kind='cubic', fill_value='extrapolate')
    omega = f_omega(t)

    return amp, mean, omega


def get_amp_freq_minmax_simple(t_min, x_min, t_max, x_max):
    # Instantaneous Frequencies:
    t_omega_max = t_max[:-1] + 0.5*sp.diff(t_max)
    omega_max = 2*sp.pi/sp.diff(t_max)

    t_omega_min = t_min[:-1] + 0.5*sp.diff(t_min)
    omega_min = 2*sp.pi/sp.diff(t_min)

    t_omega = sp.append(t_omega_max, t_omega_min)
    omega = sp.append(omega_max, omega_min)
    id_sort = sp.argsort(t_omega)
    t_omega = t_omega[id_sort]
    omega = omega[id_sort]

    # Instantaneous Mean and Amplitude:
    t_amp = sp.zeros(t_max.shape)
    amp = sp.zeros(t_max.shape)

    t_mean = sp.zeros(t_max.shape)
    mean = sp.zeros(t_max.shape)

    for i_max, t_max_i in enumerate(t_max):
        i_min = sp.absolute(t_min - t_max_i).argmin()

        t_amp[i_max] = (t_max_i + t_min[i_min])/2.
        amp[i_max] = (x_max[i_max] - x_min[i_min])/2.

        t_mean[i_max] = (t_max_i + t_min[i_min])/2.
        mean[i_max] = (x_max[i_max] + x_min[i_min])/2.

    return t_amp, amp, t_mean, mean, t_omega, omega


def resample(t, x, dt_resample):
    # Make sure you are dealing with numpy arrays:
    t = sp.array(t)
    x = sp.array(x)

    t_min = t.min()
    t_max = t.max()

    t_res = sp.arange(t_min, t_max, dt_resample)

    f_res = interp.interp1d(t, x, kind='cubic', fill_value='extrapolate')
    x_res = f_res(t_res)

    return t_res, x_res


def cluster_amplitudes(amp, eps=0.5):
    """
    cluster the amplitude points into clusters and calculates mean values of each cluster
    :param amp:
    :param eps:
    :return:
    """
    amp_rshp = amp.reshape(-1, 1)

    # center and rescale the input amplitudes:
    amp_scl = preprocessing.StandardScaler().fit_transform(amp_rshp)

    # cluster the selected amplitudes (use DBSCAN algorithm):
    db = cluster.DBSCAN(eps=eps).fit(amp_scl)
    lbls = db.labels_
    lbls_unique = set(lbls)

    # Number of clusters in labels, ignoring noise if present.
    n_class = len(lbls_unique) - (1 if -1 in lbls else 0)

    amp_class_mean = sp.zeros((n_class,))
    for j, lbl_j in enumerate(lbls_unique):
        if lbl_j != -1:
            mask_class_member = (lbls == lbl_j)
            amp_class_mean[j] = sp.mean(amp[mask_class_member])

    return n_class, amp_class_mean


def cluster_stable_segments(stable, eps=0.1):
    """
    Find stable and unstable segments of a data series (e.g. LCO as a function of velocity) and cluster them together
    to ease plotting. The only required input is a boolean vector assigning a state (stable or unstable) to each data point.

    The function is based on the DBSCAN clustering algorithm from Scikit learn library.

    :param stable: boleand vector of states (stable = True, unstable = False) for each datapoint.
    :param eps: max. distance between points belonging to the same cluster.
    :return: two lists of clusters, for stable and unstable segments respectiely.
    """

    # Create mask for stable and unstable segments. -> find submasks for each segments:
    # =================================================================================
    m_stbl = sp.array((stable==True))

    # Create index column:
    # =====================
    idx = sp.array(range(len(stable)))

    idx_stbl = idx[m_stbl]
    idx_ustbl= idx[~m_stbl]


    # center and rescale the masks used for plotting -> impove clustering procedure:
    # ==============================================================================
    idx_stbl_rshp = sp.float64(idx_stbl.reshape((-1,1)))
    idx_ustbl_rshp = sp.float64(idx_ustbl.reshape((-1, 1)))
    idx_stbl_scl = preprocessing.StandardScaler().fit_transform(idx_stbl_rshp)
    idx_ustbl_scl = preprocessing.StandardScaler().fit_transform(idx_ustbl_rshp)

    # cluster the selected masks (use DBSCAN algorithm):
    # =================================================
    db_stbl = cluster.DBSCAN(eps=eps, min_samples=1).fit(idx_stbl_scl)
    db_ustbl = cluster.DBSCAN(eps=eps, min_samples=1).fit(idx_ustbl_scl)
    l_stbl = db_stbl.labels_
    l_ustbl = db_ustbl.labels_

    l_stbl_unique = set(l_stbl)
    l_ustbl_unique = set(l_ustbl)

    # Pack submasks into arrrays:
    # ===========================
    lst_idx_stbl = []
    for lbl in l_stbl_unique:
        mask = (l_stbl == lbl)
        lst_idx_stbl.append(idx_stbl[mask])

    lst_idx_ustbl = []
    for lbl in l_ustbl_unique:
        mask = (l_ustbl == lbl)
        lst_idx_ustbl.append(idx_ustbl[mask])

    return lst_idx_stbl, lst_idx_ustbl