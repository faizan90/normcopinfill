# cython: nonecheck=False, boundscheck=False, wraparound=False, cdivision=True
from __future__ import division
import random
import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel

ctypedef double DT_D
ctypedef unsigned long DT_UL


cdef extern from 'math.h' nogil:
    cdef DT_D exp(DT_D x)
    cdef DT_D log(DT_D x)
    cdef DT_D M_PI
    cdef DT_D INFINITY


cdef extern from "c_ftns_norm_cop.h" nogil:
    cdef:
        DT_D get_demr_c(const DT_D *x_arr,
                        const DT_D *mean_ref,
                        const DT_UL *size,
                        const DT_UL *off_idx)

        DT_D get_ln_demr_c(const DT_D *x_arr,
                           const DT_D *ln_mean_ref,
                           const DT_UL *size,
                           const DT_UL *off_idx)

        DT_D get_mean_c(const DT_D *in_arr,
                        const DT_UL *size,
                        const DT_UL *off_idx)

        DT_D get_ln_mean_c(const DT_D *in_arr,
                           const DT_UL *size,
                           const DT_UL *off_idx)

        DT_D get_var_c(const DT_D *in_arr_mean,
                       const DT_D *in_arr,
                       const DT_UL *size,
                       const DT_UL *off_idx)

        DT_D get_ns_c(const DT_D *x_arr,
                      DT_D *y_arr,
                      const DT_UL *size,
                      const DT_D *demr,
                      const DT_UL *off_idx)

        DT_D get_ln_ns_c(const DT_D *x_arr,
                         DT_D *y_arr,
                         const DT_UL *size,
                         const DT_D *ln_demr,
                         const DT_UL *off_idx)

        DT_D get_kge_c(const DT_D *act_arr,
                       const DT_D *sim_arr,
                       const DT_D *act_mean,
                       const DT_D *act_std_dev,
                       const DT_UL *size,
                       const DT_UL *off_idx)

cpdef DT_D get_dist(DT_D x1, DT_D y1, DT_D x2, DT_D y2):
    """Get distance between polongs
    """
    cdef DT_D dist
    dist = ((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))**0.5
    return dist


cdef inline DT_D get_mean(DT_D[:] in_arr) nogil:
    cdef:
        DT_D _sum = 0
        long _n = in_arr.shape[0], i = 0

    for i in xrange(_n):
        _sum += in_arr[i]
    return _sum / _n


cdef inline DT_D get_variance(DT_D in_arr_mean,
                              DT_D[:] in_arr) nogil:
    cdef:
        DT_D _sum = 0
        long i, _n = in_arr.shape[0]

    for i in xrange(_n):
        _sum += (in_arr[i] - in_arr_mean)**2
    return _sum / (_n)


cdef inline DT_D get_covar(DT_D in_arr_1_mean,
                           DT_D in_arr_2_mean,
                           DT_D[:] in_arr_1,
                           DT_D[:] in_arr_2) nogil:
    cdef:
        DT_D _sum = 0
        long i = 0, _n = in_arr_1.shape[0]

    for i in xrange(_n):
        _sum += (in_arr_1[i] - in_arr_1_mean) * (in_arr_2[i] - in_arr_2_mean)
    return _sum / _n


cdef inline DT_D get_correl(DT_D in_arr_1_std_dev,
                            DT_D in_arr_2_std_dev,
                            DT_D arrs_covar) nogil:
    return arrs_covar / (in_arr_1_std_dev * in_arr_2_std_dev)


cpdef DT_D get_corrcoeff(DT_D[:] act_arr,
                         DT_D[:] sim_arr):

    cdef:
        DT_D act_mean, sim_mean, act_std_dev
        DT_D sim_std_dev, covar, correl

    act_mean = get_mean(act_arr)
    sim_mean = get_mean(sim_arr)

    act_std_dev = get_variance(act_mean, act_arr)**0.5
    sim_std_dev = get_variance(sim_mean, sim_arr)**0.5

    covar = get_covar(act_mean, sim_mean, act_arr, sim_arr)
    correl = get_correl(act_std_dev, sim_std_dev, covar)
    return correl


cpdef fill_correl_mat(DT_D[:, :] vals_arr):
    cdef:
        DT_UL i, j
        DT_UL shape = vals_arr.shape[1]
        np.ndarray[DT_D, ndim=2] corrs_arr = np.zeros((shape, shape))

    for i in xrange(shape):
        for j in xrange(shape):
            if i > j:
                corrs_arr[i, j] = get_corrcoeff(vals_arr[:, i], vals_arr[:, j])
            elif i == j:
                corrs_arr[i, j] = 1.0

    for i in range(shape):
        for j in range(shape):
            if i < j:
                corrs_arr[i, j] = corrs_arr[j, i]

    return corrs_arr


cpdef DT_D norm_cdf_py(DT_D z, DT_D mu=0.0, DT_D sig=1.0):
    cdef:
        DT_D t, q, p

    z = (z - mu) / sig
    z = z / (2**0.5)

    if z < 0:
        t = 1. / (1. + 0.5 * (-1. * z))
    else:
        t = 1. / (1. + 0.5 * z)

    q = -z**2
    q -= 1.26551223
    q += 1.00002368 * t
    q += 0.37409196 * t**2
    q += 0.09678418 * t**3
    q -= 0.18628806 * t**4
    q += 0.27886807 * t**5
    q -= 1.13520398 * t**6
    q += 1.48851587 * t**7
    q -= 0.82215223 * t**8
    q += 0.17087277 * t**9
    q = exp(q)
    q = q * t

    if z >= 0:
        p = 1 - q
    else:
        p = q - 1

    return 0.5 * (1 + p)


cpdef norm_cdf_py_arr(DT_D[:] z, DT_D mu=0.0, DT_D sig=1.0):
    cdef:
        DT_UL i, n = z.shape[0]
        np.ndarray[DT_D, ndim=1] cdf_arr = np.zeros((n), dtype=np.float64)

    for i in xrange(n):
        cdf_arr[i] = norm_cdf_py(z[i], mu=mu, sig=sig)
    return cdf_arr


cpdef DT_D norm_ppf_py(DT_D p, DT_D mu=0.0, DT_D sig=1.0):
    cdef:
        DT_D t, z

    if p == 0.0:
        return -INFINITY
    elif p == 1.0:
        return INFINITY

    if p > 0.5:
        t = (-2.0 * log(1 - p))**0.5
    else:
        t = (-2.0 * log(p))**0.5

    z = -0.322232431088 + t * (-1.0 + t * (-0.342242088547 + t * \
        (-0.020423120245 + t * -0.453642210148e-4)))
    z = z / (0.0993484626060 + t * (0.588581570495 + t * \
        (0.531103462366 + t * (0.103537752850 + t * 0.3856070063e-2))))
    z = z + t

    z = (sig * z) + mu
    if p < 0.5:
        return -z
    else:
        return z


cpdef norm_ppf_py_arr(DT_D[:] p, DT_D mu=0.0, DT_D sig=1.0):
    cdef:
        DT_UL i, n = p.shape[0]
        np.ndarray[DT_D, ndim=1] ppf_arr = np.empty((n), dtype=np.float64)

    for i in xrange(n):
        ppf_arr[i] = norm_ppf_py(p[i], mu=mu, sig=sig)
    return ppf_arr


cpdef DT_D norm_pdf_py(DT_D z, DT_D mu=0.0, DT_D sig=1.0):
    cdef:
        DT_D d

    z = (z - mu) / sig
    d = -0.5 * z**2
    d = exp(d)
    d = d / (2 * M_PI)**0.5
    return d


cpdef norm_pdf_py_arr(DT_D[:] z, DT_D mu=0.0, DT_D sig=1.0):
    cdef:
        DT_UL i, n = z.shape[0]
        np.ndarray[DT_D, ndim=1] pdf_arr = np.zeros((n), dtype=np.float64)

    for i in xrange(n):
        pdf_arr[i] = norm_pdf_py(z[i], mu=mu, sig=sig)
    return pdf_arr


cpdef dict bi_var_copula(DT_D[:] x_probs,
                         DT_D[:] y_probs,
                         DT_UL cop_bins):
    '''get the bivariate empirical copula
    '''

    cdef:
        Py_ssize_t i, j

        DT_UL tot_pts = x_probs.shape[0]

        DT_D div_cnst
        DT_D u1, u2
        DT_UL i_row, j_col

        np.ndarray[DT_UL, ndim=2] emp_freqs_arr = \
            np.zeros((cop_bins, cop_bins), dtype=np.uint32)
        np.ndarray[DT_D, ndim=2] emp_dens_arr = \
            np.zeros((cop_bins, cop_bins), dtype=np.float64)

    for i in xrange(tot_pts):
        u1 = x_probs[i]
        u2 = y_probs[i]

        i_row = <DT_UL> (u1 * cop_bins)
        j_col = <DT_UL> (u2 * cop_bins)

        emp_freqs_arr[i_row, j_col] += 1

    div_cnst = cop_bins**2 / float(tot_pts)
    for i in xrange(cop_bins):
        for j in xrange(cop_bins):
            emp_dens_arr[i, j] = emp_freqs_arr[i, j] * div_cnst

    return {'emp_freqs_arr': emp_freqs_arr,
            'emp_dens_arr': emp_dens_arr}


cpdef dict bi_var_copula_slim(DT_D[:] x_probs,
                              DT_D[:] y_probs,
                              DT_UL cop_bins):
    '''get the bivariate empirical copula
    '''

    cdef:
        Py_ssize_t i, j

        DT_UL tot_pts = x_probs.shape[0], tot_sum

        DT_D div_cnst
        DT_D u1, u2
        DT_UL i_row, j_col

        np.ndarray[DT_D, ndim=2] emp_dens_arr = \
            np.zeros((cop_bins, cop_bins), dtype=np.float64)

    tot_sum = 0
    for i in xrange(tot_pts):
        u1 = x_probs[i]
        u2 = y_probs[i]

        i_row = <DT_UL> (u1 * cop_bins)
        j_col = <DT_UL> (u2 * cop_bins)

        emp_dens_arr[i_row, j_col] += 1
        tot_sum += 1

    assert tot_pts == tot_sum, 'Error!'
    div_cnst = cop_bins**2 / float(tot_pts)
    for i in xrange(cop_bins):
        for j in xrange(cop_bins):
            emp_dens_arr[i, j] = emp_dens_arr[i, j] * div_cnst

    return {'emp_dens_arr': emp_dens_arr}


cdef DT_D bivar_gau_cop(DT_D t1, DT_D t2, DT_D rho) nogil:
    cdef:
        DT_D cop_dens
    cop_dens = exp(-0.5 * (rho / (1 - rho**2)) * \
                   ((rho*(t1**2 + t2**2)) - 2*t1*t2))
    cop_dens /= (1 - rho**2)**0.5
    return cop_dens


cpdef bivar_gau_cop_arr(DT_D rho, DT_UL cop_bins):
    cdef:
        DT_UL i, j
        DT_D p_i, p_j, z_i, z_j
        np.ndarray[DT_D, ndim=2] gau_cop_arr = \
            np.zeros((cop_bins, cop_bins), dtype=np.float64)

    for i in xrange(cop_bins):
        p_i = <DT_D> ((i + 1) / float(cop_bins + 1))
        z_i = norm_ppf_py(p_i)
        for j in xrange(cop_bins):
            p_j = <DT_D> ((j + 1) / float(cop_bins + 1))
            z_j = norm_ppf_py(p_j)
            gau_cop_arr[i, j] = bivar_gau_cop(z_i, z_j, rho)
    return gau_cop_arr


cpdef dict get_rho_tau_from_bivar_emp_dens(DT_D[:] u, DT_D[:, :] emp_dens_arr):
    '''
    Given u, v and frequencies for each u and v, get rho, tau, cummulative
        and Dcummulative values in a dict
    '''
    cdef:
        DT_UL i, j, k, l, rows_cols
        DT_D rho, tau, du, cum_emp_dens, Dcum_emp_dens
        DT_D asymm_1, asymm_2, ui, uj, dudu

    rows_cols = u.shape[0]
    du = 1.0 / <DT_D> rows_cols
    dudu = du**2
    rho = 0.0
    tau = 0.0
    asymm_1 = 0.0
    asymm_2 = 0.0

    cdef:
        np.ndarray[DT_D, ndim=2] cum_emp_dens_arr = \
            np.zeros((rows_cols, rows_cols))

    for i in xrange(rows_cols):
        for j in xrange(rows_cols):
            Dcum_emp_dens = emp_dens_arr[i, j] * dudu

            ui = u[i]
            uj = u[j]

            cum_emp_dens = 0.0
            for k in xrange(i + 1):
                for l in xrange(j + 1):
                    cum_emp_dens += emp_dens_arr[k, l]
            cum_emp_dens_arr[i, j] += cum_emp_dens * dudu

            rho += (cum_emp_dens_arr[i, j] - (u[i] * u[j])) * dudu
            tau += cum_emp_dens * Dcum_emp_dens * dudu

            # the old one
#            asymm_1 += ((ui - 0.5) * (uj - 0.5) * (ui + uj - 1) * Dcum_emp_dens)
#            asymm_2 += (-(ui - 0.5) * (uj - 0.5) * (ui - uj) * Dcum_emp_dens)

            # the new one
            asymm_1 += ((ui + uj - 1)**3) * emp_dens_arr[i, j] * dudu
            asymm_2 += ((ui - uj)**3) * emp_dens_arr[i, j] * dudu

    rho = 12.0 * rho
    tau = 4.0 * tau - 1.0

    return {'rho': rho, 'tau': tau, 'asymm_1': asymm_1, 'asymm_2': asymm_2,
            'cumm_dens': cum_emp_dens_arr}


cpdef dict get_cumm_dist_from_bivar_emp_dens(DT_D[:, :] emp_dens_arr):
    '''
    Given u and frequencies for each u, get cummulative
        values dist in a dict

        The Faster Version
    '''
    cdef:
        Py_ssize_t i, j
        DT_UL rows_cols
        DT_D du, dudu
        long double cum_emp_dens

    rows_cols = emp_dens_arr.shape[0]
    du = 1.0 / <DT_D> rows_cols
    dudu = du**2

    cdef:
        np.ndarray[DT_D, ndim=2] cum_emp_dens_arr = \
            np.zeros((rows_cols + 1, rows_cols + 1), dtype=np.float64)

    for i in xrange(1, rows_cols + 1):
        for j in xrange(1, rows_cols + 1):
            cum_emp_dens = 0.0
            cum_emp_dens = cum_emp_dens_arr[i - 1, j]
            cum_emp_dens += cum_emp_dens_arr[i, j - 1]
            cum_emp_dens -= cum_emp_dens_arr[i - 1, j - 1]

            cum_emp_dens = cum_emp_dens / dudu
            cum_emp_dens += emp_dens_arr[i - 1, j - 1]

            cum_emp_dens = cum_emp_dens * dudu
            cum_emp_dens_arr[i, j] = <DT_D> cum_emp_dens

    return {'cumm_dens': cum_emp_dens_arr[1:, 1:]}


cpdef DT_D tau_sample(DT_D[:] ranks_u, DT_D[:] ranks_v):
    '''Calculate tau_b
    '''
    cdef:
        DT_UL i, j
        DT_UL tie_u = 0, tie_v = 0
        DT_UL n_vals = ranks_u.shape[0]
        DT_D crd, drd, tau, crd_drd, diff_u, diff_v
        DT_D nan = np.nan

    crd = 0.0
    drd = 0.0

    for i in xrange(n_vals):
        for j in xrange(n_vals):
            if i > j:
                diff_u = (ranks_u[i] - ranks_u[j])
                diff_v = (ranks_v[i] - ranks_v[j])
                if diff_u == 0:
                    tie_u += 1
                if diff_v == 0:
                    tie_v += 1

                if (diff_u == 0) and (diff_v == 0):
                    tie_u -= 1
                    tie_v -= 1

                crd_drd = diff_u * diff_v

                if crd_drd > 0:
                    crd += 1
                elif crd_drd < 0:
                    drd += 1

    return (crd - drd) / ((crd + drd + tie_u) * (crd + drd + tie_v))**0.5


cpdef dict get_asymms_sample(DT_D[:] u, DT_D[:] v):
    cdef:
        DT_UL i, n_vals
        DT_D asymm_1, asymm_2

    n_vals = u.shape[0]

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in xrange(n_vals):
        asymm_1 += (u[i] + v[i] - 1)**3
        asymm_2 += (u[i] - v[i])**3

    asymm_1 = asymm_1 / n_vals
    asymm_2 = asymm_2 / n_vals

    return {'asymm_1':asymm_1, 'asymm_2':asymm_2}


cpdef dict get_asymms_population(DT_D[:] u, DT_D[:, :] emp_dens_arr):
    cdef:
        DT_UL i, j, n_vals
        DT_D asymm_1, asymm_2, dudu

    n_vals = u.shape[0]
    dudu = (u[1] - u[0])**2

    asymm_1 = 0.0
    asymm_2 = 0.0

    for i in xrange(n_vals):
        for j in xrange(n_vals):
            asymm_1 += ((u[i] + u[j] - 1)**3) * emp_dens_arr[i, j] * dudu
            asymm_2 += ((u[i] - u[j])**3) * emp_dens_arr[i, j] * dudu

    return {'asymm_1':asymm_1, 'asymm_2':asymm_2}


cpdef get_cond_cumm_probs(DT_D[:] u, DT_D[:, :] emp_dens_arr):
    cdef:
        DT_UL rows
        DT_D du

    rows = u.shape[0]
    du = u[1] - u[0]

    cdef np.ndarray[DT_D, ndim=2] cond_probs = np.zeros((2, rows), dtype=np.float64)
    # cond_probs: zero along the rows, 1 along the columns

    for i in xrange(rows):
        for j in xrange(rows):
            cond_probs[0, i] += emp_dens_arr[i, j]
        cond_probs[0, i] = cond_probs[0, i] * du

    for i in xrange(rows):
        for j in xrange(rows):
            cond_probs[1, i] += emp_dens_arr[j, i]
        cond_probs[1, i] = cond_probs[1, i] * du

    return cond_probs


cpdef DT_D get_ns_py(np.ndarray[DT_D, ndim=1, mode='c'] x_arr,
                       np.ndarray[DT_D, ndim=1, mode='c'] y_arr,
                       const DT_UL off_idx):
    cdef:
        DT_UL size = x_arr.shape[0]
        DT_D mean_ref, demr

    mean_ref = get_mean_c(&x_arr[0], &size, &off_idx)
    demr = get_demr_c(&x_arr[0], &mean_ref, &size, &off_idx)
    return get_ns_c(&x_arr[0], &y_arr[0], &size, &demr, &off_idx)


cpdef DT_D get_ln_ns_py(np.ndarray[DT_D, ndim=1, mode='c'] x_arr,
                          np.ndarray[DT_D, ndim=1, mode='c'] y_arr,
                          const DT_UL off_idx):
    cdef:
        DT_UL size = x_arr.shape[0]
        DT_D ln_mean_ref, ln_demr

    ln_mean_ref = get_ln_mean_c(&x_arr[0], &size, &off_idx)
    ln_demr = get_ln_demr_c(&x_arr[0], &ln_mean_ref, &size, &off_idx)
    return get_ln_ns_c(&x_arr[0], &y_arr[0], &size, &ln_demr, &off_idx)


cpdef DT_D get_kge_py(np.ndarray[DT_D, ndim=1, mode='c'] x_arr,
                        np.ndarray[DT_D, ndim=1, mode='c'] y_arr,
                        const DT_UL off_idx):
    cdef:
        DT_UL size = x_arr.shape[0]
        DT_D mean_ref, act_std_dev

    mean_ref = get_mean_c(&x_arr[0], &size, &off_idx)
    act_std_dev = get_var_c(&mean_ref, &x_arr[0], &size, &off_idx)**0.5
    return get_kge_c(&x_arr[0], &y_arr[0], &mean_ref, &act_std_dev, &size,
                     &off_idx)
