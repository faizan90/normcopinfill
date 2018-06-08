# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from numpy import (array,
                   append,
                   linspace,
                   nan,
                   full,
                   divide,
                   isnan,
                   where,
                   logical_and)
from scipy.interpolate import interp1d

from ..misc.misc_ftns import as_err

import pyximport
pyximport.install()
from normcop_cyftns import norm_ppf_py, norm_cdf_py, norm_pdf_py


class DiscContVars:

    def __init__(self, infill_steps_obj):
        vars_list = ['in_var_df',
                     'var_le_trs',
                     'var_ge_trs',
                     'ge_le_trs_n',
                     'save_step_vars_flag',
                     'dont_stop_flag',
                     'adj_prob_bounds',
                     'n_discret']

        for _var in vars_list:
            setattr(self, _var, getattr(infill_steps_obj, _var))
        return

    def get_disc_vars(self,
                      curr_max_var_val,
                      val_cdf_ftn,
                      curr_min_var_val,
                      mu_t,
                      sig_sq_t,
                      py_del,
                      infill_date,
                      curr_var_df,
                      py_zero,
                      step_vars_dict):

        if curr_max_var_val > self.var_ge_trs:
            val_arr = val_cdf_ftn.x[val_cdf_ftn.x >= self.var_ge_trs]
            val_arr = append(linspace(self.var_le_trs,
                                      self.var_ge_trs,
                                      self.ge_le_trs_n,
                                      endpoint=False),
                             val_arr)
        else:
            val_arr = linspace(curr_min_var_val,
                               curr_max_var_val,
                               self.ge_le_trs_n)

        assert val_arr.shape[0], as_err('\'val_arr\' is empty!')

        gy_arr = full(val_arr.shape, nan)

        for i, val in enumerate(val_arr):
            if val > self.var_ge_trs:
                _ = norm_ppf_py(val_cdf_ftn(val)) - mu_t
                gy_arr[i] = norm_cdf_py(divide(_, sig_sq_t ** 0.5))
            elif (val > self.var_le_trs) and (val <= self.var_ge_trs):
                _ = norm_ppf_py(py_del) - mu_t
                gy_arr[i] = norm_cdf_py(divide(_, sig_sq_t ** 0.5))
            else:
                values_arr = (
                    self.in_var_df.loc[infill_date,
                                       curr_var_df.columns[1:]
                                       ].dropna().values)

                if len(values_arr) > 0:
                    n_wet = (values_arr > self.var_le_trs).sum()
                    wt = divide(n_wet, float(values_arr.shape[0]))
                else:
                    wt = 0.0

                if py_zero:
                    _ = norm_ppf_py(py_zero * (1.0 + wt)) - mu_t
                    gy_arr[i] = norm_cdf_py(divide(_, sig_sq_t ** 0.5))
                else:
                    gy_arr[i] = 0.0

            assert not isnan(gy_arr[i]), as_err(
                '\'gy\' is nan (val: %0.2e)!' % val)

        if self.save_step_vars_flag:
            step_vars_dict['gy_arr_raw'] = gy_arr
            step_vars_dict['val_arr_raw'] = val_arr

        probs_idxs = gy_arr > self.adj_prob_bounds[0]
        probs_idxs = logical_and(probs_idxs,
                                 gy_arr < self.adj_prob_bounds[1])
        gy_arr = gy_arr[probs_idxs]
        val_arr = val_arr[probs_idxs]

        assert val_arr.shape[0] > 0, as_err(
            ('\'val_arr\' has less than 1 elements! '
             'min_prob: %f, max_prob: %f, val_arr: %s, date: %s') %
            (gy_arr.min(), gy_arr.max(), str(val_arr), str(infill_date)))

        assert gy_arr.shape[0] > 0, as_err(
            'Increase discretization!')
        assert gy_arr.shape[0] == val_arr.shape[0], as_err(
            'Unequal shapes of probs and vals!')

        if self.save_step_vars_flag:
            step_vars_dict['gy_arr_fin'] = gy_arr
            step_vars_dict['val_arr_fin'] = val_arr

        if gy_arr.shape[0] <= 1:
            # all probs are zero or one, hope so
            # FIXME: this might be a bug

            if gy_arr.max() > self.adj_prob_bounds[1]:
                interp_val = val_arr.max()
                gy_val = gy_arr[-1]
            else:
                interp_val = self.var_le_trs
                gy_val = gy_arr[0]

            fin_val_ppf_ftn_adj = interp1d(linspace(0, 1.0, 10),
                                           [interp_val] * 10,
                                           bounds_error=False,
                                           fill_value=(interp_val,
                                                       interp_val))
            fin_val_grad_ftn_adj = interp1d((curr_min_var_val,
                                             curr_max_var_val),
                                            (0, 0),
                                            bounds_error=False,
                                            fill_value=(0, 0))
            gy_arr_adj = array([gy_val, gy_val])
            val_arr_adj = array([interp_val, interp_val])
            pdf_arr_adj = array([0.0, 0.0])

        else:
#            if val_arr.shape[0] == 0:
#                as_err(('CRITICAL: \'val_arr\' has less than 2 elements! '
#                        'min_prob: %f, max_prob: %f, val_arr: %s') %
#                       (gy_arr.min(), gy_arr.max(), str(val_arr)))
#                if self.dont_stop_flag:
#                    return [None] * 5
#                else:
#                    raise Exception(('val_arr is invalid %s!' %
#                                     str(val_arr)))

            fin_val_ppf_ftn = interp1d(gy_arr, val_arr,
                                       bounds_error=False,
                                       fill_value=(self.var_le_trs,
                                                   curr_max_var_val))

            curr_min_var_val_adj, curr_max_var_val_adj = (
                fin_val_ppf_ftn([self.adj_prob_bounds[0],
                                 self.adj_prob_bounds[1]]))

            # do the interpolation again with adjusted bounds
            if curr_max_var_val_adj > self.var_ge_trs:
                adj_val_idxs = logical_and(val_arr >= self.var_ge_trs,
                                           val_arr <= curr_max_var_val_adj)
                val_arr_adj = val_arr[adj_val_idxs]

                if val_arr_adj.shape[0] < self.n_discret:
                    val_adj_interp = interp1d(list(range(0,
                                                   val_arr_adj.shape[0])),
                                              val_arr_adj)
                    _interp_vals = linspace(0.0,
                                            val_arr_adj.shape[0] - 1,
                                            self.n_discret)
                    val_arr_adj = val_adj_interp(_interp_vals)

                val_arr_adj = append(linspace(self.var_le_trs,
                                              self.var_ge_trs,
                                              self.ge_le_trs_n,
                                              endpoint=False),
                                     val_arr_adj)
            else:
                val_arr_adj = linspace(curr_min_var_val_adj,
                                       curr_max_var_val_adj,
                                       self.ge_le_trs_n)

            gy_arr_adj = full(val_arr_adj.shape, nan)
            pdf_arr_adj = gy_arr_adj.copy()

            for i, val_adj in enumerate(val_arr_adj):
                if val_adj > self.var_ge_trs:
                    _ = (norm_ppf_py(val_cdf_ftn(val_adj)) - mu_t)
                    z_scor = divide(_, sig_sq_t ** 0.5)
                    gy_arr_adj[i] = norm_cdf_py(z_scor)
                    pdf_arr_adj[i] = norm_pdf_py(z_scor)
                elif ((val_adj > self.var_le_trs) and
                      (val_adj <= self.var_ge_trs)):
                    _ = norm_ppf_py(py_del) - mu_t
                    z_scor = divide(_, sig_sq_t ** 0.5)
                    gy_arr_adj[i] = norm_cdf_py(z_scor)
                    pdf_arr_adj[i] = norm_pdf_py(z_scor)
                else:
                    values_arr = (
                        self.in_var_df.loc[infill_date,
                                           curr_var_df.columns[1:]
                                           ].dropna().values)
                    if py_zero:
                        _ = norm_ppf_py(py_zero) - mu_t
                        z_scor = divide(_, sig_sq_t ** 0.5)
                        gy_arr_adj[i] = norm_cdf_py(z_scor)
                        pdf_arr_adj[i] = norm_pdf_py(z_scor)
                    else:
                        gy_arr_adj[i] = pdf_arr_adj[i] = 0.0

                assert not isnan(gy_arr_adj[i]), as_err(
                    '\'gy\' is nan (val: %0.2e)!')
                assert not isnan(pdf_arr_adj[i]), as_err(
                    '\'pdf\' is nan (val: %0.2e)!' % val_adj)

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_adj_raw'] = gy_arr_adj
                step_vars_dict['val_arr_adj_raw'] = val_arr_adj
                step_vars_dict['pdf_arr_adj_raw'] = pdf_arr_adj

            adj_probs_idxs = gy_arr_adj >= self.adj_prob_bounds[0]
            adj_probs_idxs = logical_and(
                             adj_probs_idxs,
                             gy_arr_adj <= self.adj_prob_bounds[1])

            gy_arr_adj = gy_arr_adj[adj_probs_idxs]
            pdf_arr_adj = pdf_arr_adj[adj_probs_idxs]
            val_arr_adj = val_arr_adj[adj_probs_idxs]

            assert gy_arr_adj.shape[0] > 0, as_err(
                'Increase discretization!')
            assert gy_arr_adj.shape[0] == val_arr_adj.shape[0], as_err(
                'unequal shapes of probs and vals!')
            assert pdf_arr_adj.shape[0] == val_arr_adj.shape[0], as_err(
                'unequal shapes of densities and vals!')

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_adj_fin'] = gy_arr_adj
                step_vars_dict['val_arr_adj_fin'] = val_arr_adj
                step_vars_dict['pdf_arr_adj_fin'] = pdf_arr_adj

            fin_val_ppf_ftn_adj = interp1d(gy_arr_adj,
                                           val_arr_adj,
                                           bounds_error=False,
                                           fill_value=(
                                            self.var_le_trs,
                                            curr_max_var_val_adj))
            fin_val_grad_ftn_adj = interp1d(val_arr_adj,
                                            pdf_arr_adj,
                                            bounds_error=False,
                                            fill_value=(0, 0))
        return (fin_val_ppf_ftn_adj,
                fin_val_grad_ftn_adj,
                gy_arr_adj,
                val_arr_adj,
                pdf_arr_adj)

    def get_cont_vars(self,
                      val_cdf_ftn,
                      mu_t,
                      sig_sq_t,
                      step_vars_dict):

        val_arr = val_cdf_ftn.x
        probs_arr = val_cdf_ftn.y
        gy_arr = full(probs_arr.shape, nan)
        for i, prob in enumerate(probs_arr):
            assert not isnan(prob), as_err('\'prob\' is NaN!')
            _ = norm_ppf_py(prob) - mu_t
            gy_arr[i] = norm_cdf_py(divide(_, sig_sq_t ** 0.5))
            assert not isnan(gy_arr[i]), as_err(
                '\'gy\' is NaN (prob:%0.2e)!' % prob)

        if self.save_step_vars_flag:
            step_vars_dict['gy_arr_raw'] = gy_arr

        # do the interpolation again with adjusted bounds
        adj_probs_idxs = gy_arr > self.adj_prob_bounds[0]
        adj_probs_idxs = logical_and(adj_probs_idxs,
                                     gy_arr < self.adj_prob_bounds[1])

        val_arr_adj = val_arr[adj_probs_idxs]
        probs_arr_adj = probs_arr[adj_probs_idxs]

        if val_arr_adj.shape[0] <= 1:
            if gy_arr.max() > self.adj_prob_bounds[1]:
                interp_val = val_arr.max()
                gy_val = gy_arr[-1]
            else:
                interp_val = val_arr.min()
                gy_val = gy_arr[0]

            fin_val_ppf_ftn_adj = interp1d(linspace(0, 1.0, 10),
                                           [interp_val] * 10,
                                           bounds_error=False,
                                           fill_value=(interp_val,
                                                       interp_val))
            fin_val_grad_ftn = interp1d((val_arr.min(), val_arr.max()),
                                        (0, 0),
                                        bounds_error=False,
                                        fill_value=(0, 0))
            gy_arr_adj = array([gy_val, gy_val])
            val_arr_adj = array([interp_val, interp_val])
            pdf_arr_adj = array([0.0, 0.0])
        else:
#            if val_arr_adj.shape[0] == 0:
#                as_err(('CRITICAL: \'val_arr_adj\' has less than 2 elements! '
#                        'min_prob: %f, max_prob: %f, val_arr_adj: %s') %
#                       (gy_arr.min(), gy_arr.max(), str(val_arr_adj)))
#
#                import pdb
#                pdb.set_trace()
#                tre = 1
#
#                if self.dont_stop_flag:
#                    return [None] * 5
#                else:
#                    raise Exception(('val_arr_adj is invalid %s!' %
#                                     str(val_arr_adj)))

            (curr_min_var_val_adj,
             curr_max_var_val_adj) = (val_arr_adj.min(),
                                      val_arr_adj.max())

            n_vals = where(adj_probs_idxs)[0].shape[0]
            if n_vals < self.n_discret:
                val_adj_interp = interp1d(list(range(0, val_arr_adj.shape[0])),
                                          val_arr_adj)
                prob_adj_interp = interp1d(list(range(0,
                                                      val_arr_adj.shape[0])),
                                           probs_arr_adj)
                _interp_vals = linspace(0.0,
                                        val_arr_adj.shape[0] - 1,
                                        self.n_discret)
                val_arr_adj = val_adj_interp(_interp_vals)
                probs_arr_adj = prob_adj_interp(_interp_vals)

            gy_arr_adj = full(val_arr_adj.shape, nan)
            pdf_arr_adj = gy_arr_adj.copy()

            for i, adj_prob in enumerate(probs_arr_adj):
                z_scor = divide((norm_ppf_py(adj_prob) - mu_t), sig_sq_t ** 0.5)
                gy_arr_adj[i] = norm_cdf_py(z_scor)
                pdf_arr_adj[i] = norm_pdf_py(z_scor)

            if self.save_step_vars_flag:
                step_vars_dict['gy_arr_adj_fin'] = gy_arr_adj
                step_vars_dict['val_arr_adj_fin'] = val_arr_adj
                step_vars_dict['pdf_arr_adj_fin'] = pdf_arr_adj

            fin_val_ppf_ftn_adj = interp1d(gy_arr_adj,
                                           val_arr_adj,
                                           bounds_error=False,
                                           fill_value=(curr_min_var_val_adj,
                                                       curr_max_var_val_adj))
            fin_val_grad_ftn = interp1d(val_arr_adj,
                                        pdf_arr_adj,
                                        bounds_error=False,
                                        fill_value=(0, 0))
        return (fin_val_ppf_ftn_adj,
                fin_val_grad_ftn,
                gy_arr_adj,
                val_arr_adj,
                pdf_arr_adj)


if __name__ == '__main__':
    pass
