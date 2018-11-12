# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from pickle import dump
from os.path import join as os_join

from numpy import (
    nan,
    where,
    all as np_all,
    isfinite,
    any as np_any,
    isnan,
    linalg,
    matmul,
    round as np_round,
    ediff1d,
    float32,
    float64)

from scipy.interpolate import interp1d
from pandas import DataFrame, Index

from .adjust_nebs import AdjustNebs
from .transform import StepTrans
from .step_vars import StepVars
from .var_vars import DiscContVars
from .plot_cond_ftns import PlotCondFtns
from ..misc.misc_ftns import as_err, pprt
from ..cyth import fill_correl_mat


class InfillSteps:

    def __init__(self, norm_cop_obj):

        vars_list = list(vars(norm_cop_obj).keys())

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))
        return

    def infill_steps(self, set_tuple):

        self.curr_nrst_stns = None

        infill_dates = set_tuple[0]
        curr_nrst_stns = Index(list(set_tuple[1]))

        curr_var_df = self.in_var_df

        out_conf_df = DataFrame(
            index=infill_dates, columns=self.conf_ser.index, dtype=float32)

        out_add_info_df = DataFrame(
            index=infill_dates,
            dtype=float32,
            columns=['infill_status',
                     'n_neighbors_raw',
                     'n_neighbors_fin',
                     'act_val_prob',
                     'n_recs'])

        out_add_info_df.iloc[:, :] = [False, 0, 0, nan, nan]

        too_hi_corr_stns_list = []

        abj_nebs_obj = AdjustNebs(self)
        step_var_obj = StepVars(self)
        var_vars_obj = DiscContVars(self)

        if self.plot_step_cdf_pdf_flag or self.plot_diag_flag:
            plot_cond_ftns_obj = PlotCondFtns(self)

        curr_val_cdf_ftns_dict = {}
        curr_py_zeros_dict = {}
        curr_py_dels_dict = {}

        if not self.force_infill_flag:
            assert curr_var_df.shape[1] > self.n_min_nebs, as_err(
                ('\'curr_var_df\' has too few neighboring '
                 'stations (%d) in it!' %
                curr_var_df.shape[1]))

        assert curr_var_df.shape[0] >= self.min_valid_vals, as_err(
            '\'curr_var_df\' has too few records (%d)!' %
            curr_var_df.shape[0])

        date_pref = infill_dates[0].strftime('%Y%m%d%H%M')

        trans_obj = StepTrans(
            self,
            curr_var_df,
            curr_py_zeros_dict,
            curr_py_dels_dict,
            curr_val_cdf_ftns_dict,
            date_pref)

        norms_df, py_del, py_zero = trans_obj.get_cdfs_probs()

        if self.plot_diag_flag:
            trans_obj.plot_cdfs()

        assert not np_any(isnan(norms_df.values)), as_err(
            'NaNs in \'norms_df\' on %s!' % date_pref)

        full_corrs_arr = fill_correl_mat(
            norms_df.values.astype(float64))

        _nans_in_corrs = np_all(isnan(full_corrs_arr))
        if _nans_in_corrs:
            as_err((
                'Invalid values in \'full_corrs_arr\' on %s!') %
                date_pref)

            raise Exception('NaNs in full_corrs_arr!')

        full_corrs_arr = abj_nebs_obj.adj_corrs(
            infill_dates[0],
            full_corrs_arr,
            norms_df,
            too_hi_corr_stns_list,
            curr_val_cdf_ftns_dict,
            curr_nrst_stns,
            curr_var_df)

        for stn in too_hi_corr_stns_list:
            del curr_val_cdf_ftns_dict[stn]

            norms_df.drop(stn, axis=1, inplace=True)
            curr_var_df.drop(stn, axis=1, inplace=True)

        norm_cov_mat = full_corrs_arr[1:, 1:]
        cov_vec = full_corrs_arr[1:, 0]

        if cov_vec.shape[0] == 0:
            print('\n')
            pprt(['WARNING: \'cov_vec\' is empty on date:',
                  infill_dates[0]],
                 nbh=8,
                 nah=8)

            pprt(['Stations in use are:'], nbh=12)
            n_pas = len(curr_nrst_stns)
            for i_msg in range(0, n_pas, min(3, n_pas)):
                pprt(curr_nrst_stns[i_msg:(i_msg + 3)], nbh=16)

            pprt([''], nbh=24, nah=24)
            print('\n')

            raise Exception('cov_vec is empty!')

        assert norm_cov_mat.shape[0] > 0, as_err(
            '\'norm_cov_mat\' is empty!')

        inv_norm_cov_mat = linalg.inv(norm_cov_mat)

        assert cov_vec.shape[0] == inv_norm_cov_mat.shape[0], (
            as_err('Incorrect deletion of vectors!'))

        assert cov_vec.shape[0] == inv_norm_cov_mat.shape[1], (
            as_err('Incorrect deletion of vectors!'))

        sig_sq_t = 1.0 - matmul(
            cov_vec.T, matmul(inv_norm_cov_mat, cov_vec))

        if sig_sq_t <= 0:
            print('\n')
            pprt(['WARNING:',
                  (('Stn %s has an invalid conditional '
                    'variance') % self.curr_infill_stn)],
                 nbh=8,
                 nah=8)

            pprt([('\'sig_sq_t (%0.6f)\' is '
                   'less than zero!') % sig_sq_t],
                 nbh=12)

            pprt(['\'infill_date\':', infill_dates[0]], nbh=12)

            pprt(['\'best_stns\': \'covariance\''], nbh=12)
            for bstn_cov in zip(list(curr_nrst_stns), cov_vec):
                pprt(['%s: %0.7f' % (bstn_cov[0], bstn_cov[1])],
                     nbh=16)

            pprt([''], nbh=24, nah=24)
            print('\n')

            raise Exception('sig_sq_t less than zero!')

        curr_max_var_val = curr_var_df[self.curr_infill_stn].max()
        curr_min_var_val = curr_var_df[self.curr_infill_stn].min()

        if not self.force_infill_flag:
            assert curr_var_df.shape[1] > self.n_min_nebs, as_err(
            ('\'curr_var_df\' has too few neighboring '
             'stations (%d) in it!') % curr_var_df.shape[1])

        assert curr_var_df.shape[0] >= self.min_valid_vals, as_err(
            '\'curr_var_df\' has too few records (%d)!' %
            curr_var_df.shape[0])

        if self.infill_type == 'precipitation':
            assert not isnan(py_zero), as_err('\'py_zero\' is nan!')
            assert not isnan(py_del), '\'py_del\' is nan!'

        # another check to ensure neighbors are selected correctly
        _ = list(curr_val_cdf_ftns_dict.keys())
        assert len(_) == len(curr_nrst_stns) + 1, as_err(
            ('\'curr_val_cdf_ftns_dict\' has incorrect '
             'number of keys! ' +
             str(list(curr_val_cdf_ftns_dict.keys())) +
             str(list(curr_nrst_stns))))

        val_cdf_ftn = curr_val_cdf_ftns_dict[self.curr_infill_stn]

        for infill_date in infill_dates:
            date_pref = infill_date.strftime('%Y%m%d%H%M')

            u_t = step_var_obj.get_step_vars(
                 curr_var_df,
                 curr_val_cdf_ftns_dict,
                 infill_date,
                 curr_py_zeros_dict,
                 curr_py_dels_dict)

            mu_t = matmul(cov_vec.T, matmul(inv_norm_cov_mat, u_t))
            assert not isnan(mu_t), as_err('\'mu_t\' is NaN!')

            if self.debug_mode_flag:
                print(date_pref)

            if not isnan(self.in_var_df.loc[
                infill_date, self.curr_infill_stn]):

                if not self.compare_infill_flag:

                    if self.debug_mode_flag:
                        print('Value exists!')

                    continue

            if self.infill_type == 'precipitation':
                curr_vals = self.in_var_df.loc[
                    infill_date, curr_nrst_stns].values

                if np_all(curr_vals == self.var_le_trs):
                    out_conf_df.loc[infill_date] = self.var_le_trs
                    out_add_info_df.loc[infill_date, 'infill_status'] = True

                    if self.debug_mode_flag:
                        print(('All values are less than or equal to %0.2f!' %
                               self.var_le_trs))
                    continue

            out_add_info_df.loc[infill_date, 'n_neighbors_fin'] = (
                len(curr_nrst_stns))

            out_add_info_df.loc[
                infill_date, 'n_neighbors_raw'] = len(curr_nrst_stns)

            if self.infill_type == 'precipitation':
                (fin_val_ppf_ftn_adj,
                 fin_val_grad_ftn,
                 gy_arr_adj,
                 val_arr_adj,
                 pdf_arr_adj) = var_vars_obj.get_disc_vars(
                     curr_max_var_val,
                     val_cdf_ftn,
                     curr_min_var_val,
                     mu_t,
                     sig_sq_t,
                     py_del,
                     infill_date,
                     curr_var_df,
                     py_zero)

            else:
                (fin_val_ppf_ftn_adj,
                 fin_val_grad_ftn,
                 gy_arr_adj,
                 val_arr_adj,
                 pdf_arr_adj) = var_vars_obj.get_cont_vars(
                     val_cdf_ftn, mu_t, sig_sq_t)

            if fin_val_ppf_ftn_adj is None:
                pprt([(('WARNING: fin_val_ppf_ftn_adj is None on (%s, %s)!') %
                       (str(infill_date),
                        str(self.curr_infill_stn)))],
                     nbh=8,
                     nah=8)
                continue

            conf_probs = self.conf_ser.values
            conf_vals = fin_val_ppf_ftn_adj(conf_probs)
            conf_grads = fin_val_grad_ftn(conf_vals)
            out_conf_df.loc[infill_date] = np_round(conf_vals, self.n_round)
            out_add_info_df.loc[infill_date, 'infill_status'] = True

            descend_idxs = where(ediff1d(conf_vals) < 0, 1, 0)

            if np_any(descend_idxs):
                print('\n')
                pprt([(('WARNING: '
                        'Interpolated \'var_vals\' on %s at '
                        'station: %s not in  ascending order!') %
                       (str(infill_date),
                        str(self.curr_infill_stn)))],
                     nbh=8,
                     nah=8)
                pprt([self.conf_heads, list(conf_vals)], nbh=8)
                pprt(['\'gy\':\n', list(gy_arr_adj)], nbh=8)
                pprt(['\'theoretical_var_vals\':\n',
                      list(val_arr_adj)],
                     nbh=8)

                raise Exception(
                    as_err((
                        'Interpolated \'var_vals\' on %s not in ascending '
                        'order!') % str(infill_date)))
                print('\n')

            if not isnan(self.in_var_df.loc[
                infill_date, self.curr_infill_stn]):

                fin_val_cdf_ftn_adj = interp1d(
                    val_arr_adj,
                    gy_arr_adj,
                    bounds_error=False,
                    fill_value=(
                        self.adj_prob_bounds[+0], self.adj_prob_bounds[-1]))

                out_add_info_df.loc[infill_date, 'act_val_prob'] = (
                    fin_val_cdf_ftn_adj(
                        self.in_var_df.loc[infill_date, self.curr_infill_stn]))

            if self.plot_step_cdf_pdf_flag:
                plot_cond_ftns_obj.plot_cond_cdf_pdf(
                    val_arr_adj,
                    gy_arr_adj,
                    conf_vals,
                    conf_probs,
                    date_pref,
                    py_zero,
                    py_del,
                    pdf_arr_adj,
                    conf_grads)

            if self.plot_diag_flag:
                plot_cond_ftns_obj.plot_corr_mat(
                    full_corrs_arr, curr_var_df, date_pref)

            if self.debug_mode_flag:
                print('Infilled on %s!' % date_pref)

        return (out_conf_df, out_add_info_df)


if __name__ == '__main__':
    pass
