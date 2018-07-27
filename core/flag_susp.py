# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from numpy import (logical_or,
                   isnan,
                   inf,
                   logical_not,
                   nan,
                   full,
                   where,
                   logical_and,
                   divide)

import matplotlib.pyplot as plt
from pandas import DataFrame


class FlagSusp:

    def __init__(self, norm_cop_obj):
        vars_list = ['infill_dates',
                     'curr_infill_stn',
                     'fin_conf_head',
                     'flag_probs',
                     'conf_ser',
                     'flag_df',
                     '_flagged_lab',
                     'n_round',
                     'fig_size_long',
                     'out_fig_dpi']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.update_summary_df_only = False
        return

    def plot(self, args):
        '''
        Plot the flags for values that are out of the normal copula CI bounds
        '''
        (act_var,
         out_conf_df,
         out_flag_susp_loc,
         self.update_summary_df_only) = args

        interp_data_idxs = logical_or(isnan(act_var),
                                      isnan(out_conf_df[
                                              self.fin_conf_head].loc[
                                                      self.infill_dates
                                                      ].values))

        _conf_head_list = []
        for _conf_head in out_conf_df.columns:
            if self.conf_ser[_conf_head] in self.flag_probs:
                _conf_head_list.append(_conf_head)

        conf_var_vals_lo = (
            out_conf_df[_conf_head_list[0]].loc[self.infill_dates].values)
        conf_var_vals_hi = (
            out_conf_df[_conf_head_list[1]].loc[self.infill_dates].values)

        conf_var_vals_lo[isnan(conf_var_vals_lo)] = -inf
        conf_var_vals_hi[isnan(conf_var_vals_hi)] = +inf

        not_interp_data_idxs = logical_not(interp_data_idxs)

        act_var_lo = act_var.copy()
        act_var_hi = act_var.copy()

        act_var_lo[isnan(act_var)] = +inf
        act_var_hi[isnan(act_var)] = -inf

        flag_arr = full(act_var.shape[0], nan)
        conf_var_idxs_lo = where(not_interp_data_idxs,
                                 act_var_lo < conf_var_vals_lo,
                                 False)
        conf_var_idxs_hi = where(not_interp_data_idxs,
                                 act_var_hi > conf_var_vals_hi,
                                 False)
        conf_var_idxs_wi_1 = where(not_interp_data_idxs,
                                   act_var_hi >= conf_var_vals_lo,
                                   False)
        conf_var_idxs_wi_2 = where(not_interp_data_idxs,
                                   act_var_lo <= conf_var_vals_hi,
                                   False)
        conf_var_idxs_wi = logical_and(conf_var_idxs_wi_1,
                                       conf_var_idxs_wi_2)

        flag_arr[conf_var_idxs_lo] = -1
        flag_arr[conf_var_idxs_hi] = +1
        flag_arr[conf_var_idxs_wi] = +0

        flag_arr[interp_data_idxs] = nan  # just in case

        _flag_ser = self.flag_df[self.curr_infill_stn].copy()
        _flag_ser[:] = flag_arr

        n_below_lower_lims = where(conf_var_idxs_lo)[0].shape[0]
        n_above_upper_lims = where(conf_var_idxs_hi)[0].shape[0]
        n_within_lims = where(conf_var_idxs_wi)[0].shape[0]

        summ_df = DataFrame(index=[self.curr_infill_stn],
                            columns=[self._flagged_lab],
                            dtype=float)

        n_out_bds = n_below_lower_lims + n_above_upper_lims
        n_tot = n_out_bds + n_within_lims

        if n_tot:
            summ_df.iloc[0, 0] = (
                 divide(100 * n_out_bds, n_tot))
            summ_df.iloc[0, 0] = round(summ_df.iloc[0, 0], self.n_round)

        if not self.update_summary_df_only:
            flag_str = '(steps below limits: %d, ' % n_below_lower_lims
            flag_str += 'steps within limits: %d, ' % n_within_lims
            flag_str += 'steps above limits: %d)' % n_above_upper_lims

            lw, alpha = 0.8, 0.7

            plt.figure(figsize=self.fig_size_long)
            infill_ax = plt.subplot(111)
            infill_ax.plot(self.infill_dates,
                           flag_arr,
                           alpha=alpha,
                           lw=lw + 0.5,
                           ls='-')

            infill_ax.set_xlabel('Time')
            infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
            infill_ax.set_ylabel('Flag')
            infill_ax.set_yticks([-1, 0, 1])
            infill_ax.set_ylim(-2, 2)
            _y_ticks = ['below_%0.3fP' % self.flag_probs[0],
                        'within\n%0.3fP_&_%0.3fP' % (self.flag_probs[0],
                                                     self.flag_probs[1]),
                        'above_%0.3fP' % self.flag_probs[1]]
            infill_ax.set_yticklabels(_y_ticks)

            plt.suptitle(('Data quality flags for station: %s\n' %
                          self.curr_infill_stn + flag_str))
            plt.grid()
            plt.savefig(out_flag_susp_loc, dpi=self.out_fig_dpi)
            plt.close('all')
        return summ_df, _flag_ser


if __name__ == '__main__':
    pass
