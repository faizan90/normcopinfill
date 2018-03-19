# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from os.path import join as os_join

from numpy import (nan,
                   where,
                   divide,
                   logical_and,
                   all as np_all,
                   isfinite,
                   any as np_any,
                   isnan)
from matplotlib import use as plt_use
plt_use('AGG')
import matplotlib.pyplot as plt
from pandas import DataFrame
from scipy.interpolate import interp1d

from ..misc.misc_ftns import as_err

import pyximport
pyximport.install()
from normcop_cyftns import norm_ppf_py_arr


class StepTrans:
    def __init__(self,
                 infill_steps_obj,
                 curr_var_df,
                 curr_py_zeros_dict,
                 curr_py_dels_dict,
                 curr_val_cdf_ftns_dict,
                 date_pref):
        vars_list = ['infill_type',
                     'var_le_trs',
                     'var_ge_trs',
                     '_rank_method',
                     'curr_infill_stn',
                     'out_fig_fmt',
                     'stn_step_cdfs_dir',
                     'out_fig_dpi',
                     'cut_cdf_thresh']

        for _var in vars_list:
            setattr(self, _var, getattr(infill_steps_obj, _var))

        self._probs_str = 'probs'
        self._norm_str = 'norm_vals'
        self._vals_str = 'vals'

        self.curr_var_df = curr_var_df
        self.curr_py_zeros_dict = curr_py_zeros_dict
        self.curr_py_dels_dict = curr_py_dels_dict
        self.curr_val_cdf_ftns_dict = curr_val_cdf_ftns_dict
        self.date_pref = date_pref

        return

    def get_cdfs_probs(self):

        # create probability and standard normal value dfs
        self.probs_df = DataFrame(index=self.curr_var_df.index,
                                  columns=self.curr_var_df.columns,
                                  dtype=float)
        self.norms_df = self.probs_df.copy()

        self.py_del = nan
        self.py_zero = nan

        self.curr_val_cdf_df_dict = {}

        for col in self.curr_var_df.columns:
            # CDFs
            var_ser = self.curr_var_df[col].copy()

            if self.infill_type == 'precipitation':
                # get probability of zero and below threshold
                zero_idxs = where(var_ser.values == self.var_le_trs)[0]
                zero_prob = divide(float(zero_idxs.shape[0]),
                                   var_ser.shape[0])

                thresh_idxs = where(
                              logical_and(var_ser.values > self.var_le_trs,
                                          var_ser.values <=
                                          self.var_ge_trs))[0]
                thresh_prob = (zero_prob +
                               divide(float(thresh_idxs.shape[0]),
                                      var_ser.shape[0]))
                thresh_prob_orig = thresh_prob
                thresh_prob = zero_prob + (0.5 * (thresh_prob - zero_prob))
                assert zero_prob <= thresh_prob, \
                    as_err('\'zero_prob\' > \'thresh_prob\'!')

                self.curr_py_zeros_dict[col] = zero_prob * 0.5
                self.curr_py_dels_dict[col] = thresh_prob

                var_ser_copy = var_ser.copy()
                var_ser_copy[var_ser_copy <= self.var_ge_trs] = nan

                _ = var_ser_copy.rank(method=self._rank_method)
                probs_ser = _.div((var_ser_copy.count() + 1.))
                probs_ser = (thresh_prob_orig +
                             ((1.0 - thresh_prob_orig) * probs_ser))

                probs_ser.iloc[zero_idxs] = (0.5 * zero_prob)
                probs_ser.iloc[thresh_idxs] = thresh_prob

                assert thresh_prob <= probs_ser.max(), \
                    as_err('\'thresh_prob\' > \'probs_ser.max()\'!')

            else:
                _ = var_ser.rank(method=self._rank_method)
                probs_ser = _.div(var_ser.count() + 1.)

                if self.infill_type == 'discharge-censored':
                    _idxs  = probs_ser.values <= self.cut_cdf_thresh
                    mean_prob = probs_ser.loc[_idxs].mean()
                    probs_ser.loc[_idxs] = mean_prob

            assert np_all(isfinite(probs_ser.values)), \
                as_err('NaNs in \'probs_ser\'!')

            self.probs_df[col] = probs_ser
            assert not np_any(isnan(probs_ser)), \
                as_err('NaNs in \'probs_ser\' on %s' % self.date_pref)
            self.norms_df[col] = norm_ppf_py_arr(probs_ser.values)

            assert np_all(isfinite(self.norms_df[col].values)), \
                as_err('NaNs in \'norms_df[%s]\'!' % col)

            if ((col == self.curr_infill_stn) and
                (self.infill_type == 'precipitation')):
                self.py_del = thresh_prob
                self.py_zero = zero_prob * 0.5

            curr_val_cdf_df = DataFrame(index=self.curr_var_df.index,
                                        columns=[self._probs_str,
                                                 self._vals_str],
                                        dtype=float)
            curr_val_cdf_df[self._probs_str] = self.probs_df[col].copy()
            curr_val_cdf_df[self._vals_str] = var_ser.copy()

            curr_val_cdf_df.sort_values(by=self._vals_str, inplace=True)

            curr_max_prob = curr_val_cdf_df[self._probs_str].values.max()
            curr_min_prob = curr_val_cdf_df[self._probs_str].values.min()

            self.curr_val_cdf_df_dict[col] = curr_val_cdf_df

            # TODO: have parameters for approximate minima and maxima.
            #       this is becoming a problem.
            curr_val_cdf_ftn = \
                interp1d(curr_val_cdf_df[self._vals_str].values,
                         curr_val_cdf_df[self._probs_str].values,
                         fill_value=(curr_min_prob, curr_max_prob),
                         bounds_error=False)

            self.curr_val_cdf_ftns_dict[col] = curr_val_cdf_ftn

        return self.norms_df, self.py_del, self.py_zero

    def plot_cdfs(self):
        plt.figure()
        ax_1 = plt.subplot(111)
        ax_2 = ax_1.twiny()
        for col in self.curr_var_df.columns:
            curr_norm_ppf_df = DataFrame(index=self.curr_var_df.index,
                                         columns=[self._probs_str,
                                                  self._norm_str],
                                         dtype=float)
            curr_norm_ppf_df[self._probs_str] = self.probs_df[col].copy()
            curr_norm_ppf_df[self._norm_str] = self.norms_df[col].copy()

            curr_norm_ppf_df.sort_values(by=self._probs_str, inplace=True)

            # plot currently used stns CDFs
            _1 = self.curr_val_cdf_df_dict[col][self._vals_str].values
            _2 = self.curr_val_cdf_df_dict[col][self._probs_str].values
            lg_1 = ax_1.scatter(_1,
                                _2,
                                label='CDF variable',
                                alpha=0.5, color='r', s=0.5)

            lg_2 = ax_2.scatter(curr_norm_ppf_df[self._norm_str].values,
                                curr_norm_ppf_df[self._probs_str].values,
                                label='CDF ui',
                                alpha=0.9, color='b', s=0.5)

            lgs = (lg_1, lg_2)
            labs = [l.get_label() for l in lgs]
            ax_1.legend(lgs, labs, loc=4, framealpha=0.5)

            ax_1.grid()

            ax_1.set_xlabel('variable x')
            ax_2.set_xlabel('transformed variable x (ui)')
            ax_1.set_ylabel('probability')
            ax_1.set_ylim(0, 1)
            ax_2.set_ylim(0, 1)

            if self.infill_type == 'precipitation':
                plt.suptitle(('Actual and normalized value CDFs '
                              '(n=%d)\n stn: %s, date: %s\npy_zero: '
                              '%0.2f, py_del: %0.2f') %
                             (self.curr_val_cdf_df_dict[col].shape[0],
                              col,
                              self.date_pref,
                              self.py_zero,
                              self.py_del))
            else:
                plt.suptitle(('Actual and normalized value CDFs '
                              '(n=%d)\n stn: %s, date: %s' %
                              (self.curr_val_cdf_df_dict[col].shape[0],
                               col,
                               self.date_pref)))

            plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.75)
            _ = 'CDF_%s_%s.%s' % (self.date_pref, col, self.out_fig_fmt)
            out_cdf_fig_loc = os_join(self.stn_step_cdfs_dir, _)
            plt.savefig(out_cdf_fig_loc,
                        dpi=self.out_fig_dpi,
                        bbox_inches='tight')
            ax_1.cla()
            ax_2.cla()

        plt.close('all')
        return


if __name__ == '__main__':
    pass
