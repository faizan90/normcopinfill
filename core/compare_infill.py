# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from math import log as mlog

from numpy import (isnan,
                   where,
                   logical_or,
                   logical_not,
                   any as np_any,
                   divide,
                   sum as np_sum,
                   abs as np_abs,
                   argsort,
                   nan)

from matplotlib import use as plt_use
plt_use('AGG')
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.stats import rankdata

import pyximport
pyximport.install()
from normcop_cyftns import (get_corrcoeff,
                            get_kge_py,
                            get_ns_py,
                            get_ln_ns_py)


class CompareInfill:
    def __init__(self, norm_cop_obj):
        vars_list = ['n_rand_infill_values',
                     'fin_conf_head',
                     'infill_dates',
                     'curr_infill_stn',
                     '_compr_lab',
                     '_ks_lims_lab',
                     '_mean_obs_lab',
                     '_mean_infill_lab',
                     '_var_obs_lab',
                     '_var_infill_lab',
                     '_bias_lab',
                     '_mae_lab',
                     '_rmse_lab',
                     '_nse_lab',
                     '_ln_nse_lab',
                     '_kge_lab',
                     '_pcorr_lab',
                     '_scorr_lab',
                     'n_round',
                     'fig_size_long',
                     'out_fig_dpi',
                     'out_fig_fmt',
                     'ks_alpha']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.update_summary_df_only = False

        return

    def _plot_comparison(self,
                         out_conf_df,
                         interp_data_idxs,
                         alpha,
                         lw,
                         act_var,
                         n_vals,
                         bias,
                         mae,
                         rmse,
                         nse,
                         ln_nse,
                         kge,
                         correl_pe,
                         correl_sp,
                         out_compar_plot_loc):

        plt.figure(figsize=self.fig_size_long)
        infill_ax = plt.subplot(111)

        for _conf_head in out_conf_df.columns:
            conf_var_vals = where(interp_data_idxs, nan,
                                  out_conf_df[_conf_head].loc[
                                          self.infill_dates])

            infill_ax.plot(self.infill_dates,
                           conf_var_vals,
                           label=_conf_head,
                           alpha=alpha,
                           lw=lw,
                           ls='-',
                           marker='o',
                           ms=lw+0.5)

        infill_ax.plot(self.infill_dates,
                       where(interp_data_idxs, nan, act_var),
                       label='actual',
                       c='k',
                       ls='-',
                       marker='o',
                       alpha=1.0,
                       lw=lw+0.5,
                       ms=lw+1)

        infill_ax.set_xlabel('Time')
        infill_ax.set_ylabel('var_val')
        title_str = (('Observed and infilled confidence interval '
                      'comparison for station: %s (%d values)') %
                     (self.curr_infill_stn, n_vals))
        title_str += (('\n(Bias: %0.2f, Mean absolute error: %0.2f, '
                       'Root mean squared error: %0.2f)') %
                      (bias, mae, rmse))
        title_str += (('\n(NSE: %0.2f, Ln-NSE: %0.2f, KGE: %0.2f, ') %
                      (nse, ln_nse, kge))
        title_str += (('Pearson correlation: %0.2f, Spearman correlation: '
                       '%0.2f)') % (correl_pe, correl_sp))
        plt.suptitle(title_str)
        infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
        plt.grid()
        plt.legend(framealpha=0.5, loc=0)
        plt.savefig(out_compar_plot_loc, dpi=self.out_fig_dpi)
        plt.close('all')

        return

    def _plot_infill_vs_obs_cdf_time_sers(self,
                                          infill_probs,
                                          orig_probs,
                                          alpha,
                                          lw,
                                          out_compar_plot_loc):
        # plot the infilled cdf values against observed w.r.t time
        plt.figure(figsize=(6, 5.5))
        infill_ax = plt.subplot(111)
        infill_ax.plot(infill_probs,
                       orig_probs,
                       alpha=alpha,
                       lw=lw,
                       ls='-',
                       marker='o',
                       ms=lw)
        infill_ax.plot(orig_probs,
                       orig_probs,
                       alpha=0.25,
                       c='k',
                       lw=lw+6,
                       ls='-')

        infill_ax.set_xlabel('Infilled Probability')
        infill_ax.set_ylabel('Observed Probability')

        infill_ax.set_xlim(-0.05, 1.05)
        infill_ax.set_ylim(-0.05, 1.05)

        title_str = (('Observed and infilled probability comparison '
                      'for each \ninfilled value for station: '
                      '%s') % self.curr_infill_stn)
        plt.suptitle(title_str)

        plt.grid()
        _ = out_compar_plot_loc[:-(len(self.out_fig_fmt) + 1)]
        out_freq_compare_loc = _ + '_time_cdf.' + self.out_fig_fmt
        plt.savefig(out_freq_compare_loc, dpi=self.out_fig_dpi)
        plt.close('all')
        return

    def _plot_obs_vs_infill_dists(self,
                                  out_add_info_df,
                                  summ_df,
                                  alpha,
                                  lw,
                                  n_vals,
                                  out_compar_plot_loc):

        obs_probs_ser = out_add_info_df['act_val_prob'].copy()
        obs_probs_ser.dropna(inplace=True)

        if not obs_probs_ser.shape[0]:
            return summ_df

        obs_probs = obs_probs_ser.values
        obs_arg_sort = obs_probs.argsort()
        obs_probs = obs_probs[obs_arg_sort]

        _ = obs_probs_ser.rank()
        obs_probs_probs = _.div(obs_probs_ser.shape[0] + 1.)
        obs_probs_probs = obs_probs_probs.values
        obs_probs_probs = obs_probs_probs[obs_arg_sort]

        ks_d_stat = (divide((-0.5 * mlog(self.ks_alpha * 0.5)),
                            obs_probs.shape[0]))**0.5

        ks_fl = obs_probs_probs - ks_d_stat
        ks_fl[ks_fl < 0] = 0

        ks_fu = obs_probs_probs + ks_d_stat
        ks_fu[ks_fu > 1] = 1

        n_ks_fu_g = where(obs_probs < ks_fl)[0].shape[0]
        n_ks_fl_l = where(obs_probs > ks_fu)[0].shape[0]

        vals_wtn_rng = divide(100.0 * (n_vals - n_ks_fl_l - n_ks_fu_g),
                              float(n_vals))

        summ_df.loc[self.curr_infill_stn,
                    self._ks_lims_lab] = round(vals_wtn_rng, 2)

        if not self.update_summary_df_only:
            plt.figure(figsize=(6, 5.5))
            obs_probs_ax = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
            obs_probs_hist_ax = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

            obs_probs_ax.plot(obs_probs,
                              obs_probs_probs,
                              alpha=alpha,
                              lw=lw,
                              ls='-',
                              marker='o',
                              ms=lw)
            obs_probs_ax.plot(obs_probs_probs,
                              ks_fl,
                              alpha=alpha,
                              c='k',
                              lw=lw,
                              ls='--',
                              ms=lw)
            obs_probs_ax.plot(obs_probs_probs,
                              ks_fu,
                              alpha=alpha,
                              c='k',
                              lw=lw,
                              ls='--',
                              ms=lw)

            obs_probs_ax.set_xticklabels([])
            obs_probs_ax.set_xlim(-0.05, 1.05)
            obs_probs_ax.set_ylim(-0.05, 1.05)
            obs_probs_ax.set_ylabel('Theoretical Probability')
            obs_probs_ax.grid()
            obs_probs_ax.get_xaxis().set_tick_params(width=0)

            obs_probs_hist_ax.hist(obs_probs,
                                   bins=20,
                                   alpha=alpha,
                                   range=(0.0, 1.0))
            obs_probs_hist_ax.set_xlim(-0.05, 1.05)
            obs_probs_hist_ax.set_xlabel('Observed-infilled Probability')
            obs_probs_hist_ax.set_ylabel('Frequency')
            obs_probs_hist_ax.grid()

            title_str = (('Observed values\' infilled probability for '
                          'station: %s') % self.curr_infill_stn)
            title_str += (('\n%0.2f%% values within %0.0f%% KS-limits') %
                          (vals_wtn_rng, 100 * (1.0 - self.ks_alpha)))
            title_str += (('\nOut of %d, %d values below and %d above '
                           'limits') % (n_vals, n_ks_fl_l, n_ks_fu_g))
            plt.suptitle(title_str)

            _ = out_compar_plot_loc[:-(len(self.out_fig_fmt) + 1)]
            out_freq_compare_loc = _ + '_obs_probs_cdf.' + self.out_fig_fmt
            plt.savefig(out_freq_compare_loc, dpi=self.out_fig_dpi)
            plt.close('all')
        return summ_df

    def _plot_obs_vs_infill_vals_hist(self,
                                      orig_vals,
                                      infill_vals,
                                      n_vals,
                                      out_compar_plot_loc):
        # plot the CDFs of infilled and original data
        orig_sorted_val_idxs = argsort(orig_vals)
        infill_sorted_val_idxs = argsort(infill_vals)

        orig_vals = orig_vals[orig_sorted_val_idxs]
        infill_vals = infill_vals[infill_sorted_val_idxs]

        assert orig_vals.shape[0] == infill_vals.shape[0]

        plt.figure(figsize=(6, 5.5))
        infill_ax = plt.subplot(111)

        _min_var = max(orig_vals.min(), infill_vals.min())
        _max_var = max(orig_vals.max(), infill_vals.max())

        infill_ax.hist(orig_vals,
                       bins=20,
                       range=(_min_var, _max_var),
                       alpha=0.5,
                       label='observed')
        infill_ax.hist(infill_vals,
                       bins=20,
                       range=(_min_var, _max_var),
                       rwidth=0.8,
                       alpha=0.5,
                       label='infilled')

        infill_ax.set_xlabel('Variable')
        infill_ax.set_ylabel('Frequency')
        title_str = (('Observed and infilled histogram comparison '
                      '\nfor station: %s (%d values)') %
                     (self.curr_infill_stn, n_vals))

        plt.suptitle(title_str)
        plt.grid()
        plt.legend()
        _ = out_compar_plot_loc[:-(len(self.out_fig_fmt) + 1)]
        out_cdf_compare_loc = _ + '_hists.' + self.out_fig_fmt
        plt.savefig(out_cdf_compare_loc, dpi=self.out_fig_dpi)
        plt.close('all')
        return

    def _compar_ser(self,
                    act_var,
                    out_conf_df,
                    out_compar_plot_loc,
                    out_add_info_df,
                    interp_data_idxs,
                    not_interp_data_idxs,
                    summ_df,
                    lw,
                    alpha):
        # compare the observed and infill bounds and plot
        orig_vals = act_var[not_interp_data_idxs]

        if not self.n_rand_infill_values:
            infill_vals = out_conf_df[self.fin_conf_head]
        else:
            infill_vals = out_conf_df[self.fin_conf_head % 0]

        infill_vals = infill_vals.loc[self.infill_dates].values
        infill_vals = infill_vals[not_interp_data_idxs]

        n_vals = orig_vals.shape[0]

        diff = orig_vals - infill_vals

        bias = round(divide(np_sum(diff), n_vals), self.n_round)
        mae = round(divide(np_sum(np_abs(diff)), n_vals), self.n_round)
        rmse = round((divide(np_sum(diff**2), n_vals))**0.5, self.n_round)

        orig_probs = divide(rankdata(orig_vals), (n_vals + 1.))
        orig_probs_sort_idxs = argsort(orig_probs)
        orig_probs = orig_probs[orig_probs_sort_idxs]

        infill_probs = divide(rankdata(infill_vals), (n_vals + 1.))
        infill_probs = infill_probs[orig_probs_sort_idxs]

        nse = round(get_ns_py(orig_vals, infill_vals, 0),
                    self.n_round)
        ln_nse = round(get_ln_ns_py(orig_vals, infill_vals, 0),
                       self.n_round)
        kge = round(get_kge_py(orig_vals, infill_vals, 0),
                    self.n_round)

        correl_pe = round(get_corrcoeff(orig_vals, infill_vals),
                          self.n_round)
        correl_sp = round(get_corrcoeff(orig_probs, infill_probs),
                          self.n_round)

        obs_mean = round(orig_vals.mean(), self.n_round)
        infill_mean = round(infill_vals.mean(), self.n_round)
        obs_var = round(orig_vals.var(), self.n_round)
        infill_var = round(infill_vals.var(), self.n_round)

        summ_df.loc[self.curr_infill_stn,
                    [self._compr_lab,
                     self._bias_lab,
                     self._mae_lab,
                     self._rmse_lab,
                     self._nse_lab,
                     self._ln_nse_lab,
                     self._kge_lab,
                     self._pcorr_lab,
                     self._scorr_lab]] = \
            [n_vals,
             bias,
             mae,
             rmse,
             nse,
             ln_nse,
             kge,
             correl_pe,
             correl_sp]

        summ_df.loc[self.curr_infill_stn,
                    [self._mean_obs_lab,
                     self._mean_infill_lab,
                     self._var_obs_lab,
                     self._var_infill_lab]] = [obs_mean,
                                               infill_mean,
                                               obs_var,
                                               infill_var]

        if not self.update_summary_df_only:
            self._plot_comparison(out_conf_df,
                                  interp_data_idxs,
                                  alpha,
                                  lw,
                                  act_var,
                                  n_vals,
                                  bias,
                                  mae,
                                  rmse,
                                  nse,
                                  ln_nse,
                                  kge,
                                  correl_pe,
                                  correl_sp,
                                  out_compar_plot_loc)

            self._plot_infill_vs_obs_cdf_time_sers(infill_probs,
                                                   orig_probs,
                                                   alpha,
                                                   lw,
                                                   out_compar_plot_loc)

        self._plot_obs_vs_infill_dists(out_add_info_df,
                                       summ_df,
                                       alpha,
                                       lw,
                                       n_vals,
                                       out_compar_plot_loc)

        if not self.update_summary_df_only:
            self._plot_obs_vs_infill_vals_hist(orig_vals,
                                               infill_vals,
                                               n_vals,
                                               out_compar_plot_loc)

        return summ_df

    def plot_all(self, args):
        '''
        1. Plot comparison between infilled (with CIs) and observed.
        2. Plot KS limits test
        3. Plot infill and observed historgrams comparison
        '''
        (act_var,
         out_conf_df,
         out_compar_plot_loc,
         out_add_info_df,
         self.update_summary_df_only) = args

        lw, alpha = 0.8, 0.7

        if not self.n_rand_infill_values:
            interp_data_idxs = logical_or(isnan(act_var),
                                          isnan(out_conf_df[
                                                  self.fin_conf_head].loc[
                                                          self.infill_dates
                                                          ].values))
        else:
            interp_data_idxs = logical_or(isnan(act_var),
                                          isnan(out_conf_df[
                                                  self.fin_conf_head % 0].loc[
                                                          self.infill_dates
                                                          ].values))

        not_interp_data_idxs = logical_not(interp_data_idxs)

        plot_compare_cond = np_any(not_interp_data_idxs)

        summ_df_cols = [self._compr_lab,
                        self._ks_lims_lab,
                        self._mean_obs_lab,
                        self._mean_infill_lab,
                        self._var_obs_lab,
                        self._var_infill_lab,
                        self._bias_lab,
                        self._mae_lab,
                        self._rmse_lab,
                        self._nse_lab,
                        self._ln_nse_lab,
                        self._kge_lab,
                        self._pcorr_lab,
                        self._scorr_lab]

        summ_df = DataFrame(index=[self.curr_infill_stn],
                            dtype=float,
                            columns=summ_df_cols)

        if plot_compare_cond:
            self._compar_ser(act_var,
                             out_conf_df,
                             out_compar_plot_loc,
                             out_add_info_df,
                             interp_data_idxs,
                             not_interp_data_idxs,
                             summ_df,
                             lw,
                             alpha)
        return summ_df


if __name__ == '__main__':
    pass
