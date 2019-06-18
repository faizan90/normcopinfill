# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sys import exc_info
from os import mkdir as os_mkdir
from os.path import exists as os_exists, join as os_join

from numpy import linspace, mgrid, round as np_round, where, nanmax, nan
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..nebors.nrst_nebs import NrstStns
from ..nebors.rank_corr_nebs import RankCorrStns
from ..misc.misc_ftns import get_norm_rand_symms, as_err, full_tb
from ..cyth import (
    get_asymms_sample,
    get_corrcoeff,
    bi_var_copula,
    tau_sample,
    bivar_gau_cop_arr)

plt.ioff()


class ECops:

    def __init__(self, norm_cop_obj):

        vars_list_1 = [
            'nrst_stns_type',
            'max_time_lag_corr']

        vars_list_2 = [
            'in_var_df',
            'infill_stns',
            'rank_corr_stns_dict',
            'nrst_stns_dict',
            'debug_mode_flag',
            'dont_stop_flag',
            'verbose',
            'ncpus',
            '_norm_cop_pool',
            'cop_bins',
            'infill_type',
            'min_valid_vals',
            '_rank_method',
            'n_norm_symm_flds',
            '_max_symm_rands',
            'ecops_dir',
            'out_fig_dpi',
            'out_fig_fmt']

        for _var in vars_list_1:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        if not norm_cop_obj._dist_cmptd:
            NrstStns(norm_cop_obj)  # deal with this
            assert norm_cop_obj._dist_cmptd, as_err(
                'Call \'cmpt_plot_nrst_stns\' first!')

        if self.nrst_stns_type == 'dist':
            pass

        elif ((self.nrst_stns_type == 'rank') or
              (self.nrst_stns_type == 'symm')):

            if not norm_cop_obj._rank_corr_cmptd:
                RankCorrStns(norm_cop_obj)

                assert norm_cop_obj._rank_corr_cmptd, as_err(
                    'Call \'cmpt_plot_rank_corr_stns\' first!')

                vars_list_2.append('rank_corrs_df')

                if self.max_time_lag_corr:
                    vars_list_2.append('time_lags_df')

        else:
            assert False, as_err(
                'Incorrect \'nrst_stns_type\': %s' %
                str(self.nrst_stns_type))

        for _var in vars_list_2:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        if not os_exists(self.ecops_dir):
            os_mkdir(self.ecops_dir)

        if self.verbose:
            print('INFO: Plotting empirical copulas of infill stations '
                  'against others...')

        if (self.ncpus == 1) or self.debug_mode_flag:
            self._plot_ecops(self.infill_stns)

        else:
            idxs = linspace(
                0,
                len(self.infill_stns),
                self.ncpus + 1,
                endpoint=True,
                dtype='int64')

            sub_cols = []
            if self.infill_stns.shape[0] <= self.ncpus:
                sub_cols = [[stn] for stn in self.infill_stns]

            else:
                for idx in range(self.ncpus):
                    sub_cols.append(self.infill_stns[idxs[idx]:idxs[idx + 1]])

            try:
                list(self._norm_cop_pool.uimap(self._plot_ecops, sub_cols))
                self._norm_cop_pool.clear()

            except:
                self._norm_cop_pool.close()
                self._norm_cop_pool.join()
                raise RuntimeError(
                    'Failed to execute \'plot_ecops\ successfully!')
        return

    def _plot_ecop(
            self,
            infill_stn,
            x_mesh,
            y_mesh,
            cop_ax_ticks,
            cop_ax_labs,
            ecop_raw_ax,
            ecop_grid_ax,
            gau_cop_ax,
            leg_ax,
            cax,
            dens_cnst):

        self.curr_nrst_stns = self.curr_nrst_stns_dict[infill_stn]
        if self.max_time_lag_corr:
            curr_lag_ser = self.time_lags_df[infill_stn].dropna()

        ser_i = self.in_var_df[infill_stn].dropna().copy()
        ser_i_index = ser_i.index
        for other_stn in self.curr_nrst_stns:
            if infill_stn == other_stn:
                continue

            ser_j = self.in_var_df[other_stn].dropna().copy()
            index_ij = ser_i_index.intersection(ser_j.index)

            if index_ij.shape[0] < self.min_valid_vals:
                continue

            new_ser_i = ser_i.loc[index_ij].copy()
            new_ser_j = ser_j.loc[index_ij].copy()

            ranks_i = new_ser_i.rank(method=self._rank_method)
            ranks_j = new_ser_j.rank(method=self._rank_method)

            prob_i = ranks_i.div(new_ser_i.shape[0] + 1.).values
            prob_j = ranks_j.div(new_ser_j.shape[0] + 1.).values

            # plot the empirical copula
            if prob_i.min() < 0 or prob_i.max() > 1:
                assert False, as_err('\'prob_i\' values out of bounds!')
            if prob_j.min() < 0 or prob_j.max() > 1:
                assert False, as_err('\'prob_j\' values out of bounds!')

            if ((self.nrst_stns_type == 'rank') or
                (self.nrst_stns_type == 'symm')):
                correl = self.rank_corrs_df.loc[infill_stn, other_stn]
            else:
                correl = get_corrcoeff(prob_i, prob_j)

            # random normal asymmetries
            asymms_1_list = []
            asymms_2_list = []
            for _ in range(self.n_norm_symm_flds):
                as_1, as_2 = get_norm_rand_symms(
                    correl, min(self._max_symm_rands, prob_i.shape[0]))

                asymms_1_list.append(as_1)
                asymms_2_list.append(as_2)

            min_asymm_1 = min(asymms_1_list)
            max_asymm_1 = max(asymms_1_list)
            min_asymm_2 = min(asymms_2_list)
            max_asymm_2 = max(asymms_2_list)

            # Empirical copula - scatter
            ecop_raw_ax.scatter(
                prob_i,
                prob_j,
                alpha=0.9,
                color='b',
                s=0.5)

            ecop_raw_ax.set_ylabel('other station: %s' % other_stn)
            ecop_raw_ax.set_xlim(0, 1)
            ecop_raw_ax.set_ylim(0, 1)
            ecop_raw_ax.grid()
            ecop_raw_ax.set_title('Empirical Copula - Scatter')

            # Empirical copula - gridded
            cop_dict = bi_var_copula(prob_i, prob_j, self.cop_bins)
            emp_dens_arr = cop_dict['emp_dens_arr']

            max_dens_idxs = where(emp_dens_arr == emp_dens_arr.max())
            max_dens_idx_i = max_dens_idxs[0][0]
            max_dens_idx_j = max_dens_idxs[1][0]
            emp_dens_arr_copy = emp_dens_arr.copy()
            emp_dens_arr_copy[max_dens_idx_i, max_dens_idx_j] = nan
            max_dens = nanmax(emp_dens_arr_copy) * dens_cnst

            ecop_grid_ax.pcolormesh(
                x_mesh,
                y_mesh,
                emp_dens_arr,
                cmap=plt.get_cmap('Blues'),
                vmin=0,
                vmax=max_dens)

            ecop_grid_ax.set_xlabel('infill station: %s' % infill_stn)
            ecop_grid_ax.set_xticks(cop_ax_ticks)
            ecop_grid_ax.set_xticklabels(cop_ax_labs)
            ecop_grid_ax.set_yticks(cop_ax_ticks)
            ecop_grid_ax.set_yticklabels([])
            ecop_grid_ax.set_xlim(0, self.cop_bins)
            ecop_grid_ax.set_ylim(0, self.cop_bins)

            # get other copula params
            tau = tau_sample(prob_i, prob_j)

            emp_asymms = get_asymms_sample(prob_i, prob_j)
            emp_asymm_1, emp_asymm_2 = (
                emp_asymms['asymm_1'], emp_asymms['asymm_2'])

            asymm_1_str = 'within limits'
            if emp_asymm_1 < min_asymm_1:
                asymm_1_str = 'too low'
            elif emp_asymm_1 > max_asymm_1:
                asymm_1_str = 'too high'

            asymm_2_str = 'within limits'
            if emp_asymm_2 < min_asymm_2:
                asymm_2_str = 'too low'
            elif emp_asymm_2 > max_asymm_2:
                asymm_2_str = 'too high'

            emp_title_str = ''
            emp_title_str += 'Empirical copula - Gridded'
            emp_title_str += ('\n(asymm_1: %1.1E, asymm_2: %1.1E)' %
                              (emp_asymm_1, emp_asymm_2))
            emp_title_str += ('\n(asymm_1: %s, asymm_2: %s)' %
                              (asymm_1_str, asymm_2_str))

            ecop_grid_ax.set_title(emp_title_str)

            # Corresponding gaussian grid
            # TODO: adjust for precipitation case i.e. 0 and 1 ppt
            gau_cop_arr = bivar_gau_cop_arr(correl, self.cop_bins)
            _cb = gau_cop_ax.pcolormesh(
                x_mesh,
                y_mesh,
                gau_cop_arr,
                cmap=plt.get_cmap('Blues'),
                vmin=0,
                vmax=max_dens)
            gau_cop_ax.set_xticks(cop_ax_ticks)
            gau_cop_ax.set_xticklabels(cop_ax_labs)
            gau_cop_ax.set_yticks(cop_ax_ticks)
            gau_cop_ax.set_yticklabels(cop_ax_labs)
            gau_cop_ax.tick_params(
                labelleft=False,
                labelbottom=True,
                labeltop=False,
                labelright=False)
            gau_cop_ax.set_xlim(0, self.cop_bins)
            gau_cop_ax.set_ylim(0, self.cop_bins)

            gau_title_str = ''
            gau_title_str += 'Gaussian copula'
            gau_title_str += (('\n(min asymm_1: %1.1E, '
                               'max asymm_1: %1.1E)') % (
                                   min_asymm_1, max_asymm_1))
            gau_title_str += (('\n(min asymm_2: %1.1E, '
                               'max asymm_2: %1.1E)') % (
                                   min_asymm_2, max_asymm_2))
            gau_cop_ax.set_title(gau_title_str)

            leg_ax.set_axis_off()
            cb = plt.colorbar(_cb, cax=cax)
            cb.set_label('copula density')
            bounds = linspace(0, max_dens, 5)
            cb.set_ticks(bounds)
            cb.set_ticklabels(['%1.1E' % i_dens for i_dens in bounds])

            if self.max_time_lag_corr:
                curr_lag = curr_lag_ser[other_stn]

            else:
                curr_lag = 0

            title_str = ''
            title_str += ('Copula densities of stations: %s and %s' %
                          (infill_stn, other_stn))
            title_str += ('\nn = %d, corr = %0.3f, bins = %d' %
                          (prob_i.shape[0], correl, self.cop_bins))
            title_str += ('\n(rho: %0.3f, tau: %0.3f, lag: %d)' % (
                correl, tau, curr_lag))
            plt.suptitle(title_str)

            plt.subplots_adjust(hspace=0.15, wspace=1.5, top=0.75)

            out_ecop_fig_name = 'ecop_%s_vs_%s.%s' % (
                    infill_stn, other_stn, self.out_fig_fmt)
            out_ecop_fig_loc = os_join(self.ecops_dir, out_ecop_fig_name)

            plt.savefig(
                out_ecop_fig_loc,
                dpi=self.out_fig_dpi,
                bbox_inches='tight')

            ecop_raw_ax.cla()
            ecop_grid_ax.cla()
            leg_ax.cla()
            cax.cla()
        return

    def _plot_ecops(self, infill_stns):

        try:
            self.__plot_ecops(infill_stns)

        except:
            full_tb(exc_info(), self.dont_stop_flag)

        finally:
            plt.close('all')
        return

    def __plot_ecops(self, infill_stns):

        n_ticks = 6
        x_mesh, y_mesh = mgrid[0:self.cop_bins + 1, 0:self.cop_bins + 1]
        cop_ax_ticks = linspace(0, self.cop_bins, n_ticks)
        cop_ax_labs = np_round(linspace(0, 1., n_ticks, dtype='float'), 1)

        n_rows, n_cols = 1, 15 + 1

        plt.figure(figsize=(17, 6))
        ecop_raw_ax = plt.subplot2grid(
            (n_rows, n_cols),
            (0, 0),
            rowspan=1,
            colspan=5)

        ecop_grid_ax = plt.subplot2grid(
            (n_rows, n_cols),
            (0, 5),
            rowspan=1,
            colspan=5)

        gau_cop_ax = plt.subplot2grid(
            (n_rows, n_cols),
            (0, 10),
            rowspan=1,
            colspan=5)

        leg_ax = plt.subplot2grid(
            (n_rows, n_cols),
            (0, 15),
            rowspan=1,
            colspan=1)

        divider = make_axes_locatable(leg_ax)
        cax = divider.append_axes("left", size="100%", pad=0.05)

        if self.infill_type == 'discharge':
            dens_cnst = 0.5

        elif self.infill_type == 'precipitation':
            dens_cnst = 0.5

        else:
            assert False, as_err(
                'Incorrect \'infill_type\': %s!' % str(self.infill_type))

        if ((self.nrst_stns_type == 'rank') or
            (self.nrst_stns_type == 'symm')):
            self.curr_nrst_stns_dict = self.rank_corr_stns_dict

        elif self.nrst_stns_type == 'dist':
            self.curr_nrst_stns_dict = self.nrst_stns_dict

        else:
            assert False, as_err('Incorrect \'nrst_stns_type\': %s!' %
                                 str(self.nrst_stns_type))

        for infill_stn in infill_stns:
            self._plot_ecop(
                infill_stn,
                x_mesh,
                y_mesh,
                cop_ax_ticks,
                cop_ax_labs,
                ecop_raw_ax,
                ecop_grid_ax,
                gau_cop_ax,
                leg_ax,
                cax,
                dens_cnst)
        return


if __name__ == '__main__':
    pass
