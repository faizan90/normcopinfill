# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from numpy import (
    ceil,
    divide,
    unique,
    linspace,
    array,
    isnan,
    fabs,
    where,
    round as np_round)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap
from pandas import DataFrame

from ..misc.misc_ftns import as_err


class Summary:

    def __init__(self, norm_cop_obj):
        vars_list = ['summary_df',
                     '_av_vals_lab',
                     '_miss_vals_lab',
                     '_infilled_vals_lab',
                     '_max_avail_nebs_lab',
                     '_avg_avail_nebs_lab',
                     '_flagged_lab',
                     '_compr_lab',
                     '_bias_lab',
                     '_mae_lab',
                     '_rmse_lab',
                     '_nse_lab',
                     '_ln_nse_lab',
                     '_kge_lab',
                     '_pcorr_lab',
                     '_scorr_lab',
                     '_mean_obs_lab',
                     '_mean_infill_lab',
                     '_var_infill_lab',
                     'out_summary_fig',
                     '_var_obs_lab',
                     'flag_probs',
                     'n_max_nebs',
                     'n_min_nebs',
                     'ks_alpha',
                     '_infilled',
                     'out_fig_dpi',
                     'verbose']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.plot_summary()
        return

    def _plot_sub_summary(self,
                          curr_summary_df,
                          stn_idx,
                          font_size,
                          ks_lim_label,
                          r_2_g_cm,
                          r_2_g_colors,
                          g_2_r_cm,
                          clr_log_norm):

        colors_df = DataFrame(
            index=curr_summary_df.index,
            columns=curr_summary_df.columns,
            dtype=object)

        # ## available values
        avail_vals = curr_summary_df[self._av_vals_lab].copy()
        n_max_available = avail_vals.max(skipna=True)
        n_min_available = avail_vals.min(skipna=True)

        if isnan(n_max_available):
            print('WARNING: No summary to plot!')
            return

        if n_max_available == n_min_available:
            n_min_available = 0.0
        avail_vals[isnan(avail_vals)] = n_min_available

        curr_summary_df.loc[
            isnan(avail_vals), self._av_vals_lab] = n_min_available

        _avail_vals_rats = divide((avail_vals.values -
                                   n_min_available),
                                  (n_max_available -
                                   n_min_available))
        available_val_clrs = r_2_g_cm(_avail_vals_rats)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._av_vals_lab] = available_val_clrs[i]

        # ## missing values
        miss_vals = curr_summary_df[self._miss_vals_lab].copy()
        n_max_missing = max(1, miss_vals.max(skipna=True))
        n_min_missing = miss_vals.min(skipna=True)
        if n_max_missing == n_min_missing:
            n_min_missing = 0.0
        miss_vals[isnan(miss_vals)] = n_min_missing

        curr_summary_df.loc[
            isnan(miss_vals), self._miss_vals_lab] = n_min_missing

        _miss_val_rats = divide((miss_vals.values - n_min_missing),
                                (n_max_missing - n_min_missing))

        missing_val_clrs = g_2_r_cm(_miss_val_rats)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._miss_vals_lab] = missing_val_clrs[i]

        # ## infilled values
        infill_vals = curr_summary_df[self._infilled_vals_lab
                                      ].values.copy()
        infill_vals[isnan(infill_vals)] = 0.0
        _miss_vals_copy = miss_vals.values.copy()
        _miss_vals_copy[_miss_vals_copy == 0.0] = 1.0
        infilled_val_clrs = r_2_g_cm(divide(infill_vals, _miss_vals_copy))
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._infilled_vals_lab] = infilled_val_clrs[i]

        # ## max available neighbors
        max_avail_neb_vals = (
            curr_summary_df[self._max_avail_nebs_lab].values.copy())

        if self.n_max_nebs == self.n_min_nebs:
            min_nebs = 0.0
        else:
            min_nebs = self.n_min_nebs

        max_avail_neb_vals[isnan(max_avail_neb_vals)] = min_nebs
        _manvrs = divide((max_avail_neb_vals - min_nebs),
                         (self.n_max_nebs - min_nebs))
        max_nebs_clrs = r_2_g_cm(_manvrs)

        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._max_avail_nebs_lab] = max_nebs_clrs[i]

        # ## average used neighbors
        _aanrs = (
            divide(curr_summary_df[self._avg_avail_nebs_lab].values,
                   max_avail_neb_vals))
        _aanrs[isnan(_aanrs)] = 0.0
        avg_nebs_clrs = r_2_g_cm(_aanrs)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._avg_avail_nebs_lab] = avg_nebs_clrs[i]

        # ## compared
        compared_vals = curr_summary_df[self._compr_lab].values.copy()
        compared_vals[isnan(compared_vals)] = 0.0
        _ = curr_summary_df[self._infilled_vals_lab].values.copy()
        _[_ == 0.0] = 1.0
        _[isnan(_)] = 1.0
        compared_vals = divide(compared_vals, _)

        n_compare_clrs = r_2_g_cm(compared_vals)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._compr_lab] = n_compare_clrs[i]

        # ## KS limits
        for i, stn in enumerate(curr_summary_df.index):
            if (curr_summary_df.loc[stn, ks_lim_label] >=
                    (100 * (1.0 - self.ks_alpha))):
                ks_clr = r_2_g_colors[1]
            else:
                ks_clr = r_2_g_colors[0]

            colors_df.loc[stn, ks_lim_label] = ks_clr

        # ## flagged
        for i, stn in enumerate(colors_df.index):
            if (curr_summary_df.loc[stn, self._flagged_lab] >
                (100 * (1 - self.flag_probs[1] + self.flag_probs[0]))):
                flag_clr = r_2_g_colors[0]
            else:
                flag_clr = r_2_g_colors[1]

            colors_df.loc[stn, self._flagged_lab] = flag_clr

        # ## bias
        max_bias = curr_summary_df[self._bias_lab].max(skipna=True)
        min_bias = curr_summary_df[self._bias_lab].min(skipna=True)
        max_bias = max(abs(max_bias), abs(min_bias))
        if max_bias == 0.0:
            max_bias = 1.0
        if isnan(max_bias):
            max_bias = 1
        bias_vals = fabs(curr_summary_df[self._bias_lab].values)
        bias_vals[isnan(bias_vals)] = max_bias
        bias_vals = clr_log_norm(divide(bias_vals, (max_bias)))
        bias_clrs = g_2_r_cm(bias_vals.data)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._bias_lab] = bias_clrs[i]

        # ## mean absolute error
        max_mae = curr_summary_df[self._mae_lab].max(skipna=True)
        if max_mae == 0.0:
            max_mae = 1.0
        if isnan(max_mae):
            max_mae = 1.0
        mae_vals = fabs(curr_summary_df[self._mae_lab].values)
        mae_vals[isnan(mae_vals)] = max_mae
        mae_vals = clr_log_norm(divide(mae_vals, max_mae))
        mae_clrs = g_2_r_cm(mae_vals.data)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._mae_lab] = mae_clrs[i]

        # ## rmse
        max_rmse = curr_summary_df[self._rmse_lab].max(skipna=True)
        if max_rmse == 0.0:
            max_rmse = 1.0
        if isnan(max_rmse):
            max_rmse = 1.0
        rmse_vals = fabs(curr_summary_df[self._rmse_lab].values)
        rmse_vals[isnan(rmse_vals)] = max_rmse
        rmse_vals = clr_log_norm(divide(rmse_vals, max_rmse))
        rmse_clrs = g_2_r_cm(rmse_vals.data)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._rmse_lab] = rmse_clrs[i]

        # ## nse
        nse_vals = curr_summary_df[self._nse_lab].copy().values
        nse_vals[isnan(nse_vals)] = 0.0
        nse_clrs = r_2_g_cm(where(nse_vals < 0.0, 0, nse_vals))
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._nse_lab] = nse_clrs[i]

        # ## ln_nse
        ln_nse_vals = curr_summary_df[self._ln_nse_lab].copy().values
        ln_nse_vals[isnan(ln_nse_vals)] = 0.0
        ln_nse_clrs = r_2_g_cm(where(ln_nse_vals < 0.0, 0.0, ln_nse_vals))
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._ln_nse_lab] = ln_nse_clrs[i]

        # ## kge
        kge_vals = curr_summary_df[self._kge_lab].copy().values
        kge_vals[isnan(kge_vals)] = 0.0
        kge_clrs = r_2_g_cm(where(kge_vals < 0.0, 0.0, kge_vals))
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._kge_lab] = kge_clrs[i]

        # ## pcorr
        pcorr_vals = fabs(curr_summary_df[self._pcorr_lab].values)
        pcorr_vals[isnan(pcorr_vals)] = 0.0
        pcorr_clrs = r_2_g_cm(pcorr_vals)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._pcorr_lab] = pcorr_clrs[i]

        # ## scorr
        scorr_vals = fabs(curr_summary_df[self._scorr_lab].values)
        scorr_vals[isnan(scorr_vals)] = 0.0
        scorr_clrs = r_2_g_cm(scorr_vals)
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._scorr_lab] = scorr_clrs[i]

        # ## means and variances
        for i, stn in enumerate(colors_df.index):
            colors_df.loc[stn, self._mean_obs_lab] = r_2_g_colors[-1]
            colors_df.loc[stn, self._mean_infill_lab] = r_2_g_colors[-1]
            colors_df.loc[stn, self._var_obs_lab] = r_2_g_colors[-1]
            colors_df.loc[stn, self._var_infill_lab] = r_2_g_colors[-1]

        # ## plot the table
        _fs = (2 + (0.5 * curr_summary_df.shape[0]), 6)
        plt_fig = plt.figure(figsize=_fs)

        col_labs = curr_summary_df.index
        row_labs = curr_summary_df.columns

        table_text_list = np_round(curr_summary_df.values.T.astype(float), 3)
        table_text_list = table_text_list.astype(str)

        for i in range(table_text_list.shape[0]):
            for j in range(table_text_list.shape[1]):
                if table_text_list[i, j].endswith('.0'):
                    table_text_list[i, j] = table_text_list[i, j][:-2]

        row_colors = [[0.75] * 4] * row_labs.shape[0]
        col_colors = [[0.75] * 4] * col_labs.shape[0]

        table_ax = plt.table(cellText=table_text_list,
                             bbox=[0, 0, 1, 1],
                             rowLabels=row_labs,
                             colLabels=col_labs,
                             rowLoc='right',
                             colLoc='left',
                             cellLoc='right',
                             rowColours=row_colors,
                             colColours=col_colors,
                             cellColours=array(colors_df[:].T))

        table_ax.auto_set_font_size(False)
        table_ax.set_fontsize(font_size)

        # adjust header text if too wide
        renderer = plt_fig.canvas.get_renderer()
        table_cells = table_ax.get_celld()

        max_text_width = 0
        cell_tups = list(table_cells.keys())
        for cell_tup in cell_tups:
            if cell_tup[0] == 0:
                curr_text_width = table_cells[cell_tup].get_text()
                curr_text_width = curr_text_width.get_window_extent(renderer)
                curr_text_width = curr_text_width.width
                if curr_text_width > max_text_width:
                    max_text_width = curr_text_width

        table_cells = table_ax.get_celld()
        padding = table_cells[(0, 0)].PAD
        cell_width = (
            float(table_cells[(0, 0)].get_window_extent(renderer).width))
        cell_width = cell_width - (2. * padding * cell_width)

        new_font_size = font_size * divide(cell_width, max_text_width)

        if new_font_size < font_size:
            cell_tups = list(table_cells.keys())
            for cell_tup in cell_tups:
                if cell_tup[0] == 0:
                    table_cells[cell_tup].set_fontsize(new_font_size)

        plt.title('Normal Copula Infilling Summary', loc='right')
        plt.axis('off')
        out_fig = self.out_summary_fig + '_%0.2d' % (stn_idx + 1)
        plt.savefig(out_fig, dpi=self.out_fig_dpi, bbox_inches='tight')
        plt.close()
        return

    def plot_summary(self):
        '''
        Plot the summary_df as a table with formatting
        '''

        if self.verbose:
            print('INFO: Plotting summary table...')

        assert self._infilled, as_err('Call \'infill\' first!')

        clr_alpha = 0.4
        n_segs = 100
        font_size = 8
        max_columns_per_fig = 15

        cmap_name = 'my_clr_list'
        ks_lim_label = (('%%age values within %0.0f%% '
                         'KS-limits') % (100 * (1.0 - self.ks_alpha)))

        r_2_g_colors = [(1, 0, 0, clr_alpha), (0, 1, 0, clr_alpha)]  # R -> G
        r_2_g_cm = LinearSegmentedColormap.from_list(cmap_name,
                                                     r_2_g_colors,
                                                     N=n_segs)
        g_2_r_cm = LinearSegmentedColormap.from_list(
                   cmap_name, list(reversed(r_2_g_colors)), N=n_segs)

        clr_log_norm = LogNorm(0.001, 1)

        n_stns = self.summary_df.shape[0]

        if n_stns > max_columns_per_fig:
            n_figs = int(ceil(divide(n_stns, float(max_columns_per_fig))))

            cols_per_fig_idxs = unique(
                linspace(0, n_stns, n_figs + 1, dtype=int))

        else:
            cols_per_fig_idxs = array([0, n_stns])

        for stn_idx in range(cols_per_fig_idxs.shape[0] - 1):
            _1, _2 = cols_per_fig_idxs[stn_idx], cols_per_fig_idxs[stn_idx + 1]
            curr_summary_df = self.summary_df.iloc[_1:_2].copy()

            self._plot_sub_summary(curr_summary_df,
                                   stn_idx,
                                   font_size,
                                   ks_lim_label,
                                   r_2_g_cm,
                                   r_2_g_colors,
                                   g_2_r_cm,
                                   clr_log_norm)

        return


if __name__ == '__main__':
    pass
