'''
Created on Jul 27, 2018

@author: Faizan-Uni
'''
import timeit
from sys import exc_info
from os import mkdir as os_mkdir
from os.path import exists as os_exists, join as os_join

from numpy import (
    any as np_any,
    divide,
    isnan,
    linspace,
    logical_not,
    seterr,
    set_printoptions,
    where,
    float32)
import matplotlib.pyplot as plt

from pandas import (
    read_csv,
    to_datetime,
    to_numeric,
    DataFrame)

from .infill_steps import InfillSteps
from .plot_infill import PlotInfill
from .compare_infill import CompareInfill
from .flag_susp import FlagSusp
from ..misc.misc_ftns import pprt, as_err, full_tb, get_lag_ser

plt.ioff()
set_printoptions(
    precision=6,
    threshold=2000,
    linewidth=200000,
    formatter={'float': '{:0.6f}'.format})

seterr(all='ignore')


class InfillStation:

    def __init__(self, norm_cop_obj):

        vars_list = list(vars(norm_cop_obj).keys())

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))
#             print(_var, type(getattr(norm_cop_obj, _var)))

        # reduce overhead by not taking some of the objects
        # only the nebors are passed to _infill_stn upon pickling for MP
        self.in_var_df = None
        self.out_var_df = None
        self.summary_df = None
        self.flag_df = None

        self.in_coords_df = None
        self.xs = None
        self.ys = None
        self.full_date_index = None
        self.nrst_stns_list = None
        self.nrst_stns_dict = None
        self.rank_corr_stns_list = None
        self.rank_corr_stns_dict = None
        self.n_norm_symm_flds = None
        self.rank_corr_vals_ctr_df = None
        self.rank_corrs_df = None
        self.time_lags_df = None
        return

    def _infill_stn(self, args):

        (ii,
         infill_stn,
         in_var_df,
         time_lags_df,
         _parent) = args

        try:
            return self.__infill_stn(
                ii,
                infill_stn,
                in_var_df,
                time_lags_df,
                _parent)

        except:
            full_tb(exc_info(), self.dont_stop_flag)
            if not self.dont_stop_flag:
                raise Exception('Stop!')
        return

    def __infill_stn(
            self,
            ii,
            infill_stn,
            in_var_df,
            time_lags_df,
            _parent):

        curr_nrst_stns = list(in_var_df.columns[1:])

        if self.max_time_lag_corr:

            if _parent:
                # can't mess with the original one
                in_var_df = in_var_df.copy()

            for stn in curr_nrst_stns:
                _lag = time_lags_df[stn]
                in_var_df[stn] = get_lag_ser(in_var_df[stn], _lag)

            time_lags_df = None

        self.out_var_df = DataFrame(
            index=in_var_df.index, dtype=float32, columns=[infill_stn])
        self.summary_df = DataFrame(
            index=[infill_stn], columns=self._summary_cols, dtype=float32)

        if self.flag_susp_flag:
            self.flag_df = DataFrame(
                columns=[infill_stn], index=self.infill_dates, dtype=float32)

        if self.verbose:
            print('\n')
            pprt(nbh=4, msgs=[('Going through station %d of %d:  %s') %
                              (ii + 1, self.n_infill_stns, infill_stn)])

        self.in_var_df = in_var_df
        self.curr_nrst_stns = curr_nrst_stns
        self.curr_infill_stn = infill_stn

        self.stn_out_dir = os_join(self.indiv_stn_outs_dir, infill_stn)
        out_conf_df_file = (
            os_join(self.stn_out_dir,
                    'stn_%s_infill_conf_vals_df.csv' % infill_stn))
        out_add_info_file = os_join(
            self.stn_out_dir, 'add_info_df_stn_%s.csv' % infill_stn)

        # load infill
        no_out = True
        n_infilled_vals = 0
        if ((not self.overwrite_flag) and
            os_exists(out_conf_df_file) and
            os_exists(out_add_info_file)):

            if self.verbose:
                pprt(['Output exists already. Not overwriting.'], nbh=8)

            try:
                out_conf_df = read_csv(
                    out_conf_df_file,
                    sep=str(self.sep),
                    encoding='utf-8',
                    index_col=0,
                    engine='python')

                out_conf_df.index = to_datetime(
                    out_conf_df.index, format=self.time_fmt)

                out_add_info_df = read_csv(
                    out_add_info_file,
                    sep=str(self.sep),
                    encoding='utf-8',
                    index_col=0,
                    engine='python')

                out_add_info_df.index = to_datetime(
                    out_add_info_df.index, format=self.time_fmt)

                n_infilled_vals = out_conf_df.dropna().shape[0]

                _idxs = isnan(self.in_var_df.loc[
                        self.infill_dates, self.curr_infill_stn])

                _ser = self.in_var_df.loc[
                    self.infill_dates, self.curr_infill_stn]

                out_stn_ser = _ser.where(
                    logical_not(_idxs),
                    out_conf_df[self.fin_conf_head],
                    axis=0)

                self.out_var_df.loc[out_conf_df.index] = out_stn_ser

                no_out = False
            except Exception as msg:
                raise Exception(('Error while trying to read and '
                                 'update values from the existing '
                                 'dataframe:\n' + msg))

        self.summary_df.loc[self.curr_infill_stn, self._av_vals_lab] = (
            self.in_var_df[self.curr_infill_stn].dropna().shape[0])

        if self.compare_infill_flag:
            nan_idxs = list(range(self.infill_dates.shape[0]))

        else:
            nan_idxs = where(isnan(self.in_var_df.loc[
                self.infill_dates, self.curr_infill_stn].values))[0]

        n_nan_idxs = len(nan_idxs)
        if (n_infilled_vals < n_nan_idxs and (not no_out)):
            # it can happen that not all steps were infilled
            if self.verbose:
                pprt(['Existing output has insufficient infilled values!'],
                     nbh=8)
            no_out = True

        self.summary_df.loc[
            self.curr_infill_stn, self._miss_vals_lab] = n_nan_idxs

        if (self.overwrite_flag or no_out):
            if (n_nan_idxs == 0) and (not self.compare_infill_flag):
                return

            # mkdirs
            dir_list = [self.stn_out_dir]

            self.stn_infill_cdfs_dir = os_join(
                self.stn_out_dir, 'stn_infill_cdfs')

            self.stn_infill_pdfs_dir = os_join(
                self.stn_out_dir, 'stn_infill_pdfs')

            self.stn_step_cdfs_dir = os_join(
                self.stn_out_dir, 'stn_step_cdfs')

            self.stn_step_corrs_dir = os_join(
                self.stn_out_dir, 'stn_step_corrs')

            self.stn_step_vars_dir = os_join(
                self.stn_out_dir, 'stn_step_vars')

            if self.plot_step_cdf_pdf_flag:
                dir_list.extend([
                    self.stn_infill_cdfs_dir, self.stn_infill_pdfs_dir])

            if self.plot_diag_flag:
                dir_list.extend([
                    self.stn_step_cdfs_dir, self.stn_step_corrs_dir])

            if self.save_step_vars_flag:
                dir_list.extend([self.stn_step_vars_dir])

            for _dir in dir_list:
                if not os_exists(_dir):
                    os_mkdir(_dir)

            idxs = linspace(
                0,
                n_nan_idxs,
                self.ncpus + 1,
                endpoint=True,
                dtype='int64')

            if self.verbose:
                infill_start = timeit.default_timer()
                pprt(['%d steps to infill' % n_nan_idxs], nbh=8)
                pprt(['Neighbors are:'], nbh=8)

                for i_msg in range(0, len(self.curr_nrst_stns), 3):
                    pprt(self.curr_nrst_stns[i_msg:(i_msg + 3)], nbh=12)

            # initiate infill
            out_conf_df = DataFrame(
                index=self.infill_dates,
                columns=self.conf_ser.index,
                dtype=float32)

            out_add_info_df = DataFrame(
                index=self.infill_dates,
                dtype=float32,
                columns=['infill_status',
                         'n_neighbors_raw',
                         'n_neighbors_fin',
                         'act_val_prob'])

            if ((n_nan_idxs > self.thresh_mp_steps) and
                not self.stn_based_mp_infill_flag):

                use_mp_infill = True

            else:
                use_mp_infill = False

            self.infill_steps_obj = InfillSteps(self)

            if ((idxs.shape[0] == 1) or
                (self.ncpus == 1) or
                (not use_mp_infill) or
                self.debug_mode_flag):

                sub_dfs = [self._infill(self.infill_dates[nan_idxs])]

            else:
                n_sub_dates = 0
                sub_infill_dates_list = []
                for idx in range(self.ncpus):
                    sub_dates = (
                        self.infill_dates[nan_idxs[idxs[idx]:idxs[idx + 1]]])

                    sub_infill_dates_list.append(sub_dates)

                    n_sub_dates += sub_dates.shape[0]

                assert n_sub_dates == n_nan_idxs, (
                    as_err(('\'n_sub_dates\' (%d) and '
                            '\'self.infill_dates\' '
                            '(%d) of unequal length!') %
                           (n_sub_dates, self.infill_dates.shape[0])))

                try:
                    sub_dfs = list(self._norm_cop_pool.uimap(
                        self._infill, sub_infill_dates_list))
                    self._norm_cop_pool.clear()

                except Exception as msg:
                    self._norm_cop_pool.close()
                    self._norm_cop_pool.join()
                    raise Exception('MP failed 1: %s!' % msg)

            for sub_df in sub_dfs:
                sub_conf_df = sub_df[0]
                sub_add_info_df = sub_df[1]

                _ser = self.in_var_df.loc[
                    sub_conf_df.index, self.curr_infill_stn]

                _idxs = isnan(_ser)
                _idxs = logical_not(_idxs)

                sub_stn_ser = (_ser.where(
                    _idxs,
                    sub_conf_df[self.fin_conf_head],
                    axis=0)).copy()

                self.out_var_df.update(sub_stn_ser, overwrite=False)

                out_conf_df.update(sub_conf_df, overwrite=False)
                out_add_info_df.update(
                    sub_add_info_df.astype(float32), overwrite=False)

            n_infilled_vals = out_conf_df.dropna().shape[0]

            if self.verbose:
                pprt([('%d steps out of %d infilled' %
                       (n_infilled_vals, n_nan_idxs))],
                     nbh=8)

                infill_stop = timeit.default_timer()
                fin_secs = infill_stop - infill_start

                _ = divide(fin_secs, max(1, n_infilled_vals))
                pprt([(('Took %0.3f secs, %0.3e secs per '
                        'step') % (fin_secs, _))],
                     nbh=8)

            # ## prepare output
            out_conf_df = out_conf_df.apply(lambda x: to_numeric(x))
            out_conf_df.to_csv(
                out_conf_df_file,
                sep=str(self.sep),
                encoding='utf-8')

            out_add_info_df.to_csv(
                out_add_info_file, sep=str(self.sep), encoding='utf-8')

        self.summary_df.loc[
            self.curr_infill_stn, self._infilled_vals_lab] = n_infilled_vals

        _ = len(self.curr_nrst_stns)
        self.summary_df.loc[
            self.curr_infill_stn, self._max_avail_nebs_lab] = _

        _ = round(out_add_info_df['n_neighbors_fin'].dropna().mean(), 1)
        self.summary_df.loc[
            self.curr_infill_stn, self._avg_avail_nebs_lab] = _

        # make plots
        # plot number of gauges available and used
        if self.verbose:
            infill_start = timeit.default_timer()

        nebs_used_per_step_file = os_join(
            self.stn_out_dir, 'stns_used_infill_%s.png' % infill_stn)

        if self.plot_used_stns_flag and (
            self.overwrite_flag or
            no_out or
            (not os_exists(nebs_used_per_step_file))):

            lw = 0.8
            alpha = 0.7

            plt.figure(figsize=self.fig_size_long)
            infill_ax = plt.subplot(111)

            infill_ax.plot(
                self.infill_dates,
                out_add_info_df['n_neighbors_raw'].values,
                label='n_neighbors_raw',
                c='r',
                alpha=alpha,
                lw=lw + 0.5,
                ls='-')

            infill_ax.plot(
                self.infill_dates,
                out_add_info_df['n_neighbors_fin'].values,
                label='n_neighbors_fin',
                c='b',
                marker='o',
                lw=0,
                ms=2)

            infill_ax.set_xlabel('Time')
            infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
            infill_ax.set_ylabel('Stations used')
            infill_ax.set_ylim(-1, infill_ax.get_ylim()[1] + 1)

            plt.suptitle(('Number of raw available and finally used stations '
                          'for infilling at station: %s') % infill_stn)
            plt.grid()
            plt.legend(framealpha=0.5, loc=0)
            plt.savefig(
                nebs_used_per_step_file,
                dpi=self.out_fig_dpi,
                bbox_inches='tight')
            plt.close('all')

        # the original unfilled series of the infilled station
        act_var = self.in_var_df.loc[self.infill_dates, infill_stn].values

        # plot the infilled series
        plot_infill_cond = self.plot_stn_infill_flag

        if not self.overwrite_flag:
            plot_infill_cond = (plot_infill_cond and no_out)

        use_mp = not (
            self.debug_mode_flag or
            self.stn_based_mp_infill_flag or
            (self.ncpus == 1))

        compare_iter = None
        flag_susp_iter = None

        if plot_infill_cond:
            if self.verbose:
                pprt(['Plotting infill...'], nbh=8)

            out_infill_plot_loc = os_join(
                self.stn_out_dir, 'missing_infill_%s.png' % infill_stn)

            args_tup = (self, act_var, out_conf_df, out_infill_plot_loc)
            if use_mp:
                self._norm_cop_pool.uimap(PlotInfill, (args_tup,))
            else:
                PlotInfill(args_tup)

        # plot the comparison between the actual and infilled series
        if self.compare_infill_flag and np_any(act_var):
            if self.verbose:
                pprt(['Plotting infill comparison...'], nbh=8)

            if (not no_out) and (not self.overwrite_flag):
                update_summary_df_only = True

            else:
                update_summary_df_only = False

            out_compar_plot_loc = os_join(
                self.stn_out_dir, 'compare_infill_%s.png' % infill_stn)

            args_tup = (
                act_var,
                out_conf_df,
                out_compar_plot_loc,
                out_add_info_df,
                update_summary_df_only)

            compare_obj = CompareInfill(self)
            if use_mp and (not update_summary_df_only):
                compare_iter = self._norm_cop_pool.uimap(
                    compare_obj.plot_all, (args_tup,))
            else:
                self.summary_df.update(
                    compare_obj.plot_all(args_tup).astype(float32),
                    overwrite=False)

        else:
            if self.verbose:
                pprt(['Nothing to compare...'], nbh=8)

        # plot steps showing if the actual data is within the bounds
        out_flag_susp_loc = os_join(
            self.stn_out_dir, 'flag_infill_%s.png' % infill_stn)

        if self.flag_susp_flag and np_any(act_var):
            if self.verbose:
                pprt(['Plotting infill flags...'], nbh=8)

            if (not no_out) and (not self.overwrite_flag):
                update_summary_df_only = True

            else:
                update_summary_df_only = False

            flag_susp_obj = FlagSusp(self)
            args_tup = (
                act_var,
                out_conf_df,
                out_flag_susp_loc,
                update_summary_df_only)

            if use_mp and (not update_summary_df_only):
                flag_susp_iter = self._norm_cop_pool.uimap(
                    flag_susp_obj.plot, (args_tup,))

            else:
                _ = flag_susp_obj.plot(args_tup)
                self.summary_df.update(_[0], overwrite=False)
                self.flag_df.update(_[1], overwrite=False)

        else:
            if self.verbose:
                pprt(['Nothing to flag...'], nbh=8)

        if use_mp:
            self._norm_cop_pool.clear()

        if self.verbose:
            infill_stop = timeit.default_timer()
            fin_secs = infill_stop - infill_start
            pprt([('Took %0.3f secs for plotting other stuff') % fin_secs],
                 nbh=8)

        # these iters are here to make use of mp
        if compare_iter:
            self.summary_df.update(list(compare_iter)[0], overwrite=False)

        if flag_susp_iter:
            _ = list(flag_susp_iter)[0]
            self.summary_df.update(_[0], overwrite=False)
            self.flag_df.update(_[1], overwrite=False)

        return (self.summary_df, self.flag_df, self.out_var_df)

    def _infill(self, infill_dates):

        try:
            (out_conf_df, out_add_info_df) = (
                 self.infill_steps_obj.infill_steps(infill_dates))

            return out_conf_df, out_add_info_df

        except:
            plt.close('all')
            full_tb(exc_info(), self.dont_stop_flag)
            if not self.dont_stop_flag:
                raise Exception('_infill failed!')

            return (DataFrame(index=infill_dates,
                              columns=self.conf_ser.index,
                              dtype=float32),

                    DataFrame(index=infill_dates,
                              dtype=float32,
                              columns=['infill_status',
                                       'n_neighbors_raw',
                                       'n_neighbors_fin',
                                       'act_val_prob']))
        return
