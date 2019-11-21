"""
Created on %(date)s

@author: %(username)s
"""
from pickle import dump, load
from os import mkdir as os_mkdir
from os.path import exists as os_exists, join as os_join

from numpy import (
    full,
    nan,
    all as np_all,
    isfinite,
    isnan,
    where,
    abs as np_abs,
    float32,
    float16)
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pandas import DataFrame, Series

from ..misc.misc_ftns import (
    get_norm_rand_symms, get_lag_ser, as_err, ret_mp_idxs)
from ..cyth import get_asymms_sample, get_corrcoeff

plt.ioff()


class RankCorrStns:

    def __init__(self, norm_cop_obj):

        vars_list = [
            'in_var_df',
            'infill_stns',
            'infill_dates',
            'full_date_index',
            'in_coords_df',
            'nrst_stns_dict',
            'nrst_stns_list',
            'rank_corr_stns_list',
            'rank_corr_stns_dict',
            '_dist_cmptd',
            'verbose',
            'read_pickles_flag',
            'debug_mode_flag',
            'force_infill_flag',
            'dont_stop_flag',
            'plot_neighbors_flag',
            'plot_long_term_corrs_flag',
            '_rank_corr_cmptd',
            'nrst_stns_type',
            '_norm_cop_pool',
            'min_valid_vals',
            'min_corr',
            'infill_type',
            'cut_cdf_thresh',
            'ncpus',
            'n_norm_symm_flds',
            '_max_symm_rands',
            'n_min_nebs',
            'n_max_nebs',
            'out_rank_corr_plots_dir',
            '_out_rank_corr_stns_pkl_file',
            'out_long_term_corrs_dir',
            'out_fig_dpi',
            'max_time_lag_corr',
            'pkls_dir',
            'xs',
            'ys']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

#         self.in_var_df = self.in_var_df.copy()

        assert self._dist_cmptd, 'Call NrstStns first!'

        self.drop_infill_stns = []
        self.bad_stns_list = []
        self.bad_stns_neighbors_count = []

        self._got_rank_corr_stns = False
        self._plotted_rank_corr_stns = False
        self._plotted_long_term_rank_corrs = False

        if (self.nrst_stns_type == 'symm'):
            norm_cop_obj._get_ncpus()
            self._norm_cop_pool = norm_cop_obj._norm_cop_pool

        if not os_exists(self.pkls_dir):
            os_mkdir(self.pkls_dir)

        if self.max_time_lag_corr:
            self.in_var_df = self.in_var_df.reindex(self.full_date_index)

        self.get_rank_corr_stns()
        assert self._got_rank_corr_stns

        if self.plot_neighbors_flag:
            self.plot_neighbors()
            assert self._plotted_rank_corr_stns

        if self.plot_long_term_corrs_flag:
            self.plot_long_term_corrs()
            assert self._plotted_long_term_rank_corrs

        if self._norm_cop_pool is not None:
            self._norm_cop_pool.clear()

        setattr(norm_cop_obj, '_rank_corr_cmptd', True)
        setattr(norm_cop_obj, 'in_var_df', self.in_var_df)
        setattr(norm_cop_obj, 'infill_stns', self.infill_stns)
        setattr(norm_cop_obj, 'rank_corr_stns_dict', self.rank_corr_stns_dict)
        setattr(norm_cop_obj, 'rank_corr_stns_list', self.rank_corr_stns_list)
        setattr(norm_cop_obj, 'n_infill_stns', self.n_infill_stns)
        setattr(norm_cop_obj, 'rank_corrs_df', self.rank_corrs_df)
        setattr(
            norm_cop_obj, 'rank_corr_vals_ctr_df', self.rank_corr_vals_ctr_df)

        setattr(norm_cop_obj, 'time_lags_df', self.time_lags_df)
        return

    def get_rank_corr_stns(self):

        if (os_exists(self._out_rank_corr_stns_pkl_file) and
            self.read_pickles_flag):

            self._load_pickle()
            return

        if self.verbose:
            print('INFO: Computing highest correlation stations...')

        self._get_rank_corrs()

        self.rank_corr_vals_ctr_df.dropna(axis=0, how='all', inplace=True)
        self.rank_corr_vals_ctr_df.dropna(axis=1, how='all', inplace=True)

        for infill_stn in self.infill_stns:
            if infill_stn in self.rank_corrs_df.index:
                continue

            if self.dont_stop_flag:
                print(('WARNING: the station %s is not in '
                       '\'rank_corrs_df\' and is removed from the '
                       'infill stations\' list!') % infill_stn)

                if infill_stn not in self.drop_infill_stns:
                    self.drop_infill_stns.append(infill_stn)

            else:
                raise KeyError(
                    'The station %s is not in \'rank_corrs_df\' anymore!' %
                    infill_stn)

        for drop_stn in self.drop_infill_stns:
            self.infill_stns = self.infill_stns.drop(drop_stn)

        if not self.infill_stns.shape[0]:
            raise Exception('No stations to work with!')

        else:
            self.n_infill_stns = self.infill_stns.shape[0]

        for infill_stn in self.infill_stns:
            self._get_best_rank_corr(infill_stn)

        for bad_stn in self.bad_stns_list:
            self.infill_stns = self.infill_stns.drop(bad_stn)
            del self.rank_corr_stns_dict[bad_stn]

        self.rank_corrs_df.drop(
            labels=self.bad_stns_list, axis=0, inplace=True)

        self.rank_corr_vals_ctr_df.drop(
            labels=self.bad_stns_list, axis=0, inplace=True)

        if self.verbose and self.bad_stns_list:
            print('INFO: These infill station(s) had too few values'
                  'to be considered for the rest of the analysis:')
            print('Station: n_neighbors')

            for bad_stn in zip(self.bad_stns_list,
                               self.bad_stns_neighbors_count):
                print('%s: %d' % (bad_stn[0], bad_stn[1]))

        # have the rank_corr_stns_list in the in_var_df only
        _drop_stns = self.in_var_df.columns.difference(
            self.rank_corr_stns_list)
        self.in_var_df.drop(_drop_stns, axis=1, inplace=True)
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        if not self.in_var_df.shape[0]:
            raise Exception('No dates to work with!')

        if not self.dont_stop_flag:
            for infill_stn in self.infill_stns:
                assert infill_stn in self.in_var_df.columns, (
                    as_err('infill station %s not in input variable '
                           'dataframe anymore!' % infill_stn))

        # check if at least one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, as_err(
            'None of the infill dates exist in \'in_var_df\' '
            'after dropping stations and records with '
            'insufficient information!')

        if self.verbose:
            print(('INFO: \'in_var_df\' shape after calling '
                   '\'cmpt_plot_rank_corr_stns\':'), self.in_var_df.shape)

        # ## save pickle
        rank_corr_stns_pkl_dict = {}
        rank_corr_stns_pkl_cur = open(self._out_rank_corr_stns_pkl_file, 'wb')

        rank_corr_stns_pkl_dict['in_var_df'] = self.in_var_df
        rank_corr_stns_pkl_dict['infill_stns'] = self.infill_stns
        rank_corr_stns_pkl_dict['rank_corr_stns_dict'] = (
            self.rank_corr_stns_dict)
        rank_corr_stns_pkl_dict['rank_corr_stns_list'] = (
            self.rank_corr_stns_list)
        rank_corr_stns_pkl_dict['n_infill_stns'] = self.n_infill_stns
        rank_corr_stns_pkl_dict['rank_corrs_df'] = self.rank_corrs_df
        rank_corr_stns_pkl_dict['rank_corr_vals_ctr_df'] = (
            self.rank_corr_vals_ctr_df)

        rank_corr_stns_pkl_dict['time_lags_df'] = (
            self.time_lags_df)

        dump(rank_corr_stns_pkl_dict, rank_corr_stns_pkl_cur)
        rank_corr_stns_pkl_cur.close()

        self._got_rank_corr_stns = True
        return

    def plot_neighbors(self):

        if not os_exists(self.out_rank_corr_plots_dir):
            os_mkdir(self.out_rank_corr_plots_dir)

        if self.verbose:
            print('INFO: Plotting hi-correlation neighbors')

        plt_gen = (
            (_stn,
             self.in_coords_df[['X']].loc[_stn].values,
             self.in_coords_df[['Y']].loc[_stn].values,
             self.rank_corr_stns_dict[_stn],
             self.in_coords_df['X'].loc[self.rank_corr_stns_dict[_stn]],
             self.in_coords_df['Y'].loc[self.rank_corr_stns_dict[_stn]],
             self.rank_corrs_df.loc[_stn].dropna(),
             self.xs,
             self.ys,
             self.out_rank_corr_plots_dir,
             self.out_fig_dpi,
             self.time_lags_df.loc[_stn])

            for _stn in self.infill_stns)

        _plt_ftn = RankCorrStns._plot_hi_corr_stns

        mp = True if self.ncpus > 1 else False
        if mp:
            self._norm_cop_pool.map(_plt_ftn, plt_gen)
            self._norm_cop_pool.clear()

        else:
            [_plt_ftn(args) for args in plt_gen]

        self._plotted_rank_corr_stns = True
        return

    def plot_long_term_corrs(self):

        if self.verbose:
            print('INFO: Plotting long-term correlation neighbors')

        if not os_exists(self.out_long_term_corrs_dir):
            os_mkdir(self.out_long_term_corrs_dir)

        _rcorr_dict = self.rank_corr_stns_dict

        plt_gen = (
            (_stn,
             self.rank_corr_stns_dict[_stn],
             self.rank_corrs_df.loc[_stn, _rcorr_dict[_stn]],
             self.rank_corr_vals_ctr_df.loc[_stn, _rcorr_dict[_stn]],
             self.out_long_term_corrs_dir,
             self.out_fig_dpi,
             self.time_lags_df.loc[_stn])

            for _stn in self.infill_stns)

        _plt_ftn = RankCorrStns._plot_long_term_corr

        mp = True if self.ncpus > 1 else False
        if mp:
            self._norm_cop_pool.map(_plt_ftn, plt_gen)
            self._norm_cop_pool.clear()

        else:
            [_plt_ftn(args) for args in plt_gen]

        self._plotted_long_term_rank_corrs = True
        return

    def _load_pickle(self):

        if self.verbose:
            print('INFO: Loading rank correlations pickle...')

        rank_corr_stns_pkl_cur = open(self._out_rank_corr_stns_pkl_file, 'rb')
        rank_corr_stns_pkl_dict = load(rank_corr_stns_pkl_cur)

        self.in_var_df = rank_corr_stns_pkl_dict['in_var_df']
        self.infill_stns = rank_corr_stns_pkl_dict['infill_stns']
        self.rank_corr_stns_dict = (
            rank_corr_stns_pkl_dict['rank_corr_stns_dict'])
        self.rank_corr_stns_list = (
            rank_corr_stns_pkl_dict['rank_corr_stns_list'])
        self.n_infill_stns = rank_corr_stns_pkl_dict['n_infill_stns']
        self.rank_corrs_df = rank_corr_stns_pkl_dict['rank_corrs_df']
        self.rank_corr_vals_ctr_df = (
            rank_corr_stns_pkl_dict['rank_corr_vals_ctr_df'])

        self.time_lags_df = rank_corr_stns_pkl_dict['time_lags_df']

        rank_corr_stns_pkl_cur.close()

        self._got_rank_corr_stns = True
        return

    def _get_neb_subset(self):
        mp_idxs = ret_mp_idxs(len(self.infill_stns), self.ncpus)

        for i in range(mp_idxs.shape[0] - 1):
            i_stns = self.infill_stns[mp_idxs[i]: mp_idxs[i + 1]]

            i_all_stns = []
            i_nrst_stns_dict = {}

            for i_stn in i_stns:
                if i_stn not in i_all_stns:
                    i_all_stns.append(i_stn)

                i_nrst_stns_dict[i_stn] = self.nrst_stns_dict[i_stn]

                self.poss_n_corrs += len(i_nrst_stns_dict[i_stn])

                for j_stn in self.nrst_stns_dict[i_stn]:
                    if j_stn in i_all_stns:
                        continue

                    i_all_stns.append(j_stn)

            yield (
                i_stns,
                self.in_var_df[i_all_stns],
                i_nrst_stns_dict,
                self.rank_corrs_df.loc[i_stns],
                self.rank_corr_vals_ctr_df.loc[i_stns])
        return

    def _get_rank_corrs(self):

        self.rank_corrs_df = DataFrame(
            index=self.infill_stns,
            columns=self.in_var_df.columns,
            dtype=float32)

        self.rank_corr_vals_ctr_df = DataFrame(
            index=self.infill_stns,
            columns=self.in_var_df.columns,
            dtype=float16)

        if self.max_time_lag_corr:
            self.time_lags_df = self.rank_corr_vals_ctr_df.copy()
            # this somehow does not show the copy-view warning
            self.time_lags_df[:] = 0.0

        else:
            self.time_lags_df = Series(index=self.infill_stns, dtype=float16)
            self.time_lags_df[:] = ''

        # this somehow does not show the copy-view warning
        self.rank_corr_vals_ctr_df[:] = 0.0

        drop_str = ('Station %s has no neighbors that satisfy the \'%s\' '
                    'criteria and is therfore dropped!')

        self.n_infill_stns = self.infill_stns.shape[0]

        rank_corr_obj = RankCorr(self)

        self.poss_n_corrs = 0

        mp = True if self.ncpus > 1 else False
        if mp:
            mp_ress = list(self._norm_cop_pool.imap(
                rank_corr_obj._get_rank_corrs_ctr_df, self._get_neb_subset()))
            self._norm_cop_pool.clear()

        else:
            mp_ress = []
            for i, mp_var_list in enumerate(self._get_neb_subset()):
                mp_ress.append(
                    rank_corr_obj._get_rank_corrs_ctr_df(mp_var_list))

        for i, mp_res in enumerate(mp_ress):
            self.rank_corrs_df.update(mp_res[0], overwrite=True)
            self.rank_corr_vals_ctr_df.update(mp_res[1], overwrite=True)

            if self.max_time_lag_corr:
                self.time_lags_df.update(mp_res[2], overwrite=True)

            mp_ress[i] = None

        mp_res = None

        self.tot_corrs_written = (~isnan(self.rank_corrs_df.values)).sum()

        for i_stn in self.infill_stns:
            if self.rank_corrs_df.loc[i_stn].dropna().shape[0]:
                continue

#             if self.dont_stop_flag:
            self.drop_infill_stns.append(i_stn)

            if self.verbose:
                print('WARNING:',
                      drop_str % (i_stn, self.nrst_stns_type))
            continue

#             else:
#                 raise Exception(drop_str % (i_stn, self.nrst_stns_type))

        if self.verbose:
            print('INFO: %d out of possible %d correlations written' %
                  (self.tot_corrs_written, self.poss_n_corrs))
        return

    def _get_best_rank_corr(self, infill_stn):

        if infill_stn not in self.rank_corr_stns_list:
            self.rank_corr_stns_list.append(infill_stn)

        stn_correl_ser = self.rank_corrs_df.loc[infill_stn].dropna().copy()

        if stn_correl_ser.shape[0] == 0:
            raise Exception('No neighbors for station %s!' % infill_stn)

        stn_correl_ser[:] = where(
            isfinite(stn_correl_ser[:]),
            np_abs(stn_correl_ser[:]),
            nan)

        stn_correl_ser.sort_values(axis=0, ascending=False, inplace=True)

        # take the nearest n_nrn stations to the infill_stn
        for rank_corr_stn in stn_correl_ser.index:
            if rank_corr_stn in self.rank_corr_stns_list:
                continue

            self.rank_corr_stns_list.append(rank_corr_stn)

        # put the neighboring stations in a dictionary for each infill_stn
        self.rank_corr_stns_dict[infill_stn] = (
            stn_correl_ser.index[:self.n_max_nebs].tolist())

        curr_n_neighbors = len(self.rank_corr_stns_dict[infill_stn])
        if ((curr_n_neighbors < self.n_min_nebs) and (
            not self.force_infill_flag)):

            self.bad_stns_list.append(infill_stn)
            self.bad_stns_neighbors_count.append(curr_n_neighbors)

        if not self.force_infill_flag:
            assert curr_n_neighbors >= self.n_min_nebs, (
                as_err(('Rank correlation stations (n=%d) less than '
                        '\'n_min_nebs\' or no neighbor for '
                        'station: %s') % (curr_n_neighbors, infill_stn)))
        return

    @staticmethod
    def _plot_hi_corr_stns(args):

        (infill_stn,
         infill_x,
         infill_y,
         nebs,
         x_nebs,
         y_nebs,
         corrs,
         xs,
         ys,
         out_rank_corr_plots_dir,
         out_fig_dpi,
         lags) = args

        tick_font_size = 7
        n_nebs = len(nebs)

        hi_corr_stns_ax = plt.subplot(111)

        hi_corr_stns_ax.scatter(
            x_nebs,
            y_nebs,
            alpha=0.75,
            c='c',
            label='hi_corr_stn (%d)' % n_nebs)

        hi_corr_stns_ax.scatter(
            infill_x,
            infill_y,
            c='r',
            label='infill_stn')

        plt_texts = []
        for stn in nebs:
            if len(lags):
                _lab = '%s\n(%0.4f, %d)' % (stn, corrs[stn], lags[stn])

            else:
                _lab = '%s\n(%0.4f)' % (stn, corrs[stn])

            if infill_stn == stn:
                continue

            _txt_obj = hi_corr_stns_ax.text(
                x_nebs[stn],
                y_nebs[stn],
                _lab,
                va='top',
                ha='left',
                fontsize=tick_font_size)
            plt_texts.append(_txt_obj)

        _txt_obj = hi_corr_stns_ax.text(
            infill_x,
            infill_y,
            infill_stn,
            va='top',
            ha='left',
            fontsize=tick_font_size)
        plt_texts.append(_txt_obj)

        adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})
        hi_corr_stns_ax.grid()
        hi_corr_stns_ax.set_xlabel('Eastings')
        hi_corr_stns_ax.set_ylabel('Northings')
        hi_corr_stns_ax.legend(framealpha=0.5, loc=0)

        hi_corr_stns_ax.set_xlim(0.999 * xs.min(), 1.001 * xs.max())
        hi_corr_stns_ax.set_ylim(0.999 * ys.min(), 1.001 * ys.max())

        plt.setp(hi_corr_stns_ax.get_xticklabels(), rotation=90)

        plt.savefig(
            os_join(out_rank_corr_plots_dir,
                    'rank_corr_stn_%s.png' % infill_stn),
            dpi=out_fig_dpi,
            bbox_inches='tight')

        plt.close()
        return

    @staticmethod
    def _plot_long_term_corr(args):

        (infill_stn,
         nebs,
         corrs_ser,
         corrs_ctr_ser,
         out_long_term_corrs_dir,
         out_fig_dpi,
         lags) = args

        # TODO: have 15 stns at max on a row
        tick_font_size = 8

        corrs_ctr_ser[isnan(corrs_ctr_ser)] = 0

        n_stns = corrs_ser.shape[0]
        _, corrs_ax = plt.subplots(1, 1, figsize=(1.0 * n_stns, 3))

        corrs_ax.matshow(
            corrs_ser.values.reshape(1, n_stns),
            vmin=0,
            vmax=2,
            cmap=plt.get_cmap('Blues'),
            origin='lower')

        for s, neb in enumerate(nebs):
            if len(lags):
                corrs_ax.text(
                    s,
                    0,
                    '%0.4f\n(%d, %d)' % (
                        corrs_ser[neb],
                        int(corrs_ctr_ser[neb]),
                        int(lags[neb])),
                    va='center',
                    ha='center',
                    fontsize=tick_font_size,
                    rotate=45)

            else:
                corrs_ax.text(
                    s,
                    0,
                    '%0.4f\n(%d)' % (
                        corrs_ser[neb], int(corrs_ctr_ser[neb])),
                    va='center',
                    ha='center',
                    fontsize=tick_font_size,
                    rotate=45)

        corrs_ax.set_yticks([])
        corrs_ax.set_yticklabels([])
        corrs_ax.set_xticks(list(range(0, n_stns)))
        corrs_ax.set_xticklabels(nebs)

        corrs_ax.spines['left'].set_position(('outward', 10))
        corrs_ax.spines['right'].set_position(('outward', 10))
        corrs_ax.spines['top'].set_position(('outward', 10))
        corrs_ax.spines['bottom'].set_position(('outward', 10))

        corrs_ax.tick_params(
            labelleft=False,
            labelbottom=True,
            labeltop=False,
            labelright=False)

        plt.setp(corrs_ax.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.suptitle('station: %s long-term corrs' % infill_stn)

        _ = 'long_term_stn_%s_rank_corrs.png' % infill_stn
        plt.savefig(
            os_join(out_long_term_corrs_dir, _),
            dpi=out_fig_dpi,
            bbox_inches='tight')
        plt.close()
        return


class RankCorr:

    def __init__(self, rank_corr_stns_obj):

        vars_list = [
            'infill_type',
            'cut_cdf_thresh',
            'nrst_stns_type',
            'min_corr',
            'n_norm_symm_flds',
            '_max_symm_rands',
            'min_valid_vals',
            'max_time_lag_corr']

        for _var in vars_list:
            setattr(self, _var, getattr(rank_corr_stns_obj, _var))

        self.in_var_df = None
        self.nrst_stns_dict = None
        return

    def _get_rank_corrs_ctr_df(self, args):

        (infill_stns,
         self.in_var_df,
         self.nrst_stns_dict,
         self.rank_corrs_df,
         self.rank_corr_vals_ctr_df) = args

        if self.max_time_lag_corr:
            self.time_lags_df = DataFrame(
                index=infill_stns,
                columns=self.in_var_df.columns,
                dtype=float32)

        else:
            self.time_lags_df = Series(index=infill_stns, dtype=float32)

        for i_stn in infill_stns:
            self._get_rank_corr(i_stn)

        # free memory
        self.in_var_df = None
        self.nrst_stns_dict = None

        return (
            self.rank_corrs_df,
            self.rank_corr_vals_ctr_df,
            self.time_lags_df)

    def _get_rank_corr(self, i_stn):

        ser_i = self.in_var_df[i_stn]

        corrs_ser = self.rank_corrs_df.loc[i_stn]
        ctrs_ser = self.rank_corr_vals_ctr_df.loc[i_stn]

        if self.max_time_lag_corr:
            lag_ser = self.time_lags_df.loc[i_stn, :]

        assert len(ser_i.shape) == 1, as_err(
            'ser_i has more than one column!')

        for j_stn in self.nrst_stns_dict[i_stn]:
            if i_stn == j_stn:
                continue

            max_correl = 0.0
            max_ct = 0.0
            best_lag = 0

            for curr_time_lag in range(-self.max_time_lag_corr,
                                       +self.max_time_lag_corr + 1):

                ser_j = self.in_var_df[j_stn]

                if self.max_time_lag_corr:
                    ser_j = get_lag_ser(ser_j, curr_time_lag)

                assert ser_i.shape[0] == ser_j.shape[0], as_err(
                    'ser_i and ser_j have unequal shapes!')

                ij_df = DataFrame(
                    index=ser_i.index,
                    data={'i': ser_i.values, 'j': ser_j.values},
                    dtype=float32,
                    copy=True)

                ij_df.dropna(axis=0, how='any', inplace=True)

                if ij_df.shape[0] < self.min_valid_vals:
                    max_correl = nan
                    max_ct = 0.0
                    best_lag = 0
                    break

                prob_ser_i = (
                    ij_df['i'].rank().div(ij_df.shape[0] + 1.).values)
                prob_ser_j = (
                    ij_df['j'].rank().div(ij_df.shape[0] + 1.).values)

                if self.infill_type == 'discharge-censored':
                    prob_ser_i[prob_ser_i < self.cut_cdf_thresh] = (
                        0.5 * self.cut_cdf_thresh)
                    prob_ser_j[prob_ser_j < self.cut_cdf_thresh] = (
                        0.5 * self.cut_cdf_thresh)

                correl = get_corrcoeff(prob_ser_i, prob_ser_j)

                if (self.nrst_stns_type == 'symm') and correl:
                    correl = self._get_symm_corr(
                        prob_ser_i, prob_ser_j, correl)

                if ((abs(correl) > abs(max_correl)) and
                    (abs(correl) >= self.min_corr)):

                    max_correl = correl
                    max_ct = ij_df.shape[0]
                    best_lag = curr_time_lag

            if not max_ct:
                max_correl = nan

            corrs_ser[j_stn] = max_correl
            ctrs_ser[j_stn] = max_ct

            if self.max_time_lag_corr:
                lag_ser[j_stn] = best_lag
        return

    def _get_symm_corr(self, prob_ser_i, prob_ser_j, correl):

        asymms_arr = full((self.n_norm_symm_flds, 2), nan)

        for asymm_idx in range(self.n_norm_symm_flds):
            as_1, as_2 = get_norm_rand_symms(
                correl, min(self._max_symm_rands, prob_ser_i.shape[0]))

            asymms_arr[asymm_idx, 0] = as_1
            asymms_arr[asymm_idx, 1] = as_2

        assert np_all(isfinite(asymms_arr)), as_err(
            'Invalid values of asymmetries!')

        min_as_1, max_as_1 = (asymms_arr[:, 0].min(), asymms_arr[:, 0].max())
        min_as_2, max_as_2 = (asymms_arr[:, 1].min(), asymms_arr[:, 1].max())

        act_asymms = get_asymms_sample(prob_ser_i, prob_ser_j)
        act_as_1, act_as_2 = (act_asymms['asymm_1'], act_asymms['asymm_2'])

        as_1_norm = False
        as_2_norm = False

        if (act_as_1 >= min_as_1) and (act_as_1 <= max_as_1):
            as_1_norm = True

        if (act_as_2 >= min_as_2) and (act_as_2 <= max_as_2):
            as_2_norm = True

        if not (as_1_norm and as_2_norm):
            correl = nan

        return correl


if __name__ == '__main__':
    pass
