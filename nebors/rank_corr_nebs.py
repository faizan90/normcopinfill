# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from pickle import dump, load
from os import mkdir as os_mkdir
from os.path import exists as os_exists, join as os_join

from numpy import (full,
                   nan,
                   array,
                   all as np_all,
                   isfinite,
                   isnan,
                   where,
                   abs as np_abs)
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pandas import DataFrame

from .nrst_nebs import NrstStns
from ..misc.misc_ftns import get_norm_rand_symms, as_err

import pyximport
pyximport.install()
from normcop_cyftns import (get_asymms_sample, get_corrcoeff)

plt.ioff()


class RankCorrStns:

    def __init__(self, norm_cop_obj):
        self.norm_cop_obj = norm_cop_obj

        vars_list = ['in_var_df',
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
                     'max_time_lag_corr']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.in_var_df = self.in_var_df.copy()

        self.drop_infill_stns = []
        self.bad_stns_list = []
        self.bad_stns_neighbors_count = []

        self._got_rank_corr_stns = False
        self._plotted_rank_corr_stns = False
        self._plotted_long_term_rank_corrs = False

        if self._norm_cop_pool is None:
            self.norm_cop_obj._get_ncpus()
            self._norm_cop_pool = self.norm_cop_obj._norm_cop_pool

        if not self._dist_cmptd:
            NrstStns(norm_cop_obj)
            setattr(self, 'in_var_df', getattr(norm_cop_obj, 'in_var_df'))
            setattr(self, 'infill_stns', getattr(norm_cop_obj, 'infill_stns'))

        if self.max_time_lag_corr:
            return

        self.get_rank_corr_stns()
        assert self._got_rank_corr_stns

        if self.plot_neighbors_flag:
            self.plot_neighbors()
            assert self._plotted_rank_corr_stns

        if self.plot_long_term_corrs_flag:
            self.plot_long_term_corrs()
            assert self._plotted_long_term_rank_corrs

        setattr(norm_cop_obj, '_rank_corr_cmptd', True)
        setattr(norm_cop_obj, 'in_var_df', self.in_var_df)
        setattr(norm_cop_obj, 'infill_stns', self.infill_stns)
        setattr(norm_cop_obj, 'rank_corr_stns_dict', self.rank_corr_stns_dict)
        setattr(norm_cop_obj, 'rank_corr_stns_list', self.rank_corr_stns_list)
        setattr(norm_cop_obj, 'n_infill_stns', self.n_infill_stns)
        setattr(norm_cop_obj, 'rank_corrs_df', self.rank_corrs_df)
        setattr(norm_cop_obj,
                'rank_corr_vals_ctr_df',
                self.rank_corr_vals_ctr_df)
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

        rank_corr_stns_pkl_cur.close()

        self._got_rank_corr_stns = True
        return

    def _get_symm_corr(self, prob_ser_i, prob_ser_j, correl):
        if ((self.ncpus == 1) or
            (prob_ser_i.shape[0] < 5000) or
            self.debug_mode_flag):
            asymms_arr = full((self.n_norm_symm_flds, 2), nan)
            for asymm_idx in range(self.n_norm_symm_flds):
                as_1, as_2 = (
                    get_norm_rand_symms(
                            correl,
                            min(self._max_symm_rands,
                                prob_ser_i.shape[0])))
                asymms_arr[asymm_idx, 0] = as_1
                asymms_arr[asymm_idx, 1] = as_2

        else:
            correls_arr = full(self.n_norm_symm_flds, correl)
            nvals_arr = full(self.n_norm_symm_flds,
                             min(self._max_symm_rands,
                                 prob_ser_i.shape[0]))
            try:
                asymms_arr = (
                    array(list(self._norm_cop_pool.uimap(
                            get_norm_rand_symms,
                            correls_arr,
                            nvals_arr))))
            except:
                self._norm_cop_pool.close()
                self._norm_cop_pool.join()
                raise Exception(('Failed to calculate '
                                    'asymmetries!'))

        assert np_all(isfinite(asymms_arr)), as_err(
            'Invalid values of asymmetries!')

        min_as_1, max_as_1 = (asymms_arr[:, 0].min(),
                              asymms_arr[:, 0].max())
        min_as_2, max_as_2 = (asymms_arr[:, 1].min(),
                              asymms_arr[:, 1].max())

        act_asymms = get_asymms_sample(prob_ser_i, prob_ser_j)
        act_as_1, act_as_2 = (act_asymms['asymm_1'],
                              act_asymms['asymm_2'])

        as_1_norm = False
        as_2_norm = False
        if (act_as_1 >= min_as_1) and (act_as_1 <= max_as_1):
            as_1_norm = True
        if (act_as_2 >= min_as_2) and (act_as_2 <= max_as_2):
            as_2_norm = True

        if not (as_1_norm and as_2_norm):
            correl = nan

        return correl

    def _get_rank_corr(self, i_stn, tot_corrs_written):
        ser_i = self.in_var_df[i_stn].dropna().copy()
        ser_i_index = ser_i.index
        assert len(ser_i.shape) == 1, as_err(
            'ser_i has more than one column!')

        if i_stn in self.drop_infill_stns:
            return tot_corrs_written

        for j_stn in self.nrst_stns_dict[i_stn]:
            if i_stn == j_stn:
                continue

            if j_stn in self.drop_infill_stns:
                continue

            try:
                if self.loop_stns_df.loc[j_stn, i_stn]:
                    self.rank_corrs_df.loc[i_stn, j_stn] = (
                        self.rank_corrs_df.loc[j_stn, i_stn])
                    self.rank_corr_vals_ctr_df.loc[i_stn, j_stn] = (
                        self.rank_corr_vals_ctr_df.loc[j_stn, i_stn])
                    if not isnan(self.rank_corrs_df.loc[i_stn, j_stn]):
                        tot_corrs_written += 1
                    self.loop_stns_df.loc[i_stn, j_stn] = True
                    continue
            except KeyError:
                pass

            ser_j = self.in_var_df[j_stn].dropna().copy()

            index_ij = ser_i_index.intersection(ser_j.index)

            assert len(ser_j.shape) == 1, as_err(
                'ser_j has more than one column!')

            if index_ij.shape[0] <= self.min_valid_vals:
                self.loop_stns_df.loc[i_stn, j_stn] = True
                continue

            new_ser_i = ser_i.loc[index_ij].copy()
            new_ser_j = ser_j.loc[index_ij].copy()
            prob_ser_i = (
                new_ser_i.rank().div(new_ser_i.shape[0] + 1.).values)
            prob_ser_j = (
                new_ser_j.rank().div(new_ser_j.shape[0] + 1.).values)

            if self.infill_type == 'discharge-censored':
                prob_ser_i[prob_ser_i < self.cut_cdf_thresh] = (
                    self.cut_cdf_thresh)
                prob_ser_j[prob_ser_j < self.cut_cdf_thresh] = (
                    self.cut_cdf_thresh)

            correl = get_corrcoeff(prob_ser_i, prob_ser_j)

            if abs(correl) < self.min_corr:
                correl = nan
            elif (self.nrst_stns_type == 'symm') and correl:
                correl = self._get_symm_corr(prob_ser_i, prob_ser_j, correl)

            self.rank_corrs_df.loc[i_stn, j_stn] = correl
            self.rank_corr_vals_ctr_df.loc[i_stn, j_stn] = new_ser_i.shape[0]

            self.loop_stns_df.loc[i_stn, j_stn] = True
            if not isnan(correl):
                tot_corrs_written += 1

        return tot_corrs_written

    def _get_rank_corrs(self):
        self.rank_corrs_df = DataFrame(index=self.infill_stns,
                                       columns=self.in_var_df.columns,
                                       dtype=float)
        self.rank_corr_vals_ctr_df = self.rank_corrs_df.copy()

        # A DF to keep track of stations that have been looped through
        self.loop_stns_df = DataFrame(index=self.infill_stns,
                                      columns=self.in_var_df.columns,
                                      dtype=bool)
        self.loop_stns_df[:] = False
        self.rank_corr_vals_ctr_df[:] = 0.0

        drop_str = ('Station %s has no neighbors that satisfy the \'%s\' '
                    'criteria and is therfore dropped!')

        self.n_infill_stns = self.infill_stns.shape[0]

        tot_corrs_written = 0
        for i_stn in self.infill_stns:
            tot_corrs_written = self._get_rank_corr(i_stn, tot_corrs_written)
            if not self.rank_corrs_df.loc[i_stn].dropna().shape[0]:
                if self.dont_stop_flag:
                    self.drop_infill_stns.append(i_stn)
                    if self.verbose:
                        print('WARNING:', drop_str % (i_stn,
                                                      self.nrst_stns_type))
                    continue
                else:
                    raise Exception(drop_str % (i_stn, self.nrst_stns_type))

        if self.verbose:
            print('INFO: %d out of possible %d correlations written' %
                  (tot_corrs_written,
                   (self.n_infill_stns * (len(self.nrst_stns_list) - 1))))

        return

    def _get_best_rank_corr(self, infill_stn):
        if infill_stn not in self.rank_corr_stns_list:
            self.rank_corr_stns_list.append(infill_stn)
        stn_correl_ser = self.rank_corrs_df.loc[infill_stn].dropna().copy()

        if stn_correl_ser.shape[0] == 0:
            raise Exception('No neighbors for station %s!' % infill_stn)
        stn_correl_ser[:] = where(isfinite(stn_correl_ser[:]),
                                  np_abs(stn_correl_ser[:]),
                                  nan)
        stn_correl_ser.sort_values(axis=0, ascending=False, inplace=True)

        # take the nearest n_nrn stations to the infill_stn
        for rank_corr_stn in stn_correl_ser.index:
            if rank_corr_stn not in self.rank_corr_stns_list:
                self.rank_corr_stns_list.append(rank_corr_stn)

        # put the neighboring stations in a dictionary for each infill_stn
        self.rank_corr_stns_dict[infill_stn] = (
            stn_correl_ser.iloc[:self.n_max_nebs].index.tolist())

        curr_n_neighbors = len(self.rank_corr_stns_dict[infill_stn])
        if (curr_n_neighbors < self.n_min_nebs) and (
           (not self.force_infill_flag)):
            self.bad_stns_list.append(infill_stn)
            self.bad_stns_neighbors_count.append(curr_n_neighbors)

        if not self.force_infill_flag:
            assert curr_n_neighbors >= self.n_min_nebs, (
                as_err(('Rank correlation stations (n=%d) less than '
                        '\'n_min_nebs\' or no neighbor for '
                        'station: %s') % (curr_n_neighbors,
                                          infill_stn)))
        return

    def _plot_neighbor(self, infill_stn):
        tick_font_size = 5

        infill_x, infill_y = (
            self.in_coords_df[['X', 'Y']].loc[infill_stn].values)

        _nebs = self.rank_corr_stns_dict[infill_stn]
        _n_nebs = len(self.rank_corr_stns_dict[infill_stn])
        hi_corr_stns_ax = plt.subplot(111)
        hi_corr_stns_ax.scatter(infill_x,
                                infill_y,
                                c='r',
                                label='infill_stn')
        hi_corr_stns_ax.scatter(self.in_coords_df['X'].loc[_nebs],
                                self.in_coords_df['Y'].loc[_nebs],
                                alpha=0.75,
                                c='c',
                                label='hi_corr_stn (%d)' % _n_nebs)
        plt_texts = []
        _txt_obj = hi_corr_stns_ax.text(infill_x,
                                        infill_y,
                                        infill_stn,
                                        va='top',
                                        ha='left',
                                        fontsize=tick_font_size)
        plt_texts.append(_txt_obj)
        for stn in self.rank_corr_stns_dict[infill_stn]:
            _lab = '%s\n(%0.4f)' % (stn,
                                    self.rank_corrs_df.loc[infill_stn,
                                                           stn])
            if not infill_stn == stn:
                _txt_obj = hi_corr_stns_ax.text(
                    self.in_coords_df['X'].loc[stn],
                    self.in_coords_df['Y'].loc[stn],
                    _lab,
                    va='top',
                    ha='left',
                    fontsize=5)
                plt_texts.append(_txt_obj)

        adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})
        hi_corr_stns_ax.grid()
        hi_corr_stns_ax.set_xlabel('Eastings', size=tick_font_size)
        hi_corr_stns_ax.set_ylabel('Northings', size=tick_font_size)
        hi_corr_stns_ax.legend(framealpha=0.5, loc=0)
        plt.setp(hi_corr_stns_ax.get_xticklabels(),
                 size=tick_font_size)
        plt.setp(hi_corr_stns_ax.get_yticklabels(),
                 size=tick_font_size)
        plt.savefig(os_join(self.out_rank_corr_plots_dir,
                            'rank_corr_stn_%s.png' % infill_stn),
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.clf()
        return

    def _plot_long_term_corr(self, infill_stn):
        tick_font_size = 6
        curr_nebs = self.rank_corr_stns_dict[infill_stn]

        corrs_arr = self.rank_corrs_df.loc[infill_stn,
                                           curr_nebs].values
        corrs_ctr_arr = (
            self.rank_corr_vals_ctr_df.loc[infill_stn,
                                           curr_nebs].values)
        corrs_ctr_arr[isnan(corrs_ctr_arr)] = 0

        n_stns = corrs_arr.shape[0]
        _, corrs_ax = plt.subplots(1, 1, figsize=(1.0 * n_stns, 3))
        corrs_ax.matshow(corrs_arr.reshape(1, n_stns),
                         vmin=0,
                         vmax=2,
                         cmap=plt.get_cmap('Blues'),
                         origin='lower')
        for s in range(n_stns):
            corrs_ax.text(s,
                          0,
                          '%0.4f\n(%d)' % (corrs_arr[s],
                                           int(corrs_ctr_arr[s])),
                          va='center',
                          ha='center',
                          fontsize=tick_font_size)

        corrs_ax.set_yticks([])
        corrs_ax.set_yticklabels([])
        corrs_ax.set_xticks(list(range(0, n_stns)))
        corrs_ax.set_xticklabels(curr_nebs)

        corrs_ax.spines['left'].set_position(('outward', 10))
        corrs_ax.spines['right'].set_position(('outward', 10))
        corrs_ax.spines['top'].set_position(('outward', 10))
        corrs_ax.spines['bottom'].set_position(('outward', 10))

        corrs_ax.tick_params(labelleft=False,
                             labelbottom=True,
                             labeltop=False,
                             labelright=False)

        plt.setp(corrs_ax.get_xticklabels(),
                 size=tick_font_size,
                 rotation=45)
        plt.suptitle('station: %s long-term corrs' % infill_stn)
        _ = 'long_term_stn_%s_rank_corrs.png' % infill_stn
        plt.savefig(os_join(self.out_long_term_corrs_dir, _),
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.close()
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
                raise KeyError(('The station %s is not in '
                                '\'rank_corrs_df\' anymore!') % infill_stn)

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

        self.rank_corrs_df.drop(labels=self.bad_stns_list,
                                axis=0,
                                inplace=True)
        self.rank_corr_vals_ctr_df.drop(labels=self.bad_stns_list,
                                        axis=0,
                                        inplace=True)

        if self.verbose and self.bad_stns_list:
            print('INFO: These infill station(s) had too few values'
                  'to be considered for the rest of the analysis:')
            print('Station: n_neighbors')
            for bad_stn in zip(self.bad_stns_list,
                               self.bad_stns_neighbors_count):
                print('%s: %d' % (bad_stn[0], bad_stn[1]))

        # have the rank_corr_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.rank_corr_stns_list]
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

        dump(rank_corr_stns_pkl_dict, rank_corr_stns_pkl_cur, -1)
        rank_corr_stns_pkl_cur.close()

        self._got_rank_corr_stns = True
        return

    def plot_neighbors(self):
        if not os_exists(self.out_rank_corr_plots_dir):
            os_mkdir(self.out_rank_corr_plots_dir)

        for infill_stn in self.infill_stns:
            self._plot_neighbor(infill_stn)

        self._plotted_rank_corr_stns = True
        return

    def plot_long_term_corrs(self):
        if self.verbose:
            print('INFO: Plotting long-term correlation neighbors')

        if not os_exists(self.out_long_term_corrs_dir):
            os_mkdir(self.out_long_term_corrs_dir)

        for infill_stn in self.infill_stns:
            self._plot_long_term_corr(infill_stn)

        plt.close()
        self._plotted_long_term_rank_corrs = True
        return


if __name__ == '__main__':
    pass
