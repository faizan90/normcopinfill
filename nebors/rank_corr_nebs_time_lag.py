# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from os import mkdir as os_mkdir
from os.path import join as os_join, exists as os_exists

from numpy import nan, isnan
import matplotlib.pyplot as plt
from adjustText import adjust_text
from pandas import DataFrame

from .rank_corr_nebs import RankCorrStns
from ..misc.misc_ftns import as_err, get_lag_ser
from ..cyth import get_corrcoeff

plt.ioff()


class BestLagRankCorrStns(RankCorrStns):

    def __init__(self, norm_cop_obj):
        super(BestLagRankCorrStns, self).__init__(norm_cop_obj)

        assert self.max_time_lag_corr, as_err(
            'Cannot use this class if \'max_time_lag_corr\' is zero!')

        self.in_var_df = self.in_var_df.reindex(self.full_date_index)

        self.time_lags_dict = {}

        self.get_rank_corr_stns()
        assert self._got_rank_corr_stns

        self.in_var_df = self.in_var_df.reindex(self.full_date_index)

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
        setattr(
            norm_cop_obj, 'rank_corr_vals_ctr_df', self.rank_corr_vals_ctr_df)
        setattr(norm_cop_obj, 'time_lags_dict', self.time_lags_dict)
        return

    def _get_symm_corr(self, prob_ser_i, prob_ser_j, correl):
        raise NotImplementedError('Still not done!')

#    def _get_symm_corr(self, prob_ser_i, prob_ser_j, correl):
#        if ((self.ncpus == 1) or
#            (prob_ser_i.shape[0] < 5000) or
#            self.debug_mode_flag):
#            asymms_arr = full((self.n_norm_symm_flds, 2), nan)
#            for asymm_idx in range(self.n_norm_symm_flds):
#                as_1, as_2 = \
#                    get_norm_rand_symms(
#                            correl,
#                            min(self._max_symm_rands,
#                                prob_ser_i.shape[0]))
#                asymms_arr[asymm_idx, 0] = as_1
#                asymms_arr[asymm_idx, 1] = as_2
#
#        else:
#            correls_arr = full(self.n_norm_symm_flds, correl)
#            nvals_arr = full(self.n_norm_symm_flds,
#                             min(self._max_symm_rands,
#                                 prob_ser_i.shape[0]))
#            try:
#                asymms_arr = \
#                    array(list(self._norm_cop_pool.uimap(
#                            get_norm_rand_symms,
#                            correls_arr,
#                            nvals_arr)))
#            except:
#                self._norm_cop_pool.close()
#                self._norm_cop_pool.join()
#                raise Exception(('Failed to calculate '
#                                    'asymmetries!'))
#
#        assert np_all(isfinite(asymms_arr)), \
#            as_err('Invalid values of asymmetries!')
#
#        min_as_1, max_as_1 = (asymms_arr[:, 0].min(),
#                              asymms_arr[:, 0].max())
#        min_as_2, max_as_2 = (asymms_arr[:, 1].min(),
#                              asymms_arr[:, 1].max())
#
#        act_asymms = get_asymms_sample(prob_ser_i, prob_ser_j)
#        act_as_1, act_as_2 = (act_asymms['asymm_1'],
#                              act_asymms['asymm_2'])
#
#        as_1_norm = False
#        as_2_norm = False
#        if (act_as_1 >= min_as_1) and (act_as_1 <= max_as_1):
#            as_1_norm = True
#        if (act_as_2 >= min_as_2) and (act_as_2 <= max_as_2):
#            as_2_norm = True
#
#        if not (as_1_norm and as_2_norm):
#            correl = nan
#
#        return correl

    def _get_rank_corr(self, i_stn, tot_corrs_written):
        ser_i = self.in_var_df[i_stn].copy()
        assert len(ser_i.shape) == 1, as_err(
            'ser_i has more than one column!')

        if i_stn in self.drop_infill_stns:
            return tot_corrs_written

        self.time_lags_dict[i_stn] = {}

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
                    self.time_lags_dict[i_stn][j_stn] = self.time_lags_dict[j_stn][i_stn]

                    continue

            except KeyError:
                pass

            max_correl = 0.0
            max_ct = 0.0
            best_lag = 0

            for curr_time_lag in range(-self.max_time_lag_corr,
                                       +self.max_time_lag_corr + 1):
                _ = self.in_var_df[j_stn].copy()
                ser_j = get_lag_ser(_, curr_time_lag)

                assert ser_i.shape[0] == ser_j.shape[0], as_err(
                    'ser_i and ser_j have unequal shapes!')

                ij_df = DataFrame(index=ser_i.index,
                                  data={'i': ser_i.values,
                                        'j': ser_j.values},
                                  dtype=float)
                ij_df.dropna(axis=0, how='any', inplace=True)

                if ij_df.shape[0] <= self.min_valid_vals:
                    self.loop_stns_df.loc[i_stn, j_stn] = True
                    continue

                ser_i_rank = ij_df['i'].rank()
                ser_j_rank = ij_df['j'].rank()
                prob_ser_i = \
                    ser_i_rank.rank().div(ser_i_rank.shape[0] + 1.).values
                prob_ser_j = \
                    ser_j_rank.rank().div(ser_j_rank.shape[0] + 1.).values

                if self.infill_type == 'discharge-censored':
                    prob_ser_i[prob_ser_i < self.cut_cdf_thresh] = (
                        self.cut_cdf_thresh)
                    prob_ser_j[prob_ser_j < self.cut_cdf_thresh] = (
                        self.cut_cdf_thresh)

                correl = get_corrcoeff(prob_ser_i, prob_ser_j)

                if ((abs(correl) > max_correl) and
                    (abs(correl) >= self.min_corr)):
                    max_correl = correl
                    max_ct = ser_i_rank.shape[0]
                    best_lag = curr_time_lag

            # TODO: implement later
#            elif (self.nrst_stns_type == 'symm') and correl:
#                correl = self._get_symm_corr(prob_ser_i, prob_ser_j, correl)

            if not max_ct:
                max_correl = nan

            self.time_lags_dict[i_stn][j_stn] = best_lag
            self.rank_corrs_df.loc[i_stn, j_stn] = max_correl
            self.rank_corr_vals_ctr_df.loc[i_stn, j_stn] = max_ct

            self.loop_stns_df.loc[i_stn, j_stn] = True
            if not isnan(correl):
                tot_corrs_written += 1

        return tot_corrs_written

    def _get_rank_corrs(self):
        self.rank_corrs_df = DataFrame(index=self.infill_stns,
                                       columns=self.in_var_df.columns,
                                       dtype=float)
        self.rank_corr_vals_ctr_df = self.rank_corrs_df.copy()

        # A DF to keep track of stations that have been processed
        self.loop_stns_df = DataFrame(index=self.infill_stns,
                                      columns=self.in_var_df.columns,
                                      dtype=bool)
        self.loop_stns_df[:] = False
        self.rank_corr_vals_ctr_df[:] = 0.0

        # modified later
        drop_str = ('Station %s has no neighbors that satisfy the \'%s\' '
                    'criteria and is therfore dropped!')

        self.n_infill_stns = self.infill_stns.shape[0]

        tot_corrs_written = 0
        for i_stn in self.infill_stns:
            tot_corrs_written = self._get_rank_corr(i_stn,
                                                    tot_corrs_written)
            if not self.rank_corrs_df.loc[i_stn].dropna().shape[0]:
                if self.dont_stop_flag:
                    self.drop_infill_stns.append(i_stn)
                    if self.verbose:
                        print('WARNING:', drop_str % (i_stn,
                                                      self.nrst_stns_type))
                    continue
                else:
                    raise Exception(drop_str % (i_stn,
                                                self.nrst_stns_type))

        if self.verbose:
            print('INFO: %d out of possible %d correlations written' %
                  (tot_corrs_written,
                   (self.n_infill_stns * (len(self.nrst_stns_list) - 1))))

        return

    def _plot_neighbor(self, infill_stn):
        tick_font_size = 5

        (infill_x,
         infill_y) = self.in_coords_df[['X', 'Y']].loc[infill_stn].values

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
            if infill_stn == stn:
                continue

            _1 = self.rank_corrs_df.loc[infill_stn, stn]
            _2 = self.time_lags_dict[infill_stn][stn]
            _lab = '%s\n(%0.4f, %d)' % (stn, _1, _2)

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
            _1 = int(corrs_ctr_arr[s])
            _2 = self.time_lags_dict[infill_stn][curr_nebs[s]]
            _lab = '%0.4f\n(%d, %d)' % (corrs_arr[s], _1, _2)

            corrs_ax.text(s,
                          0,
                          _lab,
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
        return


if __name__ == '__main__':
    pass
