# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from pickle import dump, load
from os import mkdir as os_mkdir
from os.path import exists as os_exists, join as os_join

from adjustText import adjust_text
import matplotlib.pyplot as plt
from numpy import vectorize
from pandas import DataFrame

from ..misc.misc_ftns import as_err

import pyximport
pyximport.install()
from normcop_cyftns import get_dist

plt.ioff()


class NrstStns:
    '''
    Compute and plot nearest stations around each infill station

    A station should be near, have common steps with the infill station
    and have at least one common step in the infill period as well.
    '''

    def __init__(self, norm_cop_obj):

        self.norm_cop_obj = norm_cop_obj

        vars_list = ['in_var_df',
                     'infill_stns',
                     'infill_dates',
                     'full_date_index',
                     'in_coords_df',
                     '_dist_cmptd',
                     'plot_neighbors_flag',
                     'read_pickles_flag',
                     'dont_stop_flag',
                     'verbose',
                     'nrst_stns_list',
                     'nrst_stns_dict',
                     'xs',
                     'ys',
                     'n_min_nebs',
                     'n_max_nebs',
                     'min_valid_vals',
                     'max_time_lag_corr',
                     'out_fig_dpi',
                     '_out_nrst_stns_pkl_file',
                     'out_nebor_plots_dir']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.in_var_df = self.in_var_df.copy()

        self.bad_stns_list = []
        self.bad_stns_neighbors_count = []

        self._got_nrst_stns_flag = False
        self._plotted_nrst_stns_flag = False

        self.get_nrst_stns()
        assert self._got_nrst_stns_flag

        if self.plot_neighbors_flag:
            self.plot_nrst_stns()
            assert self._plotted_nrst_stns_flag

        setattr(norm_cop_obj, '_dist_cmptd', True)
        setattr(norm_cop_obj, 'in_var_df', self.in_var_df)
        setattr(norm_cop_obj, 'infill_stns', self.infill_stns)
        setattr(norm_cop_obj, 'nrst_stns_dict', self.nrst_stns_dict)
        setattr(norm_cop_obj, 'nrst_stns_list', self.nrst_stns_list)

        return

    def _get_nrst_stn(self, infill_stn):
        curr_nebs_list = []

        # get the x and y coordinates of the infill_stn
        infill_x, infill_y = \
            self.in_coords_df[['X', 'Y']].loc[infill_stn].values

        # calculate distances of all stations from the infill_stn
        dists = vectorize(get_dist)(infill_x, infill_y, self.xs, self.ys)

        dists_df = DataFrame(index=self.in_coords_df.index,
                             data=dists,
                             columns=['dists'],
                             dtype=float)
        dists_df.sort_values('dists', axis=0, inplace=True)
        # till here

        # take the nearest n_nrn stations to the infill_stn
        # that also have enough common records and with data existing
        # for atleast one of the infill_dates
        for nrst_stn in dists_df.index[1:]:
#            # if the infill_stn is already a neighbor of nrst_stn
#            if nrst_stn in list(self.nrst_stns_dict.keys()):
#                if infill_stn in self.nrst_stns_dict[nrst_stn]:
#                    curr_nebs_list.append(nrst_stn)
#                    continue

            _cond_2 = False
            _cond_3 = False

            _ = self.in_var_df[[infill_stn, nrst_stn]].dropna()
            _cond_2 = (_.shape[0] >= self.min_valid_vals)

            _ = self.in_var_df[nrst_stn].dropna()
            if _.index.intersection(self.infill_dates).shape[0]:
                _cond_3 = True

            if _cond_2 and _cond_3:
                if nrst_stn not in curr_nebs_list:
                    curr_nebs_list.append(nrst_stn)

                if nrst_stn not in self.nrst_stns_list:
                    self.nrst_stns_list.append(nrst_stn)

        self.nrst_stns_dict[infill_stn] = curr_nebs_list

        _ = self.nrst_stns_dict[infill_stn]
        if len(_) >= self.n_min_nebs:
            if infill_stn not in self.nrst_stns_list:
                self.nrst_stns_list.append(infill_stn)
        else:
            as_err(('Neighboring stations less than '
                    '\'n_min_nebs\' '
                    'for station: %s') % infill_stn)
            if self.dont_stop_flag:
                self.bad_stns_list.append(infill_stn)
                self.bad_stns_neighbors_count.append(len(_))
            else:
                raise Exception('Too few nebors!')
        return

    def _load_pickle(self):
        if self.verbose:
            print('INFO: Loading nearest neighbors pickle...')

        nrst_stns_pickle_cur = open(self._out_nrst_stns_pkl_file, 'rb')
        nrst_stns_pickle_dict = load(nrst_stns_pickle_cur)

        self.in_var_df = nrst_stns_pickle_dict['in_var_df']
        self.infill_stns = nrst_stns_pickle_dict['infill_stns']
        self.nrst_stns_dict = nrst_stns_pickle_dict['nrst_stns_dict']
        self.nrst_stns_list = nrst_stns_pickle_dict['nrst_stns_list']

        nrst_stns_pickle_cur.close()

        self._got_nrst_stns_flag = True
        return

    def get_nrst_stns(self):
        '''Get the neighbors
        '''

        if (os_exists(self._out_nrst_stns_pkl_file) and
            self.read_pickles_flag):
            self._load_pickle()
            return

        if self.verbose:
            print('INFO: Computing nearest stations...')

        ### cmpt nrst stns
        for infill_stn in self.infill_stns:
            self._get_nrst_stn(infill_stn)

        for bad_stn in self.bad_stns_list:
            self.infill_stns = self.infill_stns.drop(bad_stn)
            del self.nrst_stns_dict[bad_stn]

        if self.verbose and self.bad_stns_list:
            print('INFO: These infill stations had too few values '
                  'to be considered for the rest of the analysis:')
            print('Station: n_neighbors')
            for bad_stn in zip(self.bad_stns_list,
                               self.bad_stns_neighbors_count):
                print('%s: %d' % (bad_stn[0], bad_stn[1]))

        # have the nrst_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.nrst_stns_list]
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, \
                as_err(('Station %s not in input variable dataframe '
                        'anymore!') % infill_stn)

        # check if at least one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, \
            as_err('None of the infill dates exist in \'in_var_df\' '
                   'after dropping stations and records with '
                   'insufficient information!')

        if self.verbose:
            print(('INFO: \'in_var_df\' shape after calling '
                   '\'get_nrst_stns\':'), self.in_var_df.shape)

        if self.max_time_lag_corr:
            self.in_var_df = self.in_var_df.reindex(self.full_date_index)

        ### save pickle
        nrst_stns_pickle_dict = {}
        nrst_stns_pickle_cur = open(self._out_nrst_stns_pkl_file, 'wb')

        nrst_stns_pickle_dict['in_var_df'] = self.in_var_df
        nrst_stns_pickle_dict['infill_stns'] = self.infill_stns
        nrst_stns_pickle_dict['nrst_stns_dict'] = self.nrst_stns_dict
        nrst_stns_pickle_dict['nrst_stns_list'] = self.nrst_stns_list

        dump(nrst_stns_pickle_dict, nrst_stns_pickle_cur, -1)
        nrst_stns_pickle_cur.close()

        self._got_nrst_stns_flag = True
        return

    def plot_nrst_stns(self):
        '''Plot the neighbors
        '''

        assert self._got_nrst_stns_flag, 'Call \'get_nrst_stns\' first!'

        if self.verbose:
            print('INFO: Plotting nearest stations...')

        if not os_exists(self.out_nebor_plots_dir):
            os_mkdir(self.out_nebor_plots_dir)

        tick_font_size = 5
        for infill_stn in self.infill_stns:
            (infill_x,
             infill_y) = self.in_coords_df[['X',
                                            'Y']].loc[infill_stn].values
            _nebs = self.nrst_stns_dict[infill_stn]
            _lab = ('neibor_stn (%d)' %
                    len(self.nrst_stns_dict[infill_stn]))
            nrst_stns_ax = plt.subplot(111)
            nrst_stns_ax.scatter(infill_x,
                                 infill_y,
                                 c='r',
                                 label='infill_stn')
            nrst_stns_ax.scatter(self.in_coords_df['X'].loc[_nebs],
                                 self.in_coords_df['Y'].loc[_nebs],
                                 alpha=0.75,
                                 c='c',
                                 label=_lab)
            plt_texts = []
            _txt_obj = nrst_stns_ax.text(infill_x,
                                         infill_y,
                                         infill_stn,
                                         va='top',
                                         ha='left',
                                         fontsize=tick_font_size)
            plt_texts.append(_txt_obj)

            for stn in self.nrst_stns_dict[infill_stn]:
                _txt_obj = nrst_stns_ax.text(self.in_coords_df['X'].loc[stn],
                                             self.in_coords_df['Y'].loc[stn],
                                             stn,
                                             va='top',
                                             ha='left',
                                             fontsize=5)
                plt_texts.append(_txt_obj)

            adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})
            nrst_stns_ax.grid()
            nrst_stns_ax.set_xlabel('Eastings', size=tick_font_size)
            nrst_stns_ax.set_ylabel('Northings', size=tick_font_size)
            nrst_stns_ax.legend(framealpha=0.5, loc=0)
            plt.setp(nrst_stns_ax.get_xticklabels(), size=tick_font_size)
            plt.setp(nrst_stns_ax.get_yticklabels(), size=tick_font_size)
            plt.savefig(os_join(self.out_nebor_plots_dir,
                                '%s_neibor_stns.png' % infill_stn),
                        dpi=self.out_fig_dpi,
                        bbox_inches='tight')
            plt.clf()
        plt.close('all')
        self._plotted_nrst_stns_flag = True
        return


if __name__ == '__main__':
    pass