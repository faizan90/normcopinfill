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
from ..cyth import get_dist

plt.ioff()

from multiprocessing import current_process

current_process().authkey = b'all_worker_have_the_same_key'


class NrstStns:
    '''
    Compute and plot nearest stations around each infill station

    A station should be near, have common steps with the infill station
    and have at least one common step in the infill period as well.
    '''

    def __init__(self, norm_cop_obj):

        vars_list = [
            'in_var_df',
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
            'out_nebor_plots_dir',
            'pkls_dir',
            'stns_valid_dates',
            '_norm_cop_pool',
            'ncpus']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.in_var_df = self.in_var_df.copy()

        self.bad_stns_list = []
        self.bad_stns_neighbors_count = []

        self._got_nrst_stns_flag = False
        self._plotted_nrst_stns_flag = False

        if not os_exists(self.pkls_dir):
            os_mkdir(self.pkls_dir)

        if self._norm_cop_pool is None:
            norm_cop_obj._get_ncpus()
            self._norm_cop_pool = norm_cop_obj._norm_cop_pool

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

        mp = True if self.ncpus > 1 else False

        if mp:
            from multiprocessing import Manager

            mng_nrst_stns_list = Manager().list(self.nrst_stns_list)
            mng_nrst_stns_dict = Manager().dict(self.nrst_stns_dict)
            mng_bad_stns_list = Manager().list(self.bad_stns_list)
            mng_bad_stns_neighbors_count = Manager().list(
                self.bad_stns_neighbors_count)
            mng_lock = Manager().Lock()  # just to be safe

            mng_list = [
                mng_nrst_stns_list,
                mng_nrst_stns_dict,
                mng_bad_stns_list,
                mng_bad_stns_neighbors_count,
                mng_lock]

        else:
            mng_list = []

        get_nrst_stns_obj = GetNrstStns(self, mng_list)

        if mp:
            self._norm_cop_pool.map(
                get_nrst_stns_obj._get_nrst_stn, self.infill_stns)

            self._norm_cop_pool.clear()

            self.nrst_stns_list = list(mng_nrst_stns_list)
            self.nrst_stns_dict = dict(mng_nrst_stns_dict)
            self.bad_stns_list = list(mng_bad_stns_list)
            self.bad_stns_neighbors_count = list(mng_bad_stns_neighbors_count)

        else:
            for infill_stn in self.infill_stns:
                get_nrst_stns_obj._get_nrst_stn(infill_stn)

        for bad_stn in self.bad_stns_list:
            self.infill_stns = self.infill_stns.drop(bad_stn)
            del self.nrst_stns_dict[bad_stn]

        if self.verbose and self.bad_stns_list:
            print('INFO: These infill stations had too few values '
                  'to be considered for the rest of the analysis:')
            print('Station: n_neighbors')

            for bad_stn in zip(
                self.bad_stns_list, self.bad_stns_neighbors_count):

                print('%s: %d' % (bad_stn[0], bad_stn[1]))

        # have the nrst_stns_list in the in_var_df only
        self.in_var_df = self.in_var_df[self.nrst_stns_list]
        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, as_err(
                ('Station %s not in input variable dataframe anymore!') %
                infill_stn)

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
                   '\'get_nrst_stns\':'), self.in_var_df.shape)

        if self.max_time_lag_corr:
            self.in_var_df = self.in_var_df.reindex(self.full_date_index)

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

        mp = True if self.ncpus > 1 else False
        if mp:
            plt_gen = (
                (_stn,
                 self.in_coords_df[['X']].loc[_stn].values,
                 self.in_coords_df[['Y']].loc[_stn].values,
                 self.nrst_stns_dict[_stn],
                 self.in_coords_df['X'].loc[self.nrst_stns_dict[_stn]],
                 self.in_coords_df['Y'].loc[self.nrst_stns_dict[_stn]],
                 self.xs,
                 self.ys,
                 self.out_nebor_plots_dir,
                 self.out_fig_dpi)
                for _stn in self.infill_stns)

            self._norm_cop_pool.map(plot_nrst_stns, plt_gen)
            self._norm_cop_pool.clear()
        else:
            for infill_stn in self.infill_stns:
                (infill_x, infill_y) = (
                    self.in_coords_df[['X', 'Y']].loc[infill_stn].values)

                nebs = self.nrst_stns_dict[infill_stn]
                x_nebs = self.in_coords_df['X'].loc[nebs]
                y_nebs = self.in_coords_df['Y'].loc[nebs]

                args = (
                    infill_stn,
                    infill_x,
                    infill_y,
                    nebs,
                    x_nebs,
                    y_nebs,
                    self.xs,
                    self.ys,
                    self.out_nebor_plots_dir,
                    self.out_fig_dpi)

                plot_nrst_stns(args)

        self._plotted_nrst_stns_flag = True
        return


class GetNrstStns:

    def __init__(self, nrst_stns_obj, mng_list):
        vars_list = [
            'in_coords_df',
            'xs',
            'ys',
            'stns_valid_dates',
            'min_valid_vals',
            'infill_dates',
            'nrst_stns_list',
            'nrst_stns_dict',
            'n_min_nebs',
            'dont_stop_flag',
            'bad_stns_list',
            'bad_stns_neighbors_count']

        for _var in vars_list:
            setattr(self, _var, getattr(nrst_stns_obj, _var))

        if mng_list:
            self.mp = True
            self.nrst_stns_list = mng_list[0]
            self.nrst_stns_dict = mng_list[1]
            self.bad_stns_list = mng_list[2]
            self.bad_stns_neighbors_count = mng_list[3]
            self.lock = mng_list[4]

        else:
            self.mp = False

        return

    def _get_nrst_stn(self, infill_stn):

        curr_nebs_list = []

        # get the x and y coordinates of the infill_stn
        (infill_x,
         infill_y) = self.in_coords_df[['X', 'Y']].loc[infill_stn].values

        # calculate distances of all stations from the infill_stn
        dists = vectorize(get_dist)(infill_x, infill_y, self.xs, self.ys)

        dists_df = DataFrame(
            index=self.in_coords_df.index,
            data=dists,
            columns=['dists'],
            dtype=float)

        dists_df.sort_values('dists', axis=0, inplace=True)

        # take the nearest n_nrn stations to the infill_stn
        # that also have enough common records and with data existing
        # for atleast one of the infill_dates
        for nrst_stn in dists_df.index[1:]:
            _cond_2 = False
            _cond_3 = False

            _ = self.stns_valid_dates[infill_stn].intersection(
                self.stns_valid_dates[nrst_stn])

            _cond_2 = _.shape[0] >= self.min_valid_vals

            _ = self.stns_valid_dates[nrst_stn]
            if _.intersection(self.infill_dates).shape[0]:
                _cond_3 = True

            if _cond_2 and _cond_3:
                curr_nebs_list.append(nrst_stn)

                if self.mp: self.lock.acquire()
                if nrst_stn not in self.nrst_stns_list:
                    self.nrst_stns_list.append(nrst_stn)
                if self.mp: self.lock.release()

        self.nrst_stns_dict[infill_stn] = curr_nebs_list

        _ = self.nrst_stns_dict[infill_stn]
        if len(_) >= self.n_min_nebs:

            if self.mp: self.lock.acquire()
            if infill_stn not in self.nrst_stns_list:
                self.nrst_stns_list.append(infill_stn)
            if self.mp: self.lock.release()

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


def plot_nrst_stns(args):

    (infill_stn,
     infill_x,
     infill_y,
     nebs,
     x_nebs,
     y_nebs,
     xs,
     ys,
     out_nebor_plots_dir,
     out_fig_dpi) = args

    tick_font_size = 5
    n_nebs = len(nebs)

    nrst_stns_ax = plt.subplot(111)

    nrst_stns_ax.scatter(
        x_nebs,
        y_nebs,
        alpha=0.75,
        c='c',
        label='nebor_stn (%d)' % n_nebs)

    nrst_stns_ax.scatter(
        infill_x,
        infill_y,
        c='r',
        label='infill_stn')

    plt_texts = []
    for stn in nebs:
        _txt_obj = nrst_stns_ax.text(
            x_nebs[stn],
            y_nebs[stn],
            stn,
            va='top',
            ha='left',
            fontsize=5)
        plt_texts.append(_txt_obj)

    _txt_obj = nrst_stns_ax.text(
        infill_x,
        infill_y,
        infill_stn,
        va='top',
        ha='left',
        fontsize=tick_font_size)
    plt_texts.append(_txt_obj)

    adjust_text(plt_texts, only_move={'points': 'y', 'text': 'y'})
    nrst_stns_ax.grid()
    nrst_stns_ax.set_xlabel('Eastings', size=tick_font_size)
    nrst_stns_ax.set_ylabel('Northings', size=tick_font_size)
    nrst_stns_ax.legend(framealpha=0.5, loc=0)

    nrst_stns_ax.set_xlim(0.995 * xs.min(), 1.005 * xs.max())
    nrst_stns_ax.set_ylim(0.995 * ys.min(), 1.005 * ys.max())

    plt.setp(nrst_stns_ax.get_xticklabels(), size=tick_font_size)
    plt.setp(nrst_stns_ax.get_yticklabels(), size=tick_font_size)
    plt.savefig(
        os_join(out_nebor_plots_dir, '%s_neibor_stns.png' % infill_stn),
        dpi=out_fig_dpi,
        bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    pass
