"""
Spyder Editor

This is a temporary script file.
"""

from numpy import divide, repeat, tile, float32
from pandas import DataFrame, to_numeric
import matplotlib.pyplot as plt


class PlotStats:

    def __init__(self, norm_cop_obj):
        vars_list = ['in_var_df',
                     'out_var_stats_file',
                     'out_fig_dpi',
                     'verbose']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.plot_stats()
        return

    def plot_stats(self):
        '''
        Compute and plot statistics of each station
        '''
        if self.verbose:
            print('INFO: Plotting statistics of each infill station...')

        stats_cols = ['min',
                      'max',
                      'mean',
                      'stdev',
                      'CoV',
                      'skew',
                      'count']

        self.stats_df = DataFrame(
            index=self.in_var_df.columns,
            columns=stats_cols,
            dtype=float32)

        for i, stn in enumerate(self.stats_df.index):
            curr_ser = self.in_var_df[stn].dropna().copy()
            curr_min = curr_ser.min()
            curr_max = curr_ser.max()
            curr_mean = curr_ser.mean()
            curr_stdev = curr_ser.std()
            curr_coovar = divide(curr_stdev, curr_mean)
            curr_skew = curr_ser.skew()
            curr_count = curr_ser.count()
            self.stats_df.iloc[i] = [curr_min,
                                     curr_max,
                                     curr_mean,
                                     curr_stdev,
                                     curr_coovar,
                                     curr_skew,
                                     curr_count]

        self.stats_df = self.stats_df.apply(lambda x: to_numeric(x))

        tick_font_size = 6
        stats_arr = self.stats_df.values
        n_stns = stats_arr.shape[0]
        n_cols = stats_arr.shape[1]

        _, stats_ax = plt.subplots(
            1, 1, figsize=(0.45 * n_stns, 0.8 * n_cols))

        stats_ax.matshow(stats_arr.T,
                         cmap=plt.get_cmap('Blues'),
                         vmin=0,
                         vmax=0,
                         origin='upper')

        for s in zip(repeat(list(range(n_stns)), n_cols),
                     tile(list(range(n_cols)), n_stns)):
            stats_ax.text(s[0],
                          s[1],
                          ('%0.2f' % stats_arr[s[0], s[1]]).rstrip('0'),
                          va='center',
                          ha='center',
                          fontsize=tick_font_size,
                          rotation=45)

        stats_ax.set_xticks(list(range(0, n_stns)))
        stats_ax.set_xticklabels(self.stats_df.index)
        stats_ax.set_yticks(list(range(0, n_cols)))
        stats_ax.set_yticklabels(self.stats_df.columns)

        stats_ax.spines['left'].set_position(('outward', 10))
        stats_ax.spines['right'].set_position(('outward', 10))
        stats_ax.spines['top'].set_position(('outward', 10))
        stats_ax.spines['bottom'].set_position(('outward', 10))

        stats_ax.set_xlabel('Stations', size=tick_font_size)
        stats_ax.set_ylabel('Statistics', size=tick_font_size)

        stats_ax.tick_params(labelleft=True,
                             labelbottom=True,
                             labeltop=True,
                             labelright=True)

        plt.setp(stats_ax.get_xticklabels(), size=tick_font_size, rotation=45)
        plt.setp(stats_ax.get_yticklabels(), size=tick_font_size)

        plt.savefig(self.out_var_stats_file,
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.close('all')
        return


if __name__ == '__main__':
    pass
