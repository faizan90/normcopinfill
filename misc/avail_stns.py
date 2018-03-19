# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from matplotlib import use as plt_use
plt_use('AGG')
import matplotlib.pyplot as plt
from pandas import DataFrame

from ..misc.misc_ftns import as_err


class AvailStns:
    def __init__(self, norm_cop_obj):
        vars_list = ['in_var_df_orig',
                     'out_var_df',
                     'out_var_dfs_list',
                     'infill_stns',
                     'n_rand_infill_values',
                     '_infilled',
                     'out_stns_avail_fig',
                     'out_fig_dpi',
                     'fig_size_long',
                     'out_stns_avail_file',
                     'sep',
                     'verbose']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.cmpt_plot_avail_stns()
        return

    def cmpt_plot_avail_stns(self):
        '''
        To compare the number of stations before and after infilling
        '''
        if self.verbose:
            print('INFO: Computing and plotting the number of stations '
                  'available per step...')

        assert self._infilled, as_err('Call \'infill\' first!')

        avail_nrst_stns_orig_ser = self.in_var_df_orig.count(axis=1)

        avail_nrst_stns_orig_ser_infilled = \
            self.in_var_df_orig[self.infill_stns].count(axis=1)

        if not self.n_rand_infill_values:
            avail_nrst_stns_ser = self.out_var_df.count(axis=1)

            avail_nrst_stns_ser_infilled = \
                self.out_var_df[self.infill_stns].count(axis=1)
        else:
            avail_nrst_stns_ser = self.out_var_dfs_list[0].count(axis=1)

            avail_nrst_stns_ser_infilled = \
                self.out_var_dfs_list[0][self.infill_stns].count(axis=1)

        assert avail_nrst_stns_orig_ser.sum() > 0, \
            as_err('in_var_df is empty!')
        assert avail_nrst_stns_ser.sum() > 0, \
            as_err('out_var_df is empty!')

        assert avail_nrst_stns_orig_ser_infilled.sum() > 0, \
            as_err('in_var_df_infilled is empty!')
        assert avail_nrst_stns_ser_infilled.sum() > 0, \
            as_err('out_var_df_infilled is empty!')

        _out_labs_list = ['Original (all stations)',
                          'Infilled (all stations)',
                          'Original (infilled stations only)',
                          'Infilled (infilled stations only)']

        plt.figure(figsize=self.fig_size_long)
        plt.plot(avail_nrst_stns_orig_ser.index,
                 avail_nrst_stns_orig_ser.values,
                 alpha=0.8, label=_out_labs_list[0])
        plt.plot(avail_nrst_stns_ser.index,
                 avail_nrst_stns_ser.values,
                 alpha=0.8, label=_out_labs_list[1])

        plt.plot(avail_nrst_stns_orig_ser_infilled.index,
                 avail_nrst_stns_orig_ser_infilled.values,
                 alpha=0.8, label=_out_labs_list[2])
        plt.plot(avail_nrst_stns_ser_infilled.index,
                 avail_nrst_stns_ser_infilled.values,
                 alpha=0.8, label=_out_labs_list[3])

        plt.xlabel('Time')
        plt.ylabel('Number of stations with valid values')
        plt.legend(framealpha=0.5)
        plt.grid()
        plt.savefig(self.out_stns_avail_fig,
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.close()

        out_index = avail_nrst_stns_ser.index.union(
                    avail_nrst_stns_orig_ser.index)

        fin_df = DataFrame(index=out_index,
                           dtype=float,
                           columns=_out_labs_list)

        fin_df[_out_labs_list[0]] = avail_nrst_stns_orig_ser
        fin_df[_out_labs_list[1]] = avail_nrst_stns_ser

        fin_df[_out_labs_list[2]] = avail_nrst_stns_orig_ser_infilled
        fin_df[_out_labs_list[3]] = avail_nrst_stns_ser_infilled

        fin_df.to_csv(self.out_stns_avail_file,
                      sep=str(self.sep), encoding='utf-8')
        return


if __name__ == '__main__':
    pass
