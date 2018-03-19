# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

from numpy import isnan, where

from matplotlib import use as plt_use
plt_use('AGG')
import matplotlib.pyplot as plt


class PlotInfill:
    def __init__(self, args):
        norm_cop_obj = args[0]
        vars_list = ['curr_infill_stn',
                     'infill_dates',
                     'fig_size_long',
                     'plot_rand_flag',
                     'out_fig_dpi']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self._plot_infill_ser(args[1:])
        return

    def _plot_infill_ser(self, args):
        '''
        Plot what the final series looks like
        '''

        act_var, out_conf_df, out_infill_plot_loc = args

        lw, alpha = 0.8, 0.7

        plt.figure(figsize=self.fig_size_long)
        infill_ax = plt.subplot(111)

        full_data_idxs = isnan(act_var)

        for _conf_head in out_conf_df.columns:
            if (not self.plot_rand_flag) and ('rand' in _conf_head):
                break

            conf_var_vals = where(full_data_idxs,
                                  out_conf_df[_conf_head].loc[
                                          self.infill_dates], act_var)
            infill_ax.plot(self.infill_dates,
                           conf_var_vals,
                           label=_conf_head,
                           alpha=alpha,
                           lw=lw,
                           ls='-',
                           marker='o',
                           ms=lw+0.5)

        infill_ax.plot(self.infill_dates,
                       act_var,
                       label='actual',
                       c='k',
                       ls='-',
                       marker='o',
                       alpha=1.0,
                       lw=lw+0.5,
                       ms=lw+1)

        infill_ax.set_xlabel('Time')
        infill_ax.set_ylabel('var_val')
        infill_ax.set_xlim(self.infill_dates[0], self. infill_dates[-1])
        infill_ax.set_title('Infilled values for station: %s' %
                            self.curr_infill_stn)
        plt.grid()
        plt.legend(framealpha=0.5, loc=0)
        plt.savefig(out_infill_plot_loc,
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.close('all')
        return


if __name__ == '__main__':
    pass
