# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from os.path import join as os_join

from numpy import repeat, tile
from matplotlib import use as plt_use
plt_use('AGG')
import matplotlib.cm as cmaps
import matplotlib.pyplot as plt

from adjustText import adjust_text


class PlotCondFtns:
    def __init__(self, infill_steps_obj):
        vars_list = ['infill_type',
                     'curr_infill_stn',
                     'out_fig_dpi',
                     'out_fig_fmt',
                     'stn_step_corrs_dir',
                     'stn_infill_cdfs_dir',
                     'stn_infill_pdfs_dir']

        for _var in vars_list:
            setattr(self, _var, getattr(infill_steps_obj, _var))
        return

    def plot_cond_cdf_pdf(self,
                          val_arr_adj,
                          gy_arr_adj,
                          conf_vals,
                          conf_probs,
                          date_pref,
                          py_zero,
                          py_del,
                          pdf_arr_adj,
                          conf_grads):
        # plot infill cdf
        plt.figure()
        plt.plot(val_arr_adj, gy_arr_adj)
        plt.scatter(conf_vals, conf_probs)
        if self.infill_type == 'precipitation':
            plt.title(('infill CDF\n stn: %s, date: %s\npy_zero: %0.2f, '
                       'py_del: %0.2f') %
                      (self.curr_infill_stn, date_pref, py_zero, py_del))
        else:
            plt.title('infill CDF\n stn: %s, date: %s' %
                      (self.curr_infill_stn, date_pref))
        plt.grid()
        plt.xlabel('var_val')
        plt.ylabel('CDF_val')

        plt_texts = []
        for i in range(conf_probs.shape[0]):
            plt_texts.append(plt.text(conf_vals[i],
                                      conf_probs[i],
                                      ('var_%0.2f: %0.2f' %
                                       (conf_probs[i], conf_vals[i])),
                                      va='top',
                                      ha='left'))

        adjust_text(plt_texts)

        _ = 'infill_CDF_%s_%s.%s' % (self.curr_infill_stn,
                                     date_pref,
                                     self.out_fig_fmt)
        out_val_cdf_loc = os_join(self.stn_infill_cdfs_dir, _)
        plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)
        plt.savefig(out_val_cdf_loc,
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.clf()

        # plot infill pdf
        plt.plot(val_arr_adj, pdf_arr_adj)
        plt.scatter(conf_vals, conf_grads)
        if self.infill_type == 'precipitation':
            plt.title(('infill PDF\n stn: %s, date: %s\npy_zero: %0.2f, '
                       'py_del: %0.2f') %
                      (self.curr_infill_stn, date_pref, py_zero, py_del))
        else:
            plt.title('infill PDF\n stn: %s, date: %s' %
                      (self.curr_infill_stn, date_pref))
        plt.grid()
        plt.xlabel('var_val')
        plt.ylabel('PDF_val')

        plt_texts = []
        for i in range(conf_probs.shape[0]):
            plt_texts.append(plt.text(conf_vals[i],
                                      conf_grads[i],
                                      ('var_%0.2f: %0.2e' %
                                       (conf_probs[i],
                                        conf_grads[i])),
                                      va='top',
                                      ha='left'))

        adjust_text(plt_texts)
        _ = 'infill_PDF_%s_%s.%s' % (self.curr_infill_stn,
                                     date_pref,
                                     self.out_fig_fmt)
        out_val_pdf_loc = os_join(self.stn_infill_pdfs_dir, _)
        plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.85)
        plt.savefig(out_val_pdf_loc,
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.close('all')
        return

    def plot_corr_mat(self,
                      full_corrs_arr,
                      curr_var_df,
                      date_pref):
        # plot corrs
        plt.figure()
        tick_font_size = 3
        n_stns = full_corrs_arr.shape[0]

        corrs_ax = plt.subplot(111)
        corrs_ax.matshow(full_corrs_arr,
                         vmin=0,
                         vmax=1,
                         cmap=cmaps.Blues,
                         origin='lower')
        for s in zip(repeat(list(range(n_stns)), n_stns),
                     tile(list(range(n_stns)), n_stns)):
            corrs_ax.text(s[1],
                          s[0],
                          '%0.2f' % (full_corrs_arr[s[0], s[1]]),
                          va='center',
                          ha='center',
                          fontsize=tick_font_size)

        corrs_ax.set_xticks(list(range(0, n_stns)))
        corrs_ax.set_xticklabels(curr_var_df.columns)
        corrs_ax.set_yticks(list(range(0, n_stns)))
        corrs_ax.set_yticklabels(curr_var_df.columns)

        corrs_ax.spines['left'].set_position(('outward', 10))
        corrs_ax.spines['right'].set_position(('outward', 10))
        corrs_ax.spines['top'].set_position(('outward', 10))
        corrs_ax.spines['bottom'].set_position(('outward', 10))

        corrs_ax.tick_params(labelleft=True,
                             labelbottom=True,
                             labeltop=True,
                             labelright=True)

        plt.setp(corrs_ax.get_xticklabels(),
                 size=tick_font_size,
                 rotation=45)
        plt.setp(corrs_ax.get_yticklabels(), size=tick_font_size)

        _ = 'stn_corrs_%s.%s' % (date_pref, self.out_fig_fmt)
        out_corrs_fig_loc = os_join(self.stn_step_corrs_dir, _)
        plt.savefig(out_corrs_fig_loc,
                    dpi=self.out_fig_dpi,
                    bbox_inches='tight')
        plt.close('all')
        return


if __name__ == '__main__':
    pass
