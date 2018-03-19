# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

from matplotlib import use as plt_use
plt_use('AGG')
import matplotlib.pyplot as plt


class BefAll:
    def __init__(self, norm_cop_obj):
        vars_list = list(vars(norm_cop_obj).keys())

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.before_all_checks()

        setattr(norm_cop_obj, '_bef_all_chked',
                getattr(self, '_bef_all_chked'))
        return

    def before_all_checks(self):
        assert isinstance(self.n_discret, int), '\'n_discret\' is non-integer!'
        assert isinstance(self.n_round, int), '\'n_round\' is non-integer!'
        assert isinstance(self.cop_bins, int), '\'cop_bins\' is non-integer!'
        assert isinstance(self.min_corr, float), '\'max_corr\' is non-float!'
        assert isinstance(self.max_corr, float), '\'max_corr\' is non-float!'
        assert isinstance(self.ks_alpha, float), '\'ks_alpha\' is non-float!'
        assert isinstance(self.out_fig_dpi, int), \
            '\'out_fig_dpi\' is non-integer!'
        assert isinstance(self.out_fig_fmt, str), \
            '\'out_fig_dpi\' is non-unicode!'
        assert isinstance(self.n_norm_symm_flds, int), \
            '\'n_norm_symm_flds\' is a non-integer!'
        assert isinstance(self.n_rand_infill_values, int), \
            '\'n_rand_infill_values\' is non-integer'
        assert isinstance(self.thresh_mp_steps, int), \
            '\'thresh_mp_steps\' is non-integer'
        assert isinstance(self.max_time_lag_corr, int), \
            '\'max_time_lag_corr\' is non-integer'

        assert self.n_discret > 0, '\'n_discret\' less than 1!'
        assert self.n_norm_symm_flds > 0, '\'n_norm_symm_flds\' less than 1!'
        assert self.out_fig_dpi > 0, '\'out_fig_dpi\' less than 1!'
        assert self.n_round >= 0, '\'n_round\' less than 0!'
        assert self.cop_bins > 0, '\'cop_bins\' less than 1!'
        assert (self.max_corr >= 0.0) and (self.max_corr <= 1.0), \
            '\'max_corr\' (%s) not between 0 and 1' % str(self.max_corr)
        assert (self.min_corr >= 0) and (self.min_corr <= 1.0), \
            '\'max_corr\' (%s) not between 0 and 1' % str(self.max_corr)
        assert (self.ks_alpha > 0) and (self.ks_alpha < 1.0), \
            '\'ks_alpha\' (%s) not between 0 and 1' % str(self.ks_alpha)

        if self.infill_type == 'discharge-censored':
            assert isinstance(self.cut_cdf_thresh, float), \
                '\'cut_cdf_thresh\' is non-float!'
            assert (self.cut_cdf_thresh > 0) and (self.cut_cdf_thresh < 1.0), \
                ('\'cut_cdf_thresh\' (%s) not between 0 and 1' %
                 str(self.cut_cdf_thresh))
        else:
            self.cut_cdf_thresh = None

        if self.n_rand_infill_values:
            _c1 = self.n_rand_infill_values > 0
            _c2 = self.n_rand_infill_values < 10000
            assert _c1 and _c2, \
                (('\'n_rand_infill_values\' (%s) not between '
                  '0 and 10000') % str(self.n_rand_infill_values))

        if ((self.infill_type == 'precipitation') or
            (self.infill_type == 'discharge-censored')):
            assert self.nrst_stns_type != 'symm', \
                ('\'infill_type\' cannot be \'symm\' for precipitation '
                 'or censored-discharge!')

        if ((self.infill_type == 'precipitation') or
            (self.infill_type == 'discharge-censored')):
            assert self.nrst_stns_type == 'rank', \
                ('\'nrst_stns_type\' can only be \'rank\' for the given '
                 '\'infill_type\'!')

        if self.max_time_lag_corr:
            assert self.nrst_stns_type == 'rank', \
                ('\'nrst_stns_type\' can only be \'rank\' for the given '
                 '\'max_time_lag_corr\'!')

            assert self.max_time_lag_corr > 0, \
                '\'max_time_lag_corr\' cannot be less than zero!'

        assert hasattr(self.fig_size_long, '__iter__'), \
            '\'fig_size_long\' not an iterable!'
        assert len(self.fig_size_long) == 2, \
            'Only two values allowed inside \'fig_size_long\'!'

        fig = plt.figure()
        supp_fmts = list(fig.canvas.get_supported_filetypes().keys())
        assert self.out_fig_fmt in supp_fmts, \
            ('\'out_fig_fmt\' (%s) not in the supported formats list (%s)' %
             (self.out_fig_fmt, supp_fmts))

        plt.close('all')
        self._bef_all_chked = True
        return


if __name__ == '__main__':
    pass
