# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
from numpy import (array,
                   isfinite,
                   all as np_all,
                   where,
                   any as np_any,
                   ediff1d,
                   isclose,
                   nan)
from pandas import Series

from ..misc.misc_ftns import as_err


class ConfInfill:

    def __init__(self, norm_cop_obj):
        vars_list = ['conf_probs',
                     'adj_prob_bounds',
                     'conf_heads',
                     'flag_probs',
                     'fin_conf_head',
                     'verbose',
                     'flag_susp_flag',
                     '_conf_ser_cmptd']

        reassign_vars_list = ['conf_probs',
                              'adj_prob_bounds',
                              'conf_heads',
                              'flag_probs',
                              'fin_conf_head',
                              'conf_ser',
                              '_conf_ser_cmptd']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.prep_conf_ser()

        # this happens after everything is done
        for _var in reassign_vars_list:
            setattr(norm_cop_obj, _var, getattr(self, _var))
        return

    def prep_conf_ser(self):
        '''
        Check if all the variables for the calculation of confidence intervals
        are correct
        '''
        try:
            self.conf_probs = array(self.conf_probs, dtype=float)
        except:
            raise ValueError(('Some or all \'_conf_probs\' are '
                              'non-float: %s' % str(self.conf_probs)))

        try:
            self.adj_prob_bounds = array(self.adj_prob_bounds, dtype=float)
        except:
            raise ValueError(('Some or all \'_adj_prob_bounds\' are '
                              'non-float: %s' % str(self.adj_prob_bounds)))

        try:
            self.conf_heads = array(self.conf_heads, dtype=str)
        except:
            raise ValueError(('Some or all \'_conf_heads\' are '
                              'non-unicode: %s' % str(self.conf_heads)))

        try:
            self.flag_probs = array(self.flag_probs, dtype=float)
        except:
            raise ValueError(('Some or all \'_flag_probs\' are '
                              'non-float: %s' % str(self.flag_probs)))

        assert self.conf_probs.shape[0] >= 2, (
            as_err('\'_conf_probs\' cannot be less than 2 values!'))
        assert self.conf_probs.shape == self.conf_heads.shape, (
            as_err(('\'_conf_probs\' (%s) and \'_conf_heads\' (%s) '
                    'cannot have different shapes!' %
                   (self.conf_probs.shape, self.conf_heads.shape))))
        assert isfinite(np_all(self.conf_probs)), (
            'Invalid values in \'_conf_probs\': %s' % str(self.conf_probs))

        assert np_any(where(ediff1d(self.conf_probs) > 0, 1, 0)), (
            as_err('\'_conf_probs\' not ascending (%s)!' %
                   str(self.conf_probs)))

        assert max(self.conf_probs) <= self.adj_prob_bounds[1], (
            as_err(('max \'adj_prob_bounds\' < max '
                    '\'_conf_probs\' (%s)!') %
                   str(self.adj_prob_bounds)))

        assert min(self.conf_probs) > self.adj_prob_bounds[0], (
            as_err(('min \'adj_prob_bounds\' > min '
                    '\'_conf_probs\' (%s)!') %
                   str(self.adj_prob_bounds)))

        assert isfinite(np_all(self.flag_probs)), (
            as_err('Invalid values in \'_flag_probs\': %s' %
                   str(self.flag_probs)))

        assert isfinite(np_all(self.adj_prob_bounds)), (
            as_err('Invalid values in \'_adj_prob_bounds\': %s' %
                   str(self.adj_prob_bounds)))

        assert len(self.adj_prob_bounds) == 2, (
            as_err('\'adj_bounds_probs\' are not two values (%s)!' %
                   str(self.adj_prob_bounds)))

        assert self.adj_prob_bounds[0] < self.adj_prob_bounds[1], (
            as_err(('\'adj_bounds_probs\' not ascending (%s)!') %
                   str(self.adj_prob_bounds)))

        assert self.fin_conf_head in self.conf_heads, (
            as_err(('\'_fin_conf_head\': %s not in '
                    '\'_conf_heads\': %s') %
                   (self.fin_conf_head, self.conf_heads)))

        if self.verbose:
            print(('INFO: Using \'%s\' as the final infill value' %
                   str(self.fin_conf_head)))

        if self.flag_susp_flag:
            assert len(self.flag_probs) == 2, as_err(
                'Only two values allowed inside \'_flag_probs\'!')
            assert self.flag_probs[0] < self.flag_probs[1], as_err(
                '\'_flags_probs\': first value should be smaller '
                'than the last!')

            for _flag_val in self.flag_probs:
                assert np_any(isclose(_flag_val, self.conf_probs)), as_err(
                    ('\'_flag_prob\' (%s) value not in '
                     '\'_conf_probs\' (%s)!') %
                     (str(_flag_val), str(self.conf_probs)))

        self.conf_ser = Series(index=self.conf_heads,
                               data=self.conf_probs,
                               dtype=float)

        self._conf_ser_cmptd = True
        return


if __name__ == '__main__':
    pass
