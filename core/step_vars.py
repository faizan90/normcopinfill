# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""

from numpy import full, nan, isnan, divide, all as np_all, isfinite
# from pandas import Timedelta

from ..misc.misc_ftns import as_err
from ..cyth import norm_ppf_py


class StepVars:

    def __init__(self, infill_steps_obj):

        vars_list = [
            'in_var_df',
            'infill_type',
            'var_le_trs',
            'var_ge_trs',
            'freq']

        for _var in vars_list:
            setattr(self, _var, getattr(infill_steps_obj, _var))
        return

    def get_step_vars(
            self,
            curr_var_df,
            curr_val_cdf_ftns_dict,
            infill_date,
            curr_py_zeros_dict,
            curr_py_dels_dict):

        u_t = full((curr_var_df.shape[1] - 1), nan)
        cur_vals = u_t.copy()

        for i, col in enumerate(curr_var_df.columns):
            # get u_t values or the interp ftns in case of infill_stn
            if i == 0:
                val_cdf_ftn = curr_val_cdf_ftns_dict[col]
                continue

            _curr_var_val = self.in_var_df.loc[infill_date, col]

            assert not isnan(_curr_var_val), as_err('_curr_var_val is NaN!')

            if self.infill_type == 'precipitation':
                if _curr_var_val == self.var_le_trs:
                    values_arr = self.in_var_df.loc[
                        infill_date, curr_var_df.columns[1:]].dropna().values

                    if len(values_arr) > 0:
                        n_wet = (values_arr > self.var_le_trs).sum()
                        wt = divide(n_wet, float(values_arr.shape[0]))

                    else:
                        wt = 0.0

                    _ = curr_py_zeros_dict[col]
                    u_t[i - 1] = norm_ppf_py(_ * (1.0 + wt))

                elif ((_curr_var_val > self.var_le_trs) and
                      (_curr_var_val <= self.var_ge_trs)):
                    u_t[i - 1] = norm_ppf_py(curr_py_dels_dict[col])

                else:
                    u_t[i - 1] = norm_ppf_py(
                        curr_val_cdf_ftns_dict[col](_curr_var_val))

            else:
                u_t[i - 1] = norm_ppf_py(
                    curr_val_cdf_ftns_dict[col](_curr_var_val))

                cur_vals[i - 1] = _curr_var_val

        try:
            assert np_all(isfinite(u_t)), as_err('NaNs in \'u_t\'!')
        except:
            print('Stop!')
        return cur_vals, u_t, val_cdf_ftn


if __name__ == '__main__':
    pass
