# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: Faizan Anwar, IWS Uni-Stuttgart
"""
from pandas import DataFrame

from ..misc.misc_ftns import as_err, get_lag_ser

import pyximport
pyximport.install()
from normcop_cyftns import fill_correl_mat, get_corrcoeff


class BestLagCorrs:

    def __init__(self, norms_df, max_time_lag_corr):
        assert isinstance(norms_df, DataFrame), as_err(
            'norms_df not a DataFrame!')

        self.norms_df = norms_df.copy()
        self.infill_stn = norms_df.columns[0]
        self.nebs = norms_df.columns[1:]

        self.max_time_lag_corr = max_time_lag_corr
        self.forw_time_lags = 0
        self.back_time_lags = 0

        self.best_stn_lags_dict = {}
        self.full_corrs_arr = None
        return

    def _get_best_lags(self):
        ser_i = self.norms_df[self.infill_stn].copy()

        for j_stn in self.nebs:
            max_correl = 0.0
            best_lag = 0

            for curr_time_lag in range(-self.max_time_lag_corr,
                                       +self.max_time_lag_corr + 1):
                _ = self.norms_df[j_stn].copy()
                ser_j = get_lag_ser(_, curr_time_lag)

                ij_df = DataFrame(index=ser_i.index,
                                  data={'i': ser_i.values,
                                        'j': ser_j.values})
                ij_df.dropna(axis=0, how='any', inplace=True)

                correl = get_corrcoeff(ij_df['i'].values, ij_df['j'].values)

                if (abs(correl) > max_correl):
                    max_correl = correl
                    best_lag = curr_time_lag

            self.best_stn_lags_dict[j_stn] = best_lag

            if ((best_lag > 0) and (best_lag > self.forw_time_lags)):
                self.forw_time_lags = best_lag
            elif ((best_lag < 0) and (best_lag < self.back_time_lags)):
                self.back_time_lags = best_lag
        return

    def _lag_norms_df(self):
        assert self.best_stn_lags_dict, as_err('Nothing to lag!')

        for j_stn in self.nebs:
            get_lag_ser(self.norms_df[j_stn], self.best_stn_lags_dict[j_stn])
        return

    def fill_lag_correl_mat(self):
        self._get_best_lags()
        self._lag_norms_df()

        if self.forw_time_lags:
            self.norms_df = self.norms_df.iloc[self.forw_time_lags:, :]
        if self.back_time_lags:
            self.norms_df = self.norms_df.iloc[:self.back_time_lags, :]
        self.full_corrs_arr = fill_correl_mat(self.norms_df.values)
        return (self.norms_df, self.best_stn_lags_dict, self.full_corrs_arr)
