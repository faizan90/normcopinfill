'''
Created on Nov 11, 2018

@author: Faizan
'''
import timeit
from itertools import combinations

from numpy import intersect1d, where, zeros_like
from pandas import Series, to_datetime, Timedelta, Timestamp

TD = Timedelta('1s')  # time delta
MIN_T = Timestamp("1970-01-01")  # reference time for counting


class InfillDatesNeborsSets:

    def __init__(self, sub_in_var_df, norm_cop_obj):

        vars_list = [
            'n_min_nebs',
            'n_max_nebs',
            'min_valid_vals',
            'force_infill_flag',
            'take_min_stns_flag',
            'compare_infill_flag',
            'infill_dates']

        beg_time = timeit.default_timer()

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))

        self.in_var_df = sub_in_var_df
        self.curr_infill_stn = sub_in_var_df.columns[0]
        # FIXME: arrange nrst_stns such that while making combinations,
        # stns with hi_corr come first
        self.curr_nrst_stns = sub_in_var_df.columns[1:]

        # hashing speeds are faster for integers
        old_var_index = self.in_var_df.index
        new_var_index = (old_var_index - MIN_T) // TD

        if self.compare_infill_flag:
            new_infill_index = (self.infill_dates - MIN_T) // TD

        else:
            ser = self.in_var_df.loc[
                self.infill_dates, self.curr_infill_stn]

            nan_idxs = ser.isna().values

            old_infill_index = ser[nan_idxs].index
            new_infill_index = (old_infill_index - MIN_T) // TD

        self.in_var_df.index = new_var_index
        self.infill_dates = new_infill_index

        self.infill_stn_idx = (
            self.in_var_df.loc[:, self.curr_infill_stn].dropna().index)

        self.infill_stn_dates_nebs_sets_dict = {}  # filled by _make_sets

        print('\nStn: %s' % self.curr_infill_stn)
        self._make_sets()
        end_time = timeit.default_timer()

        print('Took %0.4f secs in InfillDatesNeborsSets!' % (
            end_time - beg_time))
        return

    def _make_sets(self):

        stn_valid_idxs_dict = self._get_stn_valid_idxs()

        can_infill_ser = Series(
            index=self.infill_dates,
            data=zeros_like(self.infill_dates, dtype=bool))

        if self.force_infill_flag:
            n_min_nebs = 1

        else:
            n_min_nebs = self.n_min_nebs

        if self.take_min_stns_flag:
            n_max_nebs = n_min_nebs

        else:
            n_max_nebs = self.n_max_nebs

        n_min_nebs = min(n_min_nebs, self.curr_nrst_stns.shape[0])
        n_max_nebs = min(n_max_nebs, self.curr_nrst_stns.shape[0])

        rem_dates = can_infill_ser.shape[0]
        min_valid_vals = self.min_valid_vals

        infill_stn_idx = self.infill_stn_idx

        set_ctr = 0
        for n_nebs in range(n_max_nebs, n_min_nebs - 1, -1):

            if rem_dates <= 0:
                break

            combs = combinations(self.curr_nrst_stns, n_nebs)

            for comb in combs:
                if rem_dates <= 0:
                    break

                cmn_dates = infill_stn_idx
                for stn in comb:
                    cmn_dates = intersect1d(
                        cmn_dates, stn_valid_idxs_dict[stn])

                    if cmn_dates.shape[0] <= min_valid_vals:
                        break

                # better solution?
                if cmn_dates.shape[0] <= min_valid_vals:
                    break

                can_idxs = where(~can_infill_ser.values)[0]
                itsct_idxs = can_infill_ser.iloc[can_idxs].index

                for stn in comb:
                    itsct_idxs = intersect1d(
                        itsct_idxs, stn_valid_idxs_dict[stn])

                    if not itsct_idxs.shape[0]:
                        break

                if not itsct_idxs.shape[0]:
                    continue

                can_infill_ser[itsct_idxs] = True

                set_str = 'set%d' % set_ctr

                set_dates = to_datetime(itsct_idxs, unit='s')

                set_tuple = (set_dates, comb)
                self.infill_stn_dates_nebs_sets_dict[set_str] = set_tuple

                rem_dates -= set_dates.shape[0]

                set_ctr += 1
                print('comb:', comb, cmn_dates.shape[0], set_dates.shape[0])

        print('rem_dates: %d, n_sets: %d' % (rem_dates, set_ctr))
#         self.infill_stn_dates_nebs_sets_dict['rem_dates'] = rem_dates
        assert set_ctr, 'No sets created!'
        return

    def _get_stn_valid_idxs(self):

        out_dict = {}
        for stn in self.curr_nrst_stns:
            stn_index = self.in_var_df.loc[:, stn].dropna().index
            out_dict[stn] = stn_index

        return out_dict
