'''
Created on Nov 11, 2018

@author: Faizan
'''
import timeit
from math import ceil
from copy import deepcopy
from random import shuffle
from itertools import combinations

from numpy import intersect1d, where, unique, zeros_like
from pandas import Series, to_datetime, Timedelta, Timestamp

TD = Timedelta('1s')  # time delta
MIN_T = Timestamp("1970-01-01")  # reference time for counting


class InfillDatesNeborsSets:

    '''Make sets of steps on which a set of nebors is active'''

    def __init__(self, norm_cop_obj):

        vars_list = [
            'n_min_nebs',
            'n_max_nebs',
            'min_valid_vals',
            'force_infill_flag',
            'take_min_stns_flag',
            'compare_infill_flag',
            'ncpus',
            'thresh_mp_steps',
            'stn_based_mp_infill_flag',
            'debug_mode_flag']

        for _var in vars_list:
            setattr(self, _var, getattr(norm_cop_obj, _var))
        return

    def get_sets(self, args):

        (stns,
         stn_avail_dates_dict,
         infill_dates) = args

        beg_time = timeit.default_timer()

        self.curr_infill_stn = stns[0]
        self.curr_nrst_stns = stns[1:]

        stn_avail_int_date_dict = {}
        for stn in stns:
            # hashing speeds are faster for integers
            stn_avail_int_date_dict[stn] = (
                stn_avail_dates_dict[stn] - MIN_T) // TD

        self.stn_avail_int_date_dict = stn_avail_int_date_dict

        self.infill_dates = (infill_dates - MIN_T) // TD

        self.infill_stn_idx = self.stn_avail_int_date_dict[
            self.curr_infill_stn]

        self.raw_infill_stn_dates_nebs_sets_dict = {}
        self.infill_stn_dates_nebs_sets_dict = {}

        print('\nStn: %s' % self.curr_infill_stn)

        self._make_sets()

        end_time = timeit.default_timer()

        print('Took %0.4f secs in InfillDatesNeborsSets!' % (
            end_time - beg_time))

        return self.curr_infill_stn, self.infill_stn_dates_nebs_sets_dict

    def _make_sets(self):

        '''
        - Make all possible combinations of nebors.
        - Start with the biggest combination possible.
        - Choose steps on which the nebors are active
        - Put the time(s) and nebor(s) in a tuple.
        - Do till no more nebors combination or no more infill_dates left.
        '''

        stn_valid_idxs_dict = self.stn_avail_int_date_dict

        can_infill_ser = Series(
            index=self.infill_dates,
            data=zeros_like(self.infill_dates, dtype=bool))

        n_min_nebs = self.n_min_nebs

        if self.take_min_stns_flag:
            n_max_nebs = n_min_nebs

        else:
            n_max_nebs = self.n_max_nebs

        n_min_nebs = min(n_min_nebs, len(self.curr_nrst_stns))
        n_max_nebs = min(n_max_nebs, len(self.curr_nrst_stns))

        if self.force_infill_flag:
            n_min_nebs = 1

        rem_dates = can_infill_ser.shape[0]
        min_valid_vals = self.min_valid_vals

        infill_stn_idx = self.infill_stn_idx

        set_ctr = 0
        combs_ctr = 0
        for n_nebs in range(n_max_nebs, n_min_nebs - 1, -1):

            if rem_dates <= 0:
                break

            combs = combinations(self.curr_nrst_stns, n_nebs)

            for comb in combs:
                combs_ctr += 1

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
                self.raw_infill_stn_dates_nebs_sets_dict[set_str] = set_tuple

                rem_dates -= set_dates.shape[0]

                set_ctr += 1
                print('comb:', comb, cmn_dates.shape[0], set_dates.shape[0])

#         print('rem_dates: %d, n_sets: %d, n_combs: %d' % (
#             rem_dates, set_ctr, combs_ctr))

        if not set_ctr:
            pass
#             print('No sets created!')

        else:
            self._distribute_load()
        return

    def _distribute_load(self):

        '''After finding optimal combination of nebors and time steps,
        Divide them into groups such that each thread will have an equal
        number of steps to work with in case of MP.
        '''

        nvals_to_infill = 0

        raw_dict = self.raw_infill_stn_dates_nebs_sets_dict
        n_raw_tups = len(raw_dict)

        for tup in raw_dict:
            nvals_to_infill += raw_dict[tup][0].shape[0]

        if ((nvals_to_infill > self.thresh_mp_steps) and
            not self.stn_based_mp_infill_flag):

            use_mp_infill = True

        else:
            use_mp_infill = False

        if ((self.ncpus == 1) or
            (not use_mp_infill) or
            self.debug_mode_flag):

            steps_pr_grp = nvals_to_infill
            grps_dict = {'grp%d' % i: {} for i in range(1)}

            new_sets_order = list(range(n_raw_tups))

        else:
            steps_pr_grp = ceil(nvals_to_infill / self.ncpus)
            grps_dict = {'grp%d' % i: {} for i in range(self.ncpus)}

            new_sets_order = InfillDatesNeborsSets._get_new_order(
                raw_dict, steps_pr_grp, self.ncpus, n_raw_tups)

        # just in case
        assert unique(new_sets_order).shape[0] == n_raw_tups

        tot_grpd_vals = 0
        strt_set_idx = 0

        for grp in grps_dict:
            steps_in_grp = 0
            sets_list = []

            for i in range(strt_set_idx, n_raw_tups):
                set_key = 'set%d' % new_sets_order[i]

                tup = raw_dict[set_key]
                nvals = tup[0].shape[0]

                if (steps_in_grp + nvals) > steps_pr_grp:

                    vals_to_take = steps_pr_grp - steps_in_grp

                    sets_list.append((tup[0][:vals_to_take], tup[1]))

                    raw_dict[set_key] = (
                        tup[0][vals_to_take:], tup[1])

                    tot_grpd_vals += vals_to_take
                    break

                else:

                    sets_list.append(tup)

                    del raw_dict[set_key]
                    strt_set_idx = i + 1

                    tot_grpd_vals += tup[0].shape[0]
                    steps_in_grp += tup[0].shape[0]

            nsteps = 0
            for tup in sets_list:
                nsteps += tup[0].shape[0]

            assert nsteps <= steps_pr_grp

            if sets_list:
                grps_dict[grp] = sets_list

            else:
                raise Exception('Empty sets_list?')

        assert tot_grpd_vals == nvals_to_infill

        assert not raw_dict
        print('ngrps:', len(grps_dict))

        self.infill_stn_dates_nebs_sets_dict = grps_dict
        return

    @staticmethod
    def _get_new_order(raw_dict, steps_pr_grp, ncpus, n_raw_tups):

        '''Find such a combinations of infill_dates and nrst_stns such that
        each group get steps_pr_grp and about equal number of nrst_stns sets.

        Test random combinations.
        '''

        rand_sets_order = list(range(n_raw_tups))
        new_set_order = deepcopy(rand_sets_order)

        obj_max = max(rand_sets_order)

        max_iters = n_raw_tups ** 2
        max_usuc_iters = n_raw_tups ** 1.5
        nusuc_iters = 0

        for i in range(max_iters):

            tot_grpd_vals = 0
            strt_set_idx = 0

            set_ctrs = []

            raw_ctr_dict = {}
            for key in raw_dict:
                val = raw_dict[key]
                raw_ctr_dict[key] = (val[0].shape[0], val[1])

            for _ in range(ncpus):
                steps_in_grp = 0
                sets_list = []

                for i in range(strt_set_idx, n_raw_tups):
                    set_key = 'set%d' % rand_sets_order[i]

                    tup = raw_ctr_dict[set_key]
                    nvals = tup[0]

                    if (steps_in_grp + nvals) > steps_pr_grp:

                        vals_to_take = steps_pr_grp - steps_in_grp

                        sets_list.append((vals_to_take, tup[1]))

                        raw_ctr_dict[set_key] = (
                            tup[0] - vals_to_take, tup[1])

                        tot_grpd_vals += vals_to_take
                        break

                    else:

                        sets_list.append(tup)

                        strt_set_idx = i + 1

                        tot_grpd_vals += tup[0]
                        steps_in_grp += tup[0]

                nsteps = 0
                for tup in sets_list:
                    nsteps += tup[0]

                assert nsteps <= steps_pr_grp

                set_ctrs.append(len(sets_list))

            ctrs_max = max(set_ctrs)
            if ctrs_max < obj_max:
                obj_max = ctrs_max
                nusuc_iters = 0

                new_set_order = deepcopy(rand_sets_order)

            else:
                nusuc_iters += 1

            shuffle(rand_sets_order)

            if nusuc_iters >= max_usuc_iters:
                break

        return new_set_order
