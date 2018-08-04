"""
Created on %(date)s

@author: %(username)s
"""
from math import factorial as fac
from itertools import combinations

from numpy import divide


class NebSel:

    def __init__(self, infill_steps_obj):

        vars_list = [
            'curr_infill_stn',
            'curr_nrst_stns',
            'in_var_df',
            'n_min_nebs',
            'min_valid_vals',
            'force_infill_flag',
            'debug_mode_flag']

        for _var in vars_list:
            setattr(self, _var, getattr(infill_steps_obj, _var))
        return

    def get_unique_stns_seqs(self, infill_dates):

        '''
        Make a set of available stations so that if something changes
        we don't have to find the same solution again.
        '''

        avail_stns_per_step_list = []
        sel_stns = [self.curr_infill_stn] + self.curr_nrst_stns

        for infill_date in infill_dates:
            _ = self.in_var_df.loc[infill_date, sel_stns].dropna()

            if self.curr_infill_stn not in _.index:
                continue

            if _.shape[0] < self.n_min_nebs:
                continue

            _ = _.index.drop(self.curr_infill_stn)
            _set = set(_)
            _list = [infill_date, _]

            _date_idx = None
            for i in range(len(avail_stns_per_step_list)):
                _curr_stns = avail_stns_per_step_list[i][1]

                sym_diff = set(_curr_stns) ^ _set
                if not sym_diff:
                    _date_idx = i
                    break

            _list.append(_date_idx)
            _list.append(None)  # will hold best_stns here
            avail_stns_per_step_list.append(_list)

        avail_stns_per_step_dict = {}
        for _list in avail_stns_per_step_list:
            avail_stns_per_step_dict[_list[0]] = _list

        return avail_stns_per_step_dict

    def get_best_stns(
            self,
            best_stns,
            infill_date,
            curr_var_df,
            avail_cols_raw,
            comb_idxs_dict,
            avail_stns_per_step_dict):

        '''
        Select stations based on maximum number of common available steps
        while they are greater than min_valid_vals

        Time to infill increases with increase in n_nrn_min and
        n_nrn_max if use_best_stns_flag is True
        '''

        max_n_combs = 2000

        # best_stns based on intersection of date indicies

        # alg 3 to get best_stns
        # based on intersecting dates rather than the whole dataframe

        if infill_date in list(avail_stns_per_step_dict.keys()):
            _ = avail_stns_per_step_dict[infill_date][3]

        else:
            _ = None

        if _ is not None:
            return _  # best_stns

        for i in range(
            curr_var_df.shape[1] - 1, self.n_min_nebs - 1, -1):

            if (not self.force_infill_flag) and (i < self.n_min_nebs):
                break

            n_nebs = len(avail_cols_raw)
            n_combs = divide(fac(n_nebs), (fac(i) * fac(n_nebs - i)))
            if n_combs > max_n_combs:
                continue

            combs = combinations(avail_cols_raw, i)
            for comb in combs:
                curr_nebs = list(comb)
                fin_idxs = comb_idxs_dict[curr_nebs[0]]

                invalid_comb = False

                for j in range(1, len(curr_nebs)):
                    fin_idxs = fin_idxs.intersection(
                            comb_idxs_dict[curr_nebs[j]])
                    if len(fin_idxs) < self.min_valid_vals:
                        invalid_comb = True
                        break

                if invalid_comb:
                    continue

                best_stns = curr_nebs

                if infill_date in list(avail_stns_per_step_dict.keys()):
                    avail_stns_per_step_dict[infill_date][3] = best_stns
                    idx = avail_stns_per_step_dict[infill_date][2]

                    if idx is None:
                        _ = list(avail_stns_per_step_dict.keys())
                        step_idx = _.index(infill_date)

                        for _step in list(avail_stns_per_step_dict.keys()):
                            _ = avail_stns_per_step_dict[_step][2]

                            if step_idx == _:
                                avail_stns_per_step_dict[_step][3] = best_stns

                if self.debug_mode_flag:
                    print('Found best_stns:', best_stns)
                return best_stns

        else:
            return best_stns


if __name__ == '__main__':
    pass
