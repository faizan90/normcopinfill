"""
Created on %(date)s

@author: %(username)s
"""

from numpy import unique, delete

from ..misc.misc_ftns import pprt


class AdjustNebs:

    def __init__(self, infill_steps_obj):
        vars_list = [
            'max_corr',
            'verbose',
            'curr_infill_stn']

        for _var in vars_list:
            setattr(self, _var, getattr(infill_steps_obj, _var))
        return

    def adj_corrs(
            self,
            infill_date,
            full_corrs_arr,
            norms_df,
            too_hi_corr_stns_list,
            curr_val_cdf_ftns_dict,
            avail_cols_raw,
            avail_cols_fin,
            curr_var_df):

        temp_full_corrs_arr = full_corrs_arr.copy()
        temp_full_corrs_arr[temp_full_corrs_arr >= self.max_corr] = 1.0
        temp_curr_stns = norms_df.columns.tolist()
        del_rows = []
        too_hi_corr_stns = []

        for row in range(temp_full_corrs_arr.shape[0]):
            for col in range(row + 1, temp_full_corrs_arr.shape[1]):
                if temp_full_corrs_arr[row, col] != 1.0:
                    continue

                too_hi_corr_stn = temp_curr_stns[col]
                del_rows.append(col)

                if too_hi_corr_stn not in too_hi_corr_stns:
                    too_hi_corr_stns.append(too_hi_corr_stn)

                if too_hi_corr_stn not in too_hi_corr_stns_list:
                    too_hi_corr_stns_list.append(too_hi_corr_stn)

                if too_hi_corr_stn in curr_val_cdf_ftns_dict:
                    del curr_val_cdf_ftns_dict[too_hi_corr_stn]

                if too_hi_corr_stn in avail_cols_raw:
                    avail_cols_raw.remove(too_hi_corr_stn)

                if too_hi_corr_stn in avail_cols_fin:
                    avail_cols_fin.remove(too_hi_corr_stn)

                if too_hi_corr_stn in curr_var_df.columns:
                    curr_var_df.drop(
                        labels=too_hi_corr_stn, axis=1, inplace=True)

        if not del_rows:
            return temp_full_corrs_arr

        del_rows = unique(del_rows)
        temp_full_corrs_arr = delete(temp_full_corrs_arr, del_rows, axis=0)
        temp_full_corrs_arr = delete(temp_full_corrs_arr, del_rows, axis=1)

        if self.verbose:
            print('\n')
            pprt(['WARNING: A correlation of almost '
                  'equal to one is encountered'],
                 nbh=8,
                 nah=8)

            pprt(['Infill_stn:', self.curr_infill_stn], nbh=12)
            pprt(['Stations with correlation too high:'], nbh=12)

            n_hi_corr = len(too_hi_corr_stns)
            for i_msg in range(0, n_hi_corr, min(2, n_hi_corr)):
                pprt(too_hi_corr_stns[i_msg:(i_msg + 2)], nbh=16)

            pprt(['Infill_date:', infill_date], nbh=12)
            pprt(['These stations are added to the',
                  'temporary high correlations list',
                  'and not used anymore'],
                 nbh=12)
            pprt([''], nbh=24, nah=24)
            print('\n')

        return temp_full_corrs_arr


if __name__ == '__main__':
    pass
