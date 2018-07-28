# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Faizan Anwar, IWS Uni-Stuttgart)s
"""

import os
import timeit
import time
from normcopinfill import NormCopulaInfill

if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    start = timeit.default_timer()  # to get the runtime of the program

    # the working directory
    main_dir = r'P:\Synchronize\IWS\Discharge_data_longer_series'

    # location of input variable and coordinates files
    in_var_file = os.path.join(r'final_q_data_combined_Apr2017',
                               r'neckar_q_data_combined_Apr2017.csv')

    in_coords_file = r'stn_coords/upper_neckar_cats_coords_Apr2017_combined.csv'

    # output directory, everything is saved inside this
    out_dir = r'benchmark_pre_02_oc'

    # time format in in_var_file
    time_fmt = '%Y-%m-%d'

    # names of the stations to be infilled as a list
#     infill_stns = ['427', '406']
    infill_stns = 'all'

    # names of the stations that should not be used in the process
    drop_stns = ['417', '24701', '45409', '76159', '1438', '2465']  # [] #

    # a flag to show what type interval should be infilled
        # slice: infill the station from begining to end of infill_dates_list
        # indiv: infill on specfied dates only
        # all: infill wherever there is a missing value
    infill_date_str = 'slice'

    # the time period for which it should infill, its type depends on infill_date_str
    censor_period = ['1961-01-01', '1980-12-31']

    # the number of minimum days/steps that every station should have prior to infilling
        # so that it is used in the process, stations with steps less than this are
        # dropped
    min_valid_vals = int(365 * 3)
#    min_valid_vals = int(365 * 3 / 30.)

    # type of data used:
        # for continuous data: 'discharge'
        # for discrete and continuous: 'precipitation'
    infill_type = 'discharge'

    # minimum number of neighbouring stations to use while infilling
        # while using the normal copula, n_nrn_min stations are used
        # this behavior can be changed if the parameter 'take_min_stns'
        # of the NormCopulaInfill class is set to True
    n_nrn_min = 1

    # maximum number of neighboring stations to use for infilling
        # it should be greater than n_nrn_min
        # it serves as buffer i.e. may be on some days fewer stations
        # are available so n_nrn_max ensures that we have at least
        # n_nrn_min stations
    n_nrn_max = 10

    # number of processes to initiate
        # should be equal to the number of cores at maximum to
        # achieve fastest speeds
    ncpus = 31

    # the seperator used in the input files
        # it is also used in the output
    sep = ';'

    # frequency of the steps in the input data
        # it can be any valid pandas time frequency object
    freq = 'D'

    os.chdir(main_dir)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    infill_cop = NormCopulaInfill(
        in_var_file=in_var_file,
        out_dir=out_dir,
        infill_stns=infill_stns,
        min_valid_vals=min_valid_vals,
        infill_interval_type=infill_date_str,
        infill_type=infill_type,
        infill_dates_list=censor_period,
        in_coords_file=in_coords_file,
        n_min_nebs=n_nrn_min,
        n_max_nebs=n_nrn_max,
        ncpus=ncpus,
        skip_stns=drop_stns,
        sep=sep,
        time_fmt=time_fmt,
        freq=freq,
        verbose=True)

#     infill_cop.debug_mode_flag = True
#     infill_cop.plot_diag_flag = True
#     infill_cop.plot_step_cdf_pdf_flag = True
    infill_cop.compare_infill_flag = True
    infill_cop.flag_susp_flag = True
#     infill_cop.force_infill_flag = False  # force infill if avail_cols < n_nrst_stns_min
#     infill_cop.plot_neighbors_flag = True
#    infill_cop.take_min_stns_flag = True  # to take n_nrst_stns_min stns or all available
#    infill_cop.overwrite_flag = False
#    infill_cop.read_pickles_flag = True
#    infill_cop.use_best_stns_flag = False
#     infill_cop.dont_stop_flag = False
#     infill_cop.plot_long_term_corrs_flag = True
#    infill_cop.save_step_vars_flag = True
#    infill_cop.plot_rand_flag = True
    infill_cop.stn_based_mp_infill = False

#    infill_cop.nrst_stns_type = 'dist'
    infill_cop.nrst_stns_type = 'rank'

    infill_cop.min_corr = 0.7
#     infill_cop.max_time_lag_corr = 6
#    infill_cop.cut_cdf_thresh = 0.5

#     infill_cop.cmpt_plot_nrst_stns()
#     infill_cop.cmpt_plot_rank_corr_stns()
#    infill_cop.cmpt_plot_symm_stns()
#     infill_cop.plot_stats()
#     infill_cop.plot_ecops()
    infill_cop.infill()
    infill_cop.cmpt_plot_avail_stns()
    infill_cop.plot_summary()

    stop = timeit.default_timer()  # Ending time
    print('\n\a\a\a Done with everything on %s. Total run time was about %0.4f seconds \a\a\a' % (time.asctime(), stop - start))

