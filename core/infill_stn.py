'''
The Normal copula infilling

Faizan Anwar, IWS

'''

import timeit
from sys import exc_info
from datetime import datetime
from os import mkdir as os_mkdir
from os.path import exists as os_exists, join as os_join

from numpy import (any as np_any,
                   divide,
                   intersect1d,
                   isnan,
                   linspace,
                   logical_not,
                   seterr,
                   set_printoptions,
                   where)
from pathos.multiprocessing import ProcessPool as mp_pool
import matplotlib.pyplot as plt

from pandas import (date_range,
                    read_csv,
                    to_datetime,
                    to_numeric,
                    DataFrame,
                    Index)

from .infill_steps import InfillSteps
from .plot_infill import PlotInfill
from .compare_infill import CompareInfill
from .flag_susp import FlagSusp
from ..checks.conf import ConfInfill
from ..checks.bef_all import BefAll
from ..nebors.nrst_nebs import NrstStns
from ..nebors.rank_corr_nebs import RankCorrStns
from ..nebors.rank_corr_nebs_time_lag import BestLagRankCorrStns
from ..ecops.plot_ecops import ECops
from ..misc.plot_stats import PlotStats
from ..misc.avail_stns import AvailStns
from ..misc.summary import Summary
from ..misc.misc_ftns import pprt, as_err, full_tb
from ..misc.std_logger import StdFileLoggerCtrl

plt.ioff()
set_printoptions(precision=6,
                 threshold=2000,
                 linewidth=200000,
                 formatter={'float': '{:0.6f}'.format})

seterr(all='ignore')

__all__ = ['NormCopulaInfill']


class NormCopulaInfill:
    '''
    Implementation of Andras Bardossy, Geoffrey Pegram, Infilling missing
    precipitation records - A comparison of a new copula-based method with
    other techniques, Journal of Hydrology, Volume 519, Part A,
    27 November 2014, Pages 1162-1170

    Description
    -----------

    To infill missing time series data of a given station(s) using
    neighboring stations and the multivariate normal copula.
    Can be used for infilling time series data that acts like stream
    discharge or precipitation. After initiation, calling the \'infill\'
    method will start infilling based on the criteria explained below.

    Parameters
    ----------
    in_var_file: string
        Location of the file that holds the input time series data.
        The file should have its first column as time. The header
        should be the names of the stations. Any valid separator is
        allowed. The first row should be the name of the stations.
    in_coords_file: string
        Location of the file that has stations' coordinates.
        The names of the stations should be the first column. The header
        should have the \'X\' for eastings, \'Y\' for northings. Rest
        is ignored. Any valid separator is allowed.
    out_dir: string
        Location of the output directory. Will be created if it does
        not exist. All the ouputs are stored inside this directory.
    infill_stns: list_like
        Names of the stations that should be infilled. These names
        should be in the in_var_file header and the index of
        in_coords_file.
    min_valid_vals: integer
        The minimum number of the union of one station with others
        so that all have valid values. This is different in different
        cases. e.g. for calculating the long term correlation only
        two stations are used but during infilling all stations
        should satisfy this condition with respect to each other. i.e.
        only those days are used to make the correlation matrix on
        which neighboring stations have valid values at every step.
    infill_interval_type: string
        A string that is either: \'slice\', \'indiv\', \'all\'.
        slice: all steps in between infill_dates_list are
        considered for infilling.
        indiv: only fill on these dates.
        all: fill wherever there is a missing value.
    infill_type: string
        The form of variable to be infilled. It can be \'precipitation\',
        \'discharge\' and \'discharge-censored\'.
        For \'discharge-censored\', CDF values below \'cut_cdf_thresh\' are
        assigned an average probability.
    infill_dates_list: list_like, date_like
        A list containing the dates on which infilling is done.
        The way this list is used depends on the value of
        infill_interval_type.
        if infill_interval_type is \'slice\' and infill_dates_list
        is \'[date_1, date_2]\' then all steps in between and
        including date_1 and date_2 are infilled provided that
        neighboring stations have enough valid values at every
        step.
    n_min_nebs: integer
        Number of minimum neighbors to use for infilling.
        Default is 1.
    n_max_nebs: integer
        Number of maximum stations to use for infilling.
        Normally n_min_nebs stations are used but it could be
        that at a given step one of the neighbors is missing a
        value in that case we have enough stations to choose from.
        This number depends on the given dataset. It cannot be
        less than n_min_nebs.
        Default is 1.
    ncpus: integer
        Number of processes to initiate in case of methods that allow for
        multiprocessing. Ideally equal to the number of cores available.
        Default is 1.
    skip_stns: list_like
        The names of the stations that should not be used while
        infilling. Normally, after calling the
        \'cmpt_plot_rank_corr_stns\', one can see the stations
        that do not correlate well with the infill_stns.
        Default is None.
    sep: string
        The type of separator used in in_var_file and in_coords_file.
        Both of them should be similar. The output will have this as
        the separator as well.
        Default is \';\'.
    time_fmt: string
        Format of the time in the in_var_file. This is required to
        convert the string time to a datetime object that allows
        for a more accurate indexing. Any valid time format from the datetime
        module can be used.
        Default is \'%Y-%m-%d\'.
    freq: string or pandas datetime offset object
        The type of interval used in the in_var_file. e.g. for
        days it is \'D\'. It is better to take a look at the
        pandas datetime offsets.
        Default is \'D\'.
    verbose: bool
        If True, print activity messages.
        Default is True.

    Post Initiation Parameters
    --------------------------
    Several attributes can be changed after initiating the NormCopulaInfill
    object. These should be changed before calling any of the methods.

    nrst_stns_type: string
        Criteria to select neighbors. Can be \'rank\' (rank correlation),
        \'dist\' (2D proximity) or \'symm\' (rank correlation and symmetry).
        If infill_type is \'precipitation\' then nrst_stns_type cannot be
        \'symm\'.
        NOTE: Symmetry is checked in both directions.
        Default is \'symm\' if infill_type is
        \'discharge\' and \'rank\' if infill_type is \'precipitation\' or
        '\discharge-censored\'.
    n_discrete: integer
        The number of intervals used to discretize the range of values
        while calculating the conditional distributions. The more the
        better but computation time will increase.
        Default is 300.
    n_norm_symm_flds: integer
        If nrst_stns_type is \'symm\', n_norm_symm_flds is the number of
        random bivariate normal copulas for a given correlation that are
        used to establish maximum and minimum asymmetry possible. If the
        asymmetry of the sample is within these bounds then it is taken
        as a neighbor and further used to infill. The order of these
        neighbors depend on the strength of their absolute correlation.
        Default is 200.
    fig_size_long: tuple of floats or integers
        Size of the figures (width, height) in inches. Default is
        (20, 7). This does not apply to diagnostic plots.
    out_fig_dpi: integer
        Dot-per-inch of the output figures. Default is 150.
    out_fig_fmt: string
        Output figure format. Any format supported by matplotlib.
        Default is png.
    conf_heads: list of string
        Labels of the confidence intervals that appear in the output confidence
        values dataframes and figures.
        Default is
        ['var_0.05', 'var_0.25', 'var_0.50', 'var_0.75', 'var_0.95'].
    conf_probs: list of floats
        Non-exceedence probabilities that appear in the output confidence
        values dataframes and figures. Number of values should equal that of
        conf_heads otherwise an error is raised before infilling.
        Default is [0.05, 0.25, 0.50, 0.75, 0.95].
    fin_conf_head: string
        Label of the non-exceedence probabiliy that is used as the final infill
        value in the output dataframe. If should exist in conf_heads.
        Default is the third value of conf_heads i.e. \'var_0.5\'
    adj_probs_bounds: list of two floats
        The minimum and maximum cutoff cummulative probabilities. Before and
        after which all values of the range of the conditional probability
        are dropped.
        Default is [0.0001, 0.9999].
    flag_probs: list of two floats
        The non-exceedence probabilities for the conditional probability below
        and above which a value is flagged as suspicious.
        Default is [0.05, 0.95].
    n_round: integer
        The number of decimals to round all outputs to.
        Default is 3.
    cop_bins: integer
        The number of bins to use while plotting empirical copulas.
        Default is 20.
    max_corr: float
        The maximum correlation that any two stations can have while infilling,
        beyond which the station is not used.
        Default is 0.995.
    min_corr: float
        The minimum correlation that an infill and a neighboring station
        should have in order for the neighbor to be selected as a neighbor
        while infilling. It is used with nrst_stns_type of \'rank\' or \'symm'.
        Default is 0.5.
    ks_alpha: float
        The confidence limits of the Kolmogorov-Smirnov test. Should be between
        0 and 1.
        Default is 0.05.
    n_rand_infill_values: integer
        Create n_rand_infill_values output dataframes. Each of which contains
        a randomly selected value from the conditional distribution for a given
        step.
        Default is 0.
    cut_cdf_thresh: float
        A CDF value below which all values get an average probability
        (between zero and \'cut_cdf_thresh\'). This is to handle very low
        values that are repetitive and don't need to be infilled accurately.
        This values is used when \infill_type\' is \'discharge-censored\'.
        Default is 0.8.
    max_time_lag_corr: integer
        To get a better correlation, high temporal resolution data can be
        lagged in time w.r.t the infill station (backward and forward) to get a
        better correlation and infilling values. \'max_time_lag_corr\' is the
        number of lags that can be checked to get the lag that produces maximum
        correlation.
        Default is zero.
    debug_mode_flag: bool
        Turns multiprocessing and dont_stop_flag off and allows for better
        debugging. The script has to run in the python debugger for
        debugging though.
        Default is False.
    plot_diag_flag: bool
        Plot and save outputs of each step for several variables. This allows
        for a detailed analysis of the output. It runs pretty slow.
        Default is False.
    plot_step_cdf_pdf_flag: bool
        Plot the conditional CDF and PDF for each step.
        Default is False.
    compare_infill_flag: bool
        Plot a comparison of infilled and observed values wherever they
        exist.
        Default is False.
    flag_susp_flag: bool
        Plot and save data flagging results.
        Default is False.
    force_infill_flag: bool
        Infill, if the number of available neighbors with valid values is
        atleast 1.
        Default is True.
    plot_neighbors_flag: bool
        Plot the neighbors for each infill station based.
        Default is False.
    take_min_stns_flag: bool
        Take n_min_nebs if they are more than this.
        Default is True.
    overwrite_flag: bool
        Overwrite an output if it exists otherwise incorporate it.
        Default is True.
    read_pickles_flag: bool
        Read the neigboring stations correlations pickle. This avoids
        calculating the neighboring stations again. Valid only if
        nrst_stns_type is \'rank\' or \'symm\'.
        Default is False.
    use_best_stns_flag: bool
        Find a combination of stations that have the maximum number of stations
        available with values >= min_valid_vals while infilling a given
        station.
        Default is True.
    dont_stop_flag: bool
        Continue infilling even if an error occurs at a given step.
        Default is True.
    plot_long_term_corrs_flag: bool
        After selecting neighbors, plot the neighbors w.r.t descending rank
        correlation for a given infill station. Valid only if nrst_stns_type is
        \'rank\' of \'symm\'.
        Default is False.
    save_step_vars_flag: bool
        Save values of several parameters for each step during infilling.
        Default is False.
    plot_rand_flag: bool
        Plot the n_rand_infill_values with the non-exeedence values as well.
        Default is False.
    stn_based_mp_infill: bool
        Choose how multiprocessing is used to infill. If True, each station is
        infilled by a single process i.e. simulataneous infilling of many
        stations. If False, a single station is infilled by ncpus process at
        once. Should be set to True if the number of infill stations is more
        that ncpus. Setting it False, have resulted in mysterious interpreter
        shutdowns in the past. A given process shutdown results in loss of
        infilling on that particular station or those infill dates.
        Default is True.

    Outputs
    -------
    All the outputs mentioned here are created inside the \'out_dir\'.

    infill_var_df.csv: DataFrame, text
        The infilled dataframe. Looks the same as input but with infilled
        values inplace of missing values.
    infilled_flag_var_df.csv: DataFrame, text
        If flagging is True, this dataframe holds the flags for observed
        values, if they are within or out of bounds. 0 means within, -1 means
        below and +1 means above.
    n_avail_stns_df.csv: DataFrame, text
        The number of valid values available per step before and after
        infilling. Created by calling the  cmpt_plot_avail_stns method.
    n_avail_stns_compare.png: figure
        Plot of the n_avail_stns_df.csv. Saved by calling the
        cmpt_plot_avail_stns method.
    neighbor_stns_plots: directory
        If plot_neighbors_flag is True and the cmpt_plot_nrst_stns is called
        then plots for each infill stations' neighbors are saved in this
        directory.
    rank_corr_stns_plots: directory
        If plot_neighbors_flag is True and the cmpt_plot_rank_corr_stns is
        called then plots for each infill stations' neighbors are saved in this
        directory.
    long_term_correlations: directory
        If plot_long_term_corrs_flag is True and nrst_stns_type is \'rank\' or
        \'symm\' then the long-term correlation plots of each infill station
        with other stations are saved in this directory.
    summary_df.csv: DataFrame, text
        Save a dataframe showing some before and after infilling statistics.
    summary_df.png: Figure
        If the plot_summary method is called save the summary_df.csv as a
        figure.
    norm_cop_infill_log_%s.log: text
        All activity written on screen is also saved in this file.
        A new file is created, with a new initiation of the NormCopulaInfill
        object. %s stands for YearMonthDayHourMinutesSeconds in digits (it is
        the time at the NormCopulaInfill instance creation).

    Apart from these, outputs related to each station are saved inside a
    directory having the station as its name. The number of outputs is
    based on the type of flags used. The %s stands for the stations name.
    These are:

    add_info_df_stn_%s.csv: DataFrame, text
        A file containing the infill status, number of stations available,
        number of stations used and probability of the actual value (if it
        existed) in the conditional CDF per step.
        It is always produced.
    compare_infill_%s.png: figure
        Comparison of available values versus the values estimated at
        \'conf_probs\' using the normal copula per step.
        Produced when the compare_infill_flag or plot_diag_flag is True.
    compare_infill_%s_hists.png: figure
        Comparison of the histograms of observed and infilled values on infill
        dates.
        Produced when the compare_infill_flag or plot_diag_flag is True.
    flag_infill_%s.png: figure
        A figure showing if an observed value was out of \'flag_probs\'
        in the conditional CDF for a given step.
        Produced when the flag_susp_flag is True.
    missing_infill_%s.png: figure
        A figure showing the final infilled series for the station.
        It is always produced.
    stn_%s_infill_conf_vals_df.csv: DataFrame, text
        A table showing the values of the CDF at each \'conf_probs\' at each
        step.
        It is always produced.
    stns_used_infill_%s.png: figure
        A figure showing the number of available and used station while
        infilling at every step.
        It is always produced.

    Methods
    -------
    The following methods can be called to do infilling along with other
    things.

    infill:
        Infill the missing values using the criteria mentioned above.
    plot_ecops:
        Plot the empirical copulas of each infill station against its
        neighbors.
    cmpt_plot_avail_stns:
        Plot a figure and save a dataframe of the number of stations having
        valid values before and after infilling.
    plot_stats:
        Plot statistics of all the stations in the input dataframe as a table.
    plot_summary:
        Plot the summary_df.csv as a colored table.

    '''

    def __init__(self,
                 in_var_file,
                 in_coords_file,
                 out_dir,
                 infill_stns,
                 min_valid_vals,
                 infill_interval_type,
                 infill_type,
                 infill_dates_list,
                 n_min_nebs=1,
                 n_max_nebs=1,
                 ncpus=1,
                 skip_stns=None,
                 sep=';',
                 time_fmt='%Y-%m-%d',
                 freq='D',
                 verbose=True):

        # TODO: save all variables to a file with a timestamp
        self.verbose = bool(verbose)
        self.in_var_file = str(in_var_file)
        self.out_dir = str(out_dir)
        self.infill_stns = infill_stns
        self.infill_interval_type = str(infill_interval_type)
        self.infill_type = str(infill_type)
        self.infill_dates_list = list(infill_dates_list)
        self.min_valid_vals = int(min_valid_vals)
        self.in_coords_file = str(in_coords_file)
        self.n_min_nebs = int(n_min_nebs)
        self.n_max_nebs = int(n_max_nebs)
        self.ncpus = int(ncpus)
        self.sep = str(sep)
        self.time_fmt = str(time_fmt)
        self.freq = freq

        if not os_exists(self.out_dir):
            os_mkdir(self.out_dir)

        self._out_log_file = os_join(self.out_dir,
                                     ('norm_cop_infill_log_%s.log' %
                                      datetime.now().strftime('%Y%m%d%H%M%S')))
        self.log_link = StdFileLoggerCtrl(self._out_log_file)
        print('INFO: Infilling started at:',
              datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        self.in_var_df = read_csv(self.in_var_file, sep=self.sep,
                                  index_col=0, encoding='utf-8')

        self.in_var_df.index = to_datetime(self.in_var_df.index,
                                           format=self.time_fmt)

        # Checking validity of parameters and adjustments if necessary

        assert self.in_var_df.shape[0] > 0, '\'in_var_df\' has no records!'
        assert self.in_var_df.shape[1] > 1, '\'in_var_df\' has < 2 fields!'

        self.in_var_df.columns = list(map(str, self.in_var_df.columns))
        self.in_var_df.columns = [stn.strip()
                                  for stn in self.in_var_df.columns]

        self.in_var_df_orig = self.in_var_df.copy()

        if self.verbose:
            print('INFO: \'in_var_df\' original shape:', self.in_var_df.shape)

        self.in_var_df.dropna(axis=0, how='all', inplace=True)

        if self.verbose:
            print('INFO: \'in_var_df\' shape after dropping NaN steps:',
                  self.in_var_df.shape)

        self.in_coords_df = read_csv(self.in_coords_file, sep=sep, index_col=0,
                                     encoding='utf-8')
        assert self.in_coords_df.shape[0] > 0, (
            '\'in_coords_df\' has no records!')
        assert self.in_coords_df.shape[1] >= 2, (
            '\'in_coords_df\' has < 2 fields!')

        self.in_coords_df.index = list(map(str, self.in_coords_df.index))
        self.in_coords_df.index = [stn.strip()
                                   for stn in self.in_coords_df.index]
        _ = ~self.in_coords_df.index.duplicated(keep='last')
        self.in_coords_df = self.in_coords_df[_]

        assert 'X' in self.in_coords_df.columns, (
            'Column \'X\' not in \'in_coords_df\'!')
        assert 'Y' in self.in_coords_df.columns, (
            'Column \'Y\' not in \'in_coords_df\'!')

        self.in_coords_df = self.in_coords_df[['X', 'Y']]
        self.in_coords_df.dropna(inplace=True)

        if self.verbose:
            print('INFO: \'in_coords_df\' original shape:', end=' ')
            print(self.in_coords_df.shape)

        if skip_stns is not None:
            assert hasattr(skip_stns, '__iter__'), (
               '\'skip_stns\' not an iterable!')
            self.skip_stns = list(map(str, list(skip_stns)))
            self.skip_stns = [stn.strip() for stn in self.skip_stns]

            if len(self.skip_stns) > 0:
                self.in_var_df.drop(labels=self.skip_stns,
                                    axis=1,
                                    inplace=True,
                                    errors='ignore')

                self.in_coords_df.drop(labels=self.skip_stns,
                                       axis=1,
                                       inplace=True,
                                       errors='ignore')

            if self.verbose:
                print(('INFO: \'in_var_df\' shape after dropping '
                       '\'skip_stns\':'), self.in_var_df.shape)

        assert self.min_valid_vals >= 1, (
            '\'min_valid_vals\' cannot be less than one!')

        assert self.n_min_nebs >= 1, (
            '\'n_min_nebs\' cannot be < one!')

        if self.n_max_nebs + 1 > self.in_var_df.shape[1]:
            self.n_max_nebs = self.in_var_df.shape[1] - 1
            print(('WARNING: \'n_max_nebs\' reduced to %d' %
                   self.n_max_nebs))

        assert self.n_min_nebs <= self.n_max_nebs, (
            '\'n_min_nebs\' > \'n_max_nebs\'!')

        assert ((self.infill_type == 'discharge') or
                (self.infill_type == 'precipitation') or
                (self.infill_type == 'discharge-censored')), (
            '\'infill_type\' can either be \'discharge\' or \'precipitation\''
            'or \'discharge-censored\'!')

        assert isinstance(self.ncpus, int), '\'ncpus\' not an integer!'
        assert self.ncpus >= 1, '\'ncpus\' cannot be less than one!'

        if ((self.infill_interval_type == 'slice') or
            (self.infill_interval_type == 'indiv')):
            assert hasattr(self.infill_dates_list, '__iter__'), (
               '\'infill_dates_list\' not an iterable!')

        if self.infill_interval_type == 'slice':
            assert len(self.infill_dates_list) == 2, (
                'For infill_interval_type \'slice\' only '
                'two objects inside \'infill_dates_list\' are allowed!')

            self.infill_dates_list = to_datetime(self.infill_dates_list,
                                                 format=self.time_fmt)
            assert self.infill_dates_list[1] > self.infill_dates_list[0], (
                'Infill dates not in ascending order!')

            _strt_date, _end_date = (
                to_datetime([self.infill_dates_list[0],
                             self.infill_dates_list[-1]],
                            format=self.time_fmt))

            self.infill_dates = date_range(start=_strt_date,
                                           end=_end_date,
                                           freq=self.freq)
        elif self.infill_interval_type == 'all':
            self.infill_dates_list = None
            self.infill_dates = self.in_var_df.index
        elif self.infill_interval_type == 'indiv':
            assert len(self.infill_dates_list) > 0, (
                   '\'infill_dates_list\' is empty!')
            self.infill_dates = to_datetime(self.infill_dates_list,
                                            format=self.time_fmt)
        else:
            assert False, (
                '\'infill_interval_type\' can only be \'slice\', \'all\', '
                'or \'indiv\'!')

        insuff_val_cols = self.in_var_df.columns[self.in_var_df.count() <
                                                 self.min_valid_vals]

        if len(insuff_val_cols) > 0:
            self.in_var_df.drop(labels=insuff_val_cols, axis=1, inplace=True)
            if self.verbose:
                print(('INFO: The following stations (n=%d) '
                       'are with insufficient values:\n') %
                      insuff_val_cols.shape[0], insuff_val_cols.tolist())

        self.in_var_df.dropna(axis=0, how='all', inplace=True)
        self.in_var_df.dropna(axis=1, how='all', inplace=True)

        if self.verbose:
            print(('INFO: \'in_var_df\' shape after dropping values less than '
                   '\'min_valid_vals\':'), self.in_var_df.shape)

        assert self.min_valid_vals <= self.in_var_df.shape[0], (
            'Number of stations in \'in_var_df\' less than '
            '\'min_valid_vals\' after dropping days with insufficient '
            'records!')

        commn_stns = intersect1d(self.in_var_df.columns,
                                 self.in_coords_df.index)

        self.in_var_df = self.in_var_df[commn_stns]
        self.in_coords_df = self.in_coords_df.loc[commn_stns]

        self.xs = self.in_coords_df['X'].values
        self.ys = self.in_coords_df['Y'].values

        if self.infill_stns == 'all':
            self.infill_stns = self.in_var_df.columns
        else:
            self.infill_stns = Index(self.infill_stns, dtype=object)

            _ = ~self.infill_stns.duplicated(keep='last')
            self.infill_stns = self.infill_stns[_]

        if self.verbose:
            print('INFO: \'in_var_df\' shape after station name intersection '
                  'with \'in_coords_df\':', self.in_var_df.shape)
            print('INFO: \'in_coords_df\' shape after station name '
                  'intersection with \'in_var_df\':', self.in_coords_df.shape)

        assert self.n_min_nebs < self.in_var_df.shape[1], (
            'Number of stations in \'in_var_df\' less than '
            '\'n_min_nebs\' after intersecting station names!')
        if self.n_max_nebs >= self.in_var_df.shape[1]:
            self.n_max_nebs = self.in_var_df.shape[1] - 1
            print(('WARNING: \'n_max_nebs\' set to %d after station '
                   'names intersection!') % self.n_max_nebs)

        self.n_infill_stns = len(self.infill_stns)
        assert self.n_infill_stns > 0

        self.ncpus = min(self.ncpus, self.n_infill_stns)

        for infill_stn in self.infill_stns:
            assert infill_stn in self.in_var_df.columns, (
                'Station %s not in input variable dataframe anymore!' %
                infill_stn)

        self.infill_dates = (
            self.infill_dates.intersection(self.in_var_df.index))

        assert self.infill_dates.shape[0] > 0, (
            'After the above operations, no dates to work with!')

        # check if atleast one infill date is in the in_var_df
        date_in_dates = False
        full_dates = self.in_var_df.index
        for infill_date in self.infill_dates:
            if infill_date in full_dates:
                date_in_dates = True
                break

        assert date_in_dates, (
            'No infill dates exist in \'in_var_df\' after dropping '
            'stations and records with insufficient information!')

        self.full_date_index = date_range(self.in_var_df.index[+0],
                                          self.in_var_df.index[-1],
                                          freq=self.freq)
        self.in_var_df = self.in_var_df.reindex(self.full_date_index)

## =============================================================================
# # setting some values to nan to reproduce an error
#        self.in_var_df.loc[self.infill_dates, self.infill_stns] = nan
## =============================================================================

        # ## Initiating additional required variables
        self.nrst_stns_list = []
        self.nrst_stns_dict = {}

        self.rank_corr_stns_list = []
        self.rank_corr_stns_dict = {}

        if ((self.infill_type == 'precipitation') or
            (self.infill_type == 'discharge-censored')):
            self.nrst_stns_type = 'rank'
            self._rank_method = 'max'
        else:
            self.nrst_stns_type = 'symm'  # can be rank or dist or symm
            self._rank_method = 'average'

        self.n_discret = 300
        self.n_norm_symm_flds = 200
        self._max_symm_rands = 50000
        self.thresh_mp_steps = 100

        self.fig_size_long = (20, 7)
        self.out_fig_dpi = 300
        self.out_fig_fmt = 'png'

        self.conf_heads = ['var_0.05',
                           'var_0.25',
                           'var_0.50',
                           'var_0.75',
                           'var_0.95']
        self.conf_probs = [0.05, 0.25, 0.5, 0.75, 0.95]
        self.fin_conf_head = self.conf_heads[2]
        _ = self.in_var_df.shape[0]
        self.adj_prob_bounds = [0.0001, 0.9999]
        self.adj_prob_bounds[0] = min(self.adj_prob_bounds[0],
                                      (1. / self.in_var_df.shape[0]))
        self.adj_prob_bounds[1] = max(self.adj_prob_bounds[1],
                                      1 - (1. / self.in_var_df.shape[1]))

        self.flag_probs = [0.05, 0.95]
        self.n_round = 3
        self.cop_bins = 20
        self.max_corr = 0.9995
        self.min_corr = 0.5  # drop neighbors that have less abs corr than this
        self.ks_alpha = 0.05
        self.n_rand_infill_values = 0

        if self.infill_type == 'discharge-censored':
            self.cut_cdf_thresh = 0.8
        else:
            self.cut_cdf_thresh = None

        self.max_time_lag_corr = 0

        self._norm_cop_pool = None

        self._infilled = False
        self._dist_cmptd = False
        self._rank_corr_cmptd = False
        self._conf_ser_cmptd = False
        self._bef_all_chked = False
        self._bef_infill_chked = False

        self.debug_mode_flag = False
        self.plot_diag_flag = False
        self.plot_step_cdf_pdf_flag = False
        self.compare_infill_flag = False
        self.flag_susp_flag = False
        self.force_infill_flag = True
        self.plot_neighbors_flag = False
        self.take_min_stns_flag = False
        self.overwrite_flag = True
        self.read_pickles_flag = False
        self.use_best_stns_flag = True
        self.dont_stop_flag = True
        self.plot_long_term_corrs_flag = False
        self.save_step_vars_flag = False
        self.plot_rand_flag = False
        self.stn_based_mp_infill = True

        self.out_var_file = os_join(self.out_dir, 'infilled_var_df.csv')
        self.out_var_infill_stns_file = (
            os_join(self.out_dir, 'infilled_var_df_infill_stns.csv'))
        self.out_var_infill_stn_coords_file = (
            os_join(self.out_dir, 'infilled_var_df_infill_stns_coords.csv'))
        self.out_flag_file = os_join(self.out_dir, 'infilled_flag_var_df.csv')
        self.out_stns_avail_file = os_join(self.out_dir, 'n_avail_stns_df.csv')
        self.out_stns_avail_fig = os_join(self.out_dir, 'n_avail_stns_compare')
        self.out_nebor_plots_dir = os_join(self.out_dir, 'neighbor_stns_plots')
        self.out_rank_corr_plots_dir = os_join(self.out_dir,
                                               'rank_corr_stns_plots')
        self.out_long_term_corrs_dir = os_join(self.out_dir,
                                               'long_term_correlations')
        self.out_summary_file = os_join(self.out_dir, 'summary_df.csv')
        self.out_summary_fig = os_join(self.out_dir, 'summary_df')

        self._out_rank_corr_stns_pkl_file = os_join(self.out_dir,
                                                    'rank_corr_stns_vars.pkl')
        self._out_nrst_stns_pkl_file = os_join(self.out_dir,
                                               'nrst_stns_vars.pkl')

        self.ecops_dir = os_join(self.out_dir, 'empirical_copula_plots')
        self.out_var_stats_file = os_join(self.out_dir, 'var_statistics.png')

        if self.infill_type == 'precipitation':
            self.var_le_trs = 0.0
            self.var_ge_trs = 1.0
            self.ge_le_trs_n = 1

        self.summary_df = None  # defined in _before_infill_checks method

        return

    def plot_ecops(self):
        BefAll(self)
        assert self._bef_all_chked, 'Initiate \'BefAll\' first!'

        self._get_ncpus()

        ECops(self)
        return

    def plot_stats(self):
        PlotStats(self)
        return

    def cmpt_plot_nrst_stns(self):
        BefAll(self)
        assert self._bef_all_chked, 'Initiate \'BefAll\' first!'

        NrstStns(self)
        return

    def cmpt_plot_rank_corr_stns(self):
        BefAll(self)
        assert self._bef_all_chked, 'Initiate \'BefAll\' first!'

        if self.max_time_lag_corr:
            BestLagRankCorrStns(self)
        else:
            RankCorrStns(self)
        return

    def _infill_stn(self, ii, infill_stn):
        try:
            return self.__infill_stn(ii, infill_stn)
        except:
            full_tb(exc_info(), self.dont_stop_flag)
            if not self.dont_stop_flag:
                raise Exception('Stop!')
        return

    def __infill_stn(self, ii, infill_stn):
        if self.verbose:
            print('\n')
            pprt(nbh=4, msgs=[('Going through station %d of %d:  %s') %
                              (ii + 1, self.n_infill_stns, infill_stn)])

        self.curr_infill_stn = infill_stn
        self.stn_out_dir = os_join(self.out_dir, infill_stn)
        out_conf_df_file = (
            os_join(self.stn_out_dir,
                    'stn_%s_infill_conf_vals_df.csv' % infill_stn))
        out_add_info_file = os_join(self.stn_out_dir,
                                    'add_info_df_stn_%s.csv' % infill_stn)

        # load infill
        no_out = True
        n_infilled_vals = 0
        if ((not self.overwrite_flag) and
            os_exists(out_conf_df_file) and
            os_exists(out_add_info_file)):
            if self.verbose:
                pprt(['Output exists already. Not overwriting.'], nbh=8)

            try:
                out_conf_df = read_csv(out_conf_df_file,
                                       sep=str(self.sep),
                                       encoding='utf-8',
                                       index_col=0)
                out_conf_df.index = to_datetime(out_conf_df.index,
                                                format=self.time_fmt)

                out_add_info_df = read_csv(out_add_info_file,
                                           sep=str(self.sep),
                                           encoding='utf-8',
                                           index_col=0)
                out_add_info_df.index = to_datetime(out_add_info_df.index,
                                                    format=self.time_fmt)

                n_infilled_vals = out_conf_df.dropna().shape[0]

                if not self.n_rand_infill_values:
                    _idxs = (
                        isnan(self.in_var_df_orig.loc[self.infill_dates,
                                                      self.curr_infill_stn]))
                    _ser = self.in_var_df_orig.loc[self.infill_dates,
                                                   self.curr_infill_stn]
                    out_stn_ser = _ser.where(
                        logical_not(_idxs),
                        out_conf_df[self.fin_conf_head],
                        axis=0)
                    self.out_var_df.loc[out_conf_df.index,
                                        infill_stn] = out_stn_ser
                else:
                    for rand_idx in range(self.n_rand_infill_values):
                        _idxs = self.in_var_df_orig.loc[self.infll_dates,
                                                        self.curr_infill_stn]
                        _idxs = isnan(_idxs)
                        _idxs = logical_not(_idxs)
                        _ser = self.in_var_df_orig.loc[
                            self.infill_dates, self.curr_infill_stn]
                        _lab = self.fin_conf_head % rand_idx
                        out_stn_ser = _ser.where(_idxs,
                                                 out_conf_df[_lab],
                                                 axis=0)
                        self.out_var_dfs_list[rand_idx].loc[
                            out_conf_df.index, infill_stn] = out_stn_ser

                no_out = False
            except Exception as msg:
                raise Exception(('Error while trying to read and '
                                 'update values from the existing '
                                 'dataframe:\n' + msg))

        self.summary_df.loc[self.curr_infill_stn, self._av_vals_lab] = (
            self.in_var_df[self.curr_infill_stn].dropna().shape[0])

        if self.compare_infill_flag:
            nan_idxs = list(range(self.infill_dates.shape[0]))
        else:
            nan_idxs = where(
                isnan(self.in_var_df.loc[self.infill_dates,
                                         self.curr_infill_stn].values))[0]

        n_nan_idxs = len(nan_idxs)
        if (n_infilled_vals < n_nan_idxs and (not no_out)):
            # it can happen that not all steps were infilled
            if self.verbose:
                pprt(['Existing output has insufficient infilled values!'],
                     nbh=8)
            no_out = True

        self.summary_df.loc[self.curr_infill_stn,
                            self._miss_vals_lab] = n_nan_idxs

        if self.nrst_stns_type == 'rank':
            self.curr_nrst_stns = self.rank_corr_stns_dict[infill_stn]
        elif self.nrst_stns_type == 'dist':
            self.curr_nrst_stns = self.nrst_stns_dict[infill_stn]
        elif self.nrst_stns_type == 'symm':
            self.curr_nrst_stns = self.rank_corr_stns_dict[infill_stn]
        else:
            raise Exception(as_err('Incorrect \'nrst_stns_type\': %s!' %
                                   self.nrst_stns_type))

        if (self.overwrite_flag or no_out):
            if (n_nan_idxs == 0) and (not self.compare_infill_flag):
                return

            # mkdirs
            dir_list = [self.stn_out_dir]

            self.stn_infill_cdfs_dir = os_join(self.stn_out_dir,
                                               'stn_infill_cdfs')
            self.stn_infill_pdfs_dir = os_join(self.stn_out_dir,
                                               'stn_infill_pdfs')
            self.stn_step_cdfs_dir = os_join(self.stn_out_dir,
                                             'stn_step_cdfs')
            self.stn_step_corrs_dir = os_join(self.stn_out_dir,
                                              'stn_step_corrs')
            self.stn_step_vars_dir = os_join(self.stn_out_dir,
                                             'stn_step_vars')

            if self.plot_step_cdf_pdf_flag:
                dir_list.extend([self.stn_infill_cdfs_dir,
                                 self.stn_infill_pdfs_dir])

            if self.plot_diag_flag:
                dir_list.extend([self.stn_step_cdfs_dir,
                                 self.stn_step_corrs_dir])

            if self.save_step_vars_flag:
                dir_list.extend([self.stn_step_vars_dir])

            for _dir in dir_list:
                if not os_exists(_dir):
                    os_mkdir(_dir)

            idxs = linspace(0,
                            n_nan_idxs,
                            self.ncpus + 1,
                            endpoint=True,
                            dtype='int64')

            if self.verbose:
                infill_start = timeit.default_timer()
                pprt(['%d steps to infill' % n_nan_idxs], nbh=8)
                pprt(['Neighbors are:'], nbh=8)
                for i_msg in range(0, len(self.curr_nrst_stns), 3):
                    pprt(self.curr_nrst_stns[i_msg:(i_msg + 3)], nbh=12)

            # initiate infill
            out_conf_df = DataFrame(index=self.infill_dates,
                                    columns=self.conf_ser.index,
                                    dtype=float)
            out_add_info_df = DataFrame(index=self.infill_dates,
                                        dtype=float,
                                        columns=['infill_status',
                                                 'n_neighbors_raw',
                                                 'n_neighbors_fin',
                                                 'act_val_prob'])

            if ((n_nan_idxs > self.thresh_mp_steps) and
                not self.stn_based_mp_infill):
                use_mp_infill = True
            else:
                use_mp_infill = False

            self.infill_steps_obj = InfillSteps(self)

            if ((idxs.shape[0] == 1) or
                (self.ncpus == 1) or
                (not use_mp_infill) or
                self.debug_mode_flag):
                sub_dfs = [self._infill(self.infill_dates[nan_idxs])]
            else:
                n_sub_dates = 0
                sub_infill_dates_list = []
                for idx in range(self.ncpus):
                    sub_dates = self.infill_dates[
                            nan_idxs[idxs[idx]:idxs[idx + 1]]]
                    sub_infill_dates_list.append(sub_dates)
                    n_sub_dates += sub_dates.shape[0]

                assert n_sub_dates == n_nan_idxs, (
                    as_err(('\'n_sub_dates\' (%d) and '
                            '\'self.infill_dates\' '
                            '(%d) of unequal length!') %
                           (n_sub_dates, self.infill_dates.shape[0])))

                try:
                    sub_dfs = list(self._norm_cop_pool.uimap(
                            self._infill, sub_infill_dates_list))
                    self._norm_cop_pool.clear()
                except Exception as msg:
                    self._norm_cop_pool.close()
                    self._norm_cop_pool.join()
                    raise Exception('MP failed 1: %s!' % msg)

            for sub_df in sub_dfs:
                sub_conf_df = sub_df[0]
                sub_add_info_df = sub_df[1]

                _ser = self.in_var_df_orig.loc[sub_conf_df.index,
                                               self.curr_infill_stn]
                _idxs = isnan(_ser)
                _idxs = logical_not(_idxs)

                if not self.n_rand_infill_values:
                    sub_stn_ser = (_ser.where(_idxs,
                                              sub_conf_df[self.fin_conf_head],
                                              axis=0)).copy()
                    self.out_var_df.loc[sub_stn_ser.index,
                                        self.curr_infill_stn] = sub_stn_ser
                else:
                    for rand_idx in range(self.n_rand_infill_values):
                        _lab = self.fin_conf_head % rand_idx
                        sub_stn_ser = (_ser.where(_idxs,
                                                  sub_conf_df[_lab],
                                                  axis=0)).copy()
                        self.out_var_dfs_list[rand_idx].loc[
                                sub_stn_ser.index,
                                self.curr_infill_stn] = sub_stn_ser

                out_conf_df.update(sub_conf_df)
                out_add_info_df.update(sub_add_info_df)

            n_infilled_vals = out_conf_df.dropna().shape[0]

            if self.verbose:
                pprt([('%d steps out of %d infilled' %
                       (n_infilled_vals, n_nan_idxs))],
                     nbh=8)

                infill_stop = timeit.default_timer()
                fin_secs = infill_stop - infill_start

                _ = divide(fin_secs, max(1, n_infilled_vals))
                pprt([(('Took %0.3f secs, %0.3e secs per '
                        'step') % (fin_secs, _))],
                     nbh=8)

            # ## prepare output
            out_conf_df = out_conf_df.apply(lambda x: to_numeric(x))
            out_conf_df.to_csv(out_conf_df_file,
                               sep=str(self.sep),
                               encoding='utf-8')
            out_add_info_df.to_csv(out_add_info_file,
                                   sep=str(self.sep),
                                   encoding='utf-8')

        self.summary_df.loc[self.curr_infill_stn,
                            self._infilled_vals_lab] = n_infilled_vals

        _ = len(self.curr_nrst_stns)
        self.summary_df.loc[self.curr_infill_stn,
                            self._max_avail_nebs_lab] = _

        _ = round(out_add_info_df['n_neighbors_fin'].dropna().mean(), 1)
        self.summary_df.loc[self.curr_infill_stn,
                            self._avg_avail_nebs_lab] = _

        # make plots
        # plot number of gauges available and used
        if self.verbose:
            infill_start = timeit.default_timer()

        nebs_used_per_step_file = os_join(self.stn_out_dir,
                                          ('stns_used_infill_%s.png' %
                                           infill_stn))

        if (self.overwrite_flag or
            no_out or
            (not os_exists(nebs_used_per_step_file))):
            lw = 0.8
            alpha = 0.7

            plt.figure(figsize=self.fig_size_long)
            infill_ax = plt.subplot(111)
            infill_ax.plot(self.infill_dates,
                           out_add_info_df['n_neighbors_raw'].values,
                           label='n_neighbors_raw',
                           c='r',
                           alpha=alpha,
                           lw=lw + 0.5,
                           ls='-')
            infill_ax.plot(self.infill_dates,
                           out_add_info_df['n_neighbors_fin'].values,
                           label='n_neighbors_fin',
                           c='b',
                           marker='o',
                           lw=0,
                           ms=2)

            infill_ax.set_xlabel('Time')
            infill_ax.set_xlim(self.infill_dates[0], self.infill_dates[-1])
            infill_ax.set_ylabel('Stations used')
            infill_ax.set_ylim(-1, infill_ax.get_ylim()[1] + 1)

            plt.suptitle(('Number of raw available and finally used stations '
                          'for infilling at station: %s') % infill_stn)
            plt.grid()
            plt.legend(framealpha=0.5, loc=0)
            plt.savefig(nebs_used_per_step_file,
                        dpi=self.out_fig_dpi,
                        bbox_inches='tight')
            plt.close('all')

        # the original unfilled series of the infilled station
        act_var = self.in_var_df_orig[infill_stn].loc[self.infill_dates].values

        # plot the infilled series
        out_infill_plot_loc = os_join(self.stn_out_dir,
                                      'missing_infill_%s.png' % infill_stn)
        plot_infill_cond = True

        if not self.overwrite_flag:
            plot_infill_cond = (plot_infill_cond and no_out)

        use_mp = not (self.debug_mode_flag or
                      self.stn_based_mp_infill or
                      (self.ncpus == 1))
        compare_iter = None
        flag_susp_iter = None

        if plot_infill_cond:
            if self.verbose:
                pprt(['Plotting infill...'], nbh=8)

            args_tup = (self, act_var, out_conf_df, out_infill_plot_loc)
            if use_mp:
                self._norm_cop_pool.uimap(PlotInfill, (args_tup,))
            else:
                PlotInfill(args_tup)

        # plot the comparison between the actual and infilled series
        out_compar_plot_loc = os_join(self.stn_out_dir,
                                      'compare_infill_%s.png' % infill_stn)

        if self.compare_infill_flag and np_any(act_var):
            if self.verbose:
                pprt(['Plotting infill comparison...'], nbh=8)

            if (not no_out) and (not self.overwrite_flag):
                update_summary_df_only = True
            else:
                update_summary_df_only = False

            args_tup = (act_var,
                        out_conf_df,
                        out_compar_plot_loc,
                        out_add_info_df,
                        update_summary_df_only)

            compare_obj = CompareInfill(self)
            if use_mp and (not update_summary_df_only):
                compare_iter = self._norm_cop_pool.uimap(compare_obj.plot_all,
                                                         (args_tup,))
            else:
                self.summary_df.update(compare_obj.plot_all(args_tup))
        else:
            if self.verbose:
                pprt(['Nothing to compare...'], nbh=8)

        # plot steps showing if the actual data is within the bounds
        out_flag_susp_loc = os_join(self.stn_out_dir,
                                    'flag_infill_%s.png' % infill_stn)

        if self.flag_susp_flag and np_any(act_var):
            if self.verbose:
                pprt(['Plotting infill flags...'], nbh=8)

            if (not no_out) and (not self.overwrite_flag):
                update_summary_df_only = True
            else:
                update_summary_df_only = False

            flag_susp_obj = FlagSusp(self)
            args_tup = (act_var,
                        out_conf_df,
                        out_flag_susp_loc,
                        update_summary_df_only)
            if use_mp and (not update_summary_df_only):
                flag_susp_iter = self._norm_cop_pool.uimap(flag_susp_obj.plot,
                                                           (args_tup,))
            else:
                _ = flag_susp_obj.plot(args_tup)
                self.summary_df.update(_[0])
                self.flag_df.update(_[1])
        else:
            if self.verbose:
                pprt(['Nothing to flag...'], nbh=8)

        if use_mp:
            self._norm_cop_pool.clear()

        if self.verbose:
            infill_stop = timeit.default_timer()
            fin_secs = infill_stop - infill_start
            pprt([('Took %0.3f secs for plotting other stuff') % fin_secs],
                 nbh=8)

        # these iters are here to make use of mp
        if compare_iter:
            self.summary_df.update(list(compare_iter)[0])

        if flag_susp_iter:
            _ = list(flag_susp_iter)[0]
            self.summary_df.update(_[0])
            self.flag_df.update(_[1])

        if self.stn_based_mp_infill:
            # must return a tuple in case of stn_based_mp_infill
            return (self.summary_df,
                    self.flag_df,
                    self.out_var_df,
                    self.out_var_dfs_list)
        return

    def infill(self):
        '''
        Perform the infilling based on given data
        '''
        if not self._bef_infill_chked:
            self._before_infill_checks()
        assert self._bef_infill_chked, (
            as_err('Call \'_before_infill_checks\' first!'))

        if self.debug_mode_flag and self.dont_stop_flag:
            self.dont_stop_flag = False
            if self.verbose:
                print('INFO: \'dont_stop_flag\' set to False!')

        if self.verbose:
            print('INFO: Using %d number of process(es) to infill' %
                  self.ncpus)
            print('\n')
            print('INFO: Flags:')
            pprt(['debug_mode_flag:', self.debug_mode_flag], nbh=4)
            pprt(['plot_diag_flag:', self.plot_diag_flag], nbh=4)
            pprt(['plot_step_cdf_pdf_flag:',
                  self.plot_step_cdf_pdf_flag],
                 nbh=4)
            pprt(['compare_infill_flag:', self.compare_infill_flag], nbh=4)
            pprt(['flag_susp_flag:', self.flag_susp_flag], nbh=4)
            pprt(['force_infill_flag:', self.force_infill_flag], nbh=4)
            pprt(['plot_neighbors_flag:', self.plot_neighbors_flag], nbh=4)
            pprt(['take_min_stns_flag:', self.take_min_stns_flag], nbh=4)
            pprt(['overwrite_flag:', self.overwrite_flag], nbh=4)
            pprt(['read_pickles_flag:', self.read_pickles_flag], nbh=4)
            pprt(['use_best_stns_flag:', self.use_best_stns_flag], nbh=4)
            pprt(['dont_stop_flag:', self.dont_stop_flag], nbh=4)
            pprt(['plot_long_term_corrs_flag:',
                  self.plot_long_term_corrs_flag],
                 nbh=4)
            pprt(['save_step_vars_flag:', self.save_step_vars_flag], nbh=4)
            pprt(['plot_rand_flag:', self.plot_rand_flag], nbh=4)

            print('\n')
            print('INFO: Other constants:')
            pprt(['cut_cdf_thresh:', self.cut_cdf_thresh], nbh=4)
            pprt(['max_time_lag_corr:', self.max_time_lag_corr], nbh=4)

            print('\n')
            print('INFO: Infilling...')
            print('INFO: Infilling type is:', self.infill_type)
            print(('INFO: Maximum possible records to process per station: '
                   '%d') % self.infill_dates.shape[0])
            print(('INFO: Total number of stations to infill: '
                   '%d') % len(self.infill_stns))
            if self.n_rand_infill_values:
                print('INFO: n_rand_infill_values:', self.n_rand_infill_values)

        assert self.n_infill_stns == len(self.infill_stns)

        if ((self.ncpus == 1) or
            self.debug_mode_flag or
            (not self.stn_based_mp_infill)):
            for ii, infill_stn in enumerate(self.infill_stns):
                self._infill_stn(ii, infill_stn)
        else:
            try:
                iis = list(range(0, self.n_infill_stns))
                _all_res = list(self._norm_cop_pool.map(self._infill_stn,
                                                        iis,
                                                        self.infill_stns))
                self._norm_cop_pool.clear()
            except:
                self._norm_cop_pool.close()
                self._norm_cop_pool.join()
                raise Exception('MP failed 2!')

            for _res in _all_res:
                if _res is None:
                    continue

                if not isinstance(_res, tuple):
                    print(30 * '#', '\n', 30 * '#',
                          '\n_res:', _res,
                          30 * '#', '\n', 30 * '#')
                    continue

                self.summary_df.update(_res[0])

                if self.flag_df is not None:
                    self.flag_df.update(_res[1])

                if self.out_var_df is not None:
                    self.out_var_df.update(_res[2])

                if self.out_var_dfs_list is not None:
                    for rand_idx in range(self.n_rand_infill_values):
                        _1 = self.out_var_dfs_list[rand_idx]
                        _2 = _res[3][rand_idx].loc[:, self.curr_infill_stn]
                        _1.loc[:, self.curr_infill_stn].update(_2)

        if not self.n_rand_infill_values:
            self.out_var_df.to_csv(self.out_var_file,
                                   sep=str(self.sep),
                                   encoding='utf-8')

            _ = self.out_var_df.loc[self.infill_dates, self.infill_stns]
            _.to_csv(self.out_var_infill_stns_file,
                     sep=str(self.sep),
                     encoding='utf-8')

            _ = self.in_coords_df.loc[self.infill_stns]
            _.to_csv(self.out_var_infill_stn_coords_file,
                     sep=str(self.sep),
                     encoding='utf-8')
        else:
            out_var_file_tmpl = self.out_var_file[:-4]
            for rand_idx in range(self.n_rand_infill_values):
                out_var_file = out_var_file_tmpl + '_%0.4d.csv' % rand_idx
                self.out_var_dfs_list[rand_idx].to_csv(out_var_file,
                                                       sep=str(self.sep),
                                                       encoding='utf-8')

        self.summary_df.to_csv(self.out_summary_file,
                               sep=str(self.sep),
                               encoding='utf-8')

        if self.flag_susp_flag:
            self.flag_df.to_csv(self.out_flag_file,
                                sep=str(self.sep),
                                encoding='utf-8',
                                float_format='%2.0f')
        self._infilled = True
        print('\n')
        return

    def cmpt_plot_avail_stns(self):
        AvailStns(self)
        return

    def plot_summary(self):
        Summary(self)
        return

    def _get_ncpus(self):
        '''
        Set the number of processes to be used

        Call it in the first line of any function that has mp in it
        '''

        if self._norm_cop_pool is not None:
            self._norm_cop_pool.clear()

        if self.debug_mode_flag:
            self.ncpus = 1
            if self.dont_stop_flag:
                self.dont_stop_flag = False
                if self.verbose:
                    print('INFO: \'dont_stop_flag\' set to False!')
        elif self.ncpus == 1:
            pass
        elif not hasattr(self._norm_cop_pool, '_id'):
            self._norm_cop_pool = mp_pool(nodes=self.ncpus)
        return

    def _before_infill_checks(self):
        BefAll(self)
        assert self._bef_all_chked, 'Initiate \'BefAll\' first!'

        if self.plot_diag_flag:
            self.compare_infill_flag = True
            self.force_infill_flag = True
            self.plot_step_cdf_pdf_flag = True
            self.flag_susp_flag = True
            self.save_step_vars_flag = True

        if not self._dist_cmptd:
            NrstStns(self)
            assert self._dist_cmptd, 'Call \'cmpt_plot_nrst_stns\' first!'

        if (self.nrst_stns_type == 'rank') or (self.nrst_stns_type == 'symm'):
            if self.nrst_stns_type == 'rank':
                if self.verbose:
                    print('INFO: using RANKS to get neighboring stations')

            elif self.nrst_stns_type == 'symm':
                if self.verbose:
                    print('INFO: using RANKS with SYMMETRIES to get '
                          'neighboring stations')

            if not self._rank_corr_cmptd:
                if self.max_time_lag_corr:
                    BestLagRankCorrStns(self)
                else:
                    RankCorrStns(self)
                assert self._rank_corr_cmptd, (
                    'Call \'cmpt_plot_rank_corr_stns\' first!')

        elif self.nrst_stns_type == 'dist':
            if self.verbose:
                print('INFO: using DISTANCE to get neighboring stations')

        else:
            assert False, as_err('Incorrect \'nrst_stns_type\': (%s)' %
                                 str(self.nrst_stns_type))

        self.n_infill_stns = self.infill_stns.shape[0]
        assert self.n_infill_stns, 'No stations to work with!'

        _ = self.in_var_df.dropna(axis=0, how='all')
        self.infill_dates = _.index.intersection(self.infill_dates)
        self.n_infill_dates = self.infill_dates.shape[0]
        assert self.n_infill_dates, 'No dates to work with!'

        if self.verbose:
            print('INFO: Final infill_stns count = %d' % self.n_infill_stns)
            print('INFO: Final infill_dates count = %d' % self.n_infill_dates)

        ConfInfill(self)
        assert self._conf_ser_cmptd, 'Call \'_cmpt_conf_ser\' first!'

        if self.infill_type == 'precipitation':
            assert isinstance(self.var_le_trs, float), (
                '\'var_le_trs\' is non-float!')
            assert isinstance(self.var_ge_trs, float), (
                '\'var_ge_trs\' is non-float!')
            assert isinstance(self.ge_le_trs_n, int), (
                '\'ge_le_trs_n\' is non-integer!')

            assert self.var_le_trs <= self.var_ge_trs, (
                '\'var_le_trs\' > \'var_ge_trs\'!')
            assert self.ge_le_trs_n > 0, (
                '\'self.ge_le_trs_n\' less than 1!')
        else:
            self.var_le_trs, self.var_ge_trs, self.ge_le_trs_n = 3 * [None]

        if self.flag_susp_flag:
            self.flag_df = DataFrame(columns=self.infill_stns,
                                     index=self.infill_dates,
                                     dtype=float)
            self.compare_infill_flag = True
        else:
            self.flag_df = None

        self._get_ncpus()

        if not self.n_rand_infill_values:
            self.out_var_df = self.in_var_df_orig.copy()
            self.out_var_dfs_list = None
        else:
            self.out_var_dfs_list = (
                [self.in_var_df_orig.copy()] * self.n_rand_infill_values)
            self.out_var_df = None

        self._av_vals_lab = 'Available values'
        self._miss_vals_lab = 'Missing values'
        self._infilled_vals_lab = 'Infilled values'
        self._max_avail_nebs_lab = 'Max. available neighbors'
        self._avg_avail_nebs_lab = 'Avg. neighbors used for infilling'
        self._compr_lab = 'Compared values'
        self._ks_lims_lab = (('%%age values within %0.0f%% '
                              'KS-limits') % (100 * (1.0 - self.ks_alpha)))
        self._flagged_lab = 'Flagged values %age'
        self._mean_obs_lab = 'Mean (observed)'
        self._mean_infill_lab = 'Mean (infilled)'
        self._var_obs_lab = 'Variance (observed)'
        self._var_infill_lab = 'Variance (infilled)'
        self._bias_lab = 'Bias'
        self._mae_lab = 'Mean abs. error'
        self._rmse_lab = 'Root mean sq. error'
        self._nse_lab = 'NSE'
        self._ln_nse_lab = 'Ln-NSE'
        self._kge_lab = 'KGE'
        self._pcorr_lab = 'Pearson correlation'
        self._scorr_lab = 'Spearman correlation'

        self._summary_cols = [self._av_vals_lab,
                              self._miss_vals_lab,
                              self._infilled_vals_lab,
                              self._max_avail_nebs_lab,
                              self._avg_avail_nebs_lab,
                              self._compr_lab,
                              self._ks_lims_lab,
                              self._flagged_lab,
                              self._mean_obs_lab,
                              self._mean_infill_lab,
                              self._var_obs_lab,
                              self._var_infill_lab,
                              self._bias_lab,
                              self._mae_lab,
                              self._rmse_lab,
                              self._nse_lab,
                              self._ln_nse_lab,
                              self._kge_lab,
                              self._pcorr_lab,
                              self._scorr_lab]

        self.summary_df = DataFrame(index=self.infill_stns,
                                    columns=self._summary_cols,
                                    dtype=float)

        self._bef_infill_chked = True
        return

    def _infill(self, infill_dates):
        try:
            (out_conf_df,
             out_add_info_df) = (
                 self.infill_steps_obj.infill_steps(infill_dates))
            return out_conf_df, out_add_info_df
        except:
            plt.close('all')
            full_tb(exc_info(), self.dont_stop_flag)
            if not self.dont_stop_flag:
                raise Exception('_infill failed!')

            return (DataFrame(index=infill_dates,
                              columns=self.conf_ser.index,
                              dtype=float),
                    DataFrame(index=infill_dates,
                              dtype=float,
                              columns=['infill_status',
                                       'n_neighbors_raw',
                                       'n_neighbors_fin',
                                       'act_val_prob']))
        return


if __name__ == '__main__':
    pass
