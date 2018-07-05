# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from traceback import format_exception

from numpy import divide, nan
from scipy.stats import rankdata

from ..cyth import gen_n_rns_arr, norm_ppf_py_arr, get_asymms_sample


def pprt(msgs, nbh=0, nah=0):
    '''Print a message with hashes before and after

    msgs: the messages iterable
    nbh: number of hashes before message
    nah: number of hashes after message
    '''
    if nbh:
        print('#' * nbh, end=' ')

    for msg in msgs:
        print(msg, end=' ')

    print('#' * nah)
    return


def as_err(msg):
    '''Use this for assertion errors
    '''
    n_hhs = 12 * '#'
    return n_hhs + ' ' + msg + ' ' + n_hhs


def get_norm_rand_symms(corr, n_vals):
    rv_1 = norm_ppf_py_arr(gen_n_rns_arr(n_vals))
    rv_2 = norm_ppf_py_arr(gen_n_rns_arr(n_vals))
    rv_3 = (corr * rv_1) + ((1 - corr ** 2) ** 0.5 * rv_2)

    rv_1 = divide(rankdata(rv_1), (n_vals + 1.0))
    rv_3 = divide(rankdata(rv_3), (n_vals + 1.0))

    asymms = get_asymms_sample(rv_1, rv_3)
    return asymms['asymm_1'], asymms['asymm_2']


def full_tb(sys_info, dont_stop_flag):
    exc_type, exc_value, exc_traceback = sys_info
    tb_fmt_obj = format_exception(exc_type, exc_value, exc_traceback)
    for trc in tb_fmt_obj:
        print(trc)

    if not dont_stop_flag:
        raise Exception('Stop!')
    return


def get_lag_ser(in_ser_raw, lag=0):
    in_ser = in_ser_raw.copy()
    if lag < 0:
        in_ser.values[:lag] = in_ser.values[-lag:]
        in_ser.values[lag:] = nan
    elif lag > 0:
        in_ser.values[lag:] = in_ser.values[:-lag]
        in_ser.values[:lag] = nan
    return in_ser


if __name__ == '__main__':
    pass
