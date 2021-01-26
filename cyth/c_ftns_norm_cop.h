#include <math.h>
#include <stdio.h>

typedef double DT_D;
typedef unsigned long DT_UL;

DT_D get_ns_c(const DT_D *x_arr,
              const DT_D *y_arr,
              const DT_UL *size,
              const DT_D *demr,
              const DT_UL *off_idx) {

    DT_UL i;
    DT_D numr = 0, curr_diff;

    for (i=*off_idx; i<*size; ++i) {
        curr_diff = (x_arr[i] - y_arr[i]);
        numr += (curr_diff * curr_diff);
    }

    return (1.0 - (numr / *demr));
}


DT_D get_ln_ns_c(const DT_D *x_arr,
                 const DT_D *y_arr,
                 const DT_UL *size,
                 const DT_D *ln_demr,
                 const DT_UL *off_idx) {

    DT_UL i;
    DT_D ln_numr = 0, curr_diff;

    for (i=(*off_idx); i<*size; ++i) {
        curr_diff = (log(x_arr[i]) - log(y_arr[i]));
        ln_numr += (curr_diff * curr_diff);
    }

    return (1.0 - (ln_numr / *ln_demr));
}


DT_D get_mean_c(const DT_D *in_arr,
                const DT_UL *size,
                const DT_UL *off_idx) {

    DT_D _sum = 0.0;
    DT_UL i;

    for (i=*off_idx; i<*size; ++i) {
        _sum += in_arr[i];
    }
    return _sum / (*size - *off_idx);
}


DT_D get_ln_mean_c(const DT_D *in_arr,
                   const DT_UL *size,
                   const DT_UL *off_idx) {

    DT_D _sum = 0.0;
    DT_UL i;

    for (i=*off_idx; i<*size; ++i) {
        _sum += log(in_arr[i]);
    }
    return _sum / (*size - *off_idx);
}


DT_D get_var_c(const DT_D *in_arr_mean,
               const DT_D *in_arr,
               const DT_UL *size,
               const DT_UL *off_idx) {

    DT_D _sum = 0, curr_diff = 0.0;
    DT_UL i;

    for(i=*off_idx; i<*size; ++i) {
        curr_diff = (in_arr[i] - *in_arr_mean);
        _sum += (curr_diff * curr_diff);
    }
    return _sum / (*size - *off_idx);
}


DT_D get_covar_c(const DT_D *in_arr_1_mean,
                 const DT_D *in_arr_2_mean,
                 const DT_D *in_arr_1,
                 const DT_D *in_arr_2,
                 const DT_UL *size,
                 const DT_UL *off_idx) {

    DT_D _sum = 0;
    DT_UL i;

    for(i=*off_idx; i<*size; ++i) {
        _sum += ((in_arr_1[i] - *in_arr_1_mean) * \
                 (in_arr_2[i] - *in_arr_2_mean));
    }
    return _sum / (*size - *off_idx);
}


DT_D get_corr_c(const DT_D *in_arr_1_std_dev,
                const DT_D *in_arr_2_std_dev,
                const DT_D *arrs_covar) {

    return *arrs_covar / (*in_arr_1_std_dev * *in_arr_2_std_dev);
}


DT_D get_kge_c(const DT_D *act_arr,
               const DT_D *sim_arr,
               const DT_D *act_mean,
               const DT_D *act_std_dev,
               const DT_UL *size,
               const DT_UL *off_idx) {

    DT_D sim_mean, sim_std_dev, covar;
    DT_D correl, b, g, kge;

    sim_mean = get_mean_c(sim_arr, size, off_idx);
    sim_std_dev = pow(get_var_c(&sim_mean, sim_arr, size, off_idx), 0.5);

    covar = get_covar_c(act_mean, &sim_mean, act_arr, sim_arr, size, off_idx);
    correl = get_corr_c(act_std_dev, &sim_std_dev, &covar);

    b = sim_mean / *act_mean;
    g = (sim_std_dev / sim_mean) / (*act_std_dev / *act_mean);

    kge = 1 - pow(pow((correl - 1), 2) + \
                  pow((b - 1), 2) + \
                  pow((g - 1), 2), 0.5);
    return kge;
}

DT_D get_demr_c(const DT_D *x_arr,
                const DT_D *mean_ref,
                const DT_UL *size,
                const DT_UL *off_idx) {

    DT_UL i;
    DT_D demr = 0.0, curr_diff;

    for(i=*off_idx; i<*size; ++i) {
        curr_diff = (x_arr[i] - *mean_ref);
        demr += (curr_diff * curr_diff);
    }
    return demr;
}


DT_D get_ln_demr_c(const DT_D *x_arr,
                   const DT_D *ln_mean_ref,
                   const DT_UL *size,
                   const DT_UL *off_idx) {

    DT_UL i;
    DT_D ln_demr = 0.0, curr_diff;

    for(i=*off_idx; i<*size; ++i) {
        curr_diff = (log(x_arr[i]) - *ln_mean_ref);
        ln_demr += (curr_diff * curr_diff);
    }
    return ln_demr;
}


DT_D get_corr_coeff_c(const DT_D *x_arr,
                      const DT_D *y_arr,
                      const DT_UL *size,
                      const DT_UL *off_idx) {

    DT_D x_mean, y_mean, x_std_dev, y_std_dev, covar;

    x_mean = get_mean_c(x_arr, size, off_idx);
    y_mean = get_mean_c(y_arr, size, off_idx);

    x_std_dev = pow(get_var_c(&x_mean, x_arr, size, off_idx), 0.5);
    y_std_dev = pow(get_var_c(&y_mean, y_arr, size, off_idx), 0.5);

    covar = get_covar_c(&x_mean, &y_mean, x_arr, y_arr, size, off_idx);

    return get_corr_c(&x_std_dev, &y_std_dev, &covar);
}


void del_c(const DT_UL *x_arr,
           DT_UL *y_arr,
           const long *idx,
           const DT_UL *size) {

    DT_UL i = 0, j = 0;

    while (i < *size) {
        if (i != *idx) {
            y_arr[i] = x_arr[j];
        }
        else {
            j++;
            y_arr[i] = x_arr[j];
        }

        i++;
        j++;
    }
}


void lin_regsn_c(const DT_D *x_arr,
                 const DT_D *y_arr,
                 DT_D *y_arr_interp,
                 const DT_UL *size,
                 const DT_UL *off_idx,
                 DT_D *corr,
                 DT_D *slope,
                 DT_D *intercept) {

    DT_D x_m, y_m, covar, x_s, y_s;
    DT_UL i;

    x_m = get_mean_c(x_arr, size, off_idx);
    x_s = pow(get_var_c(&x_m, x_arr, size, off_idx), 0.5);

    y_m = get_mean_c(y_arr, size, off_idx);
    y_s = pow(get_var_c(&y_m, y_arr, size, off_idx), 0.5);

    covar = get_covar_c(&x_m, &y_m, x_arr, y_arr, size, off_idx);

    *corr = get_corr_c(&x_s, &y_s, &covar);

    *slope = *corr * y_s / x_s;
    *intercept = y_m - (*slope * x_m);

    for (i=*off_idx; i<*size; ++i) {
        y_arr_interp[i] = (*slope * x_arr[i]) + *intercept;
    }
}
