"""Methods for model evaluation."""

import os
import sys
import copy
import numpy
import xarray
from scipy.stats import ks_2samp

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import histograms
import file_system_utils
import error_checking
import example_io
import prediction_io
import example_utils
import normalization
import neural_net

TOLERANCE = 1e-6

SHORTWAVE_NET_FLUX_NAME = 'net_shortwave_flux_w_m02'
SHORTWAVE_LOWEST_DOWN_FLUX_NAME = 'lowest_shortwave_down_flux_w_m02'
SHORTWAVE_HIGHEST_UP_FLUX_NAME = 'highest_shortwave_up_flux_w_m02'

LONGWAVE_NET_FLUX_NAME = 'net_longwave_flux_w_m02'
LONGWAVE_LOWEST_DOWN_FLUX_NAME = 'lowest_longwave_down_flux_w_m02'
LONGWAVE_HIGHEST_UP_FLUX_NAME = 'highest_longwave_up_flux_w_m02'

SCALAR_FIELD_DIM = 'scalar_field'
HEIGHT_DIM = 'height_m_agl'
WAVELENGTH_DIM = 'wavelength_metres'
VECTOR_FIELD_DIM = 'vector_field'
AUX_TARGET_FIELD_DIM = 'aux_target_field'
AUX_PREDICTED_FIELD_DIM = 'aux_predicted_field'
RAW_FLUX_BIN_DIM = 'raw_flux_bin'
NET_FLUX_BIN_DIM = 'net_flux_bin'
HEATING_RATE_BIN_DIM = 'heating_rate_bin'
BOOTSTRAP_REP_DIM = 'bootstrap_replicate'

SCALAR_TARGET_STDEV_KEY = 'scalar_target_stdev'
SCALAR_PREDICTION_STDEV_KEY = 'scalar_prediction_stdev'
VECTOR_TARGET_STDEV_KEY = 'vector_target_stdev'
VECTOR_PREDICTION_STDEV_KEY = 'vector_prediction_stdev'
AUX_TARGET_STDEV_KEY = 'aux_target_stdev'
AUX_PREDICTION_STDEV_KEY = 'aux_prediction_stdev'

SCALAR_MSE_KEY = 'scalar_mse'
SCALAR_MSE_BIAS_KEY = 'scalar_mse_bias'
SCALAR_MSE_VARIANCE_KEY = 'scalar_mse_variance'
SCALAR_MSE_SKILL_KEY = 'scalar_mse_skill_score'
VECTOR_MSE_KEY = 'vector_mse'
VECTOR_MSE_BIAS_KEY = 'vector_mse_bias'
VECTOR_MSE_VARIANCE_KEY = 'vector_mse_variance'
VECTOR_MSE_SKILL_KEY = 'vector_mse_skill_score'
VECTOR_FLAT_MSE_SKILL_KEY = 'vector_flat_mse_skill_score'
AUX_MSE_KEY = 'aux_mse'
AUX_MSE_BIAS_KEY = 'aux_mse_bias'
AUX_MSE_VARIANCE_KEY = 'aux_mse_variance'
AUX_MSE_SKILL_KEY = 'aux_mse_skill_score'

SCALAR_KS_STATISTIC_KEY = 'scalar_ks_statistic'
SCALAR_KS_P_VALUE_KEY = 'scalar_ks_p_value'
VECTOR_KS_STATISTIC_KEY = 'vector_ks_statistic'
VECTOR_KS_P_VALUE_KEY = 'vector_ks_p_value'
AUX_KS_STATISTIC_KEY = 'aux_ks_statistic'
AUX_KS_P_VALUE_KEY = 'aux_ks_p_value'

SCALAR_MAE_KEY = 'scalar_mae'
SCALAR_MAE_SKILL_KEY = 'scalar_mae_skill_score'
VECTOR_MAE_KEY = 'vector_mae'
VECTOR_MAE_SKILL_KEY = 'vector_mae_skill_score'
AUX_MAE_KEY = 'aux_mae'
AUX_MAE_SKILL_KEY = 'aux_mae_skill_score'
SCALAR_BIAS_KEY = 'scalar_bias'
VECTOR_BIAS_KEY = 'vector_bias'
AUX_BIAS_KEY = 'aux_bias'
SCALAR_CORRELATION_KEY = 'scalar_correlation'
VECTOR_CORRELATION_KEY = 'vector_correlation'
AUX_CORRELATION_KEY = 'aux_correlation'
SCALAR_KGE_KEY = 'scalar_kge'
VECTOR_KGE_KEY = 'vector_kge'
AUX_KGE_KEY = 'aux_kge'
VECTOR_PRMSE_KEY = 'vector_prmse'
SCALAR_RELIABILITY_KEY = 'scalar_reliability'
SCALAR_RELIABILITY_X_KEY = 'scalar_reliability_x'
SCALAR_RELIABILITY_Y_KEY = 'scalar_reliability_y'
SCALAR_RELIA_BIN_CENTER_KEY = 'scalar_reliability_bin_center'
SCALAR_RELIABILITY_COUNT_KEY = 'scalar_reliability_count'
SCALAR_INV_RELIA_BIN_CENTER_KEY = 'scalar_inv_reliability_bin_center'
SCALAR_INV_RELIABILITY_COUNT_KEY = 'scalar_inv_reliability_count'
VECTOR_RELIABILITY_KEY = 'vector_reliability'
VECTOR_RELIABILITY_X_KEY = 'vector_reliability_x'
VECTOR_RELIABILITY_Y_KEY = 'vector_reliability_y'
VECTOR_RELIA_BIN_CENTER_KEY = 'vector_reliability_bin_center'
VECTOR_RELIABILITY_COUNT_KEY = 'vector_reliability_count'
VECTOR_INV_RELIA_BIN_CENTER_KEY = 'vector_inv_reliability_bin_center'
VECTOR_INV_RELIABILITY_COUNT_KEY = 'vector_inv_reliability_count'
VECTOR_FLAT_RELIABILITY_KEY = 'vector_flat_reliability'
VECTOR_FLAT_RELIABILITY_X_KEY = 'vector_flat_reliability_x'
VECTOR_FLAT_RELIABILITY_Y_KEY = 'vector_flat_reliability_y'
VECTOR_FLAT_RELIA_BIN_CENTER_KEY = 'vector_flat_reliability_bin_center'
VECTOR_FLAT_RELIABILITY_COUNT_KEY = 'vector_flat_reliability_count'
VECTOR_FLAT_INV_RELIA_BIN_CENTER_KEY = 'vector_flat_inv_reliability_bin_center'
VECTOR_FLAT_INV_RELIABILITY_COUNT_KEY = 'vector_flat_inv_reliability_count'
AUX_RELIABILITY_KEY = 'aux_reliability'
AUX_RELIABILITY_X_KEY = 'aux_reliability_x'
AUX_RELIABILITY_Y_KEY = 'aux_reliability_y'
AUX_RELIA_BIN_CENTER_KEY = 'aux_reliability_bin_center'
AUX_RELIABILITY_COUNT_KEY = 'aux_reliability_count'
AUX_INV_RELIA_BIN_CENTER_KEY = 'aux_inv_reliability_bin_center'
AUX_INV_RELIABILITY_COUNT_KEY = 'aux_inv_reliability_count'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILE_KEY = 'prediction_file_name'

AUX_TARGET_NAMES_KEY = 'aux_target_field_names'
AUX_PREDICTED_NAMES_KEY = 'aux_predicted_field_names'
AUX_TARGET_VALS_KEY = 'aux_target_matrix'
AUX_PREDICTED_VALS_KEY = 'aux_prediction_matrix'
SHORTWAVE_SURFACE_DOWN_FLUX_INDEX_KEY = 'shortwave_surface_down_flux_index'
SHORTWAVE_TOA_UP_FLUX_INDEX_KEY = 'shortwave_toa_up_flux_index'
LONGWAVE_SURFACE_DOWN_FLUX_INDEX_KEY = 'longwave_surface_down_flux_index'
LONGWAVE_TOA_UP_FLUX_INDEX_KEY = 'longwave_toa_up_flux_index'

SCALAR_TARGET_VALS_KEY = 'scalar_target_matrix'
SCALAR_PREDICTED_VALS_KEY = 'scalar_prediction_matrix'
VECTOR_TARGET_VALS_KEY = 'vector_target_matrix'
VECTOR_PREDICTED_VALS_KEY = 'vector_prediction_matrix'


def _add_wavelength_dim_to_table(evaluation_table_xarray):
    """Adds wavelength dimension to evaluation table.

    :param evaluation_table_xarray: xarray table in format created by
        `get_scores_all_variables`.
    :return: evaluation_table_xarray: Same but including wavelength dimension.
    """

    if WAVELENGTH_DIM in evaluation_table_xarray.coords:
        return evaluation_table_xarray

    data_dict = {}
    coord_dict = {}

    for var_name in evaluation_table_xarray.data_vars:
        dimension_keys = list(evaluation_table_xarray[var_name].dims)
        data_matrix = evaluation_table_xarray[var_name].values

        if SCALAR_FIELD_DIM in dimension_keys:
            i = dimension_keys.index(SCALAR_FIELD_DIM)
            dimension_keys.insert(i, WAVELENGTH_DIM)
            data_matrix = numpy.expand_dims(data_matrix, axis=i)

        if VECTOR_FIELD_DIM in dimension_keys:
            i = dimension_keys.index(VECTOR_FIELD_DIM)
            dimension_keys.insert(i, WAVELENGTH_DIM)
            data_matrix = numpy.expand_dims(data_matrix, axis=i)

        if AUX_TARGET_FIELD_DIM in dimension_keys:
            i = dimension_keys.index(AUX_TARGET_FIELD_DIM)
            dimension_keys.insert(i, WAVELENGTH_DIM)
            data_matrix = numpy.expand_dims(data_matrix, axis=i)

        data_dict[var_name] = (copy.deepcopy(dimension_keys), data_matrix + 0.)

    for coord_name in evaluation_table_xarray.coords:
        coord_dict[coord_name] = (
            evaluation_table_xarray.coords[coord_name].values
        )

    coord_dict[WAVELENGTH_DIM] = numpy.array(
        [example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]
    )

    return xarray.Dataset(
        data_vars=data_dict, coords=coord_dict,
        attrs=evaluation_table_xarray.attrs
    )


def _check_args(
        scalar_target_matrix, scalar_prediction_matrix, vector_target_matrix,
        vector_prediction_matrix, mean_training_example_dict):
    """Error-checks input args for methods called `get_*_all_variables`.

    :param scalar_target_matrix: See doc for `get_*_all_variables`.
    :param scalar_prediction_matrix: Same.
    :param vector_target_matrix: Same.
    :param vector_prediction_matrix: Same.
    :param mean_training_example_dict: Same.
    """

    error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)

    num_examples = scalar_target_matrix.shape[0]
    num_scalar_targets = len(
        mean_training_example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    )
    num_wavelengths = len(
        mean_training_example_dict[example_utils.TARGET_WAVELENGTHS_KEY]
    )

    these_expected_dim = numpy.array(
        [num_examples, num_wavelengths, num_scalar_targets], dtype=int
    )
    error_checking.assert_is_numpy_array(
        scalar_target_matrix, exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(scalar_prediction_matrix)
    error_checking.assert_is_numpy_array(
        scalar_prediction_matrix, exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
    error_checking.assert_is_numpy_array(
        vector_target_matrix, num_dimensions=4
    )

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = len(
        mean_training_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )

    these_expected_dim = numpy.array(
        [num_examples, num_heights, num_wavelengths, num_vector_targets],
        dtype=int
    )
    error_checking.assert_is_numpy_array(
        vector_target_matrix, exact_dimensions=these_expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(
        vector_prediction_matrix
    )
    error_checking.assert_is_numpy_array(
        vector_prediction_matrix, exact_dimensions=these_expected_dim
    )


def _get_mse_one_scalar(target_values, predicted_values):
    """Computes mean squared error (MSE) for one scalar target variable.

    E = number of examples

    :param target_values: length-E numpy array of target (actual) values.
    :param predicted_values: length-E numpy array of predicted values.
    :return: mse_total: Total MSE.
    :return: mse_bias: Bias component.
    :return: mse_variance: Variance component.
    """

    mse_total = numpy.mean((target_values - predicted_values) ** 2)
    mse_bias = numpy.mean(target_values - predicted_values) ** 2
    mse_variance = mse_total - mse_bias

    return mse_total, mse_bias, mse_variance


def _get_mse_ss_one_scalar(target_values, predicted_values,
                           mean_training_target_value):
    """Computes MSE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param mean_training_target_value: Mean target value over all training
        examples.
    :return: mse_skill_score: Self-explanatory.
    """

    mse_actual = _get_mse_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )[0]
    mse_climo = _get_mse_one_scalar(
        target_values=target_values, predicted_values=mean_training_target_value
    )[0]

    return (mse_climo - mse_actual) / mse_climo


def _get_mae_one_scalar(target_values, predicted_values):
    """Computes mean absolute error (MAE) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: mean_absolute_error: Self-explanatory.
    """

    return numpy.mean(numpy.abs(target_values - predicted_values))


def _get_mae_ss_one_scalar(target_values, predicted_values,
                           mean_training_target_value):
    """Computes MAE skill score for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param mean_training_target_value: See doc for `_get_mse_ss_one_scalar`.
    :return: mae_skill_score: Self-explanatory.
    """

    mae_actual = _get_mae_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )
    mae_climo = _get_mae_one_scalar(
        target_values=target_values, predicted_values=mean_training_target_value
    )

    return (mae_climo - mae_actual) / mae_climo


def _get_bias_one_scalar(target_values, predicted_values):
    """Computes bias (mean signed error) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: bias: Self-explanatory.
    """

    return numpy.mean(predicted_values - target_values)


def _get_correlation_one_scalar(target_values, predicted_values):
    """Computes Pearson correlation for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: correlation: Self-explanatory.
    """

    numerator = numpy.sum(
        (target_values - numpy.mean(target_values)) *
        (predicted_values - numpy.mean(predicted_values))
    )
    sum_squared_target_diffs = numpy.sum(
        (target_values - numpy.mean(target_values)) ** 2
    )
    sum_squared_prediction_diffs = numpy.sum(
        (predicted_values - numpy.mean(predicted_values)) ** 2
    )

    correlation = (
        numerator /
        numpy.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
    )

    return correlation


def _get_kge_one_scalar(target_values, predicted_values):
    """Computes KGE (Kling-Gupta efficiency) for one scalar target variable.

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :return: kge: Self-explanatory.
    """

    correlation = _get_correlation_one_scalar(
        target_values=target_values, predicted_values=predicted_values
    )

    mean_target_value = numpy.mean(target_values)
    mean_predicted_value = numpy.mean(predicted_values)
    stdev_target_value = numpy.std(target_values, ddof=1)
    stdev_predicted_value = numpy.std(predicted_values, ddof=1)

    variance_bias = (
        (stdev_predicted_value / mean_predicted_value) *
        (stdev_target_value / mean_target_value) ** -1
    )
    mean_bias = mean_predicted_value / mean_target_value

    kge = 1. - numpy.sqrt(
        (correlation - 1.) ** 2 +
        (variance_bias - 1.) ** 2 +
        (mean_bias - 1.) ** 2
    )

    return kge


def _get_prmse_one_variable(target_matrix, prediction_matrix):
    """Computes profile root mean squared error (PRMSE) for one variable.

    E = number of examples
    H = number of heights

    This is "PRMSE," as opposed to "prmse," in Krasnopolsky's papers.

    :param target_matrix: E-by-H numpy array of target (actual) values.
    :param prediction_matrix: E-by-H numpy array of predicted values.
    :return: prmse: Self-explanatory.
    """

    return numpy.mean(numpy.sqrt(
        numpy.mean((target_matrix - prediction_matrix) ** 2, axis=1)
    ))


def _get_rel_curve_one_scalar(
        target_values, predicted_values, num_bins, min_bin_edge, max_bin_edge,
        invert=False):
    """Computes reliability curve for one scalar target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param num_bins: Number of bins (points in curve).
    :param min_bin_edge: Value at lower edge of first bin.
    :param max_bin_edge: Value at upper edge of last bin.
    :param invert: Boolean flag.  If True, will return inverted reliability
        curve, which bins by target value and relates target value to
        conditional mean prediction.  If False, will return normal reliability
        curve, which bins by predicted value and relates predicted value to
        conditional mean observation (target).
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    # max_bin_edge = max([max_bin_edge, numpy.finfo(float).eps])
    # min_bin_edge = min([min_bin_edge, 0.])

    bin_index_by_example = histograms.create_histogram(
        input_values=target_values if invert else predicted_values,
        num_bins=num_bins, min_value=min_bin_edge, max_value=max_bin_edge
    )[0]

    mean_predictions = numpy.full(num_bins, numpy.nan)
    mean_observations = numpy.full(num_bins, numpy.nan)
    example_counts = numpy.full(num_bins, -1, dtype=int)

    for i in range(num_bins):
        these_example_indices = numpy.where(bin_index_by_example == i)[0]

        example_counts[i] = len(these_example_indices)
        mean_predictions[i] = numpy.mean(
            predicted_values[these_example_indices]
        )
        mean_observations[i] = numpy.mean(target_values[these_example_indices])

    return mean_predictions, mean_observations, example_counts


def _get_scores_one_replicate(
        result_table_xarray, prediction_dict, replicate_index,
        example_indices_in_replicate, mean_training_example_dict,
        min_heating_rate_k_day01, max_heating_rate_k_day01,
        min_heating_rate_percentile, max_heating_rate_percentile,
        min_raw_flux_w_m02, max_raw_flux_w_m02,
        min_raw_flux_percentile, max_raw_flux_percentile,
        min_net_flux_w_m02, max_net_flux_w_m02,
        min_net_flux_percentile, max_net_flux_percentile):
    """Computes scores for one bootstrap replicate.

    E = number of examples
    H = number of heights
    T_s = number of scalar targets
    T_v = number of vector targets
    T_a = number of auxiliary targets
    W = number of wavelengths

    :param result_table_xarray: See doc for `get_scores_all_variables`.
    :param prediction_dict: Dictionary with the following keys.
        "scalar_target_matrix": numpy array (E x W x T_s) with actual values of
        scalar targets.
        "scalar_prediction_matrix": Same as "scalar_target_matrix" but with
        predicted values.
        "vector_target_matrix": numpy array (E x H x W x T_v) with actual values
        of vector targets.
        "vector_prediction_matrix": Same as "vector_target_matrix" but with
        predicted values.
        "aux_target_matrix": numpy array (E x W x T_a) with actual values of
        auxiliary targets.
        "aux_prediction_matrix": Same as "aux_target_matrix" but with
        predicted values.

    :param replicate_index: Index of current bootstrap replicate.
    :param example_indices_in_replicate: 1-D numpy array with indices of
        examples in this bootstrap replicate.
    :param mean_training_example_dict: Dictionary created by
        `normalization.create_mean_example`.
    :param min_heating_rate_k_day01: See doc for `get_scores_all_variables`.
    :param max_heating_rate_k_day01: Same.
    :param min_heating_rate_percentile: Same.
    :param max_heating_rate_percentile: Same.
    :param min_raw_flux_w_m02: Same.
    :param max_raw_flux_w_m02: Same.
    :param min_raw_flux_percentile: Same.
    :param max_raw_flux_percentile: Same.
    :param min_net_flux_w_m02: Same.
    :param max_net_flux_w_m02: Same.
    :param min_net_flux_percentile: Same.
    :param max_net_flux_percentile: Same.
    :return: result_table_xarray: Same as input but with values filled for [i]th
        bootstrap replicate, where i = `replicate_index`.
    """

    rtx = result_table_xarray
    mted = mean_training_example_dict
    r = replicate_index + 0

    full_scalar_target_matrix = prediction_dict[SCALAR_TARGET_VALS_KEY]
    full_scalar_prediction_matrix = prediction_dict[SCALAR_PREDICTED_VALS_KEY]
    full_vector_target_matrix = prediction_dict[VECTOR_TARGET_VALS_KEY]
    full_vector_prediction_matrix = prediction_dict[VECTOR_PREDICTED_VALS_KEY]
    full_aux_target_matrix = prediction_dict[AUX_TARGET_VALS_KEY]
    full_aux_prediction_matrix = prediction_dict[AUX_PREDICTED_VALS_KEY]

    scalar_target_matrix = (
        full_scalar_target_matrix[example_indices_in_replicate, ...]
    )
    scalar_prediction_matrix = (
        full_scalar_prediction_matrix[example_indices_in_replicate, ...]
    )
    vector_target_matrix = (
        full_vector_target_matrix[example_indices_in_replicate, ...]
    )
    vector_prediction_matrix = (
        full_vector_prediction_matrix[example_indices_in_replicate, ...]
    )
    aux_target_matrix = (
        full_aux_target_matrix[example_indices_in_replicate, ...]
    )
    aux_prediction_matrix = (
        full_aux_prediction_matrix[example_indices_in_replicate, ...]
    )

    num_examples = scalar_target_matrix.shape[0]
    num_wavelengths = scalar_target_matrix.shape[1]
    num_scalar_targets = scalar_target_matrix.shape[2]
    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[3]

    if AUX_TARGET_FIELD_DIM in rtx.coords:
        aux_target_field_names = rtx.coords[AUX_TARGET_FIELD_DIM].values
    else:
        aux_target_field_names = []

    num_aux_targets = len(aux_target_field_names)

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            this_climo_value = (
                mted[example_utils.SCALAR_TARGET_VALS_KEY][0, w, t]
            )

            rtx[SCALAR_TARGET_STDEV_KEY].values[w, t, r] = numpy.std(
                scalar_target_matrix[:, w, t], ddof=1
            )
            rtx[SCALAR_PREDICTION_STDEV_KEY].values[w, t, r] = numpy.std(
                scalar_prediction_matrix[:, w, t], ddof=1
            )
            rtx[SCALAR_MAE_KEY].values[w, t, r] = _get_mae_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t]
            )
            rtx[SCALAR_MAE_SKILL_KEY].values[w, t, r] = _get_mae_ss_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t],
                mean_training_target_value=this_climo_value
            )

            (
                rtx[SCALAR_MSE_KEY].values[w, t, r],
                rtx[SCALAR_MSE_BIAS_KEY].values[w, t, r],
                rtx[SCALAR_MSE_VARIANCE_KEY].values[w, t, r]
            ) = _get_mse_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t]
            )
            rtx[SCALAR_MSE_SKILL_KEY].values[w, t, r] = _get_mse_ss_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t],
                mean_training_target_value=this_climo_value
            )
            rtx[SCALAR_BIAS_KEY].values[w, t, r] = _get_bias_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t]
            )
            rtx[SCALAR_CORRELATION_KEY].values[w, t, r] = (
                _get_correlation_one_scalar(
                    target_values=scalar_target_matrix[:, w, t],
                    predicted_values=scalar_prediction_matrix[:, w, t]
                )
            )
            rtx[SCALAR_KGE_KEY].values[w, t, r] = _get_kge_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t]
            )

            if num_examples == 0:
                min_bin_edge = 0.
                max_bin_edge = 1.
            elif min_raw_flux_w_m02 is not None:
                min_bin_edge = min_raw_flux_w_m02 + 0.
                max_bin_edge = max_raw_flux_w_m02 + 0.
            else:
                min_bin_edge = numpy.percentile(
                    full_scalar_prediction_matrix[:, w, t],
                    min_raw_flux_percentile
                )
                max_bin_edge = numpy.percentile(
                    full_scalar_prediction_matrix[:, w, t],
                    max_raw_flux_percentile
                )

            max_bin_edge = max([max_bin_edge, min_bin_edge + TOLERANCE])

            (
                rtx[SCALAR_RELIABILITY_X_KEY].values[w, t, :, r],
                rtx[SCALAR_RELIABILITY_Y_KEY].values[w, t, :, r],
                these_counts
            ) = _get_rel_curve_one_scalar(
                target_values=scalar_target_matrix[:, w, t],
                predicted_values=scalar_prediction_matrix[:, w, t],
                num_bins=len(rtx.coords[RAW_FLUX_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            these_squared_diffs = (
                rtx[SCALAR_RELIABILITY_X_KEY].values[w, t, :, r] -
                rtx[SCALAR_RELIABILITY_Y_KEY].values[w, t, :, r]
            ) ** 2

            rtx[SCALAR_RELIABILITY_KEY].values[w, t, r] = (
                numpy.nansum(these_counts * these_squared_diffs) /
                numpy.sum(these_counts)
            )

            if r > 0:
                continue

            (
                rtx[SCALAR_RELIA_BIN_CENTER_KEY].values[w, t, :],
                _,
                rtx[SCALAR_RELIABILITY_COUNT_KEY].values[w, t, :]
            ) = _get_rel_curve_one_scalar(
                target_values=full_scalar_target_matrix[:, w, t],
                predicted_values=full_scalar_prediction_matrix[:, w, t],
                num_bins=len(rtx.coords[RAW_FLUX_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            (
                rtx[SCALAR_INV_RELIA_BIN_CENTER_KEY].values[w, t, :],
                _,
                rtx[SCALAR_INV_RELIABILITY_COUNT_KEY].values[w, t, :]
            ) = _get_rel_curve_one_scalar(
                target_values=full_scalar_target_matrix[:, w, t],
                predicted_values=full_scalar_prediction_matrix[:, w, t],
                num_bins=len(rtx.coords[RAW_FLUX_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=True
            )

            if full_scalar_target_matrix.size == 0:
                continue

            (
                rtx[SCALAR_KS_STATISTIC_KEY].values[w, t],
                rtx[SCALAR_KS_P_VALUE_KEY].values[w, t]
            ) = ks_2samp(
                full_scalar_target_matrix[:, w, t],
                full_scalar_prediction_matrix[:, w, t],
                alternative='two-sided', mode='auto'
            )

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            rtx[VECTOR_PRMSE_KEY].values[w, t, r] = _get_prmse_one_variable(
                target_matrix=vector_target_matrix[..., w, t],
                prediction_matrix=vector_prediction_matrix[..., w, t]
            )

            for h in range(num_heights):
                this_climo_value = (
                    mted[example_utils.VECTOR_TARGET_VALS_KEY][0, h, w, t]
                )

                rtx[VECTOR_TARGET_STDEV_KEY].values[h, w, t, r] = numpy.std(
                    vector_target_matrix[:, h, w, t], ddof=1
                )
                rtx[VECTOR_PREDICTION_STDEV_KEY].values[h, w, t, r] = numpy.std(
                    vector_prediction_matrix[:, h, w, t], ddof=1
                )
                rtx[VECTOR_MAE_KEY].values[h, w, t, r] = _get_mae_one_scalar(
                    target_values=vector_target_matrix[:, h, w, t],
                    predicted_values=vector_prediction_matrix[:, h, w, t]
                )
                rtx[VECTOR_MAE_SKILL_KEY].values[h, w, t, r] = (
                    _get_mae_ss_one_scalar(
                        target_values=vector_target_matrix[:, h, w, t],
                        predicted_values=vector_prediction_matrix[:, h, w, t],
                        mean_training_target_value=this_climo_value
                    )
                )

                (
                    rtx[VECTOR_MSE_KEY].values[h, w, t, r],
                    rtx[VECTOR_MSE_BIAS_KEY].values[h, w, t, r],
                    rtx[VECTOR_MSE_VARIANCE_KEY].values[h, w, t, r]
                ) = _get_mse_one_scalar(
                    target_values=vector_target_matrix[:, h, w, t],
                    predicted_values=vector_prediction_matrix[:, h, w, t]
                )

                rtx[VECTOR_MSE_SKILL_KEY].values[h, w, t, r] = (
                    _get_mse_ss_one_scalar(
                        target_values=vector_target_matrix[:, h, w, t],
                        predicted_values=vector_prediction_matrix[:, h, w, t],
                        mean_training_target_value=this_climo_value
                    )
                )
                rtx[VECTOR_BIAS_KEY].values[h, w, t, r] = _get_bias_one_scalar(
                    target_values=vector_target_matrix[:, h, w, t],
                    predicted_values=vector_prediction_matrix[:, h, w, t]
                )
                rtx[VECTOR_CORRELATION_KEY].values[h, w, t, r] = (
                    _get_correlation_one_scalar(
                        target_values=vector_target_matrix[:, h, w, t],
                        predicted_values=vector_prediction_matrix[:, h, w, t]
                    )
                )
                rtx[VECTOR_KGE_KEY].values[h, w, t, r] = _get_kge_one_scalar(
                    target_values=vector_target_matrix[:, h, w, t],
                    predicted_values=vector_prediction_matrix[:, h, w, t]
                )

                if num_examples == 0:
                    min_bin_edge = 0.
                    max_bin_edge = 1.
                elif min_heating_rate_k_day01 is not None:
                    min_bin_edge = min_heating_rate_k_day01 + 0.
                    max_bin_edge = max_heating_rate_k_day01 + 0.
                else:
                    min_bin_edge = numpy.percentile(
                        full_vector_prediction_matrix[:, h, w, t],
                        min_heating_rate_percentile
                    )
                    max_bin_edge = numpy.percentile(
                        full_vector_prediction_matrix[:, h, w, t],
                        max_heating_rate_percentile
                    )

                max_bin_edge = max([max_bin_edge, min_bin_edge + TOLERANCE])

                (
                    rtx[VECTOR_RELIABILITY_X_KEY].values[h, w, t, :, r],
                    rtx[VECTOR_RELIABILITY_Y_KEY].values[h, w, t, :, r],
                    these_counts
                ) = _get_rel_curve_one_scalar(
                    target_values=vector_target_matrix[:, h, w, t],
                    predicted_values=vector_prediction_matrix[:, h, w, t],
                    num_bins=len(rtx.coords[HEATING_RATE_BIN_DIM].values),
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=False
                )

                these_squared_diffs = (
                    rtx[VECTOR_RELIABILITY_X_KEY].values[h, w, t, :, r] -
                    rtx[VECTOR_RELIABILITY_Y_KEY].values[h, w, t, :, r]
                ) ** 2

                rtx[VECTOR_RELIABILITY_KEY].values[h, w, t, r] = (
                    numpy.nansum(these_counts * these_squared_diffs) /
                    numpy.sum(these_counts)
                )

                if r > 0:
                    continue

                (
                    rtx[VECTOR_RELIA_BIN_CENTER_KEY].values[h, w, t, :],
                    _,
                    rtx[VECTOR_RELIABILITY_COUNT_KEY].values[h, w, t, :]
                ) = _get_rel_curve_one_scalar(
                    target_values=full_vector_target_matrix[:, h, w, t],
                    predicted_values=full_vector_prediction_matrix[:, h, w, t],
                    num_bins=len(rtx.coords[HEATING_RATE_BIN_DIM].values),
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=False
                )

                (
                    rtx[VECTOR_INV_RELIA_BIN_CENTER_KEY].values[h, w, t, :],
                    _,
                    rtx[VECTOR_INV_RELIABILITY_COUNT_KEY].values[h, w, t, :]
                ) = _get_rel_curve_one_scalar(
                    target_values=
                    full_vector_target_matrix[:, h, w, t],
                    predicted_values=full_vector_prediction_matrix[:, h, w, t],
                    num_bins=len(rtx.coords[HEATING_RATE_BIN_DIM].values),
                    min_bin_edge=min_bin_edge,
                    max_bin_edge=max_bin_edge,
                    invert=True
                )

                if full_vector_target_matrix.size == 0:
                    continue

                (
                    rtx[VECTOR_KS_STATISTIC_KEY].values[h, w, t],
                    rtx[VECTOR_KS_P_VALUE_KEY].values[h, w, t]
                ) = ks_2samp(
                    full_vector_target_matrix[:, h, w, t],
                    full_vector_prediction_matrix[:, h, w, t],
                    alternative='two-sided', mode='auto'
                )

            flat_target_values = numpy.ravel(vector_target_matrix[..., w, t])
            flat_predicted_values = numpy.ravel(
                vector_prediction_matrix[..., w, t]
            )
            this_climo_value = numpy.mean(
                mted[example_utils.VECTOR_TARGET_VALS_KEY][0, :, w, t]
            )

            rtx[VECTOR_FLAT_MSE_SKILL_KEY].values[w, t, r] = (
                _get_mse_ss_one_scalar(
                    target_values=flat_target_values,
                    predicted_values=flat_predicted_values,
                    mean_training_target_value=this_climo_value
                )
            )

            if num_examples == 0:
                min_bin_edge = 0.
                max_bin_edge = 1.
            elif min_heating_rate_k_day01 is not None:
                min_bin_edge = min_heating_rate_k_day01 + 0.
                max_bin_edge = max_heating_rate_k_day01 + 0.
            else:
                min_bin_edge = numpy.percentile(
                    full_vector_prediction_matrix[..., w, t],
                    min_heating_rate_percentile
                )
                max_bin_edge = numpy.percentile(
                    full_vector_prediction_matrix[..., w, t],
                    max_heating_rate_percentile
                )

            max_bin_edge = max([max_bin_edge, min_bin_edge + TOLERANCE])

            (
                rtx[VECTOR_FLAT_RELIABILITY_X_KEY].values[w, t, :, r],
                rtx[VECTOR_FLAT_RELIABILITY_Y_KEY].values[w, t, :, r],
                these_counts
            ) = _get_rel_curve_one_scalar(
                target_values=flat_target_values,
                predicted_values=flat_predicted_values,
                num_bins=len(rtx.coords[HEATING_RATE_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            these_squared_diffs = (
                rtx[VECTOR_FLAT_RELIABILITY_X_KEY].values[w, t, :, r] -
                rtx[VECTOR_FLAT_RELIABILITY_Y_KEY].values[w, t, :, r]
            ) ** 2

            rtx[VECTOR_FLAT_RELIABILITY_KEY].values[w, t, r] = (
                numpy.nansum(these_counts * these_squared_diffs) /
                numpy.sum(these_counts)
            )

            if r > 0:
                continue

            full_flat_target_values = numpy.ravel(
                full_vector_target_matrix[..., w, t]
            )
            full_flat_predicted_values = numpy.ravel(
                full_vector_prediction_matrix[..., w, t]
            )

            (
                rtx[VECTOR_FLAT_RELIA_BIN_CENTER_KEY].values[w, t, :],
                _,
                rtx[VECTOR_FLAT_RELIABILITY_COUNT_KEY].values[w, t, :]
            ) = _get_rel_curve_one_scalar(
                target_values=full_flat_target_values,
                predicted_values=full_flat_predicted_values,
                num_bins=len(rtx.coords[HEATING_RATE_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            (
                rtx[VECTOR_FLAT_INV_RELIA_BIN_CENTER_KEY].values[w, t, :],
                _,
                rtx[VECTOR_FLAT_INV_RELIABILITY_COUNT_KEY].values[w, t, :]
            ) = _get_rel_curve_one_scalar(
                target_values=full_flat_target_values,
                predicted_values=full_flat_predicted_values,
                num_bins=len(rtx.coords[HEATING_RATE_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=True
            )

    for t in range(num_aux_targets):
        for w in range(num_wavelengths):
            if aux_target_field_names[t] == SHORTWAVE_NET_FLUX_NAME:
                d_idx = mted[example_utils.SCALAR_TARGET_NAMES_KEY].index(
                    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
                )
                u_idx = mted[example_utils.SCALAR_TARGET_NAMES_KEY].index(
                    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
                )
                this_climo_value = (
                    mted[example_utils.SCALAR_TARGET_VALS_KEY][0, w, d_idx] -
                    mted[example_utils.SCALAR_TARGET_VALS_KEY][0, w, u_idx]
                )
            elif aux_target_field_names[t] == LONGWAVE_NET_FLUX_NAME:
                d_idx = mted[example_utils.SCALAR_TARGET_NAMES_KEY].index(
                    example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
                )
                u_idx = mted[example_utils.SCALAR_TARGET_NAMES_KEY].index(
                    example_utils.LONGWAVE_TOA_UP_FLUX_NAME
                )
                this_climo_value = (
                    mted[example_utils.SCALAR_TARGET_VALS_KEY][0, w, d_idx] -
                    mted[example_utils.SCALAR_TARGET_VALS_KEY][0, w, u_idx]
                )
            else:
                this_climo_value = None

            rtx[AUX_TARGET_STDEV_KEY].values[w, t, r] = numpy.std(
                aux_target_matrix[:, w, t], ddof=1
            )
            rtx[AUX_PREDICTION_STDEV_KEY].values[w, t, r] = numpy.std(
                aux_prediction_matrix[:, w, t], ddof=1
            )
            rtx[AUX_MAE_KEY].values[w, t, r] = _get_mae_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t]
            )
            (
                rtx[AUX_MSE_KEY].values[w, t, r],
                rtx[AUX_MSE_BIAS_KEY].values[w, t, r],
                rtx[AUX_MSE_VARIANCE_KEY].values[w, t, r]
            ) = _get_mse_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t]
            )
            rtx[AUX_MAE_SKILL_KEY].values[w, t, r] = _get_mae_ss_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t],
                mean_training_target_value=this_climo_value
            )
            rtx[AUX_MSE_SKILL_KEY].values[w, t, r] = _get_mse_ss_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t],
                mean_training_target_value=this_climo_value
            )
            rtx[AUX_BIAS_KEY].values[w, t, r] = _get_bias_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t]
            )
            rtx[AUX_CORRELATION_KEY].values[w, t, r] = (
                _get_correlation_one_scalar(
                    target_values=aux_target_matrix[:, w, t],
                    predicted_values=aux_prediction_matrix[:, w, t]
                )
            )
            rtx[AUX_KGE_KEY].values[w, t, r] = _get_kge_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t]
            )

            if num_examples == 0:
                min_bin_edge = 0.
                max_bin_edge = 1.
            elif min_net_flux_w_m02 is not None:
                min_bin_edge = min_net_flux_w_m02 + 0.
                max_bin_edge = max_net_flux_w_m02 + 0.
            else:
                min_bin_edge = numpy.percentile(
                    full_aux_prediction_matrix[:, w, t], min_net_flux_percentile
                )
                max_bin_edge = numpy.percentile(
                    full_aux_prediction_matrix[:, w, t], max_net_flux_percentile
                )

            max_bin_edge = max([max_bin_edge, min_bin_edge + TOLERANCE])

            (
                rtx[AUX_RELIABILITY_X_KEY].values[w, t, :, r],
                rtx[AUX_RELIABILITY_Y_KEY].values[w, t, :, r],
                these_counts
            ) = _get_rel_curve_one_scalar(
                target_values=aux_target_matrix[:, w, t],
                predicted_values=aux_prediction_matrix[:, w, t],
                num_bins=len(rtx.coords[NET_FLUX_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            these_squared_diffs = (
                rtx[AUX_RELIABILITY_X_KEY].values[w, t, :, r] -
                rtx[AUX_RELIABILITY_Y_KEY].values[w, t, :, r]
            ) ** 2

            rtx[AUX_RELIABILITY_KEY].values[w, t, r] = (
                numpy.nansum(these_counts * these_squared_diffs) /
                numpy.sum(these_counts)
            )

            if r > 0:
                continue

            (
                rtx[AUX_RELIA_BIN_CENTER_KEY].values[w, t, :], _,
                rtx[AUX_RELIABILITY_COUNT_KEY].values[w, t, :]
            ) = _get_rel_curve_one_scalar(
                target_values=full_aux_target_matrix[:, w, t],
                predicted_values=full_aux_prediction_matrix[:, w, t],
                num_bins=len(rtx.coords[NET_FLUX_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=False
            )

            (
                rtx[AUX_INV_RELIA_BIN_CENTER_KEY].values[w, t, :], _,
                rtx[AUX_INV_RELIABILITY_COUNT_KEY].values[w, t, :]
            ) = _get_rel_curve_one_scalar(
                target_values=full_aux_target_matrix[:, w, t],
                predicted_values=full_aux_prediction_matrix[:, w, t],
                num_bins=len(rtx.coords[NET_FLUX_BIN_DIM].values),
                min_bin_edge=min_bin_edge,
                max_bin_edge=max_bin_edge,
                invert=True
            )

            if full_aux_target_matrix.size > 0:
                (
                    rtx[AUX_KS_STATISTIC_KEY].values[w, t],
                    rtx[AUX_KS_P_VALUE_KEY].values[w, t]
                ) = ks_2samp(
                    full_aux_target_matrix[:, w, t],
                    full_aux_prediction_matrix[:, w, t],
                    alternative='two-sided', mode='auto'
                )

    return rtx


def confidence_interval_to_polygon(
        x_value_matrix, y_value_matrix, confidence_level, same_order):
    """Turns confidence interval into polygon.

    P = number of points
    B = number of bootstrap replicates
    V = number of vertices in resulting polygon = 2 * P + 1

    :param x_value_matrix: P-by-B numpy array of x-values.
    :param y_value_matrix: P-by-B numpy array of y-values.
    :param confidence_level: Confidence level (in range 0...1).
    :param same_order: Boolean flag.  If True (False), minimum x-values will be
        matched with minimum (maximum) y-values.
    :return: polygon_coord_matrix: V-by-2 numpy array of coordinates
        (x-coordinates in first column, y-coords in second).
    """

    error_checking.assert_is_numpy_array(x_value_matrix, num_dimensions=2)
    error_checking.assert_is_numpy_array(y_value_matrix, num_dimensions=2)

    expected_dim = numpy.array([
        x_value_matrix.shape[0], y_value_matrix.shape[1]
    ], dtype=int)

    error_checking.assert_is_numpy_array(
        y_value_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_leq(confidence_level, 1.)
    error_checking.assert_is_boolean(same_order)

    min_percentile = 50 * (1. - confidence_level)
    max_percentile = 50 * (1. + confidence_level)

    x_values_bottom = numpy.nanpercentile(
        x_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    x_values_top = numpy.nanpercentile(
        x_value_matrix, max_percentile, axis=1, interpolation='linear'
    )
    y_values_bottom = numpy.nanpercentile(
        y_value_matrix, min_percentile, axis=1, interpolation='linear'
    )
    y_values_top = numpy.nanpercentile(
        y_value_matrix, max_percentile, axis=1, interpolation='linear'
    )

    real_indices = numpy.where(numpy.invert(numpy.logical_or(
        numpy.isnan(x_values_bottom), numpy.isnan(y_values_bottom)
    )))[0]

    if len(real_indices) == 0:
        return None

    x_values_bottom = x_values_bottom[real_indices]
    x_values_top = x_values_top[real_indices]
    y_values_bottom = y_values_bottom[real_indices]
    y_values_top = y_values_top[real_indices]

    x_vertices = numpy.concatenate((
        x_values_top, x_values_bottom[::-1], x_values_top[[0]]
    ))

    if same_order:
        y_vertices = numpy.concatenate((
            y_values_top, y_values_bottom[::-1], y_values_top[[0]]
        ))
    else:
        y_vertices = numpy.concatenate((
            y_values_bottom, y_values_top[::-1], y_values_bottom[[0]]
        ))

    return numpy.transpose(numpy.vstack((
        x_vertices, y_vertices
    )))


def get_aux_fields(prediction_dict, example_dict):
    """Returns auxiliary fields.

    F = number of pairs of auxiliary fields
    W = number of wavelengths
    E = number of examples

    :param prediction_dict: See doc for `prediction_io.read_file`.
    :param example_dict: Dictionary with the following keys (details for each
        key in documentation for `example_io.read_file`).
    example_dict['scalar_target_names']

    :return: aux_prediction_dict: Dictionary with the following keys.
    aux_prediction_dict['aux_target_field_names']: length-F list with names of
        target fields.
    aux_prediction_dict['aux_predicted_field_names']: length-F list with names
        of predicted fields.
    aux_prediction_dict['aux_target_matrix']: E-by-W-by-F numpy array of target
        (actual) values.
    aux_prediction_dict['aux_prediction_matrix']: E-by-W-by-F numpy array of
        predicted values.
    aux_prediction_dict['shortwave_surface_down_flux_index']: Array index of
        shortwave surface downwelling flux in `example_dict`.  If surface
        downwelling flux is not available, this is -1.
    aux_prediction_dict['longwave_surface_down_flux_index']: Same but for
        longwave.
    aux_prediction_dict['shortwave_toa_up_flux_index']: Array index of shortwave
        TOA upwelling flux in `example_dict`.  If TOA upwelling flux is not
        available, this is -1.
    aux_prediction_dict['longwave_toa_up_flux_index']: Same but for longwave.
    """

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )

    num_examples = scalar_target_matrix.shape[0]
    num_wavelengths = scalar_target_matrix.shape[1]
    these_dim = (num_examples, num_wavelengths, 0)

    aux_target_matrix = numpy.full(these_dim, numpy.nan)
    aux_prediction_matrix = numpy.full(these_dim, numpy.nan)
    aux_target_field_names = []
    aux_predicted_field_names = []

    shortwave_surface_down_flux_index = -1
    shortwave_toa_up_flux_index = -1
    longwave_surface_down_flux_index = -1
    longwave_toa_up_flux_index = -1

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    these_flux_names = [
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
        example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
    ]

    if all([n in scalar_target_names for n in these_flux_names]):
        shortwave_surface_down_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        shortwave_toa_up_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(SHORTWAVE_NET_FLUX_NAME)
        aux_predicted_field_names.append(SHORTWAVE_NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[..., [shortwave_surface_down_flux_index]] -
            scalar_target_matrix[..., [shortwave_toa_up_flux_index]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=-1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[..., [shortwave_surface_down_flux_index]] -
            scalar_prediction_matrix[..., [shortwave_toa_up_flux_index]]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=-1
        )

    these_flux_names = [
        example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME,
        example_utils.LONGWAVE_TOA_UP_FLUX_NAME
    ]

    if all([n in scalar_target_names for n in these_flux_names]):
        longwave_surface_down_flux_index = scalar_target_names.index(
            example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )
        longwave_toa_up_flux_index = scalar_target_names.index(
            example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(LONGWAVE_NET_FLUX_NAME)
        aux_predicted_field_names.append(LONGWAVE_NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[..., [longwave_surface_down_flux_index]] -
            scalar_target_matrix[..., [longwave_toa_up_flux_index]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=-1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[..., [longwave_surface_down_flux_index]] -
            scalar_prediction_matrix[..., [longwave_toa_up_flux_index]]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=-1
        )

    return {
        AUX_TARGET_NAMES_KEY: aux_target_field_names,
        AUX_PREDICTED_NAMES_KEY: aux_predicted_field_names,
        AUX_TARGET_VALS_KEY: aux_target_matrix,
        AUX_PREDICTED_VALS_KEY: aux_prediction_matrix,
        SHORTWAVE_SURFACE_DOWN_FLUX_INDEX_KEY:
            shortwave_surface_down_flux_index,
        SHORTWAVE_TOA_UP_FLUX_INDEX_KEY: shortwave_toa_up_flux_index,
        LONGWAVE_SURFACE_DOWN_FLUX_INDEX_KEY: longwave_surface_down_flux_index,
        LONGWAVE_TOA_UP_FLUX_INDEX_KEY: longwave_toa_up_flux_index
    }


def get_scores_all_variables(
        prediction_file_name, num_bootstrap_reps, num_heating_rate_bins,
        min_heating_rate_k_day01, max_heating_rate_k_day01,
        min_heating_rate_percentile, max_heating_rate_percentile,
        num_raw_flux_bins, min_raw_flux_w_m02, max_raw_flux_w_m02,
        min_raw_flux_percentile, max_raw_flux_percentile,
        num_net_flux_bins, min_net_flux_w_m02, max_net_flux_w_m02,
        min_net_flux_percentile, max_net_flux_percentile):
    """Computes desired scores for all target variables.

    :param prediction_file_name: Path to file with predictions that will be
        evaluated.  This file will be read by `prediction_io.read_file`.
    :param num_bootstrap_reps: Number of bootstrap replicates.
    :param num_heating_rate_bins: Number of heating-rate bins for reliability
        curves.
    :param min_heating_rate_k_day01: Minimum heating rate (Kelvins per day) for
        reliability curves.  If you instead want minimum heating rate to be a
        percentile over the data -- chosen independently at each height -- make
        this argument None.
    :param max_heating_rate_k_day01: Same as above but max heating rate.
    :param min_heating_rate_percentile: Determines minimum heating rate for
        reliability curves.  This percentile (ranging from 0...100) will be
        taken independently at each height.
    :param max_heating_rate_percentile: Same as above but max heating rate.
    :param num_raw_flux_bins: Number of bins for reliability curves on raw flux
        variables (surface downwelling and TOA upwelling).
    :param min_raw_flux_w_m02: Min flux for reliability curves on raw flux
        variables (surface downwelling and TOA upwelling).  If you want to
        specify min/max by percentiles instead -- chosen independently for each
        variable -- make this argument None.
    :param max_raw_flux_w_m02: Same as above but for max flux.
    :param min_raw_flux_percentile: Min flux percentile -- taken independently
        for each raw flux variable (surface downwelling and TOA upwelling) --
        for reliability curves.  If you want to specify min/max by physical
        values instead, make this argument None.
    :param max_raw_flux_percentile: Same as above but for max percentile.
    :param num_net_flux_bins: Number of net-flux bins for reliability curve.
    :param min_net_flux_w_m02: Min net flux for reliability curve.  If you want
        to specify min/max by percentiles instead, make this argument None.
    :param max_net_flux_w_m02: Same as above but for max net flux.
    :param min_net_flux_percentile: Min net-flux percentiles for reliability
        curve.  If you want to specify min/max by physical values instead, make
        this argument None.
    :param max_net_flux_percentile: Same as above but for max percentile.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_string(prediction_file_name)
    error_checking.assert_is_integer(num_bootstrap_reps)
    error_checking.assert_is_greater(num_bootstrap_reps, 0)
    error_checking.assert_is_integer(num_heating_rate_bins)
    error_checking.assert_is_geq(num_heating_rate_bins, 10)
    error_checking.assert_is_leq(num_heating_rate_bins, 100)
    error_checking.assert_is_integer(num_raw_flux_bins)
    error_checking.assert_is_geq(num_raw_flux_bins, 10)
    error_checking.assert_is_leq(num_raw_flux_bins, 100)
    error_checking.assert_is_integer(num_net_flux_bins)
    error_checking.assert_is_geq(num_net_flux_bins, 10)
    error_checking.assert_is_leq(num_net_flux_bins, 100)

    if min_heating_rate_k_day01 is None or max_heating_rate_k_day01 is None:
        error_checking.assert_is_leq(min_heating_rate_percentile, 10.)
        error_checking.assert_is_geq(max_heating_rate_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            max_heating_rate_k_day01, min_heating_rate_k_day01
        )

    if min_raw_flux_w_m02 is None or max_raw_flux_w_m02 is None:
        error_checking.assert_is_leq(min_raw_flux_percentile, 10.)
        error_checking.assert_is_geq(max_raw_flux_percentile, 90.)
    else:
        error_checking.assert_is_geq(min_raw_flux_w_m02, 0.)
        error_checking.assert_is_greater(
            max_raw_flux_w_m02, min_raw_flux_w_m02
        )

    if min_net_flux_w_m02 is None or max_net_flux_w_m02 is None:
        error_checking.assert_is_leq(min_net_flux_percentile, 10.)
        error_checking.assert_is_geq(max_net_flux_percentile, 90.)
    else:
        error_checking.assert_is_greater(
            max_net_flux_w_m02, min_net_flux_w_m02
        )

    # Do actual stuff.
    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict = prediction_io.get_ensemble_mean(prediction_dict)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY]
    wavelengths_metres = prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY]

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl,
        example_utils.TARGET_WAVELENGTHS_KEY: wavelengths_metres
    }

    normalization_file_name = (
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )
    if normalization_file_name is None:
        normalization_file_name = (
            generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
        )

    print('Reading normalization params from: "{0:s}"...'.format(
        normalization_file_name
    ))
    norm_param_table_xarray = normalization.read_params(normalization_file_name)

    if normalization.VECTOR_TARGET_DIM in norm_param_table_xarray.coords:
        mean_training_example_dict = normalization.create_mean_example(
            example_dict=example_dict,
            normalization_param_table_xarray=norm_param_table_xarray,
            use_absolute_values=False
        )
    else:
        mean_training_example_dict = normalization.create_mean_example_old(
            new_example_dict=example_dict,
            training_example_dict=example_io.read_file(normalization_file_name)
        )

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    aux_prediction_dict = get_aux_fields(
        prediction_dict=prediction_dict, example_dict=example_dict
    )
    aux_target_field_names = aux_prediction_dict[AUX_TARGET_NAMES_KEY]
    aux_predicted_field_names = aux_prediction_dict[AUX_PREDICTED_NAMES_KEY]
    aux_target_matrix = aux_prediction_dict[AUX_TARGET_VALS_KEY]
    aux_prediction_matrix = aux_prediction_dict[AUX_PREDICTED_VALS_KEY]

    prediction_dict = {
        SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
        SCALAR_PREDICTED_VALS_KEY: scalar_prediction_matrix,
        VECTOR_TARGET_VALS_KEY: vector_target_matrix,
        VECTOR_PREDICTED_VALS_KEY: vector_prediction_matrix,
        AUX_TARGET_VALS_KEY: aux_target_matrix,
        AUX_PREDICTED_VALS_KEY: aux_prediction_matrix
    }

    num_heights = vector_target_matrix.shape[1]
    num_wavelengths = vector_target_matrix.shape[2]
    num_vector_targets = vector_target_matrix.shape[3]
    num_scalar_targets = scalar_target_matrix.shape[2]
    num_aux_targets = len(aux_target_field_names)

    these_dimensions = (num_wavelengths, num_scalar_targets, num_bootstrap_reps)
    these_dim_keys = (WAVELENGTH_DIM, SCALAR_FIELD_DIM, BOOTSTRAP_REP_DIM)
    main_data_dict = {
        SCALAR_TARGET_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_MAE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_MAE_SKILL_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_MSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_MSE_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_MSE_VARIANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_MSE_SKILL_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_CORRELATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_KGE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_RELIABILITY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }

    these_dimensions = (
        num_wavelengths, num_scalar_targets,
        num_raw_flux_bins, num_bootstrap_reps
    )
    these_dim_keys = (
        WAVELENGTH_DIM, SCALAR_FIELD_DIM, RAW_FLUX_BIN_DIM, BOOTSTRAP_REP_DIM
    )
    new_dict = {
        SCALAR_RELIABILITY_X_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_RELIABILITY_Y_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (num_wavelengths, num_scalar_targets, num_raw_flux_bins)
    these_dim_keys = (WAVELENGTH_DIM, SCALAR_FIELD_DIM, RAW_FLUX_BIN_DIM)
    new_dict = {
        SCALAR_RELIA_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        SCALAR_INV_RELIA_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_INV_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (num_wavelengths, num_scalar_targets)
    these_dim_keys = (WAVELENGTH_DIM, SCALAR_FIELD_DIM)
    new_dict = {
        SCALAR_KS_STATISTIC_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        SCALAR_KS_P_VALUE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (
        num_heights, num_wavelengths, num_vector_targets, num_bootstrap_reps
    )
    these_dim_keys = (
        HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_FIELD_DIM, BOOTSTRAP_REP_DIM
    )
    new_dict = {
        VECTOR_TARGET_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_MAE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_MAE_SKILL_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_MSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_MSE_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_MSE_VARIANCE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_MSE_SKILL_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_BIAS_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_CORRELATION_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_KGE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_RELIABILITY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (num_wavelengths, num_vector_targets, num_bootstrap_reps)
    these_dim_keys = (WAVELENGTH_DIM, VECTOR_FIELD_DIM, BOOTSTRAP_REP_DIM)
    new_dict = {
        VECTOR_PRMSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (
        num_heights, num_wavelengths, num_vector_targets,
        num_heating_rate_bins, num_bootstrap_reps
    )
    these_dim_keys = (
        HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_FIELD_DIM,
        HEATING_RATE_BIN_DIM, BOOTSTRAP_REP_DIM
    )
    new_dict = {
        VECTOR_RELIABILITY_X_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_RELIABILITY_Y_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (
        num_heights, num_wavelengths, num_vector_targets, num_heating_rate_bins
    )
    these_dim_keys = (
        HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_FIELD_DIM, HEATING_RATE_BIN_DIM
    )
    new_dict = {
        VECTOR_RELIA_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        VECTOR_INV_RELIA_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_INV_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (
        num_wavelengths, num_vector_targets,
        num_heating_rate_bins, num_bootstrap_reps
    )
    these_dim_keys = (
        WAVELENGTH_DIM, VECTOR_FIELD_DIM,
        HEATING_RATE_BIN_DIM, BOOTSTRAP_REP_DIM
    )
    new_dict = {
        VECTOR_FLAT_RELIABILITY_X_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_FLAT_RELIABILITY_Y_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (num_wavelengths, num_vector_targets, num_bootstrap_reps)
    these_dim_keys = (WAVELENGTH_DIM, VECTOR_FIELD_DIM, BOOTSTRAP_REP_DIM)
    new_dict = {
        VECTOR_FLAT_MSE_SKILL_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_FLAT_RELIABILITY_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (
        num_wavelengths, num_vector_targets, num_heating_rate_bins
    )
    these_dim_keys = (WAVELENGTH_DIM, VECTOR_FIELD_DIM, HEATING_RATE_BIN_DIM)
    new_dict = {
        VECTOR_FLAT_RELIA_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_FLAT_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        ),
        VECTOR_FLAT_INV_RELIA_BIN_CENTER_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_FLAT_INV_RELIABILITY_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
        )
    }
    main_data_dict.update(new_dict)

    these_dimensions = (num_heights, num_wavelengths, num_vector_targets)
    these_dim_keys = (HEIGHT_DIM, WAVELENGTH_DIM, VECTOR_FIELD_DIM)
    new_dict = {
        VECTOR_KS_STATISTIC_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        VECTOR_KS_P_VALUE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    }
    main_data_dict.update(new_dict)

    if num_aux_targets > 0:
        these_dimensions = (
            num_wavelengths, num_aux_targets, num_bootstrap_reps
        )
        these_dim_keys = (
            WAVELENGTH_DIM, AUX_TARGET_FIELD_DIM, BOOTSTRAP_REP_DIM
        )
        new_dict = {
            AUX_TARGET_STDEV_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_PREDICTION_STDEV_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_MAE_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_MAE_SKILL_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_MSE_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_MSE_BIAS_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_MSE_VARIANCE_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_MSE_SKILL_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_BIAS_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_CORRELATION_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_KGE_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_RELIABILITY_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            )
        }
        main_data_dict.update(new_dict)

        these_dimensions = (
            num_wavelengths, num_aux_targets,
            num_net_flux_bins, num_bootstrap_reps
        )
        these_dim_keys = (
            WAVELENGTH_DIM, AUX_TARGET_FIELD_DIM,
            NET_FLUX_BIN_DIM, BOOTSTRAP_REP_DIM
        )
        new_dict = {
            AUX_RELIABILITY_X_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_RELIABILITY_Y_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            )
        }
        main_data_dict.update(new_dict)

        these_dimensions = (num_wavelengths, num_aux_targets, num_net_flux_bins)
        these_dim_keys = (
            WAVELENGTH_DIM, AUX_TARGET_FIELD_DIM, NET_FLUX_BIN_DIM
        )
        new_dict = {
            AUX_RELIA_BIN_CENTER_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_RELIABILITY_COUNT_KEY: (
                these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
            ),
            AUX_INV_RELIA_BIN_CENTER_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_INV_RELIABILITY_COUNT_KEY: (
                these_dim_keys, numpy.full(these_dimensions, 0, dtype=int)
            )
        }
        main_data_dict.update(new_dict)

        these_dimensions = (num_wavelengths, num_aux_targets)
        these_dim_keys = (WAVELENGTH_DIM, AUX_TARGET_FIELD_DIM)
        new_dict = {
            AUX_KS_STATISTIC_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            ),
            AUX_KS_P_VALUE_KEY: (
                these_dim_keys, numpy.full(these_dimensions, numpy.nan)
            )
        }
        main_data_dict.update(new_dict)

    raw_flux_bin_indices = numpy.linspace(
        0, num_raw_flux_bins - 1, num=num_raw_flux_bins, dtype=int
    )
    net_flux_bin_indices = numpy.linspace(
        0, num_net_flux_bins - 1, num=num_net_flux_bins, dtype=int
    )
    heating_rate_bin_indices = numpy.linspace(
        0, num_heating_rate_bins - 1, num=num_heating_rate_bins, dtype=int
    )
    bootstrap_indices = numpy.linspace(
        0, num_bootstrap_reps - 1, num=num_bootstrap_reps, dtype=int
    )

    metadata_dict = {
        SCALAR_FIELD_DIM:
            mean_training_example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        WAVELENGTH_DIM: wavelengths_metres,
        VECTOR_FIELD_DIM:
            mean_training_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        RAW_FLUX_BIN_DIM: raw_flux_bin_indices,
        NET_FLUX_BIN_DIM: net_flux_bin_indices,
        HEATING_RATE_BIN_DIM: heating_rate_bin_indices,
        BOOTSTRAP_REP_DIM: bootstrap_indices
    }

    if num_aux_targets > 0:
        metadata_dict[AUX_TARGET_FIELD_DIM] = aux_target_field_names
        metadata_dict[AUX_PREDICTED_FIELD_DIM] = aux_predicted_field_names

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    num_examples = scalar_target_matrix.shape[0]
    if num_examples == 0:
        num_examples = vector_target_matrix.shape[0]

    example_indices = numpy.linspace(
        0, num_examples - 1, num=num_examples, dtype=int
    )

    for i in range(num_bootstrap_reps):
        if num_bootstrap_reps == 1:
            these_indices = example_indices
        else:
            these_indices = numpy.random.choice(
                example_indices, size=num_examples, replace=True
            )

        print((
            'Computing scores for {0:d}th of {1:d} bootstrap replicates...'
        ).format(
            i + 1, num_bootstrap_reps
        ))

        result_table_xarray = _get_scores_one_replicate(
            result_table_xarray=result_table_xarray,
            prediction_dict=prediction_dict, replicate_index=i,
            example_indices_in_replicate=these_indices,
            mean_training_example_dict=mean_training_example_dict,
            min_heating_rate_k_day01=min_heating_rate_k_day01,
            max_heating_rate_k_day01=max_heating_rate_k_day01,
            min_heating_rate_percentile=min_heating_rate_percentile,
            max_heating_rate_percentile=max_heating_rate_percentile,
            min_raw_flux_w_m02=min_raw_flux_w_m02,
            max_raw_flux_w_m02=max_raw_flux_w_m02,
            min_raw_flux_percentile=min_raw_flux_percentile,
            max_raw_flux_percentile=max_raw_flux_percentile,
            min_net_flux_w_m02=min_net_flux_w_m02,
            max_net_flux_w_m02=max_net_flux_w_m02,
            min_net_flux_percentile=min_net_flux_percentile,
            max_net_flux_percentile=max_net_flux_percentile
        )

    return result_table_xarray


def find_file(
        directory_name, zenith_angle_bin=None, albedo_bin=None,
        shortwave_sfc_down_flux_bin=None, aerosol_optical_depth_bin=None,
        month=None, surface_temp_bin=None, longwave_sfc_down_flux_bin=None,
        longwave_toa_up_flux_bin=None, grid_row=None, grid_column=None,
        raise_error_if_missing=True):
    """Finds NetCDF file with evaluation results.

    :param directory_name: See doc for `prediction_io.find_file`.
    :param zenith_angle_bin: Same.
    :param albedo_bin: Same.
    :param shortwave_sfc_down_flux_bin: Same.
    :param aerosol_optical_depth_bin: Same.
    :param month: Same.
    :param surface_temp_bin: Same.
    :param longwave_sfc_down_flux_bin: Same.
    :param longwave_toa_up_flux_bin: Same.
    :param grid_row: Same.
    :param grid_column: Same.
    :param raise_error_if_missing: Same.
    :return: evaluation_file_name: File path.
    """

    prediction_file_name = prediction_io.find_file(
        directory_name=directory_name, zenith_angle_bin=zenith_angle_bin,
        albedo_bin=albedo_bin,
        shortwave_sfc_down_flux_bin=shortwave_sfc_down_flux_bin,
        aerosol_optical_depth_bin=aerosol_optical_depth_bin, month=month,
        surface_temp_bin=surface_temp_bin,
        longwave_sfc_down_flux_bin=longwave_sfc_down_flux_bin,
        longwave_toa_up_flux_bin=longwave_toa_up_flux_bin,
        grid_row=grid_row, grid_column=grid_column,
        raise_error_if_missing=raise_error_if_missing
    )

    pathless_file_name = os.path.split(prediction_file_name)[-1].replace(
        'predictions', 'evaluation'
    )

    return '{0:s}/{1:s}'.format(
        os.path.split(prediction_file_name)[0],
        pathless_file_name
    )


def write_file(result_table_xarray, netcdf_file_name):
    """Writes evaluation results to NetCDF file.

    :param result_table_xarray: xarray table produced by
        `get_scores_all_variables`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    # result_table_xarray.to_netcdf(
    #     path=netcdf_file_name, mode='w', format='NETCDF3_64BIT_OFFSET'
    # )

    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_file(netcdf_file_name):
    """Reads evaluation results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table produced by
        `get_scores_all_variables`.
    """

    return _add_wavelength_dim_to_table(xarray.open_dataset(netcdf_file_name))
