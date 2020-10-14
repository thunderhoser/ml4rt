"""Methods for model evaluation."""

import os
import sys
import numpy
import xarray

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

DEFAULT_NUM_RELIABILITY_BINS = 20
DEFAULT_MAX_BIN_EDGE_PERCENTILE = 99.

NET_FLUX_NAME = 'net_shortwave_flux_w_m02'
LOWEST_DOWN_FLUX_NAME = 'lowest_shortwave_down_flux_w_m02'
HIGHEST_UP_FLUX_NAME = 'highest_shortwave_up_flux_w_m02'

ASSUMED_TOA_HEIGHT_M_AGL = 50000
ASSUMED_SURFACE_HEIGHT_M_AGL = 10

SCALAR_FIELD_DIM = 'scalar_field'
HEIGHT_DIM = 'height_m_agl'
VECTOR_FIELD_DIM = 'vector_field'
AUX_TARGET_FIELD_DIM = 'aux_target_field'
AUX_PREDICTED_FIELD_DIM = 'aux_predicted_field'
RELIABILITY_BIN_DIM = 'reliability_bin'

SCALAR_TARGET_STDEV_KEY = 'scalar_target_stdev'
SCALAR_PREDICTION_STDEV_KEY = 'scalar_prediction_stdev'
VECTOR_TARGET_STDEV_KEY = 'vector_target_stdev'
VECTOR_PREDICTION_STDEV_KEY = 'vector_prediction_stdev'
AUX_TARGET_STDEV_KEY = 'aux_target_stdev'
AUX_PREDICTION_STDEV_KEY = 'aux_prediction_stdev'
SCALAR_MSE_KEY = 'scalar_mse'
SCALAR_MSE_SKILL_KEY = 'scalar_mse_skill_score'
VECTOR_MSE_KEY = 'vector_mse'
VECTOR_MSE_SKILL_KEY = 'vector_mse_skill_score'
AUX_MSE_KEY = 'aux_mse'
AUX_MSE_SKILL_KEY = 'aux_mse_skill_score'
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
SCALAR_RELIABILITY_X_KEY = 'scalar_reliability_x'
SCALAR_RELIABILITY_Y_KEY = 'scalar_reliability_y'
SCALAR_RELIABILITY_COUNT_KEY = 'scalar_reliability_count'
SCALAR_INV_RELIABILITY_X_KEY = 'scalar_inv_reliability_x'
SCALAR_INV_RELIABILITY_Y_KEY = 'scalar_inv_reliability_y'
SCALAR_INV_RELIABILITY_COUNT_KEY = 'scalar_inv_reliability_count'
VECTOR_RELIABILITY_X_KEY = 'vector_reliability_x'
VECTOR_RELIABILITY_Y_KEY = 'vector_reliability_y'
VECTOR_RELIABILITY_COUNT_KEY = 'vector_reliability_count'
VECTOR_INV_RELIABILITY_X_KEY = 'vector_inv_reliability_x'
VECTOR_INV_RELIABILITY_Y_KEY = 'vector_inv_reliability_y'
VECTOR_INV_RELIABILITY_COUNT_KEY = 'vector_inv_reliability_count'
AUX_RELIABILITY_X_KEY = 'aux_reliability_x'
AUX_RELIABILITY_Y_KEY = 'aux_reliability_y'
AUX_RELIABILITY_COUNT_KEY = 'aux_reliability_count'
AUX_INV_RELIABILITY_X_KEY = 'aux_inv_reliability_x'
AUX_INV_RELIABILITY_Y_KEY = 'aux_inv_reliability_y'
AUX_INV_RELIABILITY_COUNT_KEY = 'aux_inv_reliability_count'

MODEL_FILE_KEY = 'model_file_name'
PREDICTION_FILE_KEY = 'prediction_file_name'

AUX_TARGET_NAMES_KEY = 'aux_target_field_names'
AUX_PREDICTED_NAMES_KEY = 'aux_predicted_field_names'
AUX_TARGET_VALS_KEY = 'aux_target_matrix'
AUX_PREDICTED_VALS_KEY = 'aux_prediction_matrix'
SURFACE_DOWN_FLUX_INDEX_KEY = 'surface_down_flux_index'
TOA_UP_FLUX_INDEX_KEY = 'toa_up_flux_index'


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

    these_expected_dim = numpy.array(
        [num_examples, num_scalar_targets], dtype=int
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
        vector_target_matrix, num_dimensions=3
    )

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = len(
        mean_training_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )

    these_expected_dim = numpy.array(
        [num_examples, num_heights, num_vector_targets], dtype=int
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
    :return: mean_squared_error: Self-explanatory.
    """

    return numpy.mean((target_values - predicted_values) ** 2)


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
    )
    mse_climo = _get_mse_one_scalar(
        target_values=target_values, predicted_values=mean_training_target_value
    )

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


def _get_rel_curve_one_scalar(target_values, predicted_values, num_bins,
                              max_bin_edge, invert=False):
    """Computes reliability curve for one scalar target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param num_bins: Number of bins (points in curve).
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

    max_bin_edge = max([max_bin_edge, numpy.finfo(float).eps])

    bin_index_by_example = histograms.create_histogram(
        input_values=target_values if invert else predicted_values,
        num_bins=num_bins, min_value=0., max_value=max_bin_edge
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


def get_aux_fields(prediction_dict, example_dict):
    """Returns auxiliary fields.

    F = number of pairs of auxiliary fields
    E = number of examples

    :param prediction_dict: See doc for `prediction_io.read_file`.
    :param example_dict: Dictionary with the following keys (details for each
        key in documentation for `example_io.read_file`).
    example_dict['scalar_target_names']
    example_dict['vector_target_names']
    example_dict['heights_m_agl']

    :return: aux_prediction_dict: Dictionary with the following keys.
    aux_prediction_dict['aux_target_field_names']: length-F list with names of
        target fields.
    aux_prediction_dict['aux_predicted_field_names']: length-F list with names
        of predicted fields.
    aux_prediction_dict['aux_target_matrix']: E-by-F numpy array of target
        (actual) values.
    aux_prediction_dict['aux_prediction_matrix']: E-by-F numpy array of
        predicted values.
    aux_prediction_dict['surface_down_flux_index']: Array index of surface
        downwelling flux in `mean_training_example_dict`.  If surface
        downwelling flux is not available, this is -1.
    aux_prediction_dict['toa_up_flux_index']: Array index of TOA upwelling flux
        in `mean_training_example_dict`.  If TOA upwelling flux is not
        available, this is -1.
    """

    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
    )
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    scalar_target_names = example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    aux_target_field_names = []
    aux_predicted_field_names = []

    num_examples = scalar_target_matrix.shape[0]
    aux_target_matrix = numpy.full((num_examples, 0), numpy.nan)
    aux_prediction_matrix = numpy.full((num_examples, 0), numpy.nan)

    try:
        surface_down_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
        )
        toa_up_flux_index = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
        )

        aux_target_field_names.append(NET_FLUX_NAME)
        aux_predicted_field_names.append(NET_FLUX_NAME)

        this_target_matrix = (
            scalar_target_matrix[:, [surface_down_flux_index]] -
            scalar_target_matrix[:, [toa_up_flux_index]]
        )
        aux_target_matrix = numpy.concatenate(
            (aux_target_matrix, this_target_matrix), axis=1
        )

        this_prediction_matrix = (
            scalar_prediction_matrix[:, [surface_down_flux_index]] -
            scalar_prediction_matrix[:, [toa_up_flux_index]]
        )
        aux_prediction_matrix = numpy.concatenate(
            (aux_prediction_matrix, this_prediction_matrix), axis=1
        )
    except ValueError:
        surface_down_flux_index = -1
        toa_up_flux_index = -1

    vector_target_names = example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]

    if toa_up_flux_index >= 0:
        try:
            this_field_index = vector_target_names.index(
                example_utils.SHORTWAVE_UP_FLUX_NAME
            )
            this_height_index = example_utils.match_heights(
                heights_m_agl=heights_m_agl,
                desired_height_m_agl=ASSUMED_TOA_HEIGHT_M_AGL
            )

            aux_target_field_names.append(
                example_utils.SHORTWAVE_TOA_UP_FLUX_NAME
            )
            aux_predicted_field_names.append(HIGHEST_UP_FLUX_NAME)

            this_target_matrix = (
                scalar_prediction_matrix[:, [toa_up_flux_index]]
            )
            aux_target_matrix = numpy.concatenate(
                (aux_target_matrix, this_target_matrix), axis=1
            )

            this_prediction_matrix = (
                vector_prediction_matrix[
                    :, this_height_index, [this_field_index]]
            )
            aux_prediction_matrix = numpy.concatenate(
                (aux_prediction_matrix, this_prediction_matrix), axis=1
            )
        except ValueError:
            pass

    if surface_down_flux_index >= 0:
        try:
            this_field_index = vector_target_names.index(
                example_utils.SHORTWAVE_DOWN_FLUX_NAME
            )
            this_height_index = example_utils.match_heights(
                heights_m_agl=heights_m_agl,
                desired_height_m_agl=ASSUMED_SURFACE_HEIGHT_M_AGL
            )

            aux_target_field_names.append(
                example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
            )
            aux_predicted_field_names.append(LOWEST_DOWN_FLUX_NAME)

            this_target_matrix = (
                scalar_prediction_matrix[:, [surface_down_flux_index]]
            )
            aux_target_matrix = numpy.concatenate(
                (aux_target_matrix, this_target_matrix), axis=1
            )

            this_prediction_matrix = (
                vector_prediction_matrix[
                    :, this_height_index, [this_field_index]]
            )
            aux_prediction_matrix = numpy.concatenate(
                (aux_prediction_matrix, this_prediction_matrix), axis=1
            )
        except ValueError:
            pass

    return {
        AUX_TARGET_NAMES_KEY: aux_target_field_names,
        AUX_PREDICTED_NAMES_KEY: aux_predicted_field_names,
        AUX_TARGET_VALS_KEY: aux_target_matrix,
        AUX_PREDICTED_VALS_KEY: aux_prediction_matrix,
        SURFACE_DOWN_FLUX_INDEX_KEY: surface_down_flux_index,
        TOA_UP_FLUX_INDEX_KEY: toa_up_flux_index
    }


def get_scores_all_variables(
        prediction_file_name,
        num_reliability_bins=DEFAULT_NUM_RELIABILITY_BINS,
        max_bin_edge_percentile=DEFAULT_MAX_BIN_EDGE_PERCENTILE):
    """Computes desired scores for all target variables.

    :param prediction_file_name: Path to file with predictions that will be
        evaluated.  This file will be read by `prediction_io.read_file`.
    :param num_reliability_bins: [used only if `get_reliability_curve == True`]
        Number of bins for each reliability curve.
    :param max_bin_edge_percentile:
        [used only if `get_reliability_curve == True`]
        Used to find upper edge of last bin for reliability curves.  For each
        scalar target variable y, the upper edge of the last bin will be the
        [q]th percentile of y-values, where q = `max_bin_edge_percentile`.

    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    error_checking.assert_is_string(prediction_file_name)
    error_checking.assert_is_integer(num_reliability_bins)
    error_checking.assert_is_geq(num_reliability_bins, 10)
    error_checking.assert_is_leq(num_reliability_bins, 1000)
    error_checking.assert_is_geq(max_bin_edge_percentile, 90.)
    error_checking.assert_is_leq(max_bin_edge_percentile, 100.)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = prediction_dict[prediction_io.HEIGHTS_KEY]

    example_dict = {
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_utils.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_utils.HEIGHTS_KEY: heights_m_agl
    }

    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )
    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict, heights_m_agl=heights_m_agl
    )

    mean_training_example_dict = normalization.create_mean_example(
        new_example_dict=example_dict,
        training_example_dict=training_example_dict
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
    surface_down_flux_index = aux_prediction_dict[SURFACE_DOWN_FLUX_INDEX_KEY]
    toa_up_flux_index = aux_prediction_dict[TOA_UP_FLUX_INDEX_KEY]

    print(
        'Computing standard deviations of target (actual) and predicted '
        'values...'
    )

    # Standard deviations of scalar fields.
    scalar_target_stdevs = numpy.std(scalar_target_matrix, axis=0, ddof=1)
    scalar_prediction_stdevs = numpy.std(
        scalar_prediction_matrix, axis=0, ddof=1
    )

    these_dim = (SCALAR_FIELD_DIM,)
    main_data_dict = {
        SCALAR_TARGET_STDEV_KEY: (these_dim, scalar_target_stdevs),
        SCALAR_PREDICTION_STDEV_KEY: (these_dim, scalar_prediction_stdevs)
    }

    # Standard deviations of vector fields.
    vector_target_stdev_matrix = numpy.std(vector_target_matrix, axis=0, ddof=1)
    vector_prediction_stdev_matrix = numpy.std(
        vector_prediction_matrix, axis=0, ddof=1
    )

    these_dim = (HEIGHT_DIM, VECTOR_FIELD_DIM)
    new_dict = {
        VECTOR_TARGET_STDEV_KEY: (these_dim, vector_target_stdev_matrix),
        VECTOR_PREDICTION_STDEV_KEY: (these_dim, vector_prediction_stdev_matrix)
    }
    main_data_dict.update(new_dict)

    # Standard deviations of auxiliary fields.
    num_aux_targets = len(aux_target_field_names)

    if num_aux_targets > 0:
        aux_target_stdevs = numpy.std(aux_target_matrix, axis=0, ddof=1)
        aux_prediction_stdevs = numpy.std(aux_prediction_matrix, axis=0, ddof=1)

        these_dim = (AUX_TARGET_FIELD_DIM,)
        new_dict = {
            AUX_TARGET_STDEV_KEY: (these_dim, aux_target_stdevs),
            AUX_PREDICTION_STDEV_KEY: (these_dim, aux_prediction_stdevs)
        }
        main_data_dict.update(new_dict)

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[2]
    num_scalar_targets = scalar_target_matrix.shape[1]

    print('Computing mean squared errors (MSE) and MSE skill scores...')

    # Mean squared errors of scalar fields.
    scalar_mse_values = numpy.full(num_scalar_targets, numpy.nan)
    scalar_mse_skill_scores = numpy.full(num_scalar_targets, numpy.nan)

    for k in range(num_scalar_targets):
        scalar_mse_values[k] = _get_mse_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k]
        )

        this_climo_value = mean_training_example_dict[
            example_utils.SCALAR_TARGET_VALS_KEY
        ][0, k]

        scalar_mse_skill_scores[k] = _get_mse_ss_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k],
            mean_training_target_value=this_climo_value
        )

    these_dim = (SCALAR_FIELD_DIM,)
    new_dict = {
        SCALAR_MSE_KEY: (these_dim, scalar_mse_values),
        SCALAR_MSE_SKILL_KEY: (these_dim, scalar_mse_skill_scores)
    }
    main_data_dict.update(new_dict)

    # Mean squared errors of vector fields.
    vector_mse_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )
    vector_mse_ss_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )

    for j in range(num_heights):
        for k in range(num_vector_targets):
            vector_mse_matrix[j, k] = _get_mse_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k]
            )

            this_climo_value = mean_training_example_dict[
                example_utils.VECTOR_TARGET_VALS_KEY
            ][0, j, k]

            vector_mse_ss_matrix[j, k] = _get_mse_ss_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k],
                mean_training_target_value=this_climo_value
            )

    these_dim = (HEIGHT_DIM, VECTOR_FIELD_DIM)
    new_dict = {
        VECTOR_MSE_KEY: (these_dim, vector_mse_matrix),
        VECTOR_MSE_SKILL_KEY: (these_dim, vector_mse_ss_matrix)
    }
    main_data_dict.update(new_dict)

    # Mean squared errors of auxiliary fields.
    if num_aux_targets > 0:
        aux_mse_values = numpy.full(num_aux_targets, numpy.nan)
        aux_mse_skill_scores = numpy.full(num_aux_targets, numpy.nan)

        for k in range(num_aux_targets):
            aux_mse_values[k] = _get_mse_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k]
            )

            if aux_target_field_names[k] != NET_FLUX_NAME:
                continue

            this_key = example_utils.SCALAR_TARGET_VALS_KEY

            this_climo_value = (
                mean_training_example_dict[this_key][0, surface_down_flux_index]
                -
                mean_training_example_dict[this_key][0, toa_up_flux_index]
            )

            aux_mse_skill_scores[k] = _get_mse_ss_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k],
                mean_training_target_value=this_climo_value
            )

        these_dim = (AUX_TARGET_FIELD_DIM,)
        new_dict = {
            AUX_MSE_KEY: (these_dim, aux_mse_values),
            AUX_MSE_SKILL_KEY: (these_dim, aux_mse_skill_scores)
        }
        main_data_dict.update(new_dict)

    print('Computing mean absolute errors (MAE) and MAE skill scores...')

    # Mean absolute errors of scalar fields.
    scalar_mae_values = numpy.full(num_scalar_targets, numpy.nan)
    scalar_mae_skill_scores = numpy.full(num_scalar_targets, numpy.nan)

    for k in range(num_scalar_targets):
        scalar_mae_values[k] = _get_mae_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k]
        )

        this_climo_value = mean_training_example_dict[
            example_utils.SCALAR_TARGET_VALS_KEY
        ][0, k]

        scalar_mae_skill_scores[k] = _get_mae_ss_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k],
            mean_training_target_value=this_climo_value
        )

    these_dim = (SCALAR_FIELD_DIM,)
    new_dict = {
        SCALAR_MAE_KEY: (these_dim, scalar_mae_values),
        SCALAR_MAE_SKILL_KEY: (these_dim, scalar_mae_skill_scores)
    }
    main_data_dict.update(new_dict)

    # Mean absolute errors of vector fields.
    vector_mae_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )
    vector_mae_ss_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )

    for j in range(num_heights):
        for k in range(num_vector_targets):
            vector_mae_matrix[j, k] = _get_mae_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k]
            )

            this_climo_value = mean_training_example_dict[
                example_utils.VECTOR_TARGET_VALS_KEY
            ][0, j, k]

            vector_mae_ss_matrix[j, k] = _get_mae_ss_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k],
                mean_training_target_value=this_climo_value
            )

    these_dim = (HEIGHT_DIM, VECTOR_FIELD_DIM)
    new_dict = {
        VECTOR_MAE_KEY: (these_dim, vector_mae_matrix),
        VECTOR_MAE_SKILL_KEY: (these_dim, vector_mae_ss_matrix)
    }
    main_data_dict.update(new_dict)

    # Mean absolute errors of auxiliary fields.
    if num_aux_targets > 0:
        aux_mae_values = numpy.full(num_aux_targets, numpy.nan)
        aux_mae_skill_scores = numpy.full(num_aux_targets, numpy.nan)

        for k in range(num_aux_targets):
            aux_mae_values[k] = _get_mae_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k]
            )

            if aux_target_field_names[k] != NET_FLUX_NAME:
                continue

            this_key = example_utils.SCALAR_TARGET_VALS_KEY

            this_climo_value = (
                mean_training_example_dict[this_key][0, surface_down_flux_index]
                -
                mean_training_example_dict[this_key][0, toa_up_flux_index]
            )

            aux_mae_skill_scores[k] = _get_mae_ss_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k],
                mean_training_target_value=this_climo_value
            )

        these_dim = (AUX_TARGET_FIELD_DIM,)
        new_dict = {
            AUX_MAE_KEY: (these_dim, aux_mae_values),
            AUX_MAE_SKILL_KEY: (these_dim, aux_mae_skill_scores)
        }
        main_data_dict.update(new_dict)

    print('Computing biases...')

    # Biases of scalar fields.
    scalar_biases = numpy.full(num_scalar_targets, numpy.nan)

    for k in range(num_scalar_targets):
        scalar_biases[k] = _get_bias_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k]
        )

    these_dim = (SCALAR_FIELD_DIM,)
    new_dict = {
        SCALAR_BIAS_KEY: (these_dim, scalar_biases)
    }
    main_data_dict.update(new_dict)

    # Biases of vector fields.
    vector_bias_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )

    for j in range(num_heights):
        for k in range(num_vector_targets):
            vector_bias_matrix[j, k] = _get_bias_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k]
            )

    these_dim = (HEIGHT_DIM, VECTOR_FIELD_DIM)
    new_dict = {
        VECTOR_BIAS_KEY: (these_dim, vector_bias_matrix)
    }
    main_data_dict.update(new_dict)

    # Biases of auxiliary fields.
    if num_aux_targets > 0:
        aux_biases = numpy.full(num_aux_targets, numpy.nan)

        for k in range(num_aux_targets):
            aux_biases[k] = _get_bias_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k]
            )

        these_dim = (AUX_TARGET_FIELD_DIM,)
        new_dict = {
            AUX_BIAS_KEY: (these_dim, aux_biases)
        }
        main_data_dict.update(new_dict)

    print('Computing correlations and KGE values...')

    # Correlation and KGE for scalar fields.
    scalar_correlations = numpy.full(num_scalar_targets, numpy.nan)
    scalar_kge_values = numpy.full(num_scalar_targets, numpy.nan)

    for k in range(num_scalar_targets):
        scalar_correlations[k] = _get_correlation_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k]
        )
        scalar_kge_values[k] = _get_kge_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k]
        )

    these_dim = (SCALAR_FIELD_DIM,)
    new_dict = {
        SCALAR_CORRELATION_KEY: (these_dim, scalar_correlations),
        SCALAR_KGE_KEY: (these_dim, scalar_kge_values)
    }
    main_data_dict.update(new_dict)

    # Correlation and KGE for vector fields.
    vector_correlation_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )
    vector_kge_matrix = numpy.full((num_heights, num_vector_targets), numpy.nan)

    for j in range(num_heights):
        for k in range(num_vector_targets):
            vector_correlation_matrix[j, k] = _get_correlation_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k]
            )
            vector_kge_matrix[j, k] = _get_kge_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k]
            )

    these_dim = (HEIGHT_DIM, VECTOR_FIELD_DIM)
    new_dict = {
        VECTOR_CORRELATION_KEY: (these_dim, vector_correlation_matrix),
        VECTOR_KGE_KEY: (these_dim, vector_kge_matrix)
    }
    main_data_dict.update(new_dict)

    # Correlation and KGE for auxiliary fields.
    if num_aux_targets > 0:
        aux_correlations = numpy.full(num_aux_targets, numpy.nan)
        aux_kge_values = numpy.full(num_aux_targets, numpy.nan)

        for k in range(num_aux_targets):
            aux_correlations[k] = _get_correlation_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k]
            )
            aux_kge_values[k] = _get_kge_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k]
            )

        these_dim = (AUX_TARGET_FIELD_DIM,)
        new_dict = {
            AUX_CORRELATION_KEY: (these_dim, aux_correlations),
            AUX_KGE_KEY: (these_dim, aux_kge_values)
        }
        main_data_dict.update(new_dict)

    print('Computing profile root mean squared errors (PRMSE)...')

    vector_prmse_values = numpy.full(num_vector_targets, numpy.nan)

    for k in range(num_vector_targets):
        vector_prmse_values[k] = _get_prmse_one_variable(
            target_matrix=vector_target_matrix[..., k],
            prediction_matrix=vector_prediction_matrix[..., k]
        )

    these_dim = (VECTOR_FIELD_DIM,)
    new_dict = {
        VECTOR_PRMSE_KEY: (these_dim, vector_prmse_values)
    }
    main_data_dict.update(new_dict)

    print('Computing reliability curves...')

    # Reliability curves for scalar fields.
    these_dim = (num_scalar_targets, num_reliability_bins)
    scalar_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
    scalar_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
    scalar_reliability_count_matrix = numpy.full(these_dim, -1, dtype=int)
    scalar_inv_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
    scalar_inv_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
    scalar_inv_reliability_count_matrix = numpy.full(these_dim, -1, dtype=int)

    num_examples = scalar_target_matrix.shape[0]

    for k in range(num_scalar_targets):
        if num_examples == 0:
            max_bin_edge = 1.
        else:
            max_bin_edge = numpy.percentile(
                scalar_prediction_matrix[:, k], max_bin_edge_percentile
            )

        (
            scalar_reliability_x_matrix[k, :],
            scalar_reliability_y_matrix[k, :],
            scalar_reliability_count_matrix[k, :]
        ) = _get_rel_curve_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k],
            num_bins=num_reliability_bins, max_bin_edge=max_bin_edge,
            invert=False
        )

        if num_examples == 0:
            max_bin_edge = 1.
        else:
            max_bin_edge = numpy.percentile(
                scalar_target_matrix[:, k], max_bin_edge_percentile
            )

        (
            scalar_inv_reliability_y_matrix[k, :],
            scalar_inv_reliability_x_matrix[k, :],
            scalar_inv_reliability_count_matrix[k, :]
        ) = _get_rel_curve_one_scalar(
            target_values=scalar_target_matrix[:, k],
            predicted_values=scalar_prediction_matrix[:, k],
            num_bins=num_reliability_bins, max_bin_edge=max_bin_edge,
            invert=True
        )

    these_dim = (SCALAR_FIELD_DIM, RELIABILITY_BIN_DIM)
    new_dict = {
        SCALAR_RELIABILITY_X_KEY: (these_dim, scalar_reliability_x_matrix),
        SCALAR_RELIABILITY_Y_KEY: (these_dim, scalar_reliability_y_matrix),
        SCALAR_RELIABILITY_COUNT_KEY:
            (these_dim, scalar_reliability_count_matrix),
        SCALAR_INV_RELIABILITY_X_KEY:
            (these_dim, scalar_inv_reliability_x_matrix),
        SCALAR_INV_RELIABILITY_Y_KEY:
            (these_dim, scalar_inv_reliability_y_matrix),
        SCALAR_INV_RELIABILITY_COUNT_KEY:
            (these_dim, scalar_inv_reliability_count_matrix)
    }
    main_data_dict.update(new_dict)

    # Reliability curves for vector fields.
    these_dim = (num_heights, num_vector_targets, num_reliability_bins)
    vector_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
    vector_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
    vector_reliability_count_matrix = numpy.full(
        these_dim, -1, dtype=int
    )
    vector_inv_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
    vector_inv_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
    vector_inv_reliability_count_matrix = numpy.full(
        these_dim, -1, dtype=int
    )

    for j in range(num_heights):
        for k in range(num_vector_targets):
            if num_examples == 0:
                max_bin_edge = 1.
            else:
                max_bin_edge = numpy.percentile(
                    vector_prediction_matrix[:, j, k], max_bin_edge_percentile
                )

            (
                vector_reliability_x_matrix[j, k, :],
                vector_reliability_y_matrix[j, k, :],
                vector_reliability_count_matrix[j, k, :]
            ) = _get_rel_curve_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k],
                num_bins=num_reliability_bins, max_bin_edge=max_bin_edge,
                invert=False
            )

            if num_examples == 0:
                max_bin_edge = 1.
            else:
                max_bin_edge = numpy.percentile(
                    vector_target_matrix[:, j, k], max_bin_edge_percentile
                )

            (
                vector_inv_reliability_y_matrix[j, k, :],
                vector_inv_reliability_x_matrix[j, k, :],
                vector_inv_reliability_count_matrix[j, k, :]
            ) = _get_rel_curve_one_scalar(
                target_values=vector_target_matrix[:, j, k],
                predicted_values=vector_prediction_matrix[:, j, k],
                num_bins=num_reliability_bins, max_bin_edge=max_bin_edge,
                invert=True
            )

    these_dim = (HEIGHT_DIM, VECTOR_FIELD_DIM, RELIABILITY_BIN_DIM)
    new_dict = {
        VECTOR_RELIABILITY_X_KEY: (these_dim, vector_reliability_x_matrix),
        VECTOR_RELIABILITY_Y_KEY: (these_dim, vector_reliability_y_matrix),
        VECTOR_RELIABILITY_COUNT_KEY:
            (these_dim, vector_reliability_count_matrix),
        VECTOR_INV_RELIABILITY_X_KEY:
            (these_dim, vector_inv_reliability_x_matrix),
        VECTOR_INV_RELIABILITY_Y_KEY:
            (these_dim, vector_inv_reliability_y_matrix),
        VECTOR_INV_RELIABILITY_COUNT_KEY:
            (these_dim, vector_inv_reliability_count_matrix)
    }
    main_data_dict.update(new_dict)

    # Reliability curves for auxiliary fields.
    if num_aux_targets > 0:
        these_dim = (num_aux_targets, num_reliability_bins)
        aux_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
        aux_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
        aux_reliability_count_matrix = numpy.full(these_dim, -1, dtype=int)
        aux_inv_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
        aux_inv_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
        aux_inv_reliability_count_matrix = numpy.full(these_dim, -1, dtype=int)

        for k in range(num_aux_targets):
            if num_examples == 0:
                max_bin_edge = 1.
            else:
                max_bin_edge = numpy.percentile(
                    aux_prediction_matrix[:, k], max_bin_edge_percentile
                )

            (
                aux_reliability_x_matrix[k, :],
                aux_reliability_y_matrix[k, :],
                aux_reliability_count_matrix[k, :]
            ) = _get_rel_curve_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k],
                num_bins=num_reliability_bins, max_bin_edge=max_bin_edge,
                invert=False
            )

            if num_examples == 0:
                max_bin_edge = 1.
            else:
                max_bin_edge = numpy.percentile(
                    aux_target_matrix[:, k], max_bin_edge_percentile
                )

            (
                aux_inv_reliability_y_matrix[k, :],
                aux_inv_reliability_x_matrix[k, :],
                aux_inv_reliability_count_matrix[k, :]
            ) = _get_rel_curve_one_scalar(
                target_values=aux_target_matrix[:, k],
                predicted_values=aux_prediction_matrix[:, k],
                num_bins=num_reliability_bins, max_bin_edge=max_bin_edge,
                invert=True
            )

        these_dim = (AUX_TARGET_FIELD_DIM, RELIABILITY_BIN_DIM)
        new_dict = {
            AUX_RELIABILITY_X_KEY: (these_dim, aux_reliability_x_matrix),
            AUX_RELIABILITY_Y_KEY: (these_dim, aux_reliability_y_matrix),
            AUX_RELIABILITY_COUNT_KEY:
                (these_dim, aux_reliability_count_matrix),
            AUX_INV_RELIABILITY_X_KEY:
                (these_dim, aux_inv_reliability_x_matrix),
            AUX_INV_RELIABILITY_Y_KEY:
                (these_dim, aux_inv_reliability_y_matrix),
            AUX_INV_RELIABILITY_COUNT_KEY:
                (these_dim, aux_inv_reliability_count_matrix)
        }
        main_data_dict.update(new_dict)

    # Add metadata.
    bin_indices = numpy.linspace(
        0, num_reliability_bins - 1, num=num_reliability_bins, dtype=int
    )

    metadata_dict = {
        SCALAR_FIELD_DIM:
            mean_training_example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        HEIGHT_DIM: heights_m_agl,
        VECTOR_FIELD_DIM:
            mean_training_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        RELIABILITY_BIN_DIM: bin_indices
    }

    if num_aux_targets > 0:
        metadata_dict[AUX_TARGET_FIELD_DIM] = aux_target_field_names
        metadata_dict[AUX_PREDICTED_FIELD_DIM] = aux_predicted_field_names

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[MODEL_FILE_KEY] = model_file_name
    result_table_xarray.attrs[PREDICTION_FILE_KEY] = prediction_file_name

    return result_table_xarray


def find_file(
        directory_name, zenith_angle_bin=None, albedo_bin=None, month=None,
        grid_row=None, grid_column=None, raise_error_if_missing=True):
    """Finds NetCDF file with evaluation results.

    :param directory_name: See doc for `prediction_io.find_file`.
    :param zenith_angle_bin: Same.
    :param albedo_bin: Same.
    :param month: Same.
    :param grid_row: Same.
    :param grid_column: Same.
    :param raise_error_if_missing: Same.
    :return: evaluation_file_name: File path.
    """

    prediction_file_name = prediction_io.find_file(
        directory_name=directory_name, zenith_angle_bin=zenith_angle_bin,
        albedo_bin=albedo_bin, month=month,
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

    return xarray.open_dataset(netcdf_file_name)