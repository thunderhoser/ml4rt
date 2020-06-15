"""Methods for model evaluation."""

import pickle
import numpy
from gewittergefahr.gg_utils import histograms
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io

DEFAULT_NUM_RELIABILITY_BINS = 20
DEFAULT_MAX_BIN_EDGE_PERCENTILE = 99.

SCALAR_TARGET_STDEV_KEY = 'scalar_target_stdevs'
SCALAR_PREDICTION_STDEV_KEY = 'scalar_prediction_stdevs'
VECTOR_TARGET_STDEV_KEY = 'vector_target_stdevs'
VECTOR_PREDICTION_STDEV_KEY = 'vector_prediction_stdevs'
SCALAR_MSE_KEY = 'scalar_mse_values'
SCALAR_MSE_SKILL_KEY = 'scalar_mse_skill_scores'
VECTOR_MSE_KEY = 'vector_mse_matrix'
VECTOR_MSE_SKILL_KEY = 'vector_mse_ss_matrix'
SCALAR_MAE_KEY = 'scalar_mae_values'
SCALAR_MAE_SKILL_KEY = 'scalar_mae_skill_scores'
VECTOR_MAE_KEY = 'vector_mae_matrix'
VECTOR_MAE_SKILL_KEY = 'vector_mae_ss_matrix'
SCALAR_BIAS_KEY = 'scalar_biases'
VECTOR_BIAS_KEY = 'vector_bias_matrix'
SCALAR_CORRELATION_KEY = 'scalar_correlations'
VECTOR_CORRELATION_KEY = 'vector_correlation_matrix'
VECTOR_PRMSE_KEY = 'vector_prmse_values'
SCALAR_RELIABILITY_X_KEY = 'scalar_reliability_x_matrix'
SCALAR_RELIABILITY_Y_KEY = 'scalar_reliability_y_matrix'
SCALAR_RELIABILITY_COUNT_KEY = 'scalar_reliability_count_matrix'
VECTOR_RELIABILITY_X_KEY = 'vector_reliability_x_matrix'
VECTOR_RELIABILITY_Y_KEY = 'vector_reliability_y_matrix'
VECTOR_RELIABILITY_COUNT_KEY = 'vector_reliability_count_matrix'

MODEL_FILE_KEY = 'model_file_name'

REQUIRED_KEYS = [
    SCALAR_TARGET_STDEV_KEY, SCALAR_PREDICTION_STDEV_KEY,
    VECTOR_TARGET_STDEV_KEY, VECTOR_PREDICTION_STDEV_KEY, MODEL_FILE_KEY
]


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
        mean_training_example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
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
    num_vector_targets = (
        mean_training_example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
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
                              max_bin_edge):
    """Computes reliability curve for one scalar target variable.

    B = number of bins

    :param target_values: See doc for `_get_mse_one_scalar`.
    :param predicted_values: Same.
    :param num_bins: Number of bins (points in curve).
    :param max_bin_edge: Value at upper edge of last bin.
    :return: mean_predictions: length-B numpy array of x-coordinates.
    :return: mean_observations: length-B numpy array of y-coordinates.
    :return: example_counts: length-B numpy array with num examples in each bin.
    """

    bin_index_by_example = histograms.create_histogram(
        input_values=predicted_values, num_bins=num_bins, min_value=0.,
        max_value=max_bin_edge
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


def get_scores_all_variables(
        scalar_target_matrix, scalar_prediction_matrix,
        vector_target_matrix, vector_prediction_matrix,
        mean_training_example_dict, get_mse=True, get_mae=True, get_bias=True,
        get_correlation=True, get_prmse=True, get_reliability_curve=True,
        num_reliability_bins=DEFAULT_NUM_RELIABILITY_BINS,
        max_bin_edge_percentile=DEFAULT_MAX_BIN_EDGE_PERCENTILE):
    """Computes desired scores for all target variables.

    E = number of examples
    H = number of heights
    T_s = number of scalar targets
    T_v = number of vector targets
    B = number of bins for reliability curve

    :param scalar_target_matrix: numpy array (E x T_s) of target (actual)
        values.
    :param scalar_prediction_matrix: numpy array (E x T_s) of predicted values.
    :param vector_target_matrix: numpy array (E x H x T_v) of target (actual)
        values.
    :param vector_prediction_matrix: numpy array (E x H x T_v) of predicted
        values.
    :param mean_training_example_dict: Dictionary created by
        `normalization.create_mean_example`, containing climatology over
        training data for each target variable.
    :param get_mse: Boolean flag.  If True, will compute MSE and MSE skill score
        for each scalar target variable.
    :param get_mae: Boolean flag.  If True, will compute MAE and MAE skill score
        for each scalar target variable.
    :param get_bias: Boolean flag.  If True, will compute bias for each scalar
        target variable.
    :param get_correlation: Boolean flag.  If True, will compute correlation for
        each scalar target variable.
    :param get_prmse: Boolean flag.  If True, will compute profile RMSE for
        each vector target variable.
    :param get_reliability_curve: Boolean flag.  If True, will compute points in
        reliability curve for each scalar target variable.
    :param num_reliability_bins: [used only if `get_reliability_curve == True`]
        Number of bins for each reliability curve.
    :param max_bin_edge_percentile:
        [used only if `get_reliability_curve == True`]
        Used to find upper edge of last bin for reliability curves.  For each
        scalar target variable y, the upper edge of the last bin will be the
        [q]th percentile of y-values, where q = `max_bin_edge_percentile`.

    :return: evaluation_dict: Dictionary with the following keys (some may be
        missing, depending on input args).
    evaluation_dict['scalar_target_stdevs']: numpy array (length T_s) of
        standard deviations for actual values.
    evaluation_dict['scalar_prediction_stdevs']: numpy array (length T_s) of
        standard deviations for predicted values.
    evaluation_dict['vector_target_stdev_matrix']: numpy array (H x T_v) of
        standard deviations for actual values.
    evaluation_dict['vector_prediction_stdev_matrix']: numpy array (H x T_v) of
        standard deviations for predicted values.
    evaluation_dict['scalar_mse_values']: numpy array (length T_s) of mean
        squared errors.
    evaluation_dict['scalar_mse_skill_scores']: numpy array (length T_s) of MSE
        skill scores.
    evaluation_dict['vector_mse_matrix']: numpy array (H x T_v) of mean squared
        errors.
    evaluation_dict['vector_mse_ss_matrix']: numpy array (H x T_v) of MSE skill
        scores.
    evaluation_dict['scalar_mae_values']: numpy array (length T_s) of mean
        absolute errors.
    evaluation_dict['scalar_mae_skill_scores']: numpy array (length T_s) of MAE
        skill scores.
    evaluation_dict['vector_mae_matrix']: numpy array (H x T_v) of mean absolute
        errors.
    evaluation_dict['vector_mae_ss_matrix']: numpy array (H x T_v) of MAE skill
        scores.
    evaluation_dict['scalar_biases']: numpy array (length T_s) of biases.
    evaluation_dict['vector_bias_matrix']: numpy array (H x T_v) of biases.
    evaluation_dict['scalar_correlations']: numpy array (length T_s) of
        correlations.
    evaluation_dict['vector_correlation_matrix']: numpy array (H x T_v) of
        correlations.
    evaluation_dict['prmse_values']: numpy array (length T_v) of profile-RMSE
        values.
    evaluation_dict['scalar_reliability_x_matrix']: numpy array (T_s x B) of
        x-coordinates for reliability curves.
    evaluation_dict['scalar_reliability_y_matrix']: Same but for y-coordinates.
    evaluation_dict['scalar_reliability_count_matrix']: Same but for example
        counts.
    evaluation_dict['vector_reliability_x_matrix']: numpy array (H x T_v x B) of
        x-coordinates for reliability curves.
    evaluation_dict['vector_reliability_y_matrix']: Same but for y-coordinates.
    evaluation_dict['vector_reliability_count_matrix']: Same but for example
        counts.
    """

    # TODO(thunderhoser): This method could use a unit test.

    _check_args(
        scalar_target_matrix=scalar_target_matrix,
        scalar_prediction_matrix=scalar_prediction_matrix,
        mean_training_example_dict=mean_training_example_dict,
        vector_target_matrix=vector_target_matrix,
        vector_prediction_matrix=vector_prediction_matrix
    )

    error_checking.assert_is_boolean(get_mse)
    error_checking.assert_is_boolean(get_mae)
    error_checking.assert_is_boolean(get_bias)
    error_checking.assert_is_boolean(get_correlation)
    error_checking.assert_is_boolean(get_prmse)
    error_checking.assert_is_boolean(get_reliability_curve)

    if get_reliability_curve:
        error_checking.assert_is_integer(num_reliability_bins)
        error_checking.assert_is_geq(num_reliability_bins, 10)
        error_checking.assert_is_leq(num_reliability_bins, 1000)
        error_checking.assert_is_geq(max_bin_edge_percentile, 90.)
        error_checking.assert_is_leq(max_bin_edge_percentile, 100.)

    print(
        'Computing standard deviations of target (actual) and predicted '
        'values...'
    )
    evaluation_dict = {
        SCALAR_TARGET_STDEV_KEY:
            numpy.std(scalar_target_matrix, axis=0, ddof=1),
        SCALAR_PREDICTION_STDEV_KEY:
            numpy.std(scalar_prediction_matrix, axis=0, ddof=1),
        VECTOR_TARGET_STDEV_KEY:
            numpy.std(vector_target_matrix, axis=0, ddof=1),
        VECTOR_PREDICTION_STDEV_KEY:
            numpy.std(vector_prediction_matrix, axis=0, ddof=1)
    }

    num_heights = vector_target_matrix.shape[1]
    num_vector_targets = vector_target_matrix.shape[2]
    num_scalar_targets = scalar_target_matrix.shape[1]

    if get_mse:
        print('Computing mean squared errors (MSE) and MSE skill scores...')

        scalar_mse_values = numpy.full(num_scalar_targets, numpy.nan)
        scalar_mse_skill_scores = numpy.full(num_scalar_targets, numpy.nan)

        for k in num_scalar_targets:
            scalar_mse_values[k] = _get_mse_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k]
            )

            this_climo_value = mean_training_example_dict[
                example_io.SCALAR_TARGET_VALS_KEY
            ][0, k]

            scalar_mse_skill_scores[k] = _get_mse_ss_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k],
                mean_training_target_value=this_climo_value
            )

        evaluation_dict[SCALAR_MSE_KEY] = scalar_mse_values
        evaluation_dict[SCALAR_MSE_SKILL_KEY] = scalar_mse_skill_scores

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
                    example_io.VECTOR_TARGET_VALS_KEY
                ][0, j, k]

                vector_mse_ss_matrix[j, k] = _get_mse_ss_one_scalar(
                    target_values=vector_target_matrix[:, j, k],
                    predicted_values=vector_prediction_matrix[:, j, k],
                    mean_training_target_value=this_climo_value
                )

        evaluation_dict[VECTOR_MSE_KEY] = vector_mse_matrix
        evaluation_dict[VECTOR_MSE_SKILL_KEY] = vector_mse_ss_matrix

    if get_mae:
        print('Computing mean absolute errors (MAE) and MAE skill scores...')

        scalar_mae_values = numpy.full(num_scalar_targets, numpy.nan)
        scalar_mae_skill_scores = numpy.full(num_scalar_targets, numpy.nan)

        for k in num_scalar_targets:
            scalar_mae_values[k] = _get_mae_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k]
            )

            this_climo_value = mean_training_example_dict[
                example_io.SCALAR_TARGET_VALS_KEY
            ][0, k]

            scalar_mae_skill_scores[k] = _get_mae_ss_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k],
                mean_training_target_value=this_climo_value
            )

        evaluation_dict[SCALAR_MAE_KEY] = scalar_mae_values
        evaluation_dict[SCALAR_MAE_SKILL_KEY] = scalar_mae_skill_scores

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
                    example_io.VECTOR_TARGET_VALS_KEY
                ][0, j, k]

                vector_mae_ss_matrix[j, k] = _get_mae_ss_one_scalar(
                    target_values=vector_target_matrix[:, j, k],
                    predicted_values=vector_prediction_matrix[:, j, k],
                    mean_training_target_value=this_climo_value
                )

        evaluation_dict[VECTOR_MAE_KEY] = vector_mae_matrix
        evaluation_dict[VECTOR_MAE_SKILL_KEY] = vector_mae_ss_matrix

    if get_bias:
        print('Computing biases...')

        scalar_biases = numpy.full(num_scalar_targets, numpy.nan)

        for k in num_scalar_targets:
            scalar_biases[k] = _get_bias_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k]
            )

        evaluation_dict[SCALAR_BIAS_KEY] = scalar_biases

        vector_bias_matrix = numpy.full(
            (num_heights, num_vector_targets), numpy.nan
        )

        for j in range(num_heights):
            for k in range(num_vector_targets):
                vector_bias_matrix[j, k] = _get_bias_one_scalar(
                    target_values=vector_target_matrix[:, j, k],
                    predicted_values=vector_prediction_matrix[:, j, k]
                )

        evaluation_dict[VECTOR_BIAS_KEY] = vector_bias_matrix

    if get_correlation:
        print('Computing correlations...')

        scalar_correlations = numpy.full(num_scalar_targets, numpy.nan)

        for k in num_scalar_targets:
            scalar_correlations[k] = _get_correlation_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k]
            )

        evaluation_dict[SCALAR_CORRELATION_KEY] = scalar_correlations

        vector_correlation_matrix = numpy.full(
            (num_heights, num_vector_targets), numpy.nan
        )

        for j in range(num_heights):
            for k in range(num_vector_targets):
                vector_correlation_matrix[j, k] = (
                    _get_correlation_one_scalar(
                        target_values=vector_target_matrix[:, j, k],
                        predicted_values=vector_prediction_matrix[:, j, k]
                    )
                )

        evaluation_dict[VECTOR_CORRELATION_KEY] = vector_correlation_matrix

    if get_prmse:
        print('Computing profile root mean squared errors (PRMSE)...')

        vector_prmse_values = numpy.full(num_vector_targets, numpy.nan)

        for k in range(num_vector_targets):
            vector_prmse_values[k] = _get_prmse_one_variable(
                target_matrix=vector_target_matrix[..., k],
                prediction_matrix=vector_prediction_matrix[..., k]
            )

        evaluation_dict[VECTOR_PRMSE_KEY] = vector_prmse_values

    if get_reliability_curve:
        print('Computing reliability curves...')

        these_dim = (num_scalar_targets, num_reliability_bins)
        scalar_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
        scalar_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
        scalar_reliability_count_matrix = numpy.full(these_dim, -1, dtype=int)

        for k in num_scalar_targets:
            these_x, these_y, these_counts = _get_rel_curve_one_scalar(
                target_values=scalar_target_matrix[:, k],
                predicted_values=scalar_prediction_matrix[:, k],
                num_bins=num_reliability_bins,
                max_bin_edge=numpy.percentile(
                    scalar_prediction_matrix[:, k], max_bin_edge_percentile
                )
            )

            scalar_reliability_x_matrix[k, :] = these_x
            scalar_reliability_y_matrix[k, :] = these_y
            scalar_reliability_count_matrix[k, :] = these_counts

        evaluation_dict[SCALAR_RELIABILITY_X_KEY] = scalar_reliability_x_matrix
        evaluation_dict[SCALAR_RELIABILITY_Y_KEY] = scalar_reliability_y_matrix
        evaluation_dict[SCALAR_RELIABILITY_COUNT_KEY] = (
            scalar_reliability_count_matrix
        )

        these_dim = (num_heights, num_vector_targets, num_reliability_bins)
        vector_reliability_x_matrix = numpy.full(these_dim, numpy.nan)
        vector_reliability_y_matrix = numpy.full(these_dim, numpy.nan)
        vector_reliability_count_matrix = numpy.full(
            these_dim, -1, dtype=int
        )

        for j in range(num_heights):
            for k in range(num_vector_targets):
                these_x, these_y, these_counts = _get_rel_curve_one_scalar(
                    target_values=vector_target_matrix[:, j, k],
                    predicted_values=vector_prediction_matrix[:, j, k],
                    num_bins=num_reliability_bins,
                    max_bin_edge=numpy.percentile(
                        vector_prediction_matrix[:, j, k],
                        max_bin_edge_percentile
                    )
                )

                vector_reliability_x_matrix[j, k, :] = these_x
                vector_reliability_y_matrix[j, k, :] = these_y
                vector_reliability_count_matrix[j, k, :] = these_counts

        evaluation_dict[VECTOR_RELIABILITY_X_KEY] = (
            vector_reliability_x_matrix
        )
        evaluation_dict[VECTOR_RELIABILITY_Y_KEY] = (
            vector_reliability_y_matrix
        )
        evaluation_dict[VECTOR_RELIABILITY_COUNT_KEY] = (
            vector_reliability_count_matrix
        )

    return evaluation_dict


def write_file(evaluation_dict, pickle_file_name):
    """Writes evaluation results to Pickle file.

    :param evaluation_dict: Dictionary created by `get_scores_all_variables`,
        but with one extra key.
    evaluation_dict['model_file_name']: Path to model used to generate
        predictions (readable by `neural_net.read_model`).

    :param pickle_file_name: Path to output file.
    """

    missing_keys = list(set(REQUIRED_KEYS) - set(evaluation_dict.keys()))

    if len(missing_keys) != 0:
        error_string = (
            '\n{0:s}\nKeys listed above were expected, but not found, in '
            'dictionary.'
        ).format(str(missing_keys))

        raise ValueError(error_string)

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(evaluation_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads evaluation results from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: evaluation_dict: See doc for `write_file`.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    evaluation_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(REQUIRED_KEYS) - set(evaluation_dict.keys()))
    if len(missing_keys) == 0:
        return evaluation_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)
