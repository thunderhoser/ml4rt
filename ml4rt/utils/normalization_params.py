"""Methods for computing, reading, and writing normalization parameters."""

import pickle
import numpy
import pandas
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

MEAN_VALUE_COLUMN = 'mean_value'
STANDARD_DEVIATION_COLUMN = 'standard_deviation'
MIN_VALUE_COLUMN = 'min_value'
MAX_VALUE_COLUMN = 'max_value'

TABLE_COLUMNS = [
    MEAN_VALUE_COLUMN, STANDARD_DEVIATION_COLUMN,
    MIN_VALUE_COLUMN, MAX_VALUE_COLUMN
]


def _get_standard_deviation(z_score_param_dict):
    """Computes standard deviation.

    :param z_score_param_dict: See doc for `update_z_score_params`.
    :return: standard_deviation: Standard deviation.
    """

    multiplier = float(
        z_score_param_dict[NUM_VALUES_KEY]
    ) / (z_score_param_dict[NUM_VALUES_KEY] - 1)

    standard_deviation = numpy.sqrt(multiplier * (
        z_score_param_dict[MEAN_OF_SQUARES_KEY] -
        z_score_param_dict[MEAN_VALUE_KEY] ** 2
    ))

    if numpy.isnan(standard_deviation):
        standard_deviation = 0.

    return numpy.maximum(standard_deviation, numpy.finfo(float).eps)


def _get_percentile(frequency_dict, percentile_level):
    """Computes percentile.

    :param frequency_dict: See doc for `update_frequency_dict`.
    :param percentile_level: Percentile level.  Will take the [q]th percentile,
        where q = `percentile_level`.
    :return: percentile: [q]th percentile.
    """

    unique_values, counts = list(zip(
        *iter(frequency_dict.items())
    ))
    unique_values = numpy.array(unique_values)
    counts = numpy.array(counts, dtype=int)

    sort_indices = numpy.argsort(unique_values)
    unique_values = unique_values[sort_indices]
    counts = counts[sort_indices]

    cumulative_frequencies = (
        numpy.cumsum(counts).astype(float) / numpy.sum(counts)
    )
    percentile_levels = 100 * (
        (cumulative_frequencies * numpy.sum(counts) - 1) /
        (numpy.sum(counts) - 1)
    )

    if percentile_level > percentile_levels[-1]:
        return unique_values[-1]

    if percentile_level < percentile_levels[0]:
        return unique_values[0]

    interp_object = interp1d(
        x=percentile_levels, y=unique_values, kind='linear',
        bounds_error=True, assume_sorted=True
    )

    return interp_object(percentile_level)


def update_z_score_params(z_score_param_dict, new_data_matrix):
    """Uses new data to update parameters for z-score normalization.

    :param z_score_param_dict: Dictionary with the following keys.
    z_score_param_dict['num_values']: Number of values on which current
        estimates are based.
    z_score_param_dict['mean_value']: Current estimate of mean.
    z_score_param_dict['mean_of_squares']: Current estimate of mean of squares.

    :param new_data_matrix: numpy array with new data.
    :return: z_score_param_dict: Same as input but with updated values.
    """

    error_checking.assert_is_numpy_array_without_nan(new_data_matrix)

    these_means = numpy.array([
        z_score_param_dict[MEAN_VALUE_KEY], numpy.mean(new_data_matrix)
    ])
    these_weights = numpy.array([
        z_score_param_dict[NUM_VALUES_KEY], new_data_matrix.size
    ])
    z_score_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    these_means = numpy.array([
        z_score_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        z_score_param_dict[NUM_VALUES_KEY], new_data_matrix.size
    ])
    z_score_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    z_score_param_dict[NUM_VALUES_KEY] += new_data_matrix.size

    return z_score_param_dict


def update_frequency_dict(frequency_dict, new_data_matrix, rounding_base):
    """Uses new data to update frequencies for min-max normalization.

    :param frequency_dict: Dictionary, where each key is a unique measurement
        and the corresponding value is num times the measurement occurs.
    :param new_data_matrix: numpy array with new data.
    :param rounding_base: Rounding base used to discretize continuous values
        into unique values.
    :return: frequency_dict: Same as input but with updated values.
    """

    error_checking.assert_is_numpy_array_without_nan(new_data_matrix)

    new_unique_values, new_counts = numpy.unique(
        number_rounding.round_to_nearest(new_data_matrix, rounding_base),
        return_counts=True
    )

    for i in range(len(new_unique_values)):
        if new_unique_values[i] in frequency_dict:
            frequency_dict[new_unique_values[i]] += new_counts[i]
        else:
            frequency_dict[new_unique_values[i]] = new_counts[i]

    return frequency_dict


def finalize_params(z_score_dict_dict, frequency_dict_dict,
                    min_percentile_level, max_percentile_level):
    """Finalizes normalization parameters.

    :param z_score_dict_dict: Dictionary of dictionaries, where each inner
        dictionary follows the input format for `update_z_score_params`.
    :param frequency_dict_dict: Dictionary of dictionaries, where each inner
        dictionary follows the input format for `update_frequency_dict`.
    :param min_percentile_level: Percentile level used to create minimum values
        for min-max normalization.
    :param max_percentile_level: Percentile level used to create maximum values
        for min-max normalization.
    :return: normalization_table: pandas DataFrame, where the indices are outer
        keys in `z_score_dict_dict`.  For example, if `z_score_dict_dict`
        contains 80 inner dictionaries, this table will have 80 rows.  Columns
        are as follows.

    normalization_table.mean_value: Mean value.
    normalization_table.standard_deviation: Standard deviation.
    normalization_table.min_value: Minimum value.
    normalization_table.max_value: Max value.
    """

    error_checking.assert_is_geq(min_percentile_level, 0.)
    error_checking.assert_is_leq(min_percentile_level, 10.)
    error_checking.assert_is_geq(max_percentile_level, 90.)
    error_checking.assert_is_leq(max_percentile_level, 100.)

    normalization_dict = {}

    for this_key in z_score_dict_dict:
        this_standard_deviation = _get_standard_deviation(
            z_score_dict_dict[this_key]
        )
        this_min_value = _get_percentile(
            frequency_dict=frequency_dict_dict[this_key],
            percentile_level=min_percentile_level
        )
        this_max_value = _get_percentile(
            frequency_dict=frequency_dict_dict[this_key],
            percentile_level=max_percentile_level
        )

        if this_max_value - this_min_value < numpy.finfo(float).eps:
            this_mean_value = numpy.mean([this_min_value, this_max_value])
            this_min_value = this_mean_value - numpy.finfo(float).eps
            this_max_value = this_mean_value + numpy.finfo(float).eps

        normalization_dict[this_key] = numpy.array([
            z_score_dict_dict[this_key][MEAN_VALUE_KEY],
            this_standard_deviation, this_min_value, this_max_value
        ])

    normalization_table = pandas.DataFrame.from_dict(
        normalization_dict, orient='index'
    )

    column_dict_old_to_new = {
        0: MEAN_VALUE_COLUMN,
        1: STANDARD_DEVIATION_COLUMN,
        2: MIN_VALUE_COLUMN,
        3: MAX_VALUE_COLUMN
    }

    return normalization_table.rename(
        columns=column_dict_old_to_new, inplace=False
    )


def write_file(pickle_file_name, norm_table_no_height, norm_table_with_height):
    """Writes normalization parameters to Pickle file.

    :param pickle_file_name: Path to output file.
    :param norm_table_no_height: pandas DataFrame created by `finalize_params`,
        containing one set of params for each variable.  This table should be
        single-indexed (field name only).
    :param norm_table_with_height: pandas DataFrame created by
        `finalize_params`, containing one set of params for each
        variable/height.  This table should be double-indexed (field name, then
        height in metres above ground level).
    """

    error_checking.assert_columns_in_dataframe(
        norm_table_no_height, TABLE_COLUMNS
    )
    error_checking.assert_columns_in_dataframe(
        norm_table_with_height, TABLE_COLUMNS
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(norm_table_no_height, pickle_file_handle)
    pickle.dump(norm_table_with_height, pickle_file_handle)
    pickle_file_handle.close()


def read_file(pickle_file_name):
    """Reads normalization parameters from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: norm_table_no_height: See doc for `write_file`.
    :return: norm_table_with_height: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    norm_table_no_height = pickle.load(pickle_file_handle)
    norm_table_with_height = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        norm_table_no_height, TABLE_COLUMNS
    )
    error_checking.assert_columns_in_dataframe(
        norm_table_with_height, TABLE_COLUMNS
    )

    return norm_table_no_height, norm_table_with_height
