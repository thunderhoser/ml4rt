"""Unit tests for normalization_params.py."""

import copy
import unittest
import numpy
import pandas
from ml4rt.dead_code import normalization_params

TOLERANCE = 1e-6

# The following constants are used to test update_z_score_params.
ORIGINAL_Z_SCORE_DICT = {
    normalization_params.NUM_VALUES_KEY: 20,
    normalization_params.MEAN_VALUE_KEY: 5.,
    normalization_params.MEAN_OF_SQUARES_KEY: 10.
}

NEW_MATRIX_FOR_Z_SCORES = numpy.array([
    [0, 1, 2, 3, 4],
    [1, 2, 4, 2, 1]
], dtype=float)

NEW_Z_SCORE_DICT = {
    normalization_params.NUM_VALUES_KEY: 30,
    normalization_params.MEAN_VALUE_KEY: 4.,
    normalization_params.MEAN_OF_SQUARES_KEY: 8.533333
}

# The following constants are used to test update_frequency_dict.
ROUNDING_BASE = 0.001
MAIN_FREQUENCY_DICT = {
    0.001: 5,
    0.002: 3,
    0.004: 7,
    0.006: 2
}

NEW_MATRIX_FOR_FREQUENCIES = numpy.array([
    [0.003, 0.007, 0.002, 0.004],
    [0.006, 0.005, 0.002, -0.001],
    [0.006, 0.001, 0.008, 0.007]
])

NEW_FREQUENCY_DICT = {
    -0.001: 1,
    0.001: 6,
    0.002: 5,
    0.003: 1,
    0.004: 8,
    0.005: 1,
    0.006: 4,
    0.007: 2,
    0.008: 1
}

# The following constants are used to test _get_standard_deviation.
STDEV_INPUT_MATRIX = numpy.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
], dtype=float)

STANDARD_DEVIATION = numpy.std(STDEV_INPUT_MATRIX, ddof=1)

# The following constants are used to test _get_percentile.
SMALL_PERCENTILE_LEVEL = 3.125
MEDIUM_PERCENTILE_LEVEL = 43.75
LARGE_PERCENTILE_LEVEL = 93.75

SMALL_PERCENTILE = 0.001
MEDIUM_PERCENTILE = 0.002
LARGE_PERCENTILE = 0.005

# The following constants are used to test finalize_params without separate
# heights (output table should be single-indexed).
FIRST_KEY_NO_HEIGHT = 'reflectivity_dbz'
SECOND_KEY_NO_HEIGHT = 'reflectivity_column_max_dbz'
THIRD_KEY_NO_HEIGHT = 'low_level_shear_s01'

FIRST_Z_SCORE_DICT = {
    normalization_params.NUM_VALUES_KEY: 100,
    normalization_params.MEAN_VALUE_KEY: 15.,
    normalization_params.MEAN_OF_SQUARES_KEY: 300.
}
SECOND_Z_SCORE_DICT = {
    normalization_params.NUM_VALUES_KEY: 100,
    normalization_params.MEAN_VALUE_KEY: 25.,
    normalization_params.MEAN_OF_SQUARES_KEY: 1000.
}
THIRD_Z_SCORE_DICT = {
    normalization_params.NUM_VALUES_KEY: 400,
    normalization_params.MEAN_VALUE_KEY: 8e-3,
    normalization_params.MEAN_OF_SQUARES_KEY: 1e-4
}
Z_SCORE_DICT_DICT_NO_HEIGHT = {
    FIRST_KEY_NO_HEIGHT: FIRST_Z_SCORE_DICT,
    SECOND_KEY_NO_HEIGHT: SECOND_Z_SCORE_DICT,
    THIRD_KEY_NO_HEIGHT: THIRD_Z_SCORE_DICT
}

FIRST_FREQUENCY_DICT = {
    0: 5,
    5: 10,
    10: 20,
    15: 30,
    20: 20,
    25: 10,
    30: 5
}
SECOND_FREQUENCY_DICT = {
    0: 0,
    5: 5,
    10: 5,
    15: 10,
    20: 15,
    25: 30,
    30: 15,
    35: 10,
    40: 5,
    45: 5
}
THIRD_FREQUENCY_DICT = {
    -4e-3: 10,
    -1e-3: 15,
    2e-3: 25,
    5e-3: 50,
    8e-3: 200,
    11e-3: 50,
    14e-3: 25,
    17e-3: 15,
    20e-3: 10
}
FREQUENCY_DICT_DICT_NO_HEIGHT = {
    FIRST_KEY_NO_HEIGHT: FIRST_FREQUENCY_DICT,
    SECOND_KEY_NO_HEIGHT: SECOND_FREQUENCY_DICT,
    THIRD_KEY_NO_HEIGHT: THIRD_FREQUENCY_DICT
}

MIN_PERCENTILE_LEVEL = 1.
MAX_PERCENTILE_LEVEL = 99.

FIRST_PARAM_VECTOR = numpy.array([
    FIRST_Z_SCORE_DICT[normalization_params.MEAN_VALUE_KEY],
    normalization_params._get_standard_deviation(FIRST_Z_SCORE_DICT),
    normalization_params._get_percentile(
        FIRST_FREQUENCY_DICT, MIN_PERCENTILE_LEVEL
    ),
    normalization_params._get_percentile(
        FIRST_FREQUENCY_DICT, MAX_PERCENTILE_LEVEL
    )
])

SECOND_PARAM_VECTOR = numpy.array([
    SECOND_Z_SCORE_DICT[normalization_params.MEAN_VALUE_KEY],
    normalization_params._get_standard_deviation(SECOND_Z_SCORE_DICT),
    normalization_params._get_percentile(
        SECOND_FREQUENCY_DICT, MIN_PERCENTILE_LEVEL
    ),
    normalization_params._get_percentile(
        SECOND_FREQUENCY_DICT, MAX_PERCENTILE_LEVEL
    )
])

THIRD_PARAM_VECTOR = numpy.array([
    THIRD_Z_SCORE_DICT[normalization_params.MEAN_VALUE_KEY],
    normalization_params._get_standard_deviation(THIRD_Z_SCORE_DICT),
    normalization_params._get_percentile(
        THIRD_FREQUENCY_DICT, MIN_PERCENTILE_LEVEL
    ),
    normalization_params._get_percentile(
        THIRD_FREQUENCY_DICT, MAX_PERCENTILE_LEVEL
    )
])

THIS_DICT = {
    FIRST_KEY_NO_HEIGHT: FIRST_PARAM_VECTOR,
    SECOND_KEY_NO_HEIGHT: SECOND_PARAM_VECTOR,
    THIRD_KEY_NO_HEIGHT: THIRD_PARAM_VECTOR
}
NORM_TABLE_NO_HEIGHT = pandas.DataFrame.from_dict(THIS_DICT, orient='index')

COLUMN_DICT_OLD_TO_NEW = {
    0: normalization_params.MEAN_VALUE_COLUMN,
    1: normalization_params.STANDARD_DEVIATION_COLUMN,
    2: normalization_params.MIN_VALUE_COLUMN,
    3: normalization_params.MAX_VALUE_COLUMN
}
NORM_TABLE_NO_HEIGHT.rename(columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

# The following constants are used to test finalize_params with separate heights
# (output table should be double-indexed).
FIRST_KEY_WITH_HEIGHT = ('reflectivity_dbz', 1000)
SECOND_KEY_WITH_HEIGHT = ('reflectivity_column_max_dbz', 250)
THIRD_KEY_WITH_HEIGHT = ('low_level_shear_s01', 250)

Z_SCORE_DICT_DICT_WITH_HEIGHT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_Z_SCORE_DICT,
    SECOND_KEY_WITH_HEIGHT: SECOND_Z_SCORE_DICT,
    THIRD_KEY_WITH_HEIGHT: THIRD_Z_SCORE_DICT
}

FREQUENCY_DICT_DICT_WITH_HEIGHT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_FREQUENCY_DICT,
    SECOND_KEY_WITH_HEIGHT: SECOND_FREQUENCY_DICT,
    THIRD_KEY_WITH_HEIGHT: THIRD_FREQUENCY_DICT
}

THIS_DICT = {
    FIRST_KEY_WITH_HEIGHT: FIRST_PARAM_VECTOR,
    SECOND_KEY_WITH_HEIGHT: SECOND_PARAM_VECTOR,
    THIRD_KEY_WITH_HEIGHT: THIRD_PARAM_VECTOR
}
NORM_TABLE_WITH_HEIGHT = pandas.DataFrame.from_dict(THIS_DICT, orient='index')
NORM_TABLE_WITH_HEIGHT.rename(columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)


def _compare_z_score_dicts(first_z_score_dict, second_z_score_dict):
    """Compares two dictionaries with z-score parameters.

    :param first_z_score_dict: First dictionary.
    :param second_z_score_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_z_score_dict.keys())
    second_keys = list(second_z_score_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    for this_key in first_keys:
        if this_key == normalization_params.MEAN_VALUE_KEY:
            if first_z_score_dict[this_key] != second_z_score_dict[this_key]:
                return False
        else:
            if not numpy.isclose(first_z_score_dict[this_key],
                                 second_z_score_dict[this_key], atol=TOLERANCE):
                return False

    return True


def _compare_frequency_dicts(first_frequency_dict, second_frequency_dict):
    """Compares two dictionaries with measurement frequencies.

    :param first_frequency_dict: First dictionary.
    :param second_frequency_dict: Second dictionary.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys, first_values = list(zip(
        *iter(first_frequency_dict.items())
    ))
    first_keys = numpy.array(first_keys)
    first_values = numpy.array(first_values, dtype=int)

    sort_indices = numpy.argsort(first_keys)
    first_keys = first_keys[sort_indices]
    first_values = first_values[sort_indices]

    second_keys, second_values = list(zip(
        *iter(second_frequency_dict.items())
    ))
    second_keys = numpy.array(second_keys)
    second_values = numpy.array(second_values, dtype=int)

    sort_indices = numpy.argsort(second_keys)
    second_keys = second_keys[sort_indices]
    second_values = second_values[sort_indices]

    if not numpy.array_equal(first_keys, second_keys):
        return False
    if not numpy.array_equal(first_values, second_values):
        return False

    return True


def _compare_normalization_tables(first_norm_table, second_norm_table):
    """Compares two pandas DataFrame with normalization params.

    :param first_norm_table: First table.
    :param second_norm_table: Second table.
    :return: are_tables_equal: Boolean flag.
    """

    first_indices = first_norm_table.index
    second_indices = second_norm_table.index
    if len(first_indices) != len(second_indices):
        return False

    first_column_names = list(first_norm_table)
    second_column_names = list(second_norm_table)
    if set(first_column_names) != set(second_column_names):
        return False

    for this_index in first_indices:
        for this_column_name in first_column_names:
            if not numpy.isclose(
                    first_norm_table[this_column_name].loc[[this_index]],
                    second_norm_table[this_column_name].loc[[this_index]],
                    atol=TOLERANCE
            ):
                return False

    return True


class NormalizationParamsTests(unittest.TestCase):
    """Each method is a unit test for normalization_params.py."""

    def test_update_z_score_params(self):
        """Ensures correct output from update_z_score_params."""

        this_new_param_dict = normalization_params.update_z_score_params(
            z_score_param_dict=copy.deepcopy(ORIGINAL_Z_SCORE_DICT),
            new_data_matrix=NEW_MATRIX_FOR_Z_SCORES
        )

        self.assertTrue(_compare_z_score_dicts(
            this_new_param_dict, NEW_Z_SCORE_DICT
        ))

    def test_update_frequency_dict(self):
        """Ensures correct output from update_frequency_dict."""

        this_new_frequency_dict = normalization_params.update_frequency_dict(
            frequency_dict=copy.deepcopy(MAIN_FREQUENCY_DICT),
            new_data_matrix=NEW_MATRIX_FOR_FREQUENCIES,
            rounding_base=ROUNDING_BASE
        )

        self.assertTrue(_compare_frequency_dicts(
            this_new_frequency_dict, NEW_FREQUENCY_DICT
        ))

    def test_get_standard_deviation(self):
        """Ensures correct output from _get_standard_deviation."""

        z_score_param_dict = {
            normalization_params.NUM_VALUES_KEY: STDEV_INPUT_MATRIX.size,
            normalization_params.MEAN_VALUE_KEY: numpy.mean(STDEV_INPUT_MATRIX),
            normalization_params.MEAN_OF_SQUARES_KEY:
                numpy.mean(STDEV_INPUT_MATRIX ** 2)
        }

        this_standard_deviation = normalization_params._get_standard_deviation(
            z_score_param_dict
        )

        self.assertTrue(numpy.isclose(
            this_standard_deviation, STANDARD_DEVIATION, atol=TOLERANCE
        ))

    def test_get_percentile_small(self):
        """Ensures correct output from _get_percentile.

        In this case, percentile level is small.
        """

        this_percentile = normalization_params._get_percentile(
            frequency_dict=MAIN_FREQUENCY_DICT,
            percentile_level=SMALL_PERCENTILE_LEVEL
        )
        self.assertTrue(numpy.isclose(
            this_percentile, SMALL_PERCENTILE, atol=TOLERANCE
        ))

    def test_get_percentile_medium(self):
        """Ensures correct output from _get_percentile.

        In this case, percentile level is medium.
        """

        this_percentile = normalization_params._get_percentile(
            frequency_dict=MAIN_FREQUENCY_DICT,
            percentile_level=MEDIUM_PERCENTILE_LEVEL
        )
        self.assertTrue(numpy.isclose(
            this_percentile, MEDIUM_PERCENTILE, atol=TOLERANCE
        ))

    def test_get_percentile_large(self):
        """Ensures correct output from _get_percentile.

        In this case, percentile level is large.
        """

        this_percentile = normalization_params._get_percentile(
            frequency_dict=MAIN_FREQUENCY_DICT,
            percentile_level=LARGE_PERCENTILE_LEVEL
        )
        self.assertTrue(numpy.isclose(
            this_percentile, LARGE_PERCENTILE, atol=TOLERANCE
        ))

    def test_finalize_params_no_height(self):
        """Ensures correct output from finalize_params.

        In this case, the table should be single-indexed (field name only).
        """

        this_norm_table = normalization_params.finalize_params(
            z_score_dict_dict=Z_SCORE_DICT_DICT_NO_HEIGHT,
            frequency_dict_dict=FREQUENCY_DICT_DICT_NO_HEIGHT,
            min_percentile_level=MIN_PERCENTILE_LEVEL,
            max_percentile_level=MAX_PERCENTILE_LEVEL
        )

        self.assertTrue(_compare_normalization_tables(
            this_norm_table, NORM_TABLE_NO_HEIGHT
        ))

    def test_finalize_params_with_height(self):
        """Ensures correct output from finalize_params.

        In this case, the table should be double-indexed (field name and
        height).
        """

        this_norm_table = normalization_params.finalize_params(
            z_score_dict_dict=Z_SCORE_DICT_DICT_WITH_HEIGHT,
            frequency_dict_dict=FREQUENCY_DICT_DICT_WITH_HEIGHT,
            min_percentile_level=MIN_PERCENTILE_LEVEL,
            max_percentile_level=MAX_PERCENTILE_LEVEL
        )

        self.assertTrue(_compare_normalization_tables(
            this_norm_table, NORM_TABLE_WITH_HEIGHT
        ))


if __name__ == '__main__':
    unittest.main()
