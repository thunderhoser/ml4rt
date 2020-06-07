"""Pre-processing methods."""

import pickle
import numpy
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io

DUMMY_HEIGHT_M_AGL = 10

MEAN_VALUE_COLUMN = 'mean_value'
STANDARD_DEVIATION_COLUMN = 'standard_deviation'
MIN_VALUE_COLUMN = 'min_value'
MAX_VALUE_COLUMN = 'max_value'

NORM_COLUMNS_NO_HEIGHT = [
    MEAN_VALUE_COLUMN, STANDARD_DEVIATION_COLUMN,
    MIN_VALUE_COLUMN, MAX_VALUE_COLUMN
]
NORM_COLUMNS_WITH_HEIGHT = [
    MEAN_VALUE_COLUMN, STANDARD_DEVIATION_COLUMN
]

MINMAX_NORM_STRING = 'minmax'
Z_SCORE_NORM_STRING = 'z_score'
VALID_NORM_TYPE_STRINGS = [MINMAX_NORM_STRING, Z_SCORE_NORM_STRING]


def _check_normalization_type(normalization_type_string):
    """Ensures that normalization type is valid.

    :param normalization_type_string: Normalization type.
    :raises: ValueError: if
        `normalization_type_string not in VALID_NORM_TYPE_STRINGS`.
    """

    error_checking.assert_is_string(normalization_type_string)

    if normalization_type_string not in VALID_NORM_TYPE_STRINGS:
        error_string = (
            '\n\n{0:s}\nValid normalization types (listed above) do not include'
            ' "{1:s}".'
        ).format(
            str(VALID_NORM_TYPE_STRINGS), normalization_type_string
        )

        raise ValueError(error_string)


def write_normalization_file(pickle_file_name, norm_table_no_height,
                             norm_table_with_height):
    """Writes normalization parameters to Pickle file.

    :param pickle_file_name: Path to output file.
    :param norm_table_no_height: Single-indexed pandas DataFrame.  Each index
        is a field name (in the list `example_io.PREDICTOR_NAMES` or
        `example_io.TARGET_NAMES`).  Must contain the following columns.
    norm_table_no_height.mean_value: Mean value for the given field.
    norm_table_no_height.standard_deviation: Standard deviation.
    norm_table_no_height.min_value: Minimum value.
    norm_table_no_height.max_value: Max value.

    :param norm_table_with_height: Double-indexed pandas DataFrame.  Each index
        is a tuple with (field_name, height_m_agl).  `field_name` is the same as
        the single index for `norm_table_no_height`, and `height_m_agl` is in
        metres above ground level.  Must contain the following columns.
    norm_table_with_height.mean_value: Mean value for the given field.
    norm_table_with_height.standard_deviation: Standard deviation.
    norm_table_with_height.min_value: Minimum value.
    norm_table_with_height.max_value: Max value.
    """

    error_checking.assert_columns_in_dataframe(
        norm_table_no_height, NORM_COLUMNS_NO_HEIGHT
    )
    error_checking.assert_columns_in_dataframe(
        norm_table_with_height, NORM_COLUMNS_WITH_HEIGHT
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)
    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(norm_table_no_height, pickle_file_handle)
    pickle.dump(norm_table_with_height, pickle_file_handle)
    pickle_file_handle.close()


def read_normalization_file(pickle_file_name):
    """Reads normalization parameters from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: norm_table_no_height: See doc for `write_normalization_file`.
    :return: norm_table_with_height: Same.
    """

    pickle_file_handle = open(pickle_file_name, 'rb')
    norm_table_no_height = pickle.load(pickle_file_handle)
    norm_table_with_height = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    error_checking.assert_columns_in_dataframe(
        norm_table_no_height, NORM_COLUMNS_NO_HEIGHT
    )
    error_checking.assert_columns_in_dataframe(
        norm_table_with_height, NORM_COLUMNS_WITH_HEIGHT
    )

    return norm_table_no_height, norm_table_with_height


def _normalize_one_variable(
        orig_values, normalization_type_string, normalization_table_one_row,
        min_normalized_value=None, max_normalized_value=None):
    """Normalizes one variable (either predictor or target variable).

    :param orig_values: numpy array with original (not normalized) values.
    :param normalization_type_string: See doc for `normalize_data`.
    :param normalization_table_one_row: One row of input `normalization_table`
        to method `normalize_data`.
    :param min_normalized_value: See doc for `normalize_data`.
    :param max_normalized_value: Same.
    :return: normalized_values: numpy array with same shape as `orig_values`,
        containing normalized values.
    """

    if normalization_type_string == MINMAX_NORM_STRING:
        min_value = normalization_table_one_row[MIN_VALUE_COLUMN].values[0]
        max_value = normalization_table_one_row[MAX_VALUE_COLUMN].values[0]

        normalized_values = (orig_values - min_value) / (max_value - min_value)
        normalized_values = min_normalized_value + normalized_values * (
            max_normalized_value - min_normalized_value
        )
    else:
        mean_value = normalization_table_one_row[MEAN_VALUE_COLUMN].values[0]
        standard_deviation = (
            normalization_table_one_row[STANDARD_DEVIATION_COLUMN].values[0]
        )

        normalized_values = (orig_values - mean_value) / standard_deviation

    return normalized_values


def _denorm_one_variable(
        normalized_values, normalization_type_string,
        normalization_table_one_row, min_normalized_value=None,
        max_normalized_value=None):
    """Denormalizes one variable (either predictor or target variable).

    :param normalized_values: numpy array with normalized values.
    :param normalization_type_string: See doc for `_normalize_one_variable`.
    :param normalization_table_one_row: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: denorm_values: numpy array with same shape as `normalized_values`,
        containing denormalized values.
    """

    if normalization_type_string == MINMAX_NORM_STRING:
        min_value = normalization_table_one_row[MIN_VALUE_COLUMN].values[0]
        max_value = normalization_table_one_row[MAX_VALUE_COLUMN].values[0]

        denorm_values = (
            (normalized_values - min_normalized_value) /
            (max_normalized_value - min_normalized_value)
        )
        denorm_values = min_value + denorm_values * (max_value - min_value)
    else:
        mean_value = normalization_table_one_row[MEAN_VALUE_COLUMN].values[0]
        standard_deviation = (
            normalization_table_one_row[STANDARD_DEVIATION_COLUMN].values[0]
        )

        denorm_values = mean_value + standard_deviation * normalized_values

    return denorm_values


def normalize_data(
        example_dict, normalization_type_string, normalization_file_name,
        min_normalized_value=-1., max_normalized_value=1.,
        separate_heights=False, test_mode=False, normalization_table=None):
    """Normalizes data (both predictor and target variables).

    :param example_dict: Dictionary with learning examples (see doc for
        `example_io.read_file`).
    :param normalization_type_string: Normalization type (must be accepted by
        `check_normalization_type`).
    :param normalization_file_name: Path to file with normalization params (will
        be read by `read_normalization_file`).
    :param min_normalized_value:
        [used only if normalization_type_string == 'minmax']
        Minimum value after normalization.
    :param max_normalized_value:
        [used only if normalization_type_string == 'minmax']
        Max value after normalization.
    :param separate_heights: Boolean flag.  If True, will normalize separately
        at each height.
    :param test_mode: For testing only.  Leave this alone.
    :param normalization_table: For testing only.  Leave this alone.
    :return: example_dict: Same as input but with normalized values.
    """

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        normalization_table = read_normalization_file(
            normalization_file_name
        )[int(separate_heights)]

    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORM_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value
        )

    scalar_predictor_names = example_dict[example_io.SCALAR_PREDICTOR_NAMES_KEY]
    scalar_predictor_matrix = example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]

    for k in range(len(scalar_predictor_names)):
        if separate_heights:
            this_index = [(scalar_predictor_names[k], DUMMY_HEIGHT_M_AGL)]
        else:
            this_index = scalar_predictor_names[k]

        scalar_predictor_matrix[..., k] = _normalize_one_variable(
            orig_values=scalar_predictor_matrix[..., k],
            normalization_type_string=normalization_type_string,
            normalization_table_one_row=normalization_table.loc[this_index],
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY] = scalar_predictor_matrix

    scalar_target_names = example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
    scalar_target_matrix = example_dict[example_io.SCALAR_TARGET_VALS_KEY]

    for k in range(len(scalar_target_names)):
        if separate_heights:
            this_index = [(scalar_target_names[k], DUMMY_HEIGHT_M_AGL)]
        else:
            this_index = scalar_target_names[k]

        scalar_target_matrix[..., k] = _normalize_one_variable(
            orig_values=scalar_target_matrix[..., k],
            normalization_type_string=normalization_type_string,
            normalization_table_one_row=normalization_table.loc[this_index],
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    example_dict[example_io.SCALAR_TARGET_VALS_KEY] = scalar_target_matrix

    vector_predictor_names = example_dict[example_io.VECTOR_PREDICTOR_NAMES_KEY]
    vector_predictor_matrix = example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    heights_m_agl = (
        numpy.round(example_dict[example_io.HEIGHTS_KEY]).astype(int)
    )
    num_heights = len(heights_m_agl)

    for k in range(len(vector_predictor_names)):
        if separate_heights:
            for j in range(num_heights):
                this_index = [(vector_predictor_names[k], heights_m_agl[j])]

                vector_predictor_matrix[..., j, k] = _normalize_one_variable(
                    orig_values=vector_predictor_matrix[..., j, k],
                    normalization_type_string=normalization_type_string,
                    normalization_table_one_row=
                    normalization_table.loc[this_index],
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            vector_predictor_matrix[..., k] = _normalize_one_variable(
                orig_values=vector_predictor_matrix[..., k],
                normalization_type_string=normalization_type_string,
                normalization_table_one_row=
                normalization_table.loc[vector_predictor_names[k]],
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY] = vector_predictor_matrix

    vector_target_names = example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
    vector_target_matrix = example_dict[example_io.VECTOR_TARGET_VALS_KEY]

    for k in range(len(vector_target_names)):
        if separate_heights:
            for j in range(num_heights):
                this_index = [(vector_target_names[k], heights_m_agl[j])]

                vector_target_matrix[..., j, k] = _normalize_one_variable(
                    orig_values=vector_target_matrix[..., j, k],
                    normalization_type_string=normalization_type_string,
                    normalization_table_one_row=
                    normalization_table.loc[this_index],
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            vector_target_matrix[..., k] = _normalize_one_variable(
                orig_values=vector_target_matrix[..., k],
                normalization_type_string=normalization_type_string,
                normalization_table_one_row=
                normalization_table.loc[vector_target_names[k]],
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    example_dict[example_io.VECTOR_TARGET_VALS_KEY] = vector_target_matrix
    return example_dict


def denormalize_data(
        example_dict, normalization_type_string, normalization_file_name,
        min_normalized_value=-1., max_normalized_value=1.,
        separate_heights=False, test_mode=False, normalization_table=None):
    """Denormalizes data (both predictor and target variables).

    :param example_dict: See doc for `normalize_data`.
    :param normalization_type_string: Same.
    :param normalization_file_name: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param separate_heights: Same.
    :param test_mode: Same.
    :param normalization_table: Same.
    :return: example_dict: Same as input but with denormalized values.
    """

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        normalization_table = read_normalization_file(
            normalization_file_name
        )[int(separate_heights)]

    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORM_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value
        )

    scalar_predictor_names = example_dict[example_io.SCALAR_PREDICTOR_NAMES_KEY]
    scalar_predictor_matrix = example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]

    for k in range(len(scalar_predictor_names)):
        if separate_heights:
            this_index = [(scalar_predictor_names[k], DUMMY_HEIGHT_M_AGL)]
        else:
            this_index = scalar_predictor_names[k]

        scalar_predictor_matrix[..., k] = _denorm_one_variable(
            normalized_values=scalar_predictor_matrix[..., k],
            normalization_type_string=normalization_type_string,
            normalization_table_one_row=normalization_table.loc[this_index],
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY] = scalar_predictor_matrix

    scalar_target_names = example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
    scalar_target_matrix = example_dict[example_io.SCALAR_TARGET_VALS_KEY]

    for k in range(len(scalar_target_names)):
        if separate_heights:
            this_index = [(scalar_target_names[k], DUMMY_HEIGHT_M_AGL)]
        else:
            this_index = scalar_target_names[k]

        scalar_target_matrix[..., k] = _denorm_one_variable(
            normalized_values=scalar_target_matrix[..., k],
            normalization_type_string=normalization_type_string,
            normalization_table_one_row=normalization_table.loc[this_index],
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    example_dict[example_io.SCALAR_TARGET_VALS_KEY] = scalar_target_matrix

    vector_predictor_names = example_dict[example_io.VECTOR_PREDICTOR_NAMES_KEY]
    vector_predictor_matrix = example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    heights_m_agl = (
        numpy.round(example_dict[example_io.HEIGHTS_KEY]).astype(int)
    )
    num_heights = len(heights_m_agl)

    for k in range(len(vector_predictor_names)):
        if separate_heights:
            for j in range(num_heights):
                this_index = [(vector_predictor_names[k], heights_m_agl[j])]

                vector_predictor_matrix[..., j, k] = _denorm_one_variable(
                    normalized_values=vector_predictor_matrix[..., j, k],
                    normalization_type_string=normalization_type_string,
                    normalization_table_one_row=
                    normalization_table.loc[this_index],
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            vector_predictor_matrix[..., k] = _denorm_one_variable(
                normalized_values=vector_predictor_matrix[..., k],
                normalization_type_string=normalization_type_string,
                normalization_table_one_row=
                normalization_table.loc[vector_predictor_names[k]],
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY] = vector_predictor_matrix

    vector_target_names = example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
    vector_target_matrix = example_dict[example_io.VECTOR_TARGET_VALS_KEY]

    for k in range(len(vector_target_names)):
        if separate_heights:
            for j in range(num_heights):
                this_index = [(vector_target_names[k], heights_m_agl[j])]

                vector_target_matrix[..., j, k] = _denorm_one_variable(
                    normalized_values=vector_target_matrix[..., j, k],
                    normalization_type_string=normalization_type_string,
                    normalization_table_one_row=
                    normalization_table.loc[this_index],
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            vector_target_matrix[..., k] = _denorm_one_variable(
                normalized_values=vector_target_matrix[..., k],
                normalization_type_string=normalization_type_string,
                normalization_table_one_row=
                normalization_table.loc[vector_target_names[k]],
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    example_dict[example_io.VECTOR_TARGET_VALS_KEY] = vector_target_matrix
    return example_dict
