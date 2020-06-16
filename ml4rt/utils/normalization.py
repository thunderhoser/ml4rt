"""Methods for normalizing predictor and target variables."""

import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import normalization_params

DUMMY_HEIGHT_M_AGL = 10

MEAN_VALUE_COLUMN = normalization_params.MEAN_VALUE_COLUMN
STANDARD_DEVIATION_COLUMN = normalization_params.STANDARD_DEVIATION_COLUMN
MIN_VALUE_COLUMN = normalization_params.MIN_VALUE_COLUMN
MAX_VALUE_COLUMN = normalization_params.MAX_VALUE_COLUMN

MINMAX_NORM_STRING = 'minmax'
Z_SCORE_NORM_STRING = 'z_score'
VALID_NORM_TYPE_STRINGS = [MINMAX_NORM_STRING, Z_SCORE_NORM_STRING]

TARGET_NAME_TO_LOG_FLAG = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: False,
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: False,
    example_io.SHORTWAVE_UP_FLUX_NAME: True,
    example_io.SHORTWAVE_TOA_UP_FLUX_NAME: True,
    example_io.SHORTWAVE_HEATING_RATE_NAME: True
}


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
        min_value = normalization_table_one_row[MIN_VALUE_COLUMN]
        max_value = normalization_table_one_row[MAX_VALUE_COLUMN]

        if 'pandas' in str(type(min_value)):
            min_value = min_value.values[0]
            max_value = max_value.values[0]

        normalized_values = (
            (orig_values - min_value) / (max_value - min_value)
        )
        normalized_values = min_normalized_value + normalized_values * (
            max_normalized_value - min_normalized_value
        )
    else:
        mean_value = normalization_table_one_row[MEAN_VALUE_COLUMN]
        standard_deviation = (
            normalization_table_one_row[STANDARD_DEVIATION_COLUMN]
        )

        if 'pandas' in str(type(mean_value)):
            mean_value = mean_value.values[0]
            standard_deviation = standard_deviation.values[0]

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
        min_value = normalization_table_one_row[MIN_VALUE_COLUMN]
        max_value = normalization_table_one_row[MAX_VALUE_COLUMN]

        if 'pandas' in str(type(min_value)):
            min_value = min_value.values[0]
            max_value = max_value.values[0]

        denorm_values = (
            (normalized_values - min_normalized_value) /
            (max_normalized_value - min_normalized_value)
        )
        denorm_values = min_value + denorm_values * (max_value - min_value)
    else:
        mean_value = normalization_table_one_row[MEAN_VALUE_COLUMN]
        standard_deviation = (
            normalization_table_one_row[STANDARD_DEVIATION_COLUMN]
        )

        if 'pandas' in str(type(mean_value)):
            mean_value = mean_value.values[0]
            standard_deviation = standard_deviation.values[0]

        denorm_values = mean_value + standard_deviation * normalized_values

    return denorm_values


def convert_to_log_if_necessary(physical_values, field_name):
    """If necessary, converts variable from physical scale to log scale.

    :param physical_values: numpy array (any shape) with physical values.
    :param field_name: Field name (must be accepted by
        `example_io.check_field_name`).
    :return: output_values: numpy array (same shape as `input_array`) with
        either physical or log values.
    """

    error_checking.assert_is_numpy_array_without_nan(physical_values)
    example_io.check_field_name(field_name)

    if field_name not in TARGET_NAME_TO_LOG_FLAG:
        return physical_values

    if not TARGET_NAME_TO_LOG_FLAG[field_name]:
        return physical_values

    return numpy.log10(1. + physical_values)


def convert_from_log_if_necessary(input_values, field_name):
    """If necessary, converts variable from log scale to physical scale.

    This method is the inverse of `convert_to_log_if_necessary`.

    :param input_values: numpy array (any shape) with either physical or log
        values.
    :param field_name: Field name (must be accepted by
        `example_io.check_field_name`).
    :return: physical_values: numpy array (same shape as `input_array`) with
        physical values.
    """

    error_checking.assert_is_numpy_array_without_nan(input_values)
    example_io.check_field_name(field_name)

    if field_name not in TARGET_NAME_TO_LOG_FLAG:
        return input_values

    if not TARGET_NAME_TO_LOG_FLAG[field_name]:
        return input_values

    return 10 ** input_values - 1.


def normalize_data(
        example_dict, normalization_type_string, normalization_file_name,
        min_normalized_value=-1., max_normalized_value=1.,
        separate_heights=False, apply_to_predictors=True, apply_to_targets=True,
        test_mode=False, normalization_table=None):
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
    :param apply_to_predictors: Boolean flag.  If True, will normalize
        predictors.
    :param apply_to_targets: Boolean flag.  If True, will normalize targets.
    :param test_mode: For testing only.  Leave this alone.
    :param normalization_table: For testing only.  Leave this alone.
    :return: example_dict: Same as input but with normalized values.
    :raises: ValueError: if `apply_to_predictors == apply_to_targets == False`.
    """

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_boolean(apply_to_predictors)
    error_checking.assert_is_boolean(apply_to_targets)
    error_checking.assert_is_boolean(test_mode)

    if not (apply_to_predictors or apply_to_targets):
        error_string = (
            'Either `apply_to_predictors` or `apply_to_targets` must be True.'
        )
        raise ValueError(error_string)

    if not test_mode:
        normalization_table = normalization_params.read_file(
            normalization_file_name
        )[int(separate_heights)]

    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORM_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value
        )

    if apply_to_predictors:
        scalar_predictor_names = example_dict[
            example_io.SCALAR_PREDICTOR_NAMES_KEY
        ]
    else:
        scalar_predictor_names = []

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

    if apply_to_targets:
        scalar_target_names = example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
    else:
        scalar_target_names = []

    scalar_target_matrix = example_dict[example_io.SCALAR_TARGET_VALS_KEY]

    for k in range(len(scalar_target_names)):
        if separate_heights:
            this_index = [(scalar_target_names[k], DUMMY_HEIGHT_M_AGL)]
        else:
            this_index = scalar_target_names[k]

        scalar_target_matrix[..., k] = convert_to_log_if_necessary(
            physical_values=scalar_target_matrix[..., k],
            field_name=scalar_target_names[k]
        )

        scalar_target_matrix[..., k] = _normalize_one_variable(
            orig_values=scalar_target_matrix[..., k],
            normalization_type_string=normalization_type_string,
            normalization_table_one_row=normalization_table.loc[this_index],
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    example_dict[example_io.SCALAR_TARGET_VALS_KEY] = scalar_target_matrix

    if apply_to_predictors:
        vector_predictor_names = example_dict[
            example_io.VECTOR_PREDICTOR_NAMES_KEY
        ]
    else:
        vector_predictor_names = []

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

    if apply_to_targets:
        vector_target_names = example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
    else:
        vector_target_names = []

    vector_target_matrix = example_dict[example_io.VECTOR_TARGET_VALS_KEY]

    for k in range(len(vector_target_names)):
        if separate_heights:
            for j in range(num_heights):
                vector_target_matrix[..., j, k] = convert_to_log_if_necessary(
                    physical_values=vector_target_matrix[..., j, k],
                    field_name=vector_target_names[k]
                )

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
            vector_target_matrix[..., k] = convert_to_log_if_necessary(
                physical_values=vector_target_matrix[..., k],
                field_name=vector_target_names[k]
            )

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
        separate_heights=False, apply_to_predictors=True, apply_to_targets=True,
        test_mode=False, normalization_table=None):
    """Denormalizes data (both predictor and target variables).

    :param example_dict: See doc for `normalize_data`.
    :param normalization_type_string: Same.
    :param normalization_file_name: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param separate_heights: Same.
    :param apply_to_predictors: Same.
    :param apply_to_targets: Same.
    :param test_mode: Same.
    :param normalization_table: Same.
    :return: example_dict: Same as input but with denormalized values.
    :raises: ValueError: if `apply_to_predictors == apply_to_targets == False`.
    """

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_boolean(apply_to_predictors)
    error_checking.assert_is_boolean(apply_to_targets)
    error_checking.assert_is_boolean(test_mode)

    if not (apply_to_predictors or apply_to_targets):
        error_string = (
            'Either `apply_to_predictors` or `apply_to_targets` must be True.'
        )
        raise ValueError(error_string)

    if not test_mode:
        normalization_table = normalization_params.read_file(
            normalization_file_name
        )[int(separate_heights)]

    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORM_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value
        )

    if apply_to_predictors:
        scalar_predictor_names = example_dict[
            example_io.SCALAR_PREDICTOR_NAMES_KEY
        ]
    else:
        scalar_predictor_names = []

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

    if apply_to_targets:
        scalar_target_names = example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
    else:
        scalar_target_names = []

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

        scalar_target_matrix[..., k] = convert_from_log_if_necessary(
            input_values=scalar_target_matrix[..., k],
            field_name=scalar_target_names[k]
        )

    example_dict[example_io.SCALAR_TARGET_VALS_KEY] = scalar_target_matrix

    if apply_to_predictors:
        vector_predictor_names = example_dict[
            example_io.VECTOR_PREDICTOR_NAMES_KEY
        ]
    else:
        vector_predictor_names = []

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

    if apply_to_targets:
        vector_target_names = example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
    else:
        vector_target_names = []

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

                vector_target_matrix[..., j, k] = convert_from_log_if_necessary(
                    input_values=vector_target_matrix[..., j, k],
                    field_name=vector_target_names[k]
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

            vector_target_matrix[..., k] = convert_from_log_if_necessary(
                input_values=vector_target_matrix[..., k],
                field_name=vector_target_names[k]
            )

    example_dict[example_io.VECTOR_TARGET_VALS_KEY] = vector_target_matrix
    return example_dict


def create_mean_example(
        example_dict, normalization_file_name, test_mode=False,
        normalization_table=None):
    """Creates mean example (with mean value for each variable/height pair).

    :param example_dict: See doc for `normalize_data`.
    :param normalization_file_name: Same.
    :param test_mode: Same.
    :param normalization_table: Same.
    :return: mean_example_dict: See doc for `example_io.average_examples`.
    """

    error_checking.assert_is_boolean(test_mode)

    if not test_mode:
        normalization_table = normalization_params.read_file(
            normalization_file_name
        )[1]

    scalar_predictor_names = example_dict[example_io.SCALAR_PREDICTOR_NAMES_KEY]
    scalar_target_names = example_dict[example_io.SCALAR_TARGET_NAMES_KEY]
    vector_predictor_names = example_dict[example_io.VECTOR_PREDICTOR_NAMES_KEY]
    vector_target_names = example_dict[example_io.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = example_dict[example_io.HEIGHTS_KEY]

    num_scalar_predictors = len(scalar_predictor_names)
    num_scalar_targets = len(scalar_target_names)
    num_vector_predictors = len(vector_predictor_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)

    scalar_predictor_values = numpy.full(num_scalar_predictors, numpy.nan)
    scalar_target_values = numpy.full(num_scalar_targets, numpy.nan)
    vector_predictor_matrix = numpy.full(
        (num_heights, num_vector_predictors), numpy.nan
    )
    vector_target_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )

    for k in range(num_scalar_predictors):
        this_index = [(scalar_predictor_names[k], DUMMY_HEIGHT_M_AGL)]
        this_mean_value = normalization_table.loc[this_index][MEAN_VALUE_COLUMN]

        if 'pandas' in str(type(this_mean_value)):
            this_mean_value = this_mean_value.values[0]

        scalar_predictor_values[k] = this_mean_value

    for k in range(num_scalar_targets):
        this_index = [(scalar_target_names[k], DUMMY_HEIGHT_M_AGL)]
        this_mean_value = normalization_table.loc[this_index][MEAN_VALUE_COLUMN]

        if 'pandas' in str(type(this_mean_value)):
            this_mean_value = this_mean_value.values[0]

        this_mean_value = convert_from_log_if_necessary(
            input_values=numpy.array([this_mean_value]),
            field_name=scalar_target_names[k]
        )[0]

        scalar_target_values[k] = this_mean_value

    for j in range(num_heights):
        for k in range(num_vector_predictors):
            this_index = [(vector_predictor_names[k], heights_m_agl[j])]
            this_mean_value = (
                normalization_table.loc[this_index][MEAN_VALUE_COLUMN]
            )

            if 'pandas' in str(type(this_mean_value)):
                this_mean_value = this_mean_value.values[0]

            vector_predictor_matrix[j, k] = this_mean_value

        for k in range(num_vector_targets):
            this_index = [(vector_target_names[k], heights_m_agl[j])]
            this_mean_value = (
                normalization_table.loc[this_index][MEAN_VALUE_COLUMN]
            )

            if 'pandas' in str(type(this_mean_value)):
                this_mean_value = this_mean_value.values[0]

            this_mean_value = convert_from_log_if_necessary(
                input_values=numpy.array([this_mean_value]),
                field_name=vector_target_names[k]
            )[0]

            vector_target_matrix[j, k] = this_mean_value

    return {
        example_io.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        example_io.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_io.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        example_io.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_io.HEIGHTS_KEY: heights_m_agl,
        example_io.SCALAR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(scalar_predictor_values, axis=0),
        example_io.SCALAR_TARGET_VALS_KEY:
            numpy.expand_dims(scalar_target_values, axis=0),
        example_io.VECTOR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(vector_predictor_matrix, axis=0),
        example_io.VECTOR_TARGET_VALS_KEY:
            numpy.expand_dims(vector_target_matrix, axis=0)
    }
