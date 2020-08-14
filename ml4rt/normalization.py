"""Methods for normalizing predictor and target variables."""

import sys
import os.path
import numpy
import scipy.stats

# THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
#     os.path.join(os.getcwd(), os.path.expanduser(__file__))
# ))
# sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

from ml4rt import error_checking
from ml4rt import example_utils

DUMMY_HEIGHT_M_AGL = 10

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6

MINMAX_NORM_STRING = 'minmax'
Z_SCORE_NORM_STRING = 'z_score'
VALID_NORM_TYPE_STRINGS = [MINMAX_NORM_STRING, Z_SCORE_NORM_STRING]

TARGET_NAME_TO_LOG_FLAG = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: False,
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: False,
    example_utils.SHORTWAVE_UP_FLUX_NAME: True,
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: True,
    example_utils.SHORTWAVE_HEATING_RATE_NAME: True
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


def _orig_to_uniform_dist(orig_values_new, orig_values_training):
    """Converts values from original to uniform distribution.

    :param orig_values_new: numpy array of original (physical-scale) values to
        convert.
    :param orig_values_training: numpy array of original (physical-scale) values
        in training data.
    :return: uniform_values_new: numpy array (same shape as `orig_values_new`)
        with rescaled values from 0...1.
    """

    # return 0.01 * numpy.array([
    #     scipy.stats.percentileofscore(
    #         orig_values_training, x, kind='rank'
    #     )
    #     for x in orig_values_new
    # ])

    orig_values_new_1d = numpy.ravel(orig_values_new)

    indices = numpy.searchsorted(
        numpy.sort(numpy.ravel(orig_values_training)), orig_values_new_1d,
        side='left'
    ).astype(float)

    num_values = orig_values_training.size
    uniform_values_new_1d = indices / (num_values - 1)
    uniform_values_new_1d = numpy.minimum(uniform_values_new_1d, 1.)

    return numpy.reshape(uniform_values_new_1d, orig_values_new.shape)


def _uniform_to_orig_dist(uniform_values_new, orig_values_training):
    """Converts values from uniform to original distribution.

    This method is the inverse of `_orig_to_uniform_dist`.

    :param uniform_values_new: See doc for `_orig_to_uniform_dist`.
    :param orig_values_training: Same.
    :return: orig_values_new: Same.
    """

    uniform_values_new_1d = numpy.ravel(uniform_values_new)

    orig_values_new_1d = numpy.percentile(
        numpy.ravel(orig_values_training), 100 * uniform_values_new_1d,
        interpolation='midpoint'
    )

    return numpy.reshape(orig_values_new_1d, uniform_values_new.shape)


def _orig_to_normal_dist(orig_values_new, orig_values_training):
    """Converts values from original to normal distribution.

    :param orig_values_new: See doc for `_orig_to_uniform_dist`.
    :param orig_values_training: Same.
    :return: normalized_values_new: numpy array (same shape as
        `orig_values_new`) with normalized values (z-scores).
    """

    uniform_values_new = _orig_to_uniform_dist(
        orig_values_new=orig_values_new,
        orig_values_training=orig_values_training
    )

    uniform_values_new = numpy.maximum(
        uniform_values_new, MIN_CUMULATIVE_DENSITY
    )
    uniform_values_new = numpy.minimum(
        uniform_values_new, MAX_CUMULATIVE_DENSITY
    )
    return scipy.stats.norm.ppf(uniform_values_new, loc=0., scale=1.)


def _normal_to_orig_dist(normalized_values_new, orig_values_training):
    """Converts values from normal to original distribution.

    This method is the inverse of `_orig_to_normal_dist`.

    :param normalized_values_new: See doc for `_orig_to_normal_dist`.
    :param orig_values_training: Same.
    :return: orig_values_new: Same.
    """

    uniform_values_new = scipy.stats.norm.cdf(
        normalized_values_new, loc=0., scale=1.
    )

    return _uniform_to_orig_dist(
        uniform_values_new=uniform_values_new,
        orig_values_training=orig_values_training
    )


def _normalize_one_variable(
        orig_values_new, orig_values_training, normalization_type_string,
        min_normalized_value=None, max_normalized_value=None):
    """Normalizes one variable (either predictor or target variable).

    :param orig_values_new: numpy array with original (unnormalized) values to
        convert.
    :param orig_values_training: numpy array with original (unnormalized) values
        in training data.
    :param normalization_type_string: See doc for `normalize_data`.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: normalized_values_new: numpy array with same shape as
        `orig_values_new`, containing normalized values.
    """

    if normalization_type_string == MINMAX_NORM_STRING:
        normalized_values_new = _orig_to_uniform_dist(
            orig_values_new=orig_values_new,
            orig_values_training=orig_values_training
        )
        normalized_values_new = min_normalized_value + normalized_values_new * (
            max_normalized_value - min_normalized_value
        )
    else:
        normalized_values_new = _orig_to_normal_dist(
            orig_values_new=orig_values_new,
            orig_values_training=orig_values_training
        )

    return normalized_values_new


def _denorm_one_variable(
        normalized_values_new, orig_values_training, normalization_type_string,
        min_normalized_value=None, max_normalized_value=None):
    """Denormalizes one variable (either predictor or target variable).

    :param normalized_values_new: See doc for `_normalize_one_variable`.
    :param orig_values_training: Same.
    :param normalization_type_string: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :return: denorm_values_new: numpy array with same shape as
        `normalized_values_new`, containing denormalized values.
    """

    if normalization_type_string == MINMAX_NORM_STRING:
        normalized_values_new = (
            (normalized_values_new - min_normalized_value) /
            (max_normalized_value - min_normalized_value)
        )
        normalized_values_new = numpy.maximum(normalized_values_new, 0.)
        normalized_values_new = numpy.minimum(normalized_values_new, 1.)

        denorm_values = _uniform_to_orig_dist(
            uniform_values_new=normalized_values_new,
            orig_values_training=orig_values_training
        )
    else:
        denorm_values = _normal_to_orig_dist(
            normalized_values_new=normalized_values_new,
            orig_values_training=orig_values_training
        )

    return denorm_values


def normalize_data(
        new_example_dict, training_example_dict, normalization_type_string,
        min_normalized_value=-1., max_normalized_value=1.,
        separate_heights=False, apply_to_predictors=True,
        apply_to_vector_targets=True, apply_to_scalar_targets=True):
    """Normalizes data (both predictor and target variables).

    :param new_example_dict: Dictionary with learning examples to be normalized
        (see doc for `example_io.read_file`).
    :param training_example_dict: Dictionary with training examples (see doc for
        `example_io.read_file`).
    :param normalization_type_string: Normalization type (must be accepted by
        `check_normalization_type`).
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
    :param apply_to_vector_targets: Boolean flag.  If True, will normalize
        vector target variables.
    :param apply_to_scalar_targets: Boolean flag.  If True, will normalize
        scalar target variables.
    :return: example_dict: Same as input but with normalized values.
    :raises: ValueError: if `apply_to_predictors == apply_to_vector_targets ==
        apply_to_scalar_targets == False`.
    """

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_boolean(apply_to_predictors)
    error_checking.assert_is_boolean(apply_to_vector_targets)
    error_checking.assert_is_boolean(apply_to_scalar_targets)

    if not (
            apply_to_predictors
            or apply_to_vector_targets
            or apply_to_scalar_targets
    ):
        raise ValueError(
            'One of `apply_to_predictors`, `apply_to_vector_targets`, and '
            '`apply_to_scalar_targets` must be True.'
        )

    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORM_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value
        )

    if apply_to_predictors:
        scalar_predictor_names = (
            new_example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
        )
    else:
        scalar_predictor_names = []

    new_scalar_predictor_matrix = (
        new_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )

    for k in range(len(scalar_predictor_names)):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_predictor_names[k]
        )

        new_scalar_predictor_matrix[..., k] = _normalize_one_variable(
            orig_values_new=new_scalar_predictor_matrix[..., k],
            orig_values_training=these_training_values,
            normalization_type_string=normalization_type_string,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    new_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        new_scalar_predictor_matrix
    )

    if apply_to_scalar_targets:
        scalar_target_names = (
            new_example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
        )
    else:
        scalar_target_names = []

    new_scalar_target_matrix = (
        new_example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    )

    for k in range(len(scalar_target_names)):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_target_names[k]
        )

        new_scalar_target_matrix[..., k] = _normalize_one_variable(
            orig_values_new=new_scalar_target_matrix[..., k],
            orig_values_training=these_training_values,
            normalization_type_string=normalization_type_string,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    new_example_dict[example_utils.SCALAR_TARGET_VALS_KEY] = (
        new_scalar_target_matrix
    )

    if apply_to_predictors:
        vector_predictor_names = (
            new_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        )
    else:
        vector_predictor_names = []

    new_vector_predictor_matrix = (
        new_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )
    heights_m_agl = (
        numpy.round(new_example_dict[example_utils.HEIGHTS_KEY]).astype(int)
    )
    num_heights = len(heights_m_agl)

    for k in range(len(vector_predictor_names)):
        if separate_heights:
            for j in range(num_heights):
                these_training_values = example_utils.get_field_from_dict(
                    example_dict=training_example_dict,
                    field_name=vector_predictor_names[k],
                    height_m_agl=heights_m_agl[j]
                )

                new_vector_predictor_matrix[..., j, k] = (
                    _normalize_one_variable(
                        orig_values_new=new_vector_predictor_matrix[..., j, k],
                        orig_values_training=these_training_values,
                        normalization_type_string=normalization_type_string,
                        min_normalized_value=min_normalized_value,
                        max_normalized_value=max_normalized_value
                    )
                )
        else:
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_predictor_names[k]
            )

            new_vector_predictor_matrix[..., k] = _normalize_one_variable(
                orig_values_new=new_vector_predictor_matrix[..., k],
                orig_values_training=these_training_values,
                normalization_type_string=normalization_type_string,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    new_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        new_vector_predictor_matrix
    )

    if apply_to_vector_targets:
        vector_target_names = (
            new_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
        )
    else:
        vector_target_names = []

    new_vector_target_matrix = (
        new_example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
    )

    for k in range(len(vector_target_names)):
        if separate_heights:
            for j in range(num_heights):
                these_training_values = example_utils.get_field_from_dict(
                    example_dict=training_example_dict,
                    field_name=vector_target_names[k],
                    height_m_agl=heights_m_agl[j]
                )

                new_vector_target_matrix[..., j, k] = _normalize_one_variable(
                    orig_values_new=new_vector_target_matrix[..., j, k],
                    orig_values_training=these_training_values,
                    normalization_type_string=normalization_type_string,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_target_names[k]
            )

            new_vector_target_matrix[..., k] = _normalize_one_variable(
                orig_values_new=new_vector_target_matrix[..., k],
                orig_values_training=these_training_values,
                normalization_type_string=normalization_type_string,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    new_example_dict[example_utils.VECTOR_TARGET_VALS_KEY] = (
        new_vector_target_matrix
    )
    return new_example_dict


def denormalize_data(
        new_example_dict, training_example_dict, normalization_type_string,
        min_normalized_value=-1., max_normalized_value=1.,
        separate_heights=False, apply_to_predictors=True,
        apply_to_vector_targets=True, apply_to_scalar_targets=True):
    """Denormalizes data (both predictor and target variables).

    :param new_example_dict: See doc for `normalize_data`.
    :param training_example_dict: Same.
    :param normalization_type_string: Same.
    :param min_normalized_value: Same.
    :param max_normalized_value: Same.
    :param separate_heights: Same.
    :param apply_to_predictors: Same.
    :param apply_to_vector_targets: Same.
    :param apply_to_scalar_targets: Same.
    :return: example_dict: Same as input but with denormalized values.
    :raises: ValueError: if `apply_to_predictors == apply_to_vector_targets ==
        apply_to_scalar_targets == False`.
    """

    error_checking.assert_is_boolean(separate_heights)
    error_checking.assert_is_boolean(apply_to_predictors)
    error_checking.assert_is_boolean(apply_to_vector_targets)
    error_checking.assert_is_boolean(apply_to_scalar_targets)

    if not (
            apply_to_predictors
            or apply_to_vector_targets
            or apply_to_scalar_targets
    ):
        raise ValueError(
            'One of `apply_to_predictors`, `apply_to_vector_targets`, and '
            '`apply_to_scalar_targets` must be True.'
        )

    _check_normalization_type(normalization_type_string)

    if normalization_type_string == MINMAX_NORM_STRING:
        error_checking.assert_is_greater(
            max_normalized_value, min_normalized_value
        )

    if apply_to_predictors:
        scalar_predictor_names = (
            new_example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
        )
    else:
        scalar_predictor_names = []

    new_scalar_predictor_matrix = (
        new_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )

    for k in range(len(scalar_predictor_names)):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_predictor_names[k]
        )

        new_scalar_predictor_matrix[..., k] = _denorm_one_variable(
            normalized_values_new=new_scalar_predictor_matrix[..., k],
            orig_values_training=these_training_values,
            normalization_type_string=normalization_type_string,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    new_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY] = (
        new_scalar_predictor_matrix
    )

    if apply_to_scalar_targets:
        scalar_target_names = (
            new_example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
        )
    else:
        scalar_target_names = []

    new_scalar_target_matrix = (
        new_example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    )

    for k in range(len(scalar_target_names)):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_target_names[k]
        )

        new_scalar_target_matrix[..., k] = _denorm_one_variable(
            normalized_values_new=new_scalar_target_matrix[..., k],
            orig_values_training=these_training_values,
            normalization_type_string=normalization_type_string,
            min_normalized_value=min_normalized_value,
            max_normalized_value=max_normalized_value
        )

    new_example_dict[example_utils.SCALAR_TARGET_VALS_KEY] = (
        new_scalar_target_matrix
    )

    if apply_to_predictors:
        vector_predictor_names = (
            new_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        )
    else:
        vector_predictor_names = []

    new_vector_predictor_matrix = (
        new_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )
    heights_m_agl = (
        numpy.round(new_example_dict[example_utils.HEIGHTS_KEY]).astype(int)
    )
    num_heights = len(heights_m_agl)

    for k in range(len(vector_predictor_names)):
        if separate_heights:
            for j in range(num_heights):
                these_training_values = example_utils.get_field_from_dict(
                    example_dict=training_example_dict,
                    field_name=vector_predictor_names[k],
                    height_m_agl=heights_m_agl[j]
                )

                new_vector_predictor_matrix[..., j, k] = _denorm_one_variable(
                    normalized_values_new=
                    new_vector_predictor_matrix[..., j, k],
                    orig_values_training=these_training_values,
                    normalization_type_string=normalization_type_string,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_predictor_names[k]
            )

            new_vector_predictor_matrix[..., k] = _denorm_one_variable(
                normalized_values_new=new_vector_predictor_matrix[..., k],
                orig_values_training=these_training_values,
                normalization_type_string=normalization_type_string,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    new_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY] = (
        new_vector_predictor_matrix
    )

    if apply_to_vector_targets:
        vector_target_names = (
            new_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
        )
    else:
        vector_target_names = []

    new_vector_target_matrix = (
        new_example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
    )

    for k in range(len(vector_target_names)):
        if separate_heights:
            for j in range(num_heights):
                these_training_values = example_utils.get_field_from_dict(
                    example_dict=training_example_dict,
                    field_name=vector_target_names[k],
                    height_m_agl=heights_m_agl[j]
                )

                new_vector_target_matrix[..., j, k] = _denorm_one_variable(
                    normalized_values_new=new_vector_target_matrix[..., j, k],
                    orig_values_training=these_training_values,
                    normalization_type_string=normalization_type_string,
                    min_normalized_value=min_normalized_value,
                    max_normalized_value=max_normalized_value
                )
        else:
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_target_names[k]
            )

            new_vector_target_matrix[..., k] = _denorm_one_variable(
                normalized_values_new=new_vector_target_matrix[..., k],
                orig_values_training=these_training_values,
                normalization_type_string=normalization_type_string,
                min_normalized_value=min_normalized_value,
                max_normalized_value=max_normalized_value
            )

    new_example_dict[example_utils.VECTOR_TARGET_VALS_KEY] = (
        new_vector_target_matrix
    )
    return new_example_dict


def create_mean_example(new_example_dict, training_example_dict):
    """Creates mean example (with mean value for each variable/height pair).

    :param new_example_dict: See doc for `normalize_data`.
    :param training_example_dict: Same.
    :return: mean_example_dict: See doc for `example_utils.average_examples`.
    """

    scalar_predictor_names = (
        new_example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )
    scalar_target_names = (
        new_example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    )
    vector_predictor_names = (
        new_example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )
    vector_target_names = (
        new_example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = new_example_dict[example_utils.HEIGHTS_KEY]

    num_scalar_predictors = len(scalar_predictor_names)
    num_scalar_targets = len(scalar_target_names)
    num_vector_predictors = len(vector_predictor_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)

    mean_scalar_predictor_values = numpy.full(num_scalar_predictors, numpy.nan)
    mean_scalar_target_values = numpy.full(num_scalar_targets, numpy.nan)
    mean_vector_predictor_matrix = numpy.full(
        (num_heights, num_vector_predictors), numpy.nan
    )
    mean_vector_target_matrix = numpy.full(
        (num_heights, num_vector_targets), numpy.nan
    )

    for k in range(num_scalar_predictors):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_predictor_names[k]
        )
        mean_scalar_predictor_values[k] = numpy.mean(these_training_values)

    for k in range(num_scalar_targets):
        these_training_values = example_utils.get_field_from_dict(
            example_dict=training_example_dict,
            field_name=scalar_target_names[k]
        )
        mean_scalar_target_values[k] = numpy.mean(these_training_values)

    for j in range(num_heights):
        for k in range(num_vector_predictors):
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_predictor_names[k],
                height_m_agl=heights_m_agl[j]
            )
            mean_vector_predictor_matrix[j, k] = numpy.mean(
                these_training_values
            )

        for k in range(num_vector_targets):
            these_training_values = example_utils.get_field_from_dict(
                example_dict=training_example_dict,
                field_name=vector_target_names[k],
                height_m_agl=heights_m_agl[j]
            )
            mean_vector_target_matrix[j, k] = numpy.mean(these_training_values)

    return {
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        example_utils.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        example_utils.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_utils.HEIGHTS_KEY: heights_m_agl,
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(mean_scalar_predictor_values, axis=0),
        example_utils.SCALAR_TARGET_VALS_KEY:
            numpy.expand_dims(mean_scalar_target_values, axis=0),
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            numpy.expand_dims(mean_vector_predictor_matrix, axis=0),
        example_utils.VECTOR_TARGET_VALS_KEY:
            numpy.expand_dims(mean_vector_target_matrix, axis=0)
    }
