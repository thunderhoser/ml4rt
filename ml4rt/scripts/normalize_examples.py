"""Normalizes learning examples."""

import os.path
import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.utils import normalization
from ml4rt.utils import example_utils

INPUT_FILE_ARG_NAME = 'input_example_file_name'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
UNIFORMIZE_ARG_NAME = 'uniformize'
MULTIPLY_PREDICTORS_ARG_NAME = 'multiply_preds_by_layer_thickness'
MULTIPLY_HEATING_RATES_ARG_NAME = 'multiply_hr_by_layer_thickness'
PREDICTOR_NORM_TYPE_ARG_NAME = 'predictor_norm_type_string'
PREDICTOR_MIN_VALUE_ARG_NAME = 'predictor_min_norm_value'
PREDICTOR_MAX_VALUE_ARG_NAME = 'predictor_max_norm_value'
VECTOR_TARGET_NORM_TYPE_ARG_NAME = 'vector_target_norm_type_string'
VECTOR_TARGET_MIN_VALUE_ARG_NAME = 'vector_target_min_norm_value'
VECTOR_TARGET_MAX_VALUE_ARG_NAME = 'vector_target_max_norm_value'
SCALAR_TARGET_NORM_TYPE_ARG_NAME = 'scalar_target_norm_type_string'
SCALAR_TARGET_MIN_VALUE_ARG_NAME = 'scalar_target_min_norm_value'
SCALAR_TARGET_MAX_VALUE_ARG_NAME = 'scalar_target_max_norm_value'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to file with unnormalized examples.  Will be read by '
    '`example_io.read_file`.'
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file, containing unnormalized sample values that '
    'will be used to create uniform distributions.  Will be read by '
    '`example_io.read_file`.'
)
UNIFORMIZE_HELP_STRING = (
    'Boolean flag.  If 1, will convert each variable to uniform distribution '
    'and then z-scores.  If 0, will convert directly to z-scores.'
)
MULTIPLY_PREDICTORS_HELP_STRING = (
    'Boolean flag.  If 1, predictors will be multiplied by layer thickness '
    'before normalization.'
)
MULTIPLY_HEATING_RATES_HELP_STRING = (
    'Boolean flag.  If 1, heating rates be multiplied by layer thickness '
    'before normalization.'
)
PREDICTOR_NORM_TYPE_HELP_STRING = (
    'Normalization type for predictors (must be accepted by '
    '`normalization.check_normalization_type`).  If you do not want to '
    'normalize, make this an empty string ("").'
)
PREDICTOR_MIN_VALUE_HELP_STRING = (
    'Minimum value if you have chosen min-max normalization.'
)
PREDICTOR_MAX_VALUE_HELP_STRING = (
    'Max value if you have chosen min-max normalization.'
)
VECTOR_TARGET_NORM_TYPE_HELP_STRING = (
    'Same as `{0:s}` but for vector target variables.'
).format(PREDICTOR_NORM_TYPE_ARG_NAME)

VECTOR_TARGET_MIN_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for vector target variables.'
).format(PREDICTOR_MIN_VALUE_ARG_NAME)

VECTOR_TARGET_MAX_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for vector target variables.'
).format(PREDICTOR_MAX_VALUE_ARG_NAME)

SCALAR_TARGET_NORM_TYPE_HELP_STRING = (
    'Same as `{0:s}` but for scalar target variables.'
).format(PREDICTOR_NORM_TYPE_ARG_NAME)

SCALAR_TARGET_MIN_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for scalar target variables.'
).format(PREDICTOR_MIN_VALUE_ARG_NAME)

SCALAR_TARGET_MAX_VALUE_HELP_STRING = (
    'Same as `{0:s}` but for scalar target variables.'
).format(PREDICTOR_MAX_VALUE_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Normalized examples will be written here by '
    '`example_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + UNIFORMIZE_ARG_NAME, type=int, required=True,
    help=UNIFORMIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MULTIPLY_PREDICTORS_ARG_NAME, type=int, required=False, default=0,
    help=MULTIPLY_PREDICTORS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MULTIPLY_HEATING_RATES_ARG_NAME, type=int, required=False, default=0,
    help=MULTIPLY_HEATING_RATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_NORM_TYPE_ARG_NAME, type=str, required=False,
    default=normalization.Z_SCORE_NORM_STRING,
    help=PREDICTOR_NORM_TYPE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_MIN_VALUE_ARG_NAME, type=float, required=False,
    default=0., help=PREDICTOR_MIN_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTOR_MAX_VALUE_ARG_NAME, type=float, required=False,
    default=1., help=PREDICTOR_MAX_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VECTOR_TARGET_NORM_TYPE_ARG_NAME, type=str, required=True,
    help=VECTOR_TARGET_NORM_TYPE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VECTOR_TARGET_MIN_VALUE_ARG_NAME, type=float, required=False,
    default=0., help=VECTOR_TARGET_MIN_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VECTOR_TARGET_MAX_VALUE_ARG_NAME, type=float, required=False,
    default=1., help=VECTOR_TARGET_MAX_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SCALAR_TARGET_NORM_TYPE_ARG_NAME, type=str, required=True,
    help=SCALAR_TARGET_NORM_TYPE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SCALAR_TARGET_MIN_VALUE_ARG_NAME, type=float, required=False,
    default=0., help=SCALAR_TARGET_MIN_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SCALAR_TARGET_MAX_VALUE_ARG_NAME, type=float, required=False,
    default=1., help=SCALAR_TARGET_MAX_VALUE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_file_name, normalization_file_name, uniformize,
         multiply_preds_by_layer_thickness, multiply_hr_by_layer_thickness,
         predictor_norm_type_string, predictor_min_norm_value,
         predictor_max_norm_value, vector_target_norm_type_string,
         vector_target_min_norm_value, vector_target_max_norm_value,
         scalar_target_norm_type_string, scalar_target_min_norm_value,
         scalar_target_max_norm_value, output_example_dir_name):
    """Normalizes learning examples.

    This is effectively the main method.

    :param input_example_file_name: See documentation at top of file.
    :param normalization_file_name: Same.
    :param uniformize: Same.
    :param multiply_preds_by_layer_thickness: Same.
    :param multiply_hr_by_layer_thickness: Same.
    :param predictor_norm_type_string: Same.
    :param predictor_min_norm_value: Same.
    :param predictor_max_norm_value: Same.
    :param vector_target_norm_type_string: Same.
    :param vector_target_min_norm_value: Same.
    :param vector_target_max_norm_value: Same.
    :param scalar_target_norm_type_string: Same.
    :param scalar_target_min_norm_value: Same.
    :param scalar_target_max_norm_value: Same.
    :param output_example_dir_name: Same.
    """

    print('Reading unnormalized examples from: "{0:s}"...'.format(
        input_example_file_name
    ))
    example_dict = example_io.read_file(
        netcdf_file_name=input_example_file_name,
        exclude_summit_greenland=False,
        max_shortwave_heating_k_day01=numpy.inf,
        min_longwave_heating_k_day01=-1 * numpy.inf,
        max_longwave_heating_k_day01=numpy.inf
    )

    normalization_metadata_dict = (
        example_dict[example_utils.NORMALIZATION_METADATA_KEY]
    )
    assert (
        normalization_metadata_dict[example_io.NORMALIZATION_FILE_KEY] is None
    )

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    if multiply_preds_by_layer_thickness:
        example_dict = example_utils.multiply_preds_by_layer_thickness(
            example_dict
        )
        training_example_dict = example_utils.multiply_preds_by_layer_thickness(
            training_example_dict
        )

    if multiply_hr_by_layer_thickness:
        example_dict = example_utils.multiply_hr_by_layer_thickness(
            example_dict
        )
        training_example_dict = example_utils.multiply_hr_by_layer_thickness(
            training_example_dict
        )

    if predictor_norm_type_string == '':
        predictor_norm_type_string = None
    else:
        print('Applying {0:s} normalization to predictors...'.format(
            predictor_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=predictor_norm_type_string,
            uniformize=uniformize,
            min_normalized_value=predictor_min_norm_value,
            max_normalized_value=predictor_max_norm_value,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

    if vector_target_norm_type_string == '':
        vector_target_norm_type_string = None
    else:
        print('Applying {0:s} normalization to vector targets...'.format(
            vector_target_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=vector_target_norm_type_string,
            uniformize=uniformize,
            min_normalized_value=vector_target_min_norm_value,
            max_normalized_value=vector_target_max_norm_value,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=False
        )

    if scalar_target_norm_type_string == '':
        scalar_target_norm_type_string = None
    else:
        print('Applying {0:s} normalization to scalar targets...'.format(
            scalar_target_norm_type_string.upper()
        ))
        example_dict = normalization.normalize_data(
            new_example_dict=example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=scalar_target_norm_type_string,
            uniformize=uniformize,
            min_normalized_value=scalar_target_min_norm_value,
            max_normalized_value=scalar_target_max_norm_value,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

    normalization_metadata_dict = {
        example_io.NORMALIZATION_FILE_KEY: normalization_file_name,
        example_io.UNIFORMIZE_FLAG_KEY: uniformize,
        example_io.PREDICTOR_NORM_TYPE_KEY: predictor_norm_type_string,
        example_io.PREDICTOR_MIN_VALUE_KEY: predictor_min_norm_value,
        example_io.PREDICTOR_MAX_VALUE_KEY: predictor_max_norm_value,
        example_io.VECTOR_TARGET_NORM_TYPE_KEY: vector_target_norm_type_string,
        example_io.VECTOR_TARGET_MIN_VALUE_KEY: vector_target_min_norm_value,
        example_io.VECTOR_TARGET_MAX_VALUE_KEY: vector_target_max_norm_value,
        example_io.SCALAR_TARGET_NORM_TYPE_KEY: scalar_target_norm_type_string,
        example_io.SCALAR_TARGET_MIN_VALUE_KEY: scalar_target_min_norm_value,
        example_io.SCALAR_TARGET_MAX_VALUE_KEY: scalar_target_max_norm_value
    }
    example_dict[example_utils.NORMALIZATION_METADATA_KEY] = (
        normalization_metadata_dict
    )

    output_example_file_name = '{0:s}/{1:s}'.format(
        output_example_dir_name, os.path.split(input_example_file_name)[1]
    )
    print('Writing normalized examples to: "{0:s}"...'.format(
        output_example_file_name
    ))
    example_io.write_file(
        example_dict=example_dict, netcdf_file_name=output_example_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        uniformize=bool(getattr(INPUT_ARG_OBJECT, UNIFORMIZE_ARG_NAME)),
        multiply_preds_by_layer_thickness=bool(
            getattr(INPUT_ARG_OBJECT, MULTIPLY_PREDICTORS_ARG_NAME)
        ),
        multiply_hr_by_layer_thickness=bool(
            getattr(INPUT_ARG_OBJECT, MULTIPLY_HEATING_RATES_ARG_NAME)
        ),
        predictor_norm_type_string=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_NORM_TYPE_ARG_NAME
        ),
        predictor_min_norm_value=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_MIN_VALUE_ARG_NAME
        ),
        predictor_max_norm_value=getattr(
            INPUT_ARG_OBJECT, PREDICTOR_MAX_VALUE_ARG_NAME
        ),
        vector_target_norm_type_string=getattr(
            INPUT_ARG_OBJECT, VECTOR_TARGET_NORM_TYPE_ARG_NAME
        ),
        vector_target_min_norm_value=getattr(
            INPUT_ARG_OBJECT, VECTOR_TARGET_MIN_VALUE_ARG_NAME
        ),
        vector_target_max_norm_value=getattr(
            INPUT_ARG_OBJECT, VECTOR_TARGET_MAX_VALUE_ARG_NAME
        ),
        scalar_target_norm_type_string=getattr(
            INPUT_ARG_OBJECT, SCALAR_TARGET_NORM_TYPE_ARG_NAME
        ),
        scalar_target_min_norm_value=getattr(
            INPUT_ARG_OBJECT, SCALAR_TARGET_MIN_VALUE_ARG_NAME
        ),
        scalar_target_max_norm_value=getattr(
            INPUT_ARG_OBJECT, SCALAR_TARGET_MAX_VALUE_ARG_NAME
        ),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
