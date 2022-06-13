"""Applies trained neural net in inference mode."""

import os
import sys
import copy
import time
import warnings
import argparse
import numpy
import netCDF4

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import file_system_utils
import example_io
import example_utils
import normalization
import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
NUM_EXAMPLES_PER_BATCH = 500

TARGET_VALUE_KEYS = [
    example_utils.SCALAR_TARGET_VALS_KEY, example_utils.VECTOR_TARGET_VALS_KEY
]

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
PREDICTOR_DIMENSION_KEY = 'predictor_variable'
VECTOR_TARGET_DIMENSION_KEY = 'vector_target_variable'
SCALAR_TARGET_DIMENSION_KEY = 'scalar_target_variable'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
VECTOR_PREDICTION_MATRIX_KEY = 'vector_prediction_matrix'
SCALAR_PREDICTION_MATRIX_KEY = 'scalar_prediction_matrix'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
EXAMPLE_DIR_FOR_PRESSURE_ARG_NAME = 'input_example_dir_name_for_pressure'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
EXCLUDE_SUMMIT_ARG_NAME = 'exclude_summit_greenland'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
EXAMPLE_DIR_FOR_PRESSURE_HELP_STRING = (
    'Name of directory containing data examples with unnormalized pressure.  '
    'This will be used only if the neural net predicts heating rate * layer '
    'thickness.  Files therein will be found by `example_io.find_file` and '
    'read by `example_io.read_file`.'
)
TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The neural net will be applied only to'
    ' examples from `{0:s}` to `{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

EXCLUDE_SUMMIT_HELP_STRING = (
    'Boolean flag.  If 1, will not apply to examples from Summit.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (will be written by `prediction_io.write_file`).'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_FOR_PRESSURE_ARG_NAME, type=str, required=False,
    default='', help=EXAMPLE_DIR_FOR_PRESSURE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXCLUDE_SUMMIT_ARG_NAME, type=int, required=False, default=0,
    help=EXCLUDE_SUMMIT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _write_predictors_and_predictions(
        netcdf_file_name, predictor_matrix, vector_prediction_matrix,
        scalar_prediction_matrix):
    """Writes predictors and predictions to NetCDF file.

    :param netcdf_file_name: Path to output file.
    :param predictor_matrix: numpy array with predictors.
    :param vector_prediction_matrix: numpy array with vector predictions.
    :param scalar_prediction_matrix: numpy array with scalar predictions.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(
        EXAMPLE_DIMENSION_KEY, predictor_matrix.shape[0]
    )
    dataset_object.createDimension(
        HEIGHT_DIMENSION_KEY, predictor_matrix.shape[1]
    )
    dataset_object.createDimension(
        PREDICTOR_DIMENSION_KEY, predictor_matrix.shape[2]
    )
    dataset_object.createDimension(
        VECTOR_TARGET_DIMENSION_KEY, vector_prediction_matrix.shape[2]
    )
    dataset_object.createDimension(
        SCALAR_TARGET_DIMENSION_KEY, scalar_prediction_matrix.shape[1]
    )

    these_dimensions = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY, PREDICTOR_DIMENSION_KEY
    )
    dataset_object.createVariable(
        PREDICTOR_MATRIX_KEY, datatype=numpy.float32,
        dimensions=these_dimensions
    )
    dataset_object.variables[PREDICTOR_MATRIX_KEY][:] = predictor_matrix

    these_dimensions = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY, VECTOR_TARGET_DIMENSION_KEY
    )
    dataset_object.createVariable(
        VECTOR_PREDICTION_MATRIX_KEY, datatype=numpy.float32,
        dimensions=these_dimensions
    )
    dataset_object.variables[VECTOR_PREDICTION_MATRIX_KEY][:] = (
        vector_prediction_matrix
    )

    these_dimensions = (EXAMPLE_DIMENSION_KEY, SCALAR_TARGET_DIMENSION_KEY)
    dataset_object.createVariable(
        SCALAR_PREDICTION_MATRIX_KEY, datatype=numpy.float32,
        dimensions=these_dimensions
    )
    dataset_object.variables[SCALAR_PREDICTION_MATRIX_KEY][:] = (
        scalar_prediction_matrix
    )

    dataset_object.close()


def _targets_numpy_to_dict(
        scalar_target_matrix, vector_target_matrix, model_metadata_dict):
    """Converts either actual or predicted target values to dictionary.

    This method is a wrapper for `neural_net.targets_numpy_to_dict`.

    :param scalar_target_matrix: numpy array with scalar outputs (either
        predicted or actual target values).
    :param vector_target_matrix: Same but with vector outputs.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :return: example_dict: Equivalent dictionary, with keys listed in doc for
        `example_io.read_file`.
    """

    net_type_string = model_metadata_dict[neural_net.NET_TYPE_KEY]

    generator_option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )

    target_matrices = [vector_target_matrix]
    if scalar_target_matrix is not None:
        target_matrices.append(scalar_target_matrix)

    example_dict = {
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_utils.HEIGHTS_KEY:
            generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    new_example_dict = neural_net.targets_numpy_to_dict(
        target_matrices=target_matrices,
        example_dict=example_dict, net_type_string=net_type_string
    )
    for this_key in TARGET_VALUE_KEYS:
        example_dict[this_key] = new_example_dict[this_key]

    return example_dict


def _run(model_file_name, example_dir_name, example_dir_name_for_pressure,
         first_time_string, last_time_string, exclude_summit_greenland,
         output_file_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param example_dir_name_for_pressure: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param exclude_summit_greenland: Same.
    :param output_file_name: Same.
    """

    first_time_unix_sec = time_conversion.string_to_unix_sec(
        first_time_string, TIME_FORMAT
    )
    last_time_unix_sec = time_conversion.string_to_unix_sec(
        last_time_string, TIME_FORMAT
    )

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    generator_option_dict = copy.deepcopy(
        metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )
    joined_output_layer = copy.deepcopy(
        generator_option_dict[neural_net.JOINED_OUTPUT_LAYER_KEY]
    )
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec
    generator_option_dict[neural_net.JOINED_OUTPUT_LAYER_KEY] = False
    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]

    pressure_matrix_pa = numpy.array([])
    pressure_id_strings = []

    if generator_option_dict[neural_net.MULTIPLY_HR_BY_THICKNESS_KEY]:
        new_option_dict = copy.deepcopy(generator_option_dict)
        new_option_dict[neural_net.NORMALIZATION_FILE_KEY] = None
        new_option_dict[neural_net.PREDICTOR_NORM_TYPE_KEY] = None
        new_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY] = None
        new_option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY] = None
        new_option_dict[neural_net.MULTIPLY_PREDS_BY_THICKNESS_KEY] = False
        new_option_dict[neural_net.MULTIPLY_HR_BY_THICKNESS_KEY] = False
        new_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY] = [
            example_utils.PRESSURE_NAME
        ]
        new_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
            example_dir_name_for_pressure
        )

        pressure_matrix_pa, _, pressure_id_strings = neural_net.create_data(
            option_dict=new_option_dict,
            net_type_string=net_type_string,
            exclude_summit_greenland=exclude_summit_greenland
        )
        print(SEPARATOR_STRING)

        pressure_id_strings, unique_indices = numpy.unique(
            numpy.array(pressure_id_strings), return_index=True
        )
        pressure_id_strings = pressure_id_strings.tolist()
        pressure_matrix_pa = pressure_matrix_pa[unique_indices, ..., 0]

    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    predictor_matrix, target_array, example_id_strings = neural_net.create_data(
        option_dict=generator_option_dict,
        net_type_string=net_type_string,
        exclude_summit_greenland=exclude_summit_greenland
    )
    print(SEPARATOR_STRING)

    example_id_strings, unique_indices = numpy.unique(
        numpy.array(example_id_strings), return_index=True
    )
    example_id_strings = example_id_strings.tolist()
    predictor_matrix = predictor_matrix[unique_indices, ...]

    if isinstance(target_array, list):
        for k in range(len(target_array)):
            target_array[k] = target_array[k][unique_indices, ...]
    else:
        target_array = target_array[unique_indices, ...]

    if generator_option_dict[neural_net.MULTIPLY_HR_BY_THICKNESS_KEY]:
        good_pressure_indices = example_utils.find_examples(
            all_id_strings=pressure_id_strings,
            desired_id_strings=example_id_strings, allow_missing=True
        )

        if numpy.any(good_pressure_indices < 0):
            warning_string = (
                'POTENTIAL ERROR: Could not find pressure profile for {0:d} of '
                '{1:d} examples.'
            ).format(
                numpy.sum(good_pressure_indices < 0),
                len(good_pressure_indices)
            )

            warnings.warn(warning_string)

        good_overall_indices = numpy.where(good_pressure_indices >= 0)[0]
        good_pressure_indices = good_pressure_indices[good_overall_indices]

        pressure_matrix_pa = pressure_matrix_pa[good_pressure_indices, :]
        predictor_matrix = predictor_matrix[good_overall_indices, ...]
        example_id_strings = [
            example_id_strings[k] for k in good_overall_indices
        ]

        if isinstance(target_array, list):
            for k in range(len(target_array)):
                target_array[k] = target_array[k][good_overall_indices, ...]
        else:
            target_array = target_array[good_overall_indices, ...]

    exec_start_time_unix_sec = time.time()
    prediction_array = neural_net.apply_model(
        model_object=model_object, predictor_matrix=predictor_matrix,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        net_type_string=net_type_string, verbose=True
    )

    if joined_output_layer:
        num_scalar_targets = len(
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
        )

        vector_prediction_matrix = numpy.expand_dims(
            prediction_array[0][:, :-num_scalar_targets], axis=-1
        )
        scalar_prediction_matrix = prediction_array[0][:, -num_scalar_targets:]
        prediction_array = [vector_prediction_matrix, scalar_prediction_matrix]

    print(SEPARATOR_STRING)
    print('Time to apply neural net = {0:.4f} seconds'.format(
        time.time() - exec_start_time_unix_sec
    ))

    vector_target_matrix = target_array[0]
    vector_prediction_matrix = prediction_array[0]

    if len(target_array) == 2:
        scalar_target_matrix = target_array[1]
        scalar_prediction_matrix = prediction_array[1]
    else:
        scalar_target_matrix = None
        scalar_prediction_matrix = None

    target_example_dict = _targets_numpy_to_dict(
        scalar_target_matrix=scalar_target_matrix,
        vector_target_matrix=vector_target_matrix,
        model_metadata_dict=metadata_dict
    )

    prediction_example_dict = _targets_numpy_to_dict(
        scalar_target_matrix=scalar_prediction_matrix,
        vector_target_matrix=vector_prediction_matrix,
        model_metadata_dict=metadata_dict
    )

    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )
    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict,
        heights_m_agl=generator_option_dict[neural_net.HEIGHTS_KEY]
    )

    num_examples = len(example_id_strings)
    num_heights = len(prediction_example_dict[example_utils.HEIGHTS_KEY])

    this_dict = {
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: [],
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, num_heights, 0), 0.),
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, 0), 0.)
    }

    target_example_dict.update(this_dict)
    prediction_example_dict.update(this_dict)

    if (
            generator_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY]
            is not None
    ):
        print('Denormalizing predicted and actual vectors...')

        prediction_example_dict = normalization.denormalize_data(
            new_example_dict=prediction_example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=
            generator_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY],
            min_normalized_value=
            generator_option_dict[neural_net.VECTOR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=
            generator_option_dict[neural_net.VECTOR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=False
        )

        target_example_dict = normalization.denormalize_data(
            new_example_dict=target_example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=
            generator_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY],
            min_normalized_value=
            generator_option_dict[neural_net.VECTOR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=
            generator_option_dict[neural_net.VECTOR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=False
        )

    if (
            generator_option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY]
            is not None
    ):
        print('Denormalizing predicted and actual scalars...')

        prediction_example_dict = normalization.denormalize_data(
            new_example_dict=prediction_example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=
            generator_option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY],
            min_normalized_value=
            generator_option_dict[neural_net.SCALAR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=
            generator_option_dict[neural_net.SCALAR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

        target_example_dict = normalization.denormalize_data(
            new_example_dict=target_example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=
            generator_option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY],
            min_normalized_value=
            generator_option_dict[neural_net.SCALAR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=
            generator_option_dict[neural_net.SCALAR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    if generator_option_dict[neural_net.MULTIPLY_HR_BY_THICKNESS_KEY]:
        hr_indices = []

        try:
            hr_indices.append(vector_target_names.index(
                example_utils.SHORTWAVE_HEATING_RATE_NAME
            ))
        except ValueError:
            pass

        try:
            hr_indices.append(vector_target_names.index(
                example_utils.LONGWAVE_HEATING_RATE_NAME
            ))
        except ValueError:
            pass

        dummy_example_dict = {
            example_utils.VECTOR_TARGET_NAMES_KEY: [
                vector_target_names[k] for k in hr_indices
            ],
            example_utils.VECTOR_TARGET_VALS_KEY:
                target_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][
                    ..., hr_indices
                ],
            example_utils.VECTOR_PREDICTOR_NAMES_KEY: [
                example_utils.PRESSURE_NAME
            ],
            example_utils.VECTOR_PREDICTOR_VALS_KEY: numpy.expand_dims(
                pressure_matrix_pa, axis=-1
            )
        }
        dummy_example_dict = example_utils.divide_hr_by_layer_thickness(
            dummy_example_dict
        )
        target_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][
            ..., hr_indices
        ] = dummy_example_dict[example_utils.VECTOR_TARGET_VALS_KEY]

        dummy_example_dict = {
            example_utils.VECTOR_TARGET_NAMES_KEY: [
                vector_target_names[k] for k in hr_indices
            ],
            example_utils.VECTOR_TARGET_VALS_KEY:
                prediction_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][
                    ..., hr_indices
                ],
            example_utils.VECTOR_PREDICTOR_NAMES_KEY: [
                example_utils.PRESSURE_NAME
            ],
            example_utils.VECTOR_PREDICTOR_VALS_KEY: numpy.expand_dims(
                pressure_matrix_pa, axis=-1
            )
        }
        dummy_example_dict = example_utils.divide_hr_by_layer_thickness(
            dummy_example_dict
        )
        prediction_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][
            ..., hr_indices
        ] = dummy_example_dict[example_utils.VECTOR_TARGET_VALS_KEY]

    for this_target_name in [
            example_utils.SHORTWAVE_HEATING_RATE_NAME,
            example_utils.LONGWAVE_HEATING_RATE_NAME
    ]:
        try:
            k = vector_target_names.index(this_target_name)
        except ValueError:
            continue

        target_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][:, -1, k] = 0.
        prediction_example_dict[
            example_utils.VECTOR_TARGET_VALS_KEY
        ][:, -1, k] = 0.

    print('Writing predictors and predictions to: "{0:s}"...'.format(
        output_file_name
    ))
    _write_predictors_and_predictions(
        netcdf_file_name=output_file_name,
        predictor_matrix=predictor_matrix[:10, ...],
        vector_prediction_matrix=
        prediction_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][:10, ...],
        scalar_prediction_matrix=
        prediction_example_dict[example_utils.SCALAR_TARGET_VALS_KEY][:10, ...]
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_dir_name_for_pressure=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_DIR_FOR_PRESSURE_ARG_NAME
        ),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        exclude_summit_greenland=bool(getattr(
            INPUT_ARG_OBJECT, EXCLUDE_SUMMIT_ARG_NAME
        )),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
