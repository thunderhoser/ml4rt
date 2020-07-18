"""Applies trained neural net in inference mode."""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

TIME_FORMAT = '%Y-%m-%d-%H%M%S'
NUM_EXAMPLES_PER_BATCH = 5000
TARGET_VALUE_KEYS = [
    example_io.SCALAR_TARGET_VALS_KEY, example_io.VECTOR_TARGET_VALS_KEY
]

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with data examples.  Files therein will be found by '
    '`example_io.find_file` and read by `example_io.read_file`.'
)
TIME_HELP_STRING = (
    'Time (format "yyyy-mm-dd-HHMMSS").  The neural net will be applied only to'
    ' examples from `{0:s}` to `{1:s}`.'
).format(FIRST_TIME_ARG_NAME, LAST_TIME_ARG_NAME)

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
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_dir_name, first_time_string, last_time_string,
         output_file_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
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

    is_loss_constrained_mse = neural_net.determine_if_loss_constrained_mse(
        metadata_dict[neural_net.LOSS_FUNCTION_KEY]
    )

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = NUM_EXAMPLES_PER_BATCH
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    target_norm_type_string = copy.deepcopy(
        generator_option_dict[neural_net.TARGET_NORM_TYPE_KEY]
    )
    generator_option_dict[neural_net.TARGET_NORM_TYPE_KEY] = None

    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]
    generator = neural_net.data_generator(
        option_dict=generator_option_dict, for_inference=True,
        net_type_string=net_type_string, is_loss_constrained_mse=False
    )

    print(SEPARATOR_STRING)

    dummy_example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    add_heating_rate = generator_option_dict[neural_net.OMIT_HEATING_RATE_KEY]
    generator_option_dict_unnorm = copy.deepcopy(generator_option_dict)
    generator_option_dict_unnorm[neural_net.PREDICTOR_NORM_TYPE_KEY] = None

    scalar_target_matrix = None
    scalar_prediction_matrix = None
    vector_target_matrix = None
    vector_prediction_matrix = None
    example_id_strings = []

    vector_predictor_matrix_unnorm = None
    vector_target_matrix_unnorm = None

    while True:
        this_scalar_target_matrix = None
        this_scalar_prediction_matrix = None
        this_vector_target_matrix = None
        this_vector_prediction_matrix = None

        try:
            this_predictor_matrix, this_target_array, these_id_strings = (
                next(generator)
            )
        except (RuntimeError, StopIteration):
            break

        this_prediction_array = neural_net.apply_model(
            model_object=model_object, predictor_matrix=this_predictor_matrix,
            num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
            net_type_string=net_type_string,
            is_loss_constrained_mse=is_loss_constrained_mse, verbose=True
        )
        print(SEPARATOR_STRING)

        example_id_strings += these_id_strings

        if net_type_string == neural_net.CNN_TYPE_STRING:
            this_vector_target_matrix = this_target_array[0]
            this_vector_prediction_matrix = this_prediction_array[0]

            if len(this_target_array) == 2:
                this_scalar_target_matrix = this_target_array[1]
                this_scalar_prediction_matrix = this_prediction_array[1]

        elif net_type_string == neural_net.DENSE_NET_TYPE_STRING:
            this_scalar_target_matrix = this_target_array
            this_scalar_prediction_matrix = this_prediction_array[0]
        else:
            this_vector_target_matrix = this_target_array
            this_vector_prediction_matrix = this_prediction_array[0]

        if this_scalar_target_matrix is not None:
            if scalar_target_matrix is None:
                scalar_target_matrix = this_scalar_target_matrix + 0.
                scalar_prediction_matrix = this_scalar_prediction_matrix + 0.
            else:
                scalar_target_matrix = numpy.concatenate(
                    (scalar_target_matrix, this_scalar_target_matrix), axis=0
                )
                scalar_prediction_matrix = numpy.concatenate(
                    (scalar_prediction_matrix, this_scalar_prediction_matrix),
                    axis=0
                )

        if this_vector_target_matrix is not None:
            if vector_target_matrix is None:
                vector_target_matrix = this_vector_target_matrix + 0.
                vector_prediction_matrix = this_vector_prediction_matrix + 0.
            else:
                vector_target_matrix = numpy.concatenate(
                    (vector_target_matrix, this_vector_target_matrix), axis=0
                )
                vector_prediction_matrix = numpy.concatenate(
                    (vector_prediction_matrix, this_vector_prediction_matrix),
                    axis=0
                )

        if add_heating_rate:
            this_generator = neural_net.data_generator_specific_examples(
                option_dict=generator_option_dict_unnorm,
                net_type_string=net_type_string,
                example_id_strings=these_id_strings
            )
            this_predictor_matrix_unnorm, these_target_matrices_unnorm = (
                next(this_generator)
            )

            if not isinstance(these_target_matrices_unnorm, list):
                these_target_matrices_unnorm = [these_target_matrices_unnorm]

            this_example_dict = neural_net.predictors_numpy_to_dict(
                predictor_matrix=this_predictor_matrix_unnorm,
                example_dict=dummy_example_dict, net_type_string=net_type_string
            )
            this_vector_predictor_matrix_unnorm = (
                this_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
            )

            this_example_dict = neural_net.targets_numpy_to_dict(
                target_matrices=these_target_matrices_unnorm,
                example_dict=dummy_example_dict, net_type_string=net_type_string
            )
            this_vector_target_matrix_unnorm = (
                this_example_dict[example_io.VECTOR_TARGET_VALS_KEY]
            )

            if vector_predictor_matrix_unnorm is None:
                vector_predictor_matrix_unnorm = (
                    this_vector_predictor_matrix_unnorm + 0.
                )
                vector_target_matrix_unnorm = (
                    this_vector_target_matrix_unnorm + 0.
                )
            else:
                vector_predictor_matrix_unnorm = numpy.concatenate((
                    vector_predictor_matrix_unnorm,
                    vector_predictor_matrix_unnorm
                ), axis=0)

                vector_target_matrix_unnorm = numpy.concatenate((
                    vector_target_matrix_unnorm,
                    this_vector_target_matrix_unnorm
                ), axis=0)

    # TODO(thunderhoser): Do heating-rate bullshit here.
    if add_heating_rate:
        num_examples = vector_target_matrix_unnorm.shape[0]
        dummy_times_unix_sec = numpy.full(num_examples, 0, dtype=int)

        this_example_dict = {
            example_io.VECTOR_PREDICTOR_NAMES_KEY:
                generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
            example_io.VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix_unnorm,
            example_io.VECTOR_TARGET_NAMES_KEY:
                generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
            example_io.VECTOR_TARGET_VALS_KEY: vector_target_matrix_unnorm,
            example_io.VALID_TIMES_KEY: dummy_times_unix_sec,
            example_io.HEIGHTS_KEY:
                generator_option_dict[neural_net.HEIGHTS_KEY]
        }

        this_example_dict = example_io.fluxes_to_heating_rate(this_example_dict)

    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )
    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

    print('Denormalizing target (actual) and predicted values...')

    # TODO(thunderhoser): Put this code in `neural_net.targets_numpy_to_dict`.
    if net_type_string == neural_net.CNN_TYPE_STRING:
        these_target_matrices = [vector_target_matrix]

        if scalar_target_matrix is not None:
            these_target_matrices.append(scalar_target_matrix)

    elif net_type_string == neural_net.U_NET_TYPE_STRING:
        these_target_matrices = [vector_target_matrix]
    else:
        these_target_matrices = [scalar_target_matrix]

    new_example_dict = neural_net.targets_numpy_to_dict(
        target_matrices=these_target_matrices,
        example_dict=dummy_example_dict, net_type_string=net_type_string
    )

    target_example_dict = copy.deepcopy(dummy_example_dict)
    for this_key in TARGET_VALUE_KEYS:
        target_example_dict[this_key] = new_example_dict[this_key]

    # TODO(thunderhoser): Put this code in `neural_net.targets_numpy_to_dict`.
    if net_type_string == neural_net.CNN_TYPE_STRING:
        these_prediction_matrices = [vector_prediction_matrix]

        if scalar_prediction_matrix is not None:
            these_prediction_matrices.append(scalar_prediction_matrix)

    elif net_type_string == neural_net.U_NET_TYPE_STRING:
        these_prediction_matrices = [vector_prediction_matrix]
    else:
        these_prediction_matrices = [scalar_prediction_matrix]

    new_example_dict = neural_net.targets_numpy_to_dict(
        target_matrices=these_prediction_matrices,
        example_dict=dummy_example_dict, net_type_string=net_type_string
    )

    prediction_example_dict = copy.deepcopy(dummy_example_dict)
    for this_key in TARGET_VALUE_KEYS:
        prediction_example_dict[this_key] = new_example_dict[this_key]

    prediction_example_dict = normalization.denormalize_data(
        new_example_dict=prediction_example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=target_norm_type_string,
        min_normalized_value=
        generator_option_dict[neural_net.TARGET_MIN_NORM_VALUE_KEY],
        max_normalized_value=
        generator_option_dict[neural_net.TARGET_MAX_NORM_VALUE_KEY],
        separate_heights=True, apply_to_predictors=False,
        apply_to_targets=True
    )

    print('Writing target (actual) and predicted values to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=
        target_example_dict[example_io.SCALAR_TARGET_VALS_KEY],
        vector_target_matrix=
        target_example_dict[example_io.VECTOR_TARGET_VALS_KEY],
        scalar_prediction_matrix=
        prediction_example_dict[example_io.SCALAR_TARGET_VALS_KEY],
        vector_prediction_matrix=
        prediction_example_dict[example_io.VECTOR_TARGET_VALS_KEY],
        example_id_strings=example_id_strings, model_file_name=model_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
