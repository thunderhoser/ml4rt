"""Applies trained neural net in inference mode."""

import copy
import os.path
import argparse
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io
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
    'Path to output file (will be written by `neural_net.write_predictions`).'
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
    metadata_dict = neural_net.read_metadata(metafile_name)

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = NUM_EXAMPLES_PER_BATCH
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    # TODO(thunderhoser): Allow generators to return example IDs.

    scalar_target_matrix = None
    scalar_prediction_matrix = None
    vector_target_matrix = None
    vector_prediction_matrix = None
    example_id_strings = []

    is_cnn = metadata_dict[neural_net.IS_CNN_KEY]

    if is_cnn:
        generator = neural_net.cnn_generator(
            option_dict=generator_option_dict, for_inference=True
        )
    else:
        generator = neural_net.dense_net_generator(
            option_dict=generator_option_dict, for_inference=True
        )

    print(SEPARATOR_STRING)

    while True:
        this_vector_target_matrix = None
        this_vector_prediction_matrix = None

        try:
            if is_cnn:
                this_predictor_matrix, this_target_array, these_id_strings = (
                    next(generator)
                )
            else:
                this_predictor_matrix, this_target_array, these_id_strings = (
                    next(generator)
                )
        except StopIteration:
            break

        print(this_predictor_matrix.shape)
        print(len(these_id_strings))

        example_id_strings += these_id_strings

        if is_cnn:
            this_vector_target_matrix = this_target_array[0]
            this_scalar_target_matrix = this_target_array[1]

            this_vector_prediction_matrix, this_scalar_prediction_matrix = (
                neural_net.apply_cnn(
                    model_object=model_object,
                    predictor_matrix=this_predictor_matrix,
                    num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
                )
            )
        else:
            this_scalar_target_matrix = this_target_array

            this_scalar_prediction_matrix = neural_net.apply_dense_net(
                model_object=model_object,
                predictor_matrix=this_predictor_matrix,
                num_examples_per_batch=NUM_EXAMPLES_PER_BATCH, verbose=True
            )

        print(SEPARATOR_STRING)

        if scalar_target_matrix is None:
            scalar_target_matrix = this_scalar_target_matrix + 0.
            scalar_prediction_matrix = this_scalar_prediction_matrix + 0.

            if is_cnn:
                vector_target_matrix = this_vector_target_matrix + 0.
                vector_prediction_matrix = this_vector_prediction_matrix + 0.

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
    target_example_dict = {
        example_io.SCALAR_PREDICTOR_VALS_KEY: None,
        example_io.VECTOR_PREDICTOR_VALS_KEY: None,
        example_io.SCALAR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    prediction_example_dict = copy.deepcopy(target_example_dict)

    new_example_dict = neural_net.targets_numpy_to_dict(
        target_matrices=[vector_target_matrix, scalar_target_matrix],
        example_dict=target_example_dict, for_cnn=is_cnn
    )

    for this_key in TARGET_VALUE_KEYS:
        target_example_dict[this_key] = new_example_dict[this_key]

    new_example_dict = neural_net.targets_numpy_to_dict(
        target_matrices=[vector_prediction_matrix, scalar_prediction_matrix],
        example_dict=prediction_example_dict, for_cnn=is_cnn
    )

    for this_key in TARGET_VALUE_KEYS:
        prediction_example_dict[this_key] = new_example_dict[this_key]

    target_example_dict = normalization.denormalize_data(
        new_example_dict=target_example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=
        generator_option_dict[neural_net.TARGET_NORM_TYPE_KEY],
        min_normalized_value=
        generator_option_dict[neural_net.TARGET_MIN_NORM_VALUE_KEY],
        max_normalized_value=
        generator_option_dict[neural_net.TARGET_MAX_NORM_VALUE_KEY],
        separate_heights=True, apply_to_predictors=False,
        apply_to_targets=True
    )

    prediction_example_dict = normalization.denormalize_data(
        new_example_dict=prediction_example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=
        generator_option_dict[neural_net.TARGET_NORM_TYPE_KEY],
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

    neural_net.write_predictions(
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
