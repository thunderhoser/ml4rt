"""Applies trained neural net in inference mode."""

import os
import sys
import copy
import time
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import prediction_io
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

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
FIRST_TIME_ARG_NAME = 'first_time_string'
LAST_TIME_ARG_NAME = 'last_time_string'
NUM_DROPOUT_ITERS_ARG_NAME = 'num_dropout_iterations'
NUM_BNN_ITERS_ARG_NAME = 'num_bnn_iterations'
MAX_ENSEMBLE_SIZE_ARG_NAME = 'max_ensemble_size'
EXCLUDE_SUMMIT_ARG_NAME = 'exclude_summit_greenland'
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

NUM_DROPOUT_ITERS_HELP_STRING = (
    'Number of iterations for Monte Carlo dropout.  If you do not want to use '
    'MC dropout, make this argument <= 0.'
)
NUM_BNN_ITERS_HELP_STRING = (
    'Number of iterations for Bayesian neural net.  If the neural net is not '
    'Bayesian, make this argument <= 0.'
)
MAX_ENSEMBLE_SIZE_HELP_STRING = (
    'Max ensemble size.  If the NN does uncertainty quantification and yields '
    'more ensemble members, the desired number of members will be randomly '
    'selected.'
)
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
    '--' + FIRST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_TIME_ARG_NAME, type=str, required=True, help=TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DROPOUT_ITERS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_DROPOUT_ITERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_BNN_ITERS_ARG_NAME, type=int, required=False, default=0,
    help=NUM_BNN_ITERS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ENSEMBLE_SIZE_ARG_NAME, type=int, required=False, default=1e10,
    help=MAX_ENSEMBLE_SIZE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXCLUDE_SUMMIT_ARG_NAME, type=int, required=False, default=0,
    help=EXCLUDE_SUMMIT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


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


def _apply_model_once(model_object, model_metadata_dict, predictor_matrix,
                      use_dropout):
    """Applies model once.

    :param model_object: Trained instance of `keras.models.Model` or
        `keras.models.Sequential`.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metafile`.
    :param predictor_matrix: numpy array of predictors.
    :param use_dropout: Boolean flag.
    :return: vector_prediction_matrix: numpy array of predictions for profile-
        based target variables.
    :return: scalar_prediction_matrix: numpy array of predictions for scalar
        target variables.  If the model does not predict scalar target
        variables, this will be None.
    """

    generator_option_dict = copy.deepcopy(
        model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    )
    joined_output_layer = copy.deepcopy(
        generator_option_dict[neural_net.JOINED_OUTPUT_LAYER_KEY]
    )
    net_type_string = model_metadata_dict[neural_net.NET_TYPE_KEY]

    exec_start_time_unix_sec = time.time()
    prediction_array = neural_net.apply_model(
        model_object=model_object, predictor_matrix=predictor_matrix,
        num_examples_per_batch=NUM_EXAMPLES_PER_BATCH,
        net_type_string=net_type_string, use_dropout=use_dropout, verbose=True
    )

    print(SEPARATOR_STRING)
    print('Time to apply neural net = {0:.4f} seconds'.format(
        time.time() - exec_start_time_unix_sec
    ))

    if joined_output_layer:
        num_scalar_targets = len(
            generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
        )

        vector_prediction_matrix = numpy.expand_dims(
            prediction_array[0][..., :-num_scalar_targets, :], axis=-2
        )
        scalar_prediction_matrix = (
            prediction_array[0][..., -num_scalar_targets:, :]
        )
        prediction_array = [
            vector_prediction_matrix, scalar_prediction_matrix
        ]

    vector_prediction_matrix = prediction_array[0]
    if len(prediction_array) > 1:
        scalar_prediction_matrix = prediction_array[1]
    else:
        scalar_prediction_matrix = None

    return vector_prediction_matrix, scalar_prediction_matrix


def _run(model_file_name, example_dir_name, first_time_string, last_time_string,
         num_dropout_iterations, num_bnn_iterations, max_ensemble_size,
         exclude_summit_greenland, output_file_name):
    """Applies trained neural net in inference mode.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_dir_name: Same.
    :param first_time_string: Same.
    :param last_time_string: Same.
    :param num_dropout_iterations: Same.
    :param num_bnn_iterations: Same.
    :param max_ensemble_size: Same.
    :param exclude_summit_greenland: Same.
    :param output_file_name: Same.
    """

    max_ensemble_size = max([max_ensemble_size, 1])

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
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec
    generator_option_dict[neural_net.JOINED_OUTPUT_LAYER_KEY] = False
    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]

    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = example_dir_name
    generator_option_dict[neural_net.NUM_DEEP_SUPER_LAYERS_KEY] = 0

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

    if num_dropout_iterations > 1:
        num_bnn_iterations = 0
        num_iterations = num_dropout_iterations
    elif num_bnn_iterations > 1:
        num_dropout_iterations = 0
        num_iterations = num_bnn_iterations
    else:
        num_iterations = 0

    if num_iterations > 1:
        vector_prediction_matrix = None
        scalar_prediction_matrix = None
        ensemble_size_per_iter = -1
        ensemble_size = -1

        for k in range(num_iterations):
            this_vector_prediction_matrix, this_scalar_prediction_matrix = (
                _apply_model_once(
                    model_object=model_object,
                    model_metadata_dict=metadata_dict,
                    predictor_matrix=predictor_matrix,
                    use_dropout=num_dropout_iterations > 1
                )
            )

            if k == 0:
                max_ensemble_size_per_iter = int(numpy.ceil(
                    float(max_ensemble_size) / num_iterations
                ))
                ensemble_size_per_iter = min([
                    this_vector_prediction_matrix.shape[-1],
                    max_ensemble_size_per_iter
                ])
                ensemble_size = ensemble_size_per_iter * num_iterations

                vector_prediction_matrix = numpy.full(
                    this_vector_prediction_matrix.shape[:-1] + (ensemble_size,),
                    numpy.nan
                )

                if this_scalar_prediction_matrix is not None:
                    scalar_prediction_matrix = numpy.full(
                        this_scalar_prediction_matrix.shape[:-1] +
                        (ensemble_size,),
                        numpy.nan
                    )

            if this_vector_prediction_matrix.shape[-1] > ensemble_size_per_iter:
                ensemble_indices = numpy.linspace(
                    0, this_vector_prediction_matrix.shape[-1] - 1,
                    num=this_vector_prediction_matrix.shape[-1], dtype=int
                )
                ensemble_indices = numpy.random.choice(
                    ensemble_indices, size=ensemble_size_per_iter, replace=False
                )

                this_vector_prediction_matrix = (
                    this_vector_prediction_matrix[..., ensemble_indices]
                )
                if this_scalar_prediction_matrix is not None:
                    this_scalar_prediction_matrix = (
                        this_scalar_prediction_matrix[..., ensemble_indices]
                    )

            first_index = k * ensemble_size_per_iter
            last_index = first_index + ensemble_size_per_iter
            vector_prediction_matrix[..., first_index:last_index] = (
                this_vector_prediction_matrix + 0.
            )

            if this_scalar_prediction_matrix is not None:
                scalar_prediction_matrix[..., first_index:last_index] = (
                    this_scalar_prediction_matrix + 0.
                )
    else:
        vector_prediction_matrix, scalar_prediction_matrix = _apply_model_once(
            model_object=model_object,
            model_metadata_dict=metadata_dict,
            predictor_matrix=predictor_matrix, use_dropout=False
        )

        while len(vector_prediction_matrix.shape) < 3:
            vector_prediction_matrix = numpy.expand_dims(
                vector_prediction_matrix, axis=-1
            )

        while len(scalar_prediction_matrix.shape) < 3:
            scalar_prediction_matrix = numpy.expand_dims(
                scalar_prediction_matrix, axis=-1
            )

    print('VECTOR PREDICTIONS')
    print(vector_prediction_matrix.shape)

    ensemble_size = vector_prediction_matrix.shape[-1]
    if max_ensemble_size < ensemble_size:
        ensemble_indices = numpy.linspace(
            0, ensemble_size - 1, num=ensemble_size, dtype=int
        )
        ensemble_indices = numpy.random.choice(
            ensemble_indices, size=max_ensemble_size, replace=False
        )

        vector_prediction_matrix = vector_prediction_matrix[
            ..., ensemble_indices
        ]

        if scalar_prediction_matrix is not None:
            scalar_prediction_matrix = scalar_prediction_matrix[
                ..., ensemble_indices
            ]

    vector_target_matrix = target_array[0]
    if len(target_array) > 1:
        scalar_target_matrix = target_array[1]
    else:
        scalar_target_matrix = None

    target_example_dict = _targets_numpy_to_dict(
        scalar_target_matrix=scalar_target_matrix,
        vector_target_matrix=vector_target_matrix,
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
    num_heights = len(target_example_dict[example_utils.HEIGHTS_KEY])

    this_dict = {
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: [],
        example_utils.VECTOR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, num_heights, 0), 0.),
        example_utils.SCALAR_PREDICTOR_NAMES_KEY: [],
        example_utils.SCALAR_PREDICTOR_VALS_KEY:
            numpy.full((num_examples, 0), 0.)
    }
    target_example_dict.update(this_dict)

    ensemble_size = vector_prediction_matrix.shape[-1]
    prediction_example_dict_by_member = [dict()] * ensemble_size

    for k in range(ensemble_size):
        prediction_example_dict_by_member[k] = _targets_numpy_to_dict(
            scalar_target_matrix=scalar_prediction_matrix[..., k],
            vector_target_matrix=vector_prediction_matrix[..., k],
            model_metadata_dict=metadata_dict
        )
        prediction_example_dict_by_member[k].update(this_dict)

    if (
            generator_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY]
            is not None
    ):
        print('Denormalizing predicted and actual vectors...')

        target_example_dict = normalization.denormalize_data(
            new_example_dict=target_example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=
            generator_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY],
            uniformize=generator_option_dict[neural_net.UNIFORMIZE_FLAG_KEY],
            min_normalized_value=
            generator_option_dict[neural_net.VECTOR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=
            generator_option_dict[neural_net.VECTOR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=False
        )

        for k in range(ensemble_size):
            (
                prediction_example_dict_by_member[k]
            ) = normalization.denormalize_data(
                new_example_dict=prediction_example_dict_by_member[k],
                training_example_dict=training_example_dict,
                normalization_type_string=
                generator_option_dict[neural_net.VECTOR_TARGET_NORM_TYPE_KEY],
                uniformize=
                generator_option_dict[neural_net.UNIFORMIZE_FLAG_KEY],
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

        target_example_dict = normalization.denormalize_data(
            new_example_dict=target_example_dict,
            training_example_dict=training_example_dict,
            normalization_type_string=
            generator_option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY],
            uniformize=generator_option_dict[neural_net.UNIFORMIZE_FLAG_KEY],
            min_normalized_value=
            generator_option_dict[neural_net.SCALAR_TARGET_MIN_VALUE_KEY],
            max_normalized_value=
            generator_option_dict[neural_net.SCALAR_TARGET_MAX_VALUE_KEY],
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=False, apply_to_scalar_targets=True
        )

        for k in range(ensemble_size):
            (
                prediction_example_dict_by_member[k]
            ) = normalization.denormalize_data(
                new_example_dict=prediction_example_dict_by_member[k],
                training_example_dict=training_example_dict,
                normalization_type_string=
                generator_option_dict[neural_net.SCALAR_TARGET_NORM_TYPE_KEY],
                uniformize=
                generator_option_dict[neural_net.UNIFORMIZE_FLAG_KEY],
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

    for this_target_name in [
            example_utils.SHORTWAVE_HEATING_RATE_NAME,
            example_utils.LONGWAVE_HEATING_RATE_NAME
    ]:
        try:
            j = vector_target_names.index(this_target_name)
        except ValueError:
            continue

        target_example_dict[example_utils.VECTOR_TARGET_VALS_KEY][:, -1, j] = 0.

        for k in range(ensemble_size):
            prediction_example_dict_by_member[k][
                example_utils.VECTOR_TARGET_VALS_KEY
            ][:, -1, j] = 0.

    scalar_prediction_matrix = numpy.stack([
        d[example_utils.SCALAR_TARGET_VALS_KEY]
        for d in prediction_example_dict_by_member
    ], axis=-1)

    vector_prediction_matrix = numpy.stack([
        d[example_utils.VECTOR_TARGET_VALS_KEY]
        for d in prediction_example_dict_by_member
    ], axis=-1)

    print('Writing target (actual) and predicted values to: "{0:s}"...'.format(
        output_file_name
    ))
    prediction_io.write_file(
        netcdf_file_name=output_file_name,
        scalar_target_matrix=
        target_example_dict[example_utils.SCALAR_TARGET_VALS_KEY],
        vector_target_matrix=
        target_example_dict[example_utils.VECTOR_TARGET_VALS_KEY],
        scalar_prediction_matrix=scalar_prediction_matrix,
        vector_prediction_matrix=vector_prediction_matrix,
        heights_m_agl=target_example_dict[example_utils.HEIGHTS_KEY],
        example_id_strings=example_id_strings,
        model_file_name=model_file_name,
        isotonic_model_file_name=None, uncertainty_calib_model_file_name=None,
        normalization_file_name=None
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        first_time_string=getattr(INPUT_ARG_OBJECT, FIRST_TIME_ARG_NAME),
        last_time_string=getattr(INPUT_ARG_OBJECT, LAST_TIME_ARG_NAME),
        num_dropout_iterations=getattr(
            INPUT_ARG_OBJECT, NUM_DROPOUT_ITERS_ARG_NAME
        ),
        num_bnn_iterations=getattr(INPUT_ARG_OBJECT, NUM_BNN_ITERS_ARG_NAME),
        max_ensemble_size=getattr(INPUT_ARG_OBJECT, MAX_ENSEMBLE_SIZE_ARG_NAME),
        exclude_summit_greenland=bool(getattr(
            INPUT_ARG_OBJECT, EXCLUDE_SUMMIT_ARG_NAME
        )),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
