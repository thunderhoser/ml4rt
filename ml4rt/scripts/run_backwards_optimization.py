"""Runs backwards optimization."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import backwards_optimization as bwo

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
LAYER_ARG_NAME = 'layer_name'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
NUM_ITERATIONS_ARG_NAME = 'num_iterations'
LEARNING_RATE_ARG_NAME = 'learning_rate'
L2_WEIGHT_ARG_NAME = 'l2_weight'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

# TODO(thunderhoser): Allow different init functions.

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_FILE_HELP_STRING = (
    'Path to file with data examples.  Will be read by `example_io.read_file`.'
)
EXAMPLE_INDICES_HELP_STRING = (
    'Indices of examples to use.  If you do not want to use specific examples, '
    'leave this alone.'
)
NUM_EXAMPLES_HELP_STRING = (
    '[used only if `{0:s}` is not specified] Number of examples to use (these '
    'will be selected randomly).  If you want to use all examples, leave this '
    'alone.'
).format(EXAMPLE_INDICES_ARG_NAME)

LAYER_HELP_STRING = 'See doc for `saliency.check_metadata`.'
NEURON_INDICES_HELP_STRING = 'See doc for `saliency.check_metadata`.'
IDEAL_ACTIVATION_HELP_STRING = 'See doc for `saliency.check_metadata`.'
NUM_ITERATIONS_HELP_STRING = 'See doc for `saliency.check_metadata`.'
LEARNING_RATE_HELP_STRING = 'See doc for `saliency.check_metadata`.'
L2_WEIGHT_HELP_STRING = 'See doc for `saliency.check_metadata`.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`backwards_optimizatiom.write_standard_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_INDICES_ARG_NAME, type=int, nargs='+', required=False,
    default=[-1], help=EXAMPLE_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_ARG_NAME, type=str, required=True, help=LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=True,
    help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ITERATIONS_ARG_NAME, type=int, required=False,
    default=bwo.DEFAULT_NUM_ITERATIONS, help=NUM_ITERATIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEARNING_RATE_ARG_NAME, type=float, required=False,
    default=bwo.DEFAULT_LEARNING_RATE, help=LEARNING_RATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + L2_WEIGHT_ARG_NAME, type=float, required=False,
    default=bwo.DEFAULT_L2_WEIGHT, help=L2_WEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, example_indices, num_examples,
         layer_name, neuron_indices, ideal_activation, num_iterations,
         learning_rate, l2_weight, output_file_name):
    """Runs backwards optimization.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param example_indices: Same.
    :param num_examples: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
    :param output_file_name: Same.
    """

    example_dict = example_io.read_file(example_file_name)

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    year = example_io.file_name_to_year(example_file_name)
    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(year)
    )

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
        os.path.split(example_file_name)[0]
    )
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = len(
        example_dict[example_io.VALID_TIMES_KEY]
    )
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec
    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]

    generator = neural_net.data_generator(
        option_dict=generator_option_dict, for_inference=True,
        net_type_string=net_type_string, is_loss_constrained_mse=False
    )

    print(SEPARATOR_STRING)
    predictor_matrix, _, example_id_strings = next(generator)
    print(SEPARATOR_STRING)

    example_indices = misc_utils.subset_examples(
        indices_to_keep=example_indices, num_examples_to_keep=num_examples,
        num_examples_total=len(example_id_strings)
    )

    predictor_matrix = predictor_matrix[example_indices, ...]
    example_id_strings = [example_id_strings[i] for i in example_indices]

    num_examples = len(example_id_strings)
    bwo_dict = None

    for i in range(num_examples):
        this_bwo_dict = bwo.optimize_input_for_neuron(
            model_object=model_object,
            init_function_or_matrix=predictor_matrix[i, ...],
            layer_name=layer_name, neuron_indices=neuron_indices,
            ideal_activation=ideal_activation, num_iterations=num_iterations,
            learning_rate=learning_rate, l2_weight=l2_weight
        )

        if i == num_examples - 1:
            print(SEPARATOR_STRING)
        else:
            print(MINOR_SEPARATOR_STRING)

        if bwo_dict is None:
            these_dim = numpy.array(
                (num_examples,) +
                this_bwo_dict[bwo.INITIAL_PREDICTORS_KEY].shape[1:],
                dtype=int
            )

            bwo_dict = {
                bwo.INITIAL_PREDICTORS_KEY: numpy.full(these_dim, numpy.nan),
                bwo.FINAL_PREDICTORS_KEY: numpy.full(these_dim, numpy.nan),
                bwo.INITIAL_ACTIVATIONS_KEY: numpy.full(num_examples, numpy.nan),
                bwo.FINAL_ACTIVATIONS_KEY: numpy.full(num_examples, numpy.nan)
            }

        bwo_dict[bwo.INITIAL_PREDICTORS_KEY][i, ...] = (
            this_bwo_dict[bwo.INITIAL_PREDICTORS_KEY][0, ...]
        )
        bwo_dict[bwo.FINAL_PREDICTORS_KEY][i, ...] = (
            this_bwo_dict[bwo.FINAL_PREDICTORS_KEY][0, ...]
        )
        bwo_dict[bwo.INITIAL_ACTIVATIONS_KEY][i] = (
            this_bwo_dict[bwo.INITIAL_ACTIVATION_KEY]
        )
        bwo_dict[bwo.FINAL_ACTIVATIONS_KEY][i] = (
            this_bwo_dict[bwo.FINAL_ACTIVATION_KEY]
        )

    new_example_dict = neural_net.predictors_numpy_to_dict(
        predictor_matrix=bwo_dict[bwo.INITIAL_PREDICTORS_KEY],
        example_dict=example_dict, net_type_string=net_type_string
    )
    init_scalar_predictor_matrix = (
        new_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    )
    init_vector_predictor_matrix = (
        new_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    )

    new_example_dict = neural_net.predictors_numpy_to_dict(
        predictor_matrix=bwo_dict[bwo.FINAL_PREDICTORS_KEY],
        example_dict=example_dict, net_type_string=net_type_string
    )
    final_scalar_predictor_matrix = (
        new_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    )
    final_vector_predictor_matrix = (
        new_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    )

    print('Writing results to file: "{0:s}"...'.format(output_file_name))
    bwo.write_standard_file(
        netcdf_file_name=output_file_name,
        init_scalar_predictor_matrix=init_scalar_predictor_matrix,
        final_scalar_predictor_matrix=final_scalar_predictor_matrix,
        init_vector_predictor_matrix=init_vector_predictor_matrix,
        final_vector_predictor_matrix=final_vector_predictor_matrix,
        initial_activations=bwo_dict[bwo.INITIAL_ACTIVATIONS_KEY],
        final_activations=bwo_dict[bwo.FINAL_ACTIVATIONS_KEY],
        example_id_strings=example_id_strings, model_file_name=model_file_name,
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, num_iterations=num_iterations,
        learning_rate=learning_rate, l2_weight=l2_weight
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_ARG_NAME),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        num_iterations=getattr(INPUT_ARG_OBJECT, NUM_ITERATIONS_ARG_NAME),
        learning_rate=getattr(INPUT_ARG_OBJECT, LEARNING_RATE_ARG_NAME),
        l2_weight=getattr(INPUT_ARG_OBJECT, L2_WEIGHT_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
