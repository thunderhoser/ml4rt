"""Runs backwards optimization."""

import argparse
import numpy
from ml4rt.machine_learning import backwards_optimization as bwo

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
