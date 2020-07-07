"""Runs the Grad-CAM (gradient-weighted class-activation maps) algorithm."""

import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import gradcam
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
ACTIVATION_LAYER_ARG_NAME = 'activation_layer_name'
VECTOR_OUT_LAYER_ARG_NAME = 'vector_output_layer_name'
NEURON_INDICES_ARG_NAME = 'output_neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
OUTPUT_FILE_ARG_NAME = 'output_gradcam_file_name'

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

ACTIVATION_LAYER_HELP_STRING = 'See doc for `gradcam.check_metadata`.'
VECTOR_OUT_LAYER_HELP_STRING = 'See doc for `gradcam.check_metadata`.'
NEURON_INDICES_HELP_STRING = 'See doc for `gradcam.check_metadata`.'
IDEAL_ACTIVATION_HELP_STRING = 'See doc for `gradcam.check_metadata`.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `gradcam.write_standard_file`.'
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
    '--' + ACTIVATION_LAYER_ARG_NAME, type=str, required=True,
    help=ACTIVATION_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VECTOR_OUT_LAYER_ARG_NAME, type=str, required=True,
    help=VECTOR_OUT_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs=2, required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=gradcam.DEFAULT_IDEAL_ACTIVATION, help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, example_indices, num_examples,
         activation_layer_name, vector_output_layer_name, output_neuron_indices,
         ideal_activation, output_file_name):
    """Runs the Grad-CAM (gradient-weighted class-activation maps) algorithm.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param example_indices: Same.
    :param num_examples: Same.
    :param activation_layer_name: Same.
    :param vector_output_layer_name: Same.
    :param output_neuron_indices: Same.
    :param ideal_activation: Same.
    :param output_file_name: Same.
    :raises: ValueError: if neural-net type is not CNN or U-net.
    """

    this_example_dict = example_io.read_file(example_file_name)
    num_examples_total = len(this_example_dict[example_io.VALID_TIMES_KEY])

    example_indices = misc_utils.subset_examples(
        indices_to_keep=example_indices, num_examples_to_keep=num_examples,
        num_examples_total=num_examples_total
    )

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]
    valid_net_type_strings = [
        neural_net.CNN_TYPE_STRING, neural_net.U_NET_TYPE_STRING
    ]

    if net_type_string not in valid_net_type_strings:
        error_string = (
            '\nThis script does not work for net type "{0:s}".  Works only for '
            'those listed below:\n{1:s}'
        ).format(net_type_string, str(valid_net_type_strings))

        raise ValueError(error_string)

    year = example_io.file_name_to_year(example_file_name)
    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(year)
    )

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
        os.path.split(example_file_name)[0]
    )
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = num_examples_total
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    generator = neural_net.data_generator(
        option_dict=generator_option_dict, for_inference=True,
        net_type_string=metadata_dict[neural_net.NET_TYPE_KEY],
        is_loss_constrained_mse=False
    )

    print(SEPARATOR_STRING)
    predictor_matrix, _, example_id_strings = next(generator)
    print(SEPARATOR_STRING)

    predictor_matrix = predictor_matrix[example_indices, ...]
    example_id_strings = [example_id_strings[i] for i in example_indices]

    num_examples = predictor_matrix.shape[0]
    num_heights = predictor_matrix.shape[1]
    class_activation_matrix = numpy.full((num_examples, num_heights), numpy.nan)

    for i in range(num_examples):
        if numpy.mod(i, 10) == 0:
            print('Have run Grad-CAM for {0:d} of {1:d} examples...'.format(
                i, num_examples
            ))

        class_activation_matrix[i, :] = gradcam.run_gradcam(
            model_object=model_object,
            predictor_matrix=predictor_matrix[i, ...],
            activation_layer_name=activation_layer_name,
            vector_output_layer_name=vector_output_layer_name,
            output_neuron_indices=output_neuron_indices,
            ideal_activation=ideal_activation
        )

    print('Have run Grad-CAM for all {0:d} examples!\n'.format(num_examples))

    print('Writing class-activation maps to: "{0:s}"...'.format(
        output_file_name
    ))
    gradcam.write_standard_file(
        netcdf_file_name=output_file_name,
        class_activation_matrix=class_activation_matrix,
        example_id_strings=example_id_strings,
        model_file_name=model_file_name,
        activation_layer_name=activation_layer_name,
        vector_output_layer_name=vector_output_layer_name,
        output_neuron_indices=output_neuron_indices,
        ideal_activation=ideal_activation
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
        activation_layer_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_LAYER_ARG_NAME
        ),
        vector_output_layer_name=getattr(
            INPUT_ARG_OBJECT, VECTOR_OUT_LAYER_ARG_NAME
        ),
        output_neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
