"""Makes saliency map for each example, according to one model."""

import os
import sys
import copy
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_utils
import misc as misc_utils
import neural_net
import saliency

# TODO(thunderhoser): The input arg `is_layer_output` is a HACK.  I can't find a
# reasonable automated way to determine if a layer is output, because Keras.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
EXAMPLE_ID_FILE_ARG_NAME = 'input_example_id_file_name'
LAYER_ARG_NAME = 'layer_name'
IS_LAYER_OUTPUT_ARG_NAME = 'is_layer_output'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
OUTPUT_FILE_ARG_NAME = 'output_saliency_file_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
EXAMPLE_FILE_HELP_STRING = (
    '[use only if you want random examples] Path to file with data examples.  '
    'Will be read by `example_io.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    '[use only if you want random examples] Number of examples to use.  If you '
    'want to use all examples in `{0:s}`, leave this alone.'
).format(EXAMPLE_FILE_ARG_NAME)

EXAMPLE_DIR_HELP_STRING = (
    '[use only if you want specific examples] Name of directory with data '
    'examples.  Files therein will be found by `example_io.find_file` and read '
    'by `example_io.read_file`.'
)
EXAMPLE_ID_FILE_HELP_STRING = (
    '[use only if you want specific examples] Path to file with desired IDs.  '
    'Will be read by `misc.read_example_ids_from_netcdf`.'
)

LAYER_HELP_STRING = 'See doc for `saliency.check_metadata`.'
IS_LAYER_OUTPUT_HELP_STRING = (
    'Boolean flag.  If 1, `{0:s}` is an output layer.  If 0, it is not an '
    'output layer.'
).format(LAYER_ARG_NAME)

NEURON_INDICES_HELP_STRING = 'See doc for `saliency.check_metadata`.'
IDEAL_ACTIVATION_HELP_STRING = 'See doc for `saliency.check_metadata`.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `saliency.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_ID_FILE_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_ID_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_ARG_NAME, type=str, required=True, help=LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IS_LAYER_OUTPUT_ARG_NAME, type=int, required=True,
    help=IS_LAYER_OUTPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=saliency.DEFAULT_IDEAL_ACTIVATION, help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, num_examples, example_dir_name,
         example_id_file_name, layer_name, is_layer_output, neuron_indices,
         ideal_activation, output_file_name):
    """Makes saliency map for each example, according to one model.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param layer_name: Same.
    :param is_layer_output: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param output_file_name: Same.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)

    predictor_matrix, _, example_id_strings = (
        misc_utils.get_examples_for_inference(
            model_metadata_dict=metadata_dict,
            example_file_name=example_file_name,
            num_examples=num_examples, example_dir_name=example_dir_name,
            example_id_file_name=example_id_file_name
        )
    )
    print(SEPARATOR_STRING)

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    if is_layer_output:
        dummy_example_dict = {
            example_utils.SCALAR_TARGET_NAMES_KEY:
                generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY],
            example_utils.VECTOR_TARGET_NAMES_KEY:
                generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
            example_utils.HEIGHTS_KEY:
                generator_option_dict[neural_net.HEIGHTS_KEY]
        }

        target_field_name, target_height_m_agl = (
            neural_net.neuron_indices_to_target_var(
                neuron_indices=neuron_indices,
                example_dict=copy.deepcopy(dummy_example_dict),
                net_type_string=metadata_dict[neural_net.NET_TYPE_KEY]
            )
        )
    else:
        target_field_name = None
        target_height_m_agl = None

    print('Target field and height = {0:s}, {1:s}'.format(
        str(target_field_name), str(target_height_m_agl)
    ))

    print('Computing saliency for neuron {0:s} in layer "{1:s}"...'.format(
        str(neuron_indices), layer_name
    ))
    saliency_matrix = saliency.get_saliency_one_neuron(
        model_object=model_object, predictor_matrix=predictor_matrix,
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )

    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]

    if net_type_string == neural_net.DENSE_NET_TYPE_STRING:
        dummy_example_dict = {
            example_utils.SCALAR_PREDICTOR_NAMES_KEY:
                generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
            example_utils.VECTOR_PREDICTOR_NAMES_KEY:
                generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
            example_utils.HEIGHTS_KEY:
                generator_option_dict[neural_net.HEIGHTS_KEY]
        }

        dummy_example_dict = neural_net.predictors_numpy_to_dict(
            predictor_matrix=saliency_matrix, example_dict=dummy_example_dict,
            net_type_string=metadata_dict[neural_net.NET_TYPE_KEY]
        )
        scalar_saliency_matrix = (
            dummy_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
        )
        vector_saliency_matrix = (
            dummy_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
        )
    else:
        num_scalar_predictors = len(
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY]
        )
        scalar_saliency_matrix = saliency_matrix[..., -num_scalar_predictors:]
        vector_saliency_matrix = saliency_matrix[..., :-num_scalar_predictors]

    print('Writing saliency maps to: "{0:s}"...'.format(output_file_name))
    saliency.write_file(
        netcdf_file_name=output_file_name,
        scalar_saliency_matrix=scalar_saliency_matrix,
        vector_saliency_matrix=vector_saliency_matrix,
        example_id_strings=example_id_strings, model_file_name=model_file_name,
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation, target_field_name=target_field_name,
        target_height_m_agl=target_height_m_agl
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_ID_FILE_ARG_NAME
        ),
        layer_name=getattr(INPUT_ARG_OBJECT, LAYER_ARG_NAME),
        is_layer_output=bool(
            getattr(INPUT_ARG_OBJECT, IS_LAYER_OUTPUT_ARG_NAME)
        ),
        neuron_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, NEURON_INDICES_ARG_NAME), dtype=int
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
