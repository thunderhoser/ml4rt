"""Runs Grad-CAM for each example and target variable."""

import os.path
import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import gradcam
from ml4rt.machine_learning import neural_net
from ml4rt.scripts import run_gradcam

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = run_gradcam.MODEL_FILE_ARG_NAME
EXAMPLE_FILE_ARG_NAME = run_gradcam.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = run_gradcam.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = run_gradcam.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = run_gradcam.EXAMPLE_ID_FILE_ARG_NAME
ACTIVATION_LAYER_ARG_NAME = run_gradcam.ACTIVATION_LAYER_ARG_NAME
VECTOR_OUT_LAYER_ARG_NAME = run_gradcam.VECTOR_OUT_LAYER_ARG_NAME
IDEAL_ACTIVATION_ARG_NAME = run_gradcam.IDEAL_ACTIVATION_ARG_NAME
OUTPUT_FILE_ARG_NAME = run_gradcam.OUTPUT_FILE_ARG_NAME

MODEL_FILE_HELP_STRING = run_gradcam.MODEL_FILE_HELP_STRING
EXAMPLE_FILE_HELP_STRING = run_gradcam.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = run_gradcam.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = run_gradcam.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = run_gradcam.EXAMPLE_ID_FILE_HELP_STRING
ACTIVATION_LAYER_HELP_STRING = run_gradcam.ACTIVATION_LAYER_HELP_STRING
VECTOR_OUT_LAYER_HELP_STRING = run_gradcam.VECTOR_OUT_LAYER_HELP_STRING
IDEAL_ACTIVATION_HELP_STRING = run_gradcam.IDEAL_ACTIVATION_HELP_STRING
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `gradcam.write_all_targets_file`.'
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
    '--' + ACTIVATION_LAYER_ARG_NAME, type=str, required=True,
    help=ACTIVATION_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VECTOR_OUT_LAYER_ARG_NAME, type=str, required=True,
    help=VECTOR_OUT_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=gradcam.DEFAULT_IDEAL_ACTIVATION, help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, num_examples, example_dir_name,
         example_id_file_name, activation_layer_name, vector_output_layer_name,
         ideal_activation, output_file_name):
    """Runs Grad-CAM for each example and target variable.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param activation_layer_name: Same.
    :param vector_output_layer_name: Same.
    :param ideal_activation: Same.
    :param output_file_name: Same.
    :raises: ValueError: if neural-net type is not CNN or U-net.
    """

    print('Reading model from: "{0:s}"...'.format(model_file_name))
    model_object = neural_net.read_model(model_file_name)

    metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(metafile_name))
    metadata_dict = neural_net.read_metafile(metafile_name)
    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

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

    predictor_matrix, _, example_id_strings = (
        misc_utils.get_examples_for_inference(
            model_metadata_dict=metadata_dict,
            example_file_name=example_file_name,
            num_examples=num_examples, example_dir_name=example_dir_name,
            example_id_file_name=example_id_file_name
        )
    )
    print(SEPARATOR_STRING)

    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    dummy_example_dict = {
        example_io.SCALAR_TARGET_NAMES_KEY: [],
        example_io.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_io.HEIGHTS_KEY: heights_m_agl
    }

    num_examples = len(example_id_strings)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)

    class_activation_matrix = numpy.full(
        (num_examples, num_heights, num_heights, num_vector_targets), numpy.nan
    )

    for i in range(num_examples):
        print((
            'Have run Grad-CAM (all target variables) for {0:d} of {1:d} '
            'examples...'
        ).format(
            i, num_examples
        ))

        for k in range(num_vector_targets):
            for j in range(num_heights):
                these_neuron_indices = neural_net.target_var_to_neuron_indices(
                    example_dict=dummy_example_dict,
                    net_type_string=net_type_string,
                    target_name=vector_target_names[k],
                    height_m_agl=heights_m_agl[j]
                )

                class_activation_matrix[i, :, j, k] = gradcam.run_gradcam(
                    model_object=model_object,
                    predictor_matrix=predictor_matrix[i, ...],
                    activation_layer_name=activation_layer_name,
                    vector_output_layer_name=vector_output_layer_name,
                    output_neuron_indices=these_neuron_indices,
                    ideal_activation=ideal_activation
                )

    print((
        'Have run Grad-CAM (all target variables) for all {0:d} examples!\n'
    ).format(
        num_examples
    ))

    print('Writing class-activation maps to: "{0:s}"...'.format(
        output_file_name
    ))
    gradcam.write_all_targets_file(
        netcdf_file_name=output_file_name,
        class_activation_matrix=class_activation_matrix,
        example_id_strings=example_id_strings,
        model_file_name=model_file_name,
        activation_layer_name=activation_layer_name,
        vector_output_layer_name=vector_output_layer_name,
        ideal_activation=ideal_activation
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
        activation_layer_name=getattr(
            INPUT_ARG_OBJECT, ACTIVATION_LAYER_ARG_NAME
        ),
        vector_output_layer_name=getattr(
            INPUT_ARG_OBJECT, VECTOR_OUT_LAYER_ARG_NAME
        ),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
