"""Runs backwards optimization."""

import copy
import os.path
import argparse
import numpy
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import backwards_optimization as bwo
from ml4rt.scripts import make_saliency_maps

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_ID_FILE_ARG_NAME
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
EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_ID_FILE_HELP_STRING
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


def _run(model_file_name, example_file_name, num_examples, example_dir_name,
         example_id_file_name, layer_name, neuron_indices, ideal_activation,
         num_iterations, learning_rate, l2_weight, output_file_name):
    """Runs backwards optimization.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param layer_name: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param num_iterations: Same.
    :param learning_rate: Same.
    :param l2_weight: Same.
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
        misc_utils.get_examples_from_inference(
            model_metadata_dict=metadata_dict,
            example_file_name=example_file_name,
            num_examples=num_examples, example_dir_name=example_dir_name,
            example_id_file_name=example_id_file_name
        )
    )
    print(SEPARATOR_STRING)

    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    normalization_file_name = (
        generator_option_dict[neural_net.NORMALIZATION_FILE_KEY]
    )

    print((
        'Reading training examples (for normalization) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)

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
                bwo.INITIAL_ACTIVATIONS_KEY:
                    numpy.full(num_examples, numpy.nan),
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

    if example_file_name == '':
        example_file_name = example_io.find_many_files(
            example_dir_name=example_dir_name,
            first_time_unix_sec=0, last_time_unix_sec=int(1e12),
            raise_error_if_any_missing=False, raise_error_if_all_missing=True
        )[0]

    first_example_dict = example_io.read_file(example_file_name)
    net_type_string = metadata_dict[neural_net.NET_TYPE_KEY]

    init_example_dict = copy.deepcopy(first_example_dict)
    this_example_dict = neural_net.predictors_numpy_to_dict(
        predictor_matrix=bwo_dict[bwo.INITIAL_PREDICTORS_KEY],
        example_dict=init_example_dict, net_type_string=net_type_string
    )
    init_example_dict.update(this_example_dict)

    init_example_dict = normalization.denormalize_data(
        new_example_dict=init_example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=
        generator_option_dict[neural_net.PREDICTOR_NORM_TYPE_KEY],
        min_normalized_value=
        generator_option_dict[neural_net.PREDICTOR_MIN_NORM_VALUE_KEY],
        max_normalized_value=
        generator_option_dict[neural_net.PREDICTOR_MAX_NORM_VALUE_KEY],
        separate_heights=True, apply_to_predictors=True,
        apply_to_targets=False
    )
    init_scalar_predictor_matrix = (
        init_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    )
    init_vector_predictor_matrix = (
        init_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    )

    final_example_dict = copy.deepcopy(first_example_dict)
    this_example_dict = neural_net.predictors_numpy_to_dict(
        predictor_matrix=bwo_dict[bwo.FINAL_PREDICTORS_KEY],
        example_dict=final_example_dict, net_type_string=net_type_string
    )
    final_example_dict.update(this_example_dict)

    final_example_dict = normalization.denormalize_data(
        new_example_dict=final_example_dict,
        training_example_dict=training_example_dict,
        normalization_type_string=
        generator_option_dict[neural_net.PREDICTOR_NORM_TYPE_KEY],
        min_normalized_value=
        generator_option_dict[neural_net.PREDICTOR_MIN_NORM_VALUE_KEY],
        max_normalized_value=
        generator_option_dict[neural_net.PREDICTOR_MAX_NORM_VALUE_KEY],
        separate_heights=True, apply_to_predictors=True,
        apply_to_targets=False
    )
    final_scalar_predictor_matrix = (
        final_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    )
    final_vector_predictor_matrix = (
        final_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
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
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_ID_FILE_ARG_NAME
        ),
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
