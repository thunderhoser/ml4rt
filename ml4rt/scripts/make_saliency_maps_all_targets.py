"""Makes saliency map for each example and target, according to one model."""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.utils import misc as misc_utils
from ml4rt.io import example_io
from ml4rt.scripts import make_saliency_maps
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import saliency

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
SENTINEL_VALUE = -123456.

MODEL_FILE_ARG_NAME = make_saliency_maps.MODEL_FILE_ARG_NAME
EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
EXAMPLE_INDICES_ARG_NAME = make_saliency_maps.EXAMPLE_INDICES_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
IDEAL_ACTIVATION_ARG_NAME = make_saliency_maps.IDEAL_ACTIVATION_ARG_NAME
SCALAR_LAYER_ARG_NAME = 'scalar_output_layer_name'
VECTOR_LAYER_ARG_NAME = 'vector_output_layer_name'
OUTPUT_FILE_ARG_NAME = 'output_saliency_file_name'

MODEL_FILE_HELP_STRING = make_saliency_maps.MODEL_FILE_HELP_STRING
EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
EXAMPLE_INDICES_HELP_STRING = make_saliency_maps.EXAMPLE_INDICES_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
IDEAL_ACTIVATION_HELP_STRING = make_saliency_maps.IDEAL_ACTIVATION_HELP_STRING
SCALAR_LAYER_HELP_STRING = (
    'Name of layer that outputs scalar target variables.  If there are no '
    'scalar target variables, leave this alone.'
)
VECTOR_LAYER_HELP_STRING = (
    'Name of layer that outputs vector target variables.  If there are no '
    'vector target variables, leave this alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`saliency.write_all_targets_file`.'
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
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SCALAR_LAYER_ARG_NAME, type=str, required=False, default='',
    help=SCALAR_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VECTOR_LAYER_ARG_NAME, type=str, required=False, default='',
    help=VECTOR_LAYER_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(model_file_name, example_file_name, example_indices, num_examples,
         ideal_activation, scalar_output_layer_name, vector_output_layer_name,
         output_file_name):
    """Makes saliency map for each example and target, according to one model.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param example_indices: Same.
    :param num_examples: Same.
    :param ideal_activation: Same.
    :param scalar_output_layer_name: Same.
    :param vector_output_layer_name: Same.
    :param output_file_name: Same.
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
    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    # TODO(thunderhoser): Modularize generation (and maybe subsetting) of
    # examples.
    year = example_io.file_name_to_year(example_file_name)
    first_time_unix_sec, last_time_unix_sec = (
        time_conversion.first_and_last_times_in_year(year)
    )

    generator_option_dict[neural_net.EXAMPLE_DIRECTORY_KEY] = (
        os.path.split(example_file_name)[0]
    )
    generator_option_dict[neural_net.BATCH_SIZE_KEY] = num_examples_total
    generator_option_dict[neural_net.FIRST_TIME_KEY] = first_time_unix_sec
    generator_option_dict[neural_net.LAST_TIME_KEY] = last_time_unix_sec

    generator = neural_net.data_generator(
        option_dict=generator_option_dict, for_inference=True,
        net_type_string=net_type_string, is_loss_constrained_mse=False
    )

    print(SEPARATOR_STRING)
    predictor_matrix, _, example_id_strings = next(generator)
    print(SEPARATOR_STRING)

    predictor_matrix = predictor_matrix[example_indices, ...]
    example_id_strings = [example_id_strings[i] for i in example_indices]

    scalar_predictor_names = (
        generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY]
    )
    vector_predictor_names = (
        generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
    )
    scalar_target_names = (
        generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    )
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    num_scalar_predictors = len(scalar_predictor_names)
    num_vector_predictors = len(vector_predictor_names)
    num_scalar_targets = len(scalar_target_names)
    num_vector_targets = len(vector_target_names)
    num_heights = len(heights_m_agl)
    num_examples = len(example_id_strings)

    saliency_matrix_scalar_p_scalar_t = numpy.full(
        (num_examples, num_scalar_predictors, num_scalar_targets), numpy.nan
    )
    saliency_matrix_vector_p_scalar_t = numpy.full(
        (num_examples, num_heights, num_vector_predictors, num_scalar_targets),
        numpy.nan
    )
    saliency_matrix_scalar_p_vector_t = numpy.full(
        (num_examples, num_scalar_predictors, num_heights, num_vector_targets),
        numpy.nan
    )
    saliency_matrix_vector_p_vector_t = numpy.full(
        (num_examples, num_heights, num_vector_predictors, num_heights,
         num_vector_targets),
        numpy.nan
    )

    dummy_example_dict = {
        example_io.SCALAR_PREDICTOR_NAMES_KEY: scalar_predictor_names,
        example_io.VECTOR_PREDICTOR_NAMES_KEY: vector_predictor_names,
        example_io.SCALAR_TARGET_NAMES_KEY: scalar_target_names,
        example_io.VECTOR_TARGET_NAMES_KEY: vector_target_names,
        example_io.HEIGHTS_KEY: heights_m_agl
    }

    for k in range(num_scalar_targets):
        these_neuron_indices = neural_net.target_var_to_neuron_indices(
            example_dict=copy.deepcopy(dummy_example_dict),
            net_type_string=net_type_string, target_name=scalar_target_names[k]
        )

        print('Computing saliency for "{0:s}"...'.format(
            scalar_target_names[k]
        ))

        this_saliency_matrix = saliency.get_saliency_one_neuron(
            model_object=model_object, predictor_matrix=predictor_matrix,
            layer_name=scalar_output_layer_name,
            neuron_indices=these_neuron_indices,
            ideal_activation=ideal_activation
        )

        new_example_dict = neural_net.predictors_numpy_to_dict(
            predictor_matrix=this_saliency_matrix,
            example_dict=copy.deepcopy(dummy_example_dict),
            net_type_string=net_type_string
        )
        saliency_matrix_scalar_p_scalar_t[..., k] = (
            new_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
        )
        saliency_matrix_vector_p_scalar_t[..., k] = (
            new_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
        )

    print(SEPARATOR_STRING)

    for k in range(num_vector_targets):
        for j in range(num_heights):
            these_neuron_indices = neural_net.target_var_to_neuron_indices(
                example_dict=copy.deepcopy(dummy_example_dict),
                net_type_string=net_type_string,
                target_name=vector_target_names[k],
                height_m_agl=heights_m_agl[j]
            )

            print('Computing saliency for "{0:s}" at {1:d} m AGL...'.format(
                vector_target_names[k], int(numpy.round(heights_m_agl[j]))
            ))

            this_layer_name = (
                scalar_output_layer_name
                if net_type_string == neural_net.DENSE_NET_TYPE_STRING
                else vector_output_layer_name
            )

            this_saliency_matrix = saliency.get_saliency_one_neuron(
                model_object=model_object, predictor_matrix=predictor_matrix,
                layer_name=this_layer_name, neuron_indices=these_neuron_indices,
                ideal_activation=ideal_activation
            )

            new_example_dict = neural_net.predictors_numpy_to_dict(
                predictor_matrix=this_saliency_matrix,
                example_dict=copy.deepcopy(dummy_example_dict),
                net_type_string=net_type_string
            )
            saliency_matrix_scalar_p_vector_t[..., j, k] = (
                new_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
            )
            saliency_matrix_vector_p_vector_t[..., j, k] = (
                new_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
            )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        ideal_activation=getattr(INPUT_ARG_OBJECT, IDEAL_ACTIVATION_ARG_NAME),
        scalar_output_layer_name=getattr(
            INPUT_ARG_OBJECT, SCALAR_LAYER_ARG_NAME
        ),
        vector_output_layer_name=getattr(
            INPUT_ARG_OBJECT, VECTOR_LAYER_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
