"""Makes saliency map for each example, according to one model."""

import copy
import os.path
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import saliency

# TODO(thunderhoser): The input arg `is_layer_output` is a HACK.  I can't find a
# reasonable automated way to determine if a layer is output, because Keras.

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
SENTINEL_VALUE = -123456.

MODEL_FILE_ARG_NAME = 'input_model_file_name'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
LAYER_ARG_NAME = 'layer_name'
IS_LAYER_OUTPUT_ARG_NAME = 'is_layer_output'
NEURON_INDICES_ARG_NAME = 'neuron_indices'
IDEAL_ACTIVATION_ARG_NAME = 'ideal_activation'
OUTPUT_FILE_ARG_NAME = 'output_saliency_file_name'

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
IS_LAYER_OUTPUT_HELP_STRING = (
    'Boolean flag.  If 1, `{0:s}` is an output layer.  If 0, it is not an '
    'output layer.'
).format(LAYER_ARG_NAME)

NEURON_INDICES_HELP_STRING = 'See doc for `saliency.check_metadata`.'
IDEAL_ACTIVATION_HELP_STRING = (
    'See doc for `saliency.check_metadata`.  For no ideal activation, leave '
    'this alone.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by `saliency.write_standard_file`.'
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
    '--' + IS_LAYER_OUTPUT_ARG_NAME, type=int, required=True,
    help=IS_LAYER_OUTPUT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NEURON_INDICES_ARG_NAME, type=int, nargs='+', required=True,
    help=NEURON_INDICES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + IDEAL_ACTIVATION_ARG_NAME, type=float, required=False,
    default=SENTINEL_VALUE, help=IDEAL_ACTIVATION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _subset_examples(indices_to_keep, num_examples_to_keep, num_examples_total):
    """Subsets examples.

    :param indices_to_keep: 1-D numpy array with indices to keep.  If None, will
        use `num_examples_to_keep` instead.
    :param num_examples_to_keep: Number of examples to keep.  If None, will use
        `indices_to_keep` instead.
    :param num_examples_total: Total number of examples available.
    :return: indices_to_keep: See input doc.
    :raises: ValueError: if both `indices_to_keep` and `num_examples_to_keep`
        are None.
    """

    # TODO(thunderhoser): Put this method somewhere more general.

    if len(indices_to_keep) == 1 and indices_to_keep[0] < 0:
        indices_to_keep = None
    if indices_to_keep is not None:
        num_examples_to_keep = None
    if num_examples_to_keep < 1:
        num_examples_to_keep = None

    if indices_to_keep is None and num_examples_to_keep is None:
        error_string = (
            'Input args {0:s} and {1:s} cannot both be empty.'
        ).format(EXAMPLE_INDICES_ARG_NAME, NUM_EXAMPLES_ARG_NAME)

        raise ValueError(error_string)

    if indices_to_keep is not None:
        error_checking.assert_is_geq_numpy_array(indices_to_keep, 0)
        error_checking.assert_is_less_than_numpy_array(
            indices_to_keep, num_examples_total
        )

        return indices_to_keep

    indices_to_keep = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )

    if num_examples_to_keep >= num_examples_total:
        return indices_to_keep

    return numpy.random.choice(
        indices_to_keep, size=num_examples_to_keep, replace=False
    )


def _run(model_file_name, example_file_name, example_indices, num_examples,
         layer_name, is_layer_output, neuron_indices, ideal_activation,
         output_file_name):
    """Makes saliency map for each example, according to one model.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param example_file_name: Same.
    :param example_indices: Same.
    :param num_examples: Same.
    :param layer_name: Same.
    :param is_layer_output: Same.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param output_file_name: Same.
    """

    this_example_dict = example_io.read_file(example_file_name)
    num_examples_total = len(this_example_dict[example_io.VALID_TIMES_KEY])

    example_indices = _subset_examples(
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
    generator_option_dict = metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    dummy_example_dict = {
        example_io.SCALAR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.SCALAR_PREDICTOR_NAMES_KEY],
        example_io.VECTOR_PREDICTOR_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY],
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY]
    }

    if is_layer_output:
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
        net_type_string=metadata_dict[neural_net.NET_TYPE_KEY],
        is_loss_constrained_mse=False
    )

    print(SEPARATOR_STRING)
    predictor_matrix, _, example_id_strings = next(generator)
    print(SEPARATOR_STRING)

    predictor_matrix = predictor_matrix[example_indices, ...]
    example_id_strings = [example_id_strings[i] for i in example_indices]

    print('Computing saliency for neuron {0:s} in layer "{1:s}"...'.format(
        str(neuron_indices), layer_name
    ))
    saliency_matrix = saliency.get_saliency_one_neuron(
        model_object=model_object, predictor_matrix=predictor_matrix,
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )

    dummy_example_dict = neural_net.predictors_numpy_to_dict(
        predictor_matrix=saliency_matrix, example_dict=dummy_example_dict,
        net_type_string=metadata_dict[neural_net.NET_TYPE_KEY]
    )
    scalar_saliency_matrix = (
        dummy_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY]
    )
    vector_saliency_matrix = (
        dummy_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY]
    )

    print('Writing saliency maps to: "{0:s}"...'.format(output_file_name))
    saliency.write_standard_file(
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
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
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
