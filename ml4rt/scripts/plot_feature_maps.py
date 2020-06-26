"""Plots feature maps for each layer/example pair, for a single neural net."""

import os.path
import argparse
import numpy
import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import feature_map_plotting
from ml4rt.io import example_io
from ml4rt.machine_learning import neural_net

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
FIGURE_RESOLUTION_DPI = 300

MODEL_FILE_ARG_NAME = 'input_model_file_name'
LAYER_NAMES_ARG_NAME = 'layer_names'
EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
LAYER_NAMES_HELP_STRING = (
    'List of layer names.  Feature maps will be plotted for each layer/example '
    'pair.'
)
EXAMPLE_FILE_HELP_STRING = (
    'Path to file with data examples.  Will be read by `example_io.read_file`.'
)
EXAMPLE_INDICES_HELP_STRING = (
    'Indices of examples to plot.  If you do not want to plot specific '
    'examples, leave this alone.'
)
NUM_EXAMPLES_HELP_STRING = (
    '[used only if `{0:s}` is not specified] Number of examples to plot (these '
    'will be selected randomly).  If you want to plot all examples, leave this '
    'alone.'
).format(EXAMPLE_INDICES_ARG_NAME)

OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=True,
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAYER_NAMES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAYER_NAMES_HELP_STRING
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
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


def _plot_feature_maps_one_layer(
        feature_matrix, example_id_strings, layer_name, output_dir_name):
    """Plots feature maps for one layer.

    E = number of examples
    H = number of heights
    C = number of channels

    :param feature_matrix: E-by-H-by-C numpy array of feature maps.
    :param example_id_strings: length-E list of example IDs.
    :param layer_name: Name of layer that generated feature maps.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    error_checking.assert_is_numpy_array(feature_matrix, num_dimensions=3)
    num_examples = feature_matrix.shape[0]

    # TODO(thunderhoser): Maybe define colour limits differently?
    max_colour_value = numpy.percentile(numpy.absolute(feature_matrix), 99)
    min_colour_value = -1 * max_colour_value

    for i in range(num_examples):
        this_figure_object, this_axes_object_matrix = (
            feature_map_plotting.plot_many_1d_feature_maps(
                feature_matrix=feature_matrix[i, ...],
                colour_map_object=COLOUR_MAP_OBJECT,
                min_colour_value=min_colour_value,
                max_colour_value=max_colour_value)
        )

        plotting_utils.plot_linear_colour_bar(
            axes_object_or_matrix=this_axes_object_matrix,
            data_matrix=feature_matrix[i, ...],
            colour_map_object=COLOUR_MAP_OBJECT,
            min_value=min_colour_value, max_value=max_colour_value,
            orientation_string='horizontal', padding=0.01,
            extend_min=True, extend_max=True
        )

        this_title_string = 'Layer "{0:s}", example "{1:s}"'.format(
            layer_name, example_id_strings[i]
        )
        this_figure_object.suptitle(this_title_string, fontsize=25)

        this_file_name = '{0:s}/{1:s}.jpg'.format(
            output_dir_name, example_id_strings[i]
        )

        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)


def _run(model_file_name, layer_names, example_file_name, example_indices,
         num_examples, output_dir_name):
    """Plots feature maps for each layer/example pair, for a single neural net.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param layer_names: Same.
    :param example_file_name: Same.
    :param example_indices: Same.
    :param num_examples: Same.
    :param output_dir_name: Same.
    :raises: ValueError: if neural-net type is not CNN or U-net.
    """

    example_dict = example_io.read_file(example_file_name)
    num_examples_total = len(example_dict[example_io.VALID_TIMES_KEY])

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
        net_type_string=net_type_string, use_custom_cnn_loss=False
    )

    print(SEPARATOR_STRING)
    predictor_matrix, _, example_id_strings = next(generator)
    print(SEPARATOR_STRING)

    predictor_matrix = predictor_matrix[example_indices, ...]
    example_id_strings = [example_id_strings[i] for i in example_indices]

    num_layers = len(layer_names)
    feature_matrix_by_layer = [numpy.array([])] * num_layers

    for k in range(num_layers):
        print('Creating feature maps for layer "{0:s}"...'.format(
            layer_names[k]
        ))

        feature_matrix_by_layer[k] = neural_net.get_feature_maps(
            model_object=model_object, predictor_matrix=predictor_matrix,
            num_examples_per_batch=predictor_matrix.shape[0],
            feature_layer_name=layer_names[k], verbose=False
        )

    print('\n')

    for k in range(num_layers):
        this_output_dir_name = '{0:s}/{1:s}'.format(
            output_dir_name, layer_names[k]
        )

        file_system_utils.mkdir_recursive_if_necessary(
            directory_name=this_output_dir_name
        )

        _plot_feature_maps_one_layer(
            feature_matrix=feature_matrix_by_layer[k],
            example_id_strings=example_id_strings,
            layer_name=layer_names[k],
            output_dir_name=this_output_dir_name
        )

        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        layer_names=getattr(INPUT_ARG_OBJECT, LAYER_NAMES_ARG_NAME),
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
