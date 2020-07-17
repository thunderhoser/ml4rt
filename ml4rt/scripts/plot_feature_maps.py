"""Plots feature maps for each layer/example pair, for a single neural net."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import plotting_utils
from gewittergefahr.plotting import feature_map_plotting
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import neural_net
from ml4rt.scripts import make_saliency_maps

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

COLOUR_MAP_OBJECT = pyplot.get_cmap('seismic')
FIGURE_RESOLUTION_DPI = 300

MODEL_FILE_ARG_NAME = 'input_model_file_name'
LAYER_NAMES_ARG_NAME = 'layer_names'
EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_ID_FILE_ARG_NAME
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_FILE_HELP_STRING = (
    'Path to trained model.  Will be read by `neural_net.read_model`.'
)
LAYER_NAMES_HELP_STRING = (
    'List of layer names.  Feature maps will be plotted for each layer/example '
    'pair.'
)
EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_ID_FILE_HELP_STRING
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
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
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


def _run(model_file_name, layer_names, example_file_name, num_examples,
         example_dir_name, example_id_file_name, output_dir_name):
    """Plots feature maps for each layer/example pair, for a single neural net.

    This is effectively the main method.

    :param model_file_name: See documentation at top of file.
    :param layer_names: Same.
    :param example_file_name: Same.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param output_dir_name: Same.
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
        misc_utils.get_examples_from_inference(
            model_metadata_dict=metadata_dict,
            example_file_name=example_file_name,
            num_examples=num_examples, example_dir_name=example_dir_name,
            example_id_file_name=example_id_file_name
        )
    )
    print(SEPARATOR_STRING)

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
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_ID_FILE_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
