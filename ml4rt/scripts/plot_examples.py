"""Plots profiles (vector predictor and target variables) for each example."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io
from ml4rt.utils import misc as misc_utils
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting
from ml4rt.scripts import make_saliency_maps
from ml4rt.scripts import plot_backwards_optimization as plot_bwo

MINOR_SEPARATOR_STRING = '\n\n' + '-' * 50 + '\n\n'

PREDICTOR_NAMES_BY_SET = [
    plot_bwo.FIRST_PREDICTOR_NAMES, plot_bwo.SECOND_PREDICTOR_NAMES,
    plot_bwo.THIRD_PREDICTOR_NAMES
]
PREDICTOR_COLOURS_BY_SET = [
    plot_bwo.FIRST_PREDICTOR_COLOURS, plot_bwo.SECOND_PREDICTOR_COLOURS,
    plot_bwo.THIRD_PREDICTOR_COLOURS
]

LINE_WIDTH = 2
FIGURE_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_FILE_ARG_NAME
NUM_EXAMPLES_ARG_NAME = make_saliency_maps.NUM_EXAMPLES_ARG_NAME
EXAMPLE_DIR_ARG_NAME = make_saliency_maps.EXAMPLE_DIR_ARG_NAME
EXAMPLE_ID_FILE_ARG_NAME = make_saliency_maps.EXAMPLE_ID_FILE_ARG_NAME
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
MODEL_FILE_ARG_NAME = 'model_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_FILE_HELP_STRING
NUM_EXAMPLES_HELP_STRING = make_saliency_maps.NUM_EXAMPLES_HELP_STRING
EXAMPLE_DIR_HELP_STRING = make_saliency_maps.EXAMPLE_DIR_HELP_STRING
EXAMPLE_ID_FILE_HELP_STRING = make_saliency_maps.EXAMPLE_ID_FILE_HELP_STRING
USE_LOG_SCALE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use logarithmic (linear) scale for height '
    'axis.'
)
MODEL_FILE_HELP_STRING = (
    '[optional] Path to model (readable by `neural_net.read_model`).  If '
    'specified, this script will plot only the variables/heights used by the '
    'model.'
)
OUTPUT_DIR_HELP_STRING = 'Name of output directory (figures will be saved here).'

INPUT_ARG_PARSER = argparse.ArgumentParser()
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
    '--' + USE_LOG_SCALE_ARG_NAME, type=int, required=False, default=1,
    help=USE_LOG_SCALE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_FILE_ARG_NAME, type=str, required=False, default='',
    help=MODEL_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_example(
        example_dict, example_index, example_id_string, use_log_scale,
        output_dir_name):
    """Plots data for one example.

    :param example_dict: See doc for `example_io.read_file`.
    :param example_index: Will plot results for example with this array index.
    :param example_id_string: Example ID.
    :param use_log_scale: See documentation at top of file.
    :param output_dir_name: Name of output directory.  Figures will be saved
        here.
    """

    num_predictor_sets = len(PREDICTOR_NAMES_BY_SET)

    for k in range(num_predictor_sets):
        these_flags = numpy.array([
            n in example_dict[example_io.VECTOR_PREDICTOR_NAMES_KEY]
            for n in PREDICTOR_NAMES_BY_SET[k]
        ], dtype=bool)

        these_indices = numpy.where(these_flags)[0]
        if len(these_indices) == 0:
            continue

        predictor_names = [PREDICTOR_NAMES_BY_SET[k][i] for i in these_indices]
        predictor_colours = [
            PREDICTOR_COLOURS_BY_SET[k][i] for i in these_indices
        ]

        handle_dict = profile_plotting.plot_predictors(
            example_dict=example_dict, example_index=example_index,
            predictor_names=predictor_names,
            predictor_colours=predictor_colours,
            predictor_line_widths=
            numpy.full(len(these_indices), LINE_WIDTH),
            predictor_line_styles=['solid'] * len(these_indices),
            use_log_scale=use_log_scale
        )

        output_file_name = '{0:s}/{1:s}_predictor-set-{2:d}.jpg'.format(
            output_dir_name, example_id_string.replace('_', '-'), k
        )
        figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

        print('Saving figure to: "{0:s}"...'.format(output_file_name))
        figure_object.savefig(
            output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
            bbox_inches='tight'
        )
        pyplot.close(figure_object)

    handle_dict = profile_plotting.plot_targets(
        example_dict=example_dict, example_index=example_index,
        use_log_scale=use_log_scale, line_width=LINE_WIDTH, line_style='solid'
    )

    output_file_name = '{0:s}/{1:s}_targets.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI, pad_inches=0,
        bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(example_file_name, num_examples, example_dir_name,
         example_id_file_name, use_log_scale, model_file_name, output_dir_name):
    """Plots profiles (vector predictor and target variables) for each example.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param num_examples: Same.
    :param example_dir_name: Same.
    :param example_id_file_name: Same.
    :param use_log_scale: Same.
    :param model_file_name: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    example_dict = misc_utils.get_raw_examples(
        example_file_name=example_file_name, num_examples=num_examples,
        example_dir_name=example_dir_name,
        example_id_file_name=example_id_file_name
    )

    example_id_strings = example_io.create_example_ids(example_dict)

    if model_file_name != '':
        model_metafile_name = neural_net.find_metafile(
            os.path.split(model_file_name)[0]
        )

        print('Reading model metadata from: "{0:s}"...\n'.format(
            model_metafile_name
        ))
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)

        generator_option_dict = (
            model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
        )
        vector_predictor_names = (
            generator_option_dict[neural_net.VECTOR_PREDICTOR_NAMES_KEY]
        )
        all_field_names = (
            vector_predictor_names + example_io.DEFAULT_VECTOR_TARGET_NAMES
        )

        example_dict = example_io.subset_by_field(
            example_dict=example_dict, field_names=all_field_names
        )

    num_examples = len(example_id_strings)
    for i in range(num_examples):
        _plot_one_example(
            example_dict=example_dict, example_index=i,
            example_id_string=example_id_strings[i],
            use_log_scale=use_log_scale, output_dir_name=output_dir_name
        )

        print(MINOR_SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        example_id_file_name=getattr(
            INPUT_ARG_OBJECT, EXAMPLE_ID_FILE_ARG_NAME
        ),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        model_file_name=getattr(INPUT_ARG_OBJECT, MODEL_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
