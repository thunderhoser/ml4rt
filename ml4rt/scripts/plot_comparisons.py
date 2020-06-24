"""Plots comparisons between predicted and actual (target) profiles."""

import os.path
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting

FIGURE_RESOLUTION_DPI = 300

TARGET_NAME_TO_VERBOSE = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: 'Downwelling shortwave flux',
    example_io.SHORTWAVE_UP_FLUX_NAME: 'Upwelling shortwave flux',
    example_io.SHORTWAVE_HEATING_RATE_NAME: 'Shortwave heating rate'
}

TARGET_NAME_TO_UNITS = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_io.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$'
}

TARGET_NAME_TO_COLOUR = {
    example_io.SHORTWAVE_DOWN_FLUX_NAME:
        profile_plotting.DOWNWELLING_FLUX_COLOUR,
    example_io.SHORTWAVE_UP_FLUX_NAME: profile_plotting.UPWELLING_FLUX_COLOUR,
    example_io.SHORTWAVE_HEATING_RATE_NAME: profile_plotting.HEATING_RATE_COLOUR
}

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
EXAMPLE_INDICES_ARG_NAME = 'example_indices'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, containing both predicted and actual (target) '
    'profiles.  Will be read by `prediction_io.read_file`.'
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
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
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


def _plot_comparisons_fancy(
        vector_target_matrix, vector_prediction_matrix, model_metadata_dict,
        output_dir_name):
    """Plots fancy comparisons (with all target variables in the same plot).

    E = number of examples
    H = number of heights
    T = number of target variables

    :param vector_target_matrix: E-by-H-by-T numpy array of target (actual)
        values.
    :param vector_prediction_matrix: E-by-H-by-T numpy array of predicted
        values.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metadata`.
    :param output_dir_name: Path to output directory (figures will be saved
        here).
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    target_example_dict = {
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY],
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.VECTOR_TARGET_VALS_KEY: vector_target_matrix
    }

    prediction_example_dict = {
        example_io.HEIGHTS_KEY: generator_option_dict[neural_net.HEIGHTS_KEY],
        example_io.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_io.VECTOR_TARGET_VALS_KEY: vector_prediction_matrix
    }

    num_examples = vector_target_matrix.shape[0]

    for i in range(num_examples):
        this_handle_dict = profile_plotting.plot_targets(
            example_dict=target_example_dict, example_index=i,
            use_log_scale=False, line_style='solid', handle_dict=None
        )
        profile_plotting.plot_targets(
            example_dict=prediction_example_dict, example_index=i,
            use_log_scale=False, line_style='dashed',
            handle_dict=this_handle_dict
        )

        this_file_name = '{0:s}/comparison_example{1:06d}.jpg'.format(
            output_dir_name, i
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))

        this_figure_object = (
            this_handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
        )
        this_figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(this_figure_object)


def _plot_comparisons_simple(
        vector_target_matrix, vector_prediction_matrix, model_metadata_dict,
        output_dir_name):
    """Plots simple comparisons (with each target var in a different plot).

    :param vector_target_matrix: See doc for `_plot_comparisons_fancy`.
    :param vector_prediction_matrix: Same.
    :param model_metadata_dict: Same.
    :param output_dir_name: Same.
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    target_names = generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    num_examples = vector_target_matrix.shape[0]
    num_target_vars = len(target_names)

    for i in range(num_examples):
        for k in range(num_target_vars):
            this_figure_object, this_axes_object = (
                profile_plotting.plot_one_variable(
                    values=vector_target_matrix[i, :, k],
                    heights_m_agl=heights_m_agl, use_log_scale=False,
                    line_colour=TARGET_NAME_TO_COLOUR[target_names[k]],
                    line_style='solid', figure_object=None
                )
            )

            profile_plotting.plot_one_variable(
                values=vector_prediction_matrix[i, :, k],
                heights_m_agl=heights_m_agl, use_log_scale=False,
                line_colour=TARGET_NAME_TO_COLOUR[target_names[k]],
                line_style='dashed', figure_object=this_figure_object
            )

            this_axes_object.set_xlabel('{0:s} ({1:s})'.format(
                TARGET_NAME_TO_VERBOSE[target_names[k]],
                TARGET_NAME_TO_UNITS[target_names[k]]
            ))

            this_file_name = '{0:s}/comparison_{1:s}_example{2:06d}.jpg'.format(
                output_dir_name, target_names[k].replace('_', '-'), i
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)


def _run(prediction_file_name, example_indices, num_examples, output_dir_name):
    """Plots comparisons between predicted and actual (target) profiles.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param example_indices: Same.
    :param num_examples: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    if len(example_indices) == 1 and example_indices[0] < 0:
        example_indices = None
    if example_indices is not None:
        num_examples = None
    if num_examples < 1:
        num_examples = None

    print((
        'Reading predicted and actual (target) profiles from: "{0:s}"...'
    ).format(
        prediction_file_name
    ))

    prediction_dict = prediction_io.read_file(prediction_file_name)
    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )

    num_examples_total = vector_target_matrix.shape[0]

    if example_indices is not None:
        error_checking.assert_is_geq_numpy_array(example_indices, 0)
        error_checking.assert_is_less_than_numpy_array(
            example_indices, num_examples_total
        )

        vector_target_matrix = vector_target_matrix[example_indices, ...]
        vector_prediction_matrix = (
            vector_prediction_matrix[example_indices, ...]
        )

    if num_examples >= num_examples_total:
        num_examples = None

    if num_examples is not None:
        example_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )
        example_indices = numpy.random.choice(
            example_indices, size=num_examples, replace=False
        )

        vector_target_matrix = vector_target_matrix[example_indices, ...]
        vector_prediction_matrix = (
            vector_prediction_matrix[example_indices, ...]
        )

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metadata(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    vector_target_names = (
        generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    if all([t in vector_target_names for t in example_io.VECTOR_TARGET_NAMES]):
        _plot_comparisons_fancy(
            vector_target_matrix=vector_target_matrix,
            vector_prediction_matrix=vector_prediction_matrix,
            model_metadata_dict=model_metadata_dict,
            output_dir_name=output_dir_name
        )
    else:
        _plot_comparisons_simple(
            vector_target_matrix=vector_target_matrix,
            vector_prediction_matrix=vector_prediction_matrix,
            model_metadata_dict=model_metadata_dict,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        example_indices=numpy.array(
            getattr(INPUT_ARG_OBJECT, EXAMPLE_INDICES_ARG_NAME), dtype=int
        ),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
