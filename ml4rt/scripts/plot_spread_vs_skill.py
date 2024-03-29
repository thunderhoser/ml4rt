"""Plots spread-skill relationship for each target variable."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.utils import spread_skill_utils as ss_utils
from ml4rt.plotting import evaluation_plotting
from ml4rt.plotting import uq_evaluation_plotting as uq_eval_plotting

ERROR_PROFILE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_file_name'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file.  Will be read by `ss_utils.read_results`.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, output_dir_name):
    """Plots spread-skill relationship for each target variable.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    result_table_xarray = ss_utils.read_results(input_file_name)
    t = result_table_xarray

    for this_var_name in t.coords[ss_utils.SCALAR_FIELD_DIM].values:
        figure_object, _ = uq_eval_plotting.plot_spread_vs_skill(
            result_table_xarray=result_table_xarray,
            target_var_name=this_var_name
        )

        this_figure_file_name = '{0:s}/spread_vs_skill_{1:s}.jpg'.format(
            output_dir_name, this_var_name.replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        figure_object.savefig(
            this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for this_var_name in t.coords[ss_utils.AUX_TARGET_FIELD_DIM].values:
        figure_object, _ = uq_eval_plotting.plot_spread_vs_skill(
            result_table_xarray=result_table_xarray,
            target_var_name=this_var_name
        )

        this_figure_file_name = '{0:s}/spread_vs_skill_{1:s}.jpg'.format(
            output_dir_name, this_var_name.replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        figure_object.savefig(
            this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    for this_var_name in t.coords[ss_utils.VECTOR_FIELD_DIM].values:
        figure_object, _ = uq_eval_plotting.plot_spread_vs_skill(
            result_table_xarray=result_table_xarray,
            target_var_name=this_var_name
        )

        this_figure_file_name = '{0:s}/spread_vs_skill_{1:s}.jpg'.format(
            output_dir_name, this_var_name.replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(this_figure_file_name))
        figure_object.savefig(
            this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        for this_height_m_agl in t.coords[ss_utils.HEIGHT_DIM].values:
            figure_object, _ = uq_eval_plotting.plot_spread_vs_skill(
                result_table_xarray=result_table_xarray,
                target_var_name=this_var_name,
                target_height_m_agl=this_height_m_agl
            )

            this_figure_file_name = (
                '{0:s}/spread_vs_skill_{1:s}_{2:05d}-m-agl.jpg'
            ).format(
                output_dir_name, this_var_name.replace('_', '-'),
                int(numpy.round(this_height_m_agl))
            )
            print('Saving figure to file: "{0:s}"...'.format(
                this_figure_file_name
            ))
            figure_object.savefig(
                this_figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(figure_object)

    for j in range(len(t.coords[ss_utils.VECTOR_FIELD_DIM].values)):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        evaluation_plotting.plot_score_profile(
            heights_m_agl=t.coords[ss_utils.HEIGHT_DIM].values,
            score_values=t[ss_utils.VECTOR_SSREL_KEY].values[j, :],
            score_name=evaluation_plotting.SSREL_NAME,
            line_colour=ERROR_PROFILE_COLOUR, line_width=4, line_style='solid',
            use_log_scale=True, axes_object=axes_object,
            are_axes_new=True
        )

        this_var_name = t.coords[ss_utils.VECTOR_FIELD_DIM].values[j]
        axes_object.set_xlabel(
            'Spread-skill reliability (SSREL; {0:s})'.format(
                uq_eval_plotting.TARGET_NAME_TO_UNITS[this_var_name]
            )
        )
        axes_object.set_title('SSREL for {0:s}'.format(
            uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[this_var_name]
        ))

        figure_file_name = '{0:s}/ssrel_{1:s}.jpg'.format(
            output_dir_name, this_var_name.replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        evaluation_plotting.plot_score_profile(
            heights_m_agl=t.coords[ss_utils.HEIGHT_DIM].values,
            score_values=t[ss_utils.VECTOR_SSRAT_KEY].values[j, :],
            score_name=evaluation_plotting.SSRAT_NAME,
            line_colour=ERROR_PROFILE_COLOUR, line_width=4, line_style='solid',
            use_log_scale=True, axes_object=axes_object,
            are_axes_new=True
        )

        axes_object.set_xlim([0., 2.])

        this_var_name = t.coords[ss_utils.VECTOR_FIELD_DIM].values[j]
        axes_object.set_xlabel('Spread-skill ratio (SSRAT)')
        axes_object.set_title('SSRAT for {0:s}\nMin value = {1:.2f}'.format(
            uq_eval_plotting.TARGET_NAME_ABBREV_TO_FANCY[this_var_name],
            numpy.nanmin(t[ss_utils.VECTOR_SSRAT_KEY].values[j, :])
        ))

        figure_file_name = '{0:s}/ssrat_{1:s}.jpg'.format(
            output_dir_name, this_var_name.replace('_', '-')
        )
        print('Saving figure to file: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
