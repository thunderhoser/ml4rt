"""Creates 7-panel figure comparing evaluation metrics across models."""

import os
import glob
import argparse
import numpy
from matplotlib import pyplot
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.plotting import imagemagick_utils
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation
from ml4rt.utils import pit_utils
from ml4rt.utils import spread_skill_utils as ss_utils
from ml4rt.utils import discard_test_utils as dt_utils

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
SINGLE_ERROR_METRIC_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
FIRST_ERROR_METRIC_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SECOND_ERROR_METRIC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

DEFAULT_FONT_SIZE = 40
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

CONVERT_EXE_NAME = '/usr/bin/convert'
TITLE_FONT_SIZE = 250
TITLE_FONT_NAME = 'DejaVu-Sans-Bold'

NUM_PANEL_ROWS = 3
NUM_PANEL_COLUMNS = 3
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

MODEL_EVAL_DIRS_ARG_NAME = 'input_model_evaluation_dir_names'
MODEL_DESCRIPTIONS_ARG_NAME = 'model_description_strings'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

MODEL_EVAL_DIRS_HELP_STRING = (
    'Space-separated list of paths to input directories, one per model.'
)
MODEL_DESCRIPTIONS_HELP_STRING = (
    'Space-separated list of model descriptions, one per model.  Within each '
    'list item, underscores will be replaced by spaces.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_EVAL_DIRS_ARG_NAME, type=str, nargs='+', required=True,
    help=MODEL_EVAL_DIRS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTIONS_ARG_NAME, type=str, nargs='+', required=True,
    help=MODEL_DESCRIPTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_bar_graph(
        first_error_values, second_error_values, first_error_metric_name,
        second_error_metric_name, y_label_string, title_string,
        model_description_strings, plotting_ssrat, output_file_name):
    """Plots one bar graph, comparing either 1 or 2 error metrics across models.

    M = number of models

    :param first_error_values: length-M numpy array with values of first error
        metric.
    :param second_error_values: length-M numpy array with values of second error
        metric.  If plotting only one error metric, make this None.
    :param first_error_metric_name: Name of first error metric (string).
    :param second_error_metric_name: Name of second error metric (string).  If
        plotting only one error metric, make this None.
    :param y_label_string: Title for y-axis.
    :param title_string: Figure title.
    :param model_description_strings: length-M numpy array of model descriptions
        (to be used as x-tick labels).
    :param plotting_ssrat: Boolean flag.  If True (False), the metric being
        plotted is spread-skill ratio (something else).
    :param output_file_name: Path to output file.  Figure will be saved here.
    """

    num_models = len(model_description_strings)
    num_scores = 1 + int(second_error_values is not None)

    bar_width = 1. / (num_scores + 1)
    bar_offset_by_metric = numpy.linspace(
        0, bar_width * (num_scores - 1), num=num_scores, dtype=float
    )
    first_x_coords = numpy.linspace(
        0, num_models - 1, num=num_models, dtype=float
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    this_colour = (
        SINGLE_ERROR_METRIC_COLOUR if num_scores == 1
        else FIRST_ERROR_METRIC_COLOUR
    )
    axes_object.bar(
        x=first_x_coords + bar_offset_by_metric[0], height=first_error_values,
        width=bar_width, align='center', color=this_colour, linewidth=0,
        label=first_error_metric_name
    )

    if num_scores == 2:
        axes_object.bar(
            x=first_x_coords + bar_offset_by_metric[1],
            height=second_error_values, width=bar_width, align='center',
            color=SECOND_ERROR_METRIC_COLOUR, linewidth=0,
            label=second_error_metric_name
        )

    if plotting_ssrat:
        reference_line_y_coords = numpy.array([1, 1], dtype=float)
        reference_line_x_coords = axes_object.get_xlim()
        axes_object.plot(
            reference_line_x_coords, reference_line_y_coords,
            color=REFERENCE_LINE_COLOUR, linewidth=4, linestyle='dashed'
        )

    axes_object.set_ylabel(y_label_string)
    axes_object.set_xticks(first_x_coords + numpy.mean(bar_offset_by_metric))
    axes_object.set_xticklabels(model_description_strings, rotation=90.)
    axes_object.set_title(title_string)

    if num_scores == 2:
        axes_object.legend(loc='upper left', ncol=num_scores)

    if plotting_ssrat:
        axes_object.set_xlim(reference_line_x_coords)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _overlay_text(
        image_file_name, x_offset_from_left_px, y_offset_from_top_px,
        text_string):
    """Creates two figures showing overall evaluation of uncertainty quant (UQ).

    :param image_file_name: Path to image file.
    :param x_offset_from_left_px: Left-relative x-coordinate (pixels).
    :param y_offset_from_top_px: Top-relative y-coordinate (pixels).
    :param text_string: String to overlay.
    :raises: ValueError: if ImageMagick command (which is ultimately a Unix
        command) fails.
    """

    command_string = (
        '"{0:s}" "{1:s}" -pointsize {2:d} -font "{3:s}" '
        '-fill "rgb(0, 0, 0)" -annotate {4:+d}{5:+d} "{6:s}" "{1:s}"'
    ).format(
        CONVERT_EXE_NAME, image_file_name, TITLE_FONT_SIZE, TITLE_FONT_NAME,
        x_offset_from_left_px, y_offset_from_top_px, text_string
    )

    exit_code = os.system(command_string)
    if exit_code == 0:
        return

    raise ValueError(imagemagick_utils.ERROR_STRING)


def _run(model_evaluation_dir_names, model_description_strings,
         output_dir_name):
    """Creates 7-panel figure comparing evaluation metrics across models.

    This is effectively the main method.

    :param model_evaluation_dir_names: See documentation at top of file.
    :param model_description_strings: Same.
    :param output_dir_name: Same.
    """

    num_models = len(model_evaluation_dir_names)
    expected_dim = numpy.array([num_models], dtype=int)

    error_checking.assert_is_numpy_array(
        numpy.array(model_description_strings), exact_dimensions=expected_dim
    )
    model_description_strings = [
        s.replace('_', ' ') for s in model_description_strings
    ]

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    heating_rate_mae_values_k_day01 = numpy.full(num_models, numpy.nan)
    flux_mae_values_w_m02 = numpy.full(num_models, numpy.nan)
    heating_rate_rel_values_k2_day02 = numpy.full(num_models, numpy.nan)
    flux_rel_values_w2_m04 = numpy.full(num_models, numpy.nan)
    heating_rate_ssrel_values_k_day01 = numpy.full(num_models, numpy.nan)
    flux_ssrel_values_w_m02 = numpy.full(num_models, numpy.nan)
    heating_rate_ssrat_values = numpy.full(num_models, numpy.nan)
    flux_ssrat_values = numpy.full(num_models, numpy.nan)
    heating_rate_pitd_values = numpy.full(num_models, numpy.nan)
    flux_pitd_values = numpy.full(num_models, numpy.nan)
    heating_rate_mf_values = numpy.full(num_models, numpy.nan)
    flux_mf_values = numpy.full(num_models, numpy.nan)
    heating_rate_cat_error_freqs = numpy.full(num_models, numpy.nan)
    flux_cat_error_freqs = numpy.full(num_models, numpy.nan)

    for i in range(num_models):
        these_prediction_file_names = glob.glob(
            '{0:s}/predictions_part*.nc'.format(model_evaluation_dir_names[i])
        )
        these_prediction_dicts = []
        for this_file_name in these_prediction_file_names:
            print('Reading data from: "{0:s}"...'.format(this_file_name))
            these_prediction_dicts.append(
                prediction_io.read_file(this_file_name)
            )

        this_prediction_dict = prediction_io.concat_predictions(
            these_prediction_dicts
        )

        this_target_matrix = this_prediction_dict[
            prediction_io.VECTOR_TARGETS_KEY
        ]
        assert this_target_matrix.shape[2] == 1
        this_target_matrix = this_target_matrix[..., 0]

        this_prediction_matrix = this_prediction_dict[
            prediction_io.VECTOR_PREDICTIONS_KEY
        ][..., 0, :]
        this_mean_prediction_matrix = numpy.mean(
            this_prediction_matrix, axis=-1
        )
        this_pit_matrix = numpy.full(this_target_matrix.shape, numpy.nan)

        for j in range(this_target_matrix.shape[0]):
            for k in range(this_target_matrix.shape[1]):
                this_pit_matrix[j, k] = 0.01 * percentileofscore(
                    a=this_prediction_matrix[j, k, :],
                    score=this_target_matrix[j, k], kind='mean'
                )

        this_large_error_flag_matrix = (
            numpy.absolute(this_target_matrix - this_mean_prediction_matrix)
            >= 1
        )
        this_extreme_pit_flag_matrix = numpy.logical_or(
            this_pit_matrix < 0.025, this_pit_matrix > 0.975
        )
        heating_rate_cat_error_freqs[i] = numpy.mean(numpy.logical_and(
            this_large_error_flag_matrix, this_extreme_pit_flag_matrix
        ))

        this_target_matrix = this_prediction_dict[
            prediction_io.SCALAR_TARGETS_KEY
        ]
        assert this_target_matrix.shape[1] == 2
        this_target_matrix = numpy.concatenate((
            this_target_matrix,
            this_target_matrix[:, [0]] - this_target_matrix[:, [1]]
        ), axis=1)

        this_prediction_matrix = this_prediction_dict[
            prediction_io.SCALAR_PREDICTIONS_KEY
        ]
        this_prediction_matrix = numpy.concatenate((
            this_prediction_matrix,
            this_prediction_matrix[:, [0], :] - this_prediction_matrix[:, [1], :]
        ), axis=1)

        this_mean_prediction_matrix = numpy.mean(
            this_prediction_matrix, axis=-1
        )
        this_pit_matrix = numpy.full(this_target_matrix.shape, numpy.nan)

        for j in range(this_target_matrix.shape[0]):
            for k in range(this_target_matrix.shape[1]):
                this_pit_matrix[j, k] = 0.01 * percentileofscore(
                    a=this_prediction_matrix[j, k, :],
                    score=this_target_matrix[j, k], kind='mean'
                )

        this_large_error_flag_matrix = (
            numpy.absolute(this_target_matrix - this_mean_prediction_matrix)
            >= 1
        )
        this_extreme_pit_flag_matrix = numpy.logical_or(
            this_pit_matrix < 0.025, this_pit_matrix > 0.975
        )
        flux_cat_error_freqs[i] = numpy.mean(numpy.logical_and(
            this_large_error_flag_matrix, this_extreme_pit_flag_matrix
        ))

        this_file_name = '{0:s}/evaluation.nc'.format(
            model_evaluation_dir_names[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_eval_table_xarray = evaluation.read_file(this_file_name)

        this_mae_matrix = numpy.concatenate((
            this_eval_table_xarray[evaluation.SCALAR_MAE_KEY].values,
            this_eval_table_xarray[evaluation.AUX_MAE_KEY].values
        ), axis=0)

        assert this_mae_matrix.shape[0] == 3
        flux_mae_values_w_m02[i] = numpy.mean(
            numpy.nanmean(this_mae_matrix, axis=1)
        )

        this_mae_matrix = (
            this_eval_table_xarray[evaluation.VECTOR_MAE_KEY].values
        )
        assert this_mae_matrix.shape[1] == 1
        this_mae_matrix = this_mae_matrix[:, 0, :]
        heating_rate_mae_values_k_day01[i] = numpy.mean(
            numpy.nanmean(this_mae_matrix, axis=1)
        )

        this_reliability_matrix = numpy.concatenate((
            this_eval_table_xarray[evaluation.SCALAR_RELIABILITY_KEY].values,
            this_eval_table_xarray[evaluation.AUX_RELIABILITY_KEY].values
        ), axis=0)

        flux_rel_values_w2_m04[i] = numpy.mean(
            numpy.nanmean(this_reliability_matrix, axis=1)
        )

        this_reliability_matrix = this_eval_table_xarray[
            evaluation.VECTOR_FLAT_RELIABILITY_KEY
        ].values
        heating_rate_rel_values_k2_day02[i] = numpy.nanmean(
            this_reliability_matrix
        )

        this_file_name = '{0:s}/spread_vs_skill.nc'.format(
            model_evaluation_dir_names[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_ss_table_xarray = ss_utils.read_results(this_file_name)

        these_ssrel = numpy.concatenate((
            this_ss_table_xarray[ss_utils.SCALAR_SSREL_KEY].values,
            this_ss_table_xarray[ss_utils.AUX_SSREL_KEY].values
        ))
        flux_ssrel_values_w_m02[i] = numpy.mean(these_ssrel)

        these_ssrat = numpy.concatenate((
            this_ss_table_xarray[ss_utils.SCALAR_SSRAT_KEY].values,
            this_ss_table_xarray[ss_utils.AUX_SSRAT_KEY].values
        ))
        flux_ssrat_values[i] = numpy.mean(these_ssrat)

        heating_rate_ssrel_values_k_day01[i] = (
            this_ss_table_xarray[ss_utils.VECTOR_FLAT_SSREL_KEY].values[0]
        )
        heating_rate_ssrat_values[i] = (
            this_ss_table_xarray[ss_utils.VECTOR_FLAT_SSRAT_KEY].values[0]
        )

        this_file_name = '{0:s}/pit_histograms.nc'.format(
            model_evaluation_dir_names[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_pit_table_xarray = pit_utils.read_results(this_file_name)

        these_pitd = numpy.concatenate((
            this_pit_table_xarray[pit_utils.SCALAR_PITD_KEY].values,
            this_pit_table_xarray[pit_utils.AUX_PITD_KEY].values
        ))
        flux_pitd_values[i] = numpy.mean(these_pitd)

        heating_rate_pitd_values[i] = (
            this_pit_table_xarray[pit_utils.VECTOR_FLAT_PITD_KEY].values[0]
        )

        this_file_name = '{0:s}/discard_test_for_heating_rates.nc'.format(
            model_evaluation_dir_names[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_dt_table_xarray = dt_utils.read_results(this_file_name)

        these_mf = numpy.concatenate((
            this_dt_table_xarray[dt_utils.SCALAR_MONO_FRACTION_KEY].values,
            this_dt_table_xarray[dt_utils.AUX_MONO_FRACTION_KEY].values
        ))
        flux_mf_values[i] = numpy.mean(these_mf)

        heating_rate_mf_values[i] = this_dt_table_xarray[
            dt_utils.VECTOR_FLAT_MONO_FRACTION_KEY
        ].values[0]

    # TODO(thunderhoser): Make sure none of the code in the above for-loop is
    # fucked up.

    panel_file_names = [
        '{0:s}/heating_rate_mae_ssrel.jpg'.format(output_dir_name)
    ]
    _plot_one_bar_graph(
        first_error_values=heating_rate_mae_values_k_day01,
        second_error_values=heating_rate_ssrel_values_k_day01,
        first_error_metric_name='MAE',
        second_error_metric_name='SSREL',
        y_label_string=r'Error metric (MAE or SSREL; K day$^{-1}$)',
        title_string='MAE and SSREL for heating rates',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/flux_mae_ssrel.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=flux_mae_values_w_m02,
        second_error_values=flux_ssrel_values_w_m02,
        first_error_metric_name='MAE',
        second_error_metric_name='SSREL',
        y_label_string=r'Error metric (MAE or SSREL; W m$^{-2}$)',
        title_string='MAE and SSREL for flux components',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/heating_rate_reliability.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=heating_rate_rel_values_k2_day02,
        second_error_values=None,
        first_error_metric_name='REL',
        second_error_metric_name=None,
        y_label_string=r'Reliability (K$^{2}$ day$^{-2}$)',
        title_string='Reliability (REL) for heating rates',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/flux_reliability.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=flux_rel_values_w2_m04,
        second_error_values=None,
        first_error_metric_name='REL',
        second_error_metric_name=None,
        y_label_string=r'Reliability (W$^{2}$ m$^{-4}$)',
        title_string='Reliability (REL) for flux components',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/ssrat.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=heating_rate_ssrat_values,
        second_error_values=flux_ssrat_values,
        first_error_metric_name='Heating-rate SSRAT',
        second_error_metric_name='Flux SSRAT',
        y_label_string='Spread-skill ratio',
        title_string='SSRAT for heating rates and flux components',
        model_description_strings=model_description_strings,
        plotting_ssrat=True, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/pitd.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=heating_rate_pitd_values,
        second_error_values=flux_pitd_values,
        first_error_metric_name='Heating-rate PITD',
        second_error_metric_name='Flux PITD',
        y_label_string='Probability-integral-transform deviation',
        title_string='PITD for heating rates and flux components',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/mono_fraction.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=heating_rate_mf_values,
        second_error_values=flux_mf_values,
        first_error_metric_name='Heating-rate MF',
        second_error_metric_name='Flux MF',
        y_label_string='Monotonicity fraction',
        title_string='MF for heating rates and flux components',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/mono_fraction.jpg'.format(output_dir_name)
    )
    _plot_one_bar_graph(
        first_error_values=heating_rate_cat_error_freqs,
        second_error_values=flux_cat_error_freqs,
        first_error_metric_name='Heating-rate CEF',
        second_error_metric_name='Flux CEF',
        y_label_string='Catastrophic-error frequency',
        title_string='CEF for heating rates and flux components',
        model_description_strings=model_description_strings,
        plotting_ssrat=False, output_file_name=panel_file_names[-1]
    )

    letter_label = None

    for i in range(len(panel_file_names)):
        print('Adding letter label to panel: "{0:s}"...'.format(
            panel_file_names[i]
        ))

        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            border_width_pixels=TITLE_FONT_SIZE
        )

        if letter_label is None:
            letter_label = 'a'
        else:
            letter_label = chr(ord(letter_label) + 1)

        _overlay_text(
            image_file_name=panel_file_names[i],
            x_offset_from_left_px=0, y_offset_from_top_px=TITLE_FONT_SIZE,
            text_string='({0:s})'.format(letter_label)
        )
        imagemagick_utils.trim_whitespace(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            border_width_pixels=10
        )
        imagemagick_utils.resize_image(
            input_file_name=panel_file_names[i],
            output_file_name=panel_file_names[i],
            output_size_pixels=PANEL_SIZE_PX
        )

    concat_figure_file_name = '{0:s}/overall_model_comparison.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=NUM_PANEL_ROWS, num_panel_columns=NUM_PANEL_COLUMNS
    )
    imagemagick_utils.resize_image(
        input_file_name=concat_figure_file_name,
        output_file_name=concat_figure_file_name,
        output_size_pixels=CONCAT_FIGURE_SIZE_PX
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        model_evaluation_dir_names=getattr(
            INPUT_ARG_OBJECT, MODEL_EVAL_DIRS_ARG_NAME
        ),
        model_description_strings=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTIONS_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
