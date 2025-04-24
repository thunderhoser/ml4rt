"""Creates 7-panel figure comparing evaluation metrics across models."""

import os
import sys
import glob
import argparse
from multiprocessing import Pool
import numpy
from matplotlib import pyplot
from scipy.stats import percentileofscore

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import imagemagick_utils
import prediction_io
import evaluation
import pit_utils
import spread_skill_utils as ss_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
NUM_SLICES_FOR_MULTIPROCESSING = 24

FIGURE_WIDTH_INCHES = 20
FIGURE_HEIGHT_INCHES = 12
FIGURE_RESOLUTION_DPI = 300

REFERENCE_LINE_COLOUR = numpy.full(3, 152. / 255)
SINGLE_ERROR_METRIC_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
FIRST_ERROR_METRIC_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
SECOND_ERROR_METRIC_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255

DEFAULT_FONT_SIZE = 30
pyplot.rc('font', size=DEFAULT_FONT_SIZE)
pyplot.rc('axes', titlesize=DEFAULT_FONT_SIZE)
pyplot.rc('axes', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('xtick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('ytick', labelsize=DEFAULT_FONT_SIZE)
pyplot.rc('legend', fontsize=DEFAULT_FONT_SIZE)
pyplot.rc('figure', titlesize=DEFAULT_FONT_SIZE)

PANEL_SIZE_PX = int(1e7)
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


def __get_slices_for_multiprocessing(num_examples):
    """Returns slices for multiprocessing.

    Each slice consists of many examples.

    K = number of slices

    :param num_examples: Total number of examples.
    :return: start_indices: length-K numpy array with index of each start
        example.
    :return: end_indices: length-K numpy array with index of each end example.
    """

    slice_indices_normalized = numpy.linspace(
        0, 1, num=NUM_SLICES_FOR_MULTIPROCESSING + 1, dtype=float
    )

    start_indices = numpy.round(
        num_examples * slice_indices_normalized[:-1]
    ).astype(int)

    end_indices = numpy.round(
        num_examples * slice_indices_normalized[1:]
    ).astype(int)

    return start_indices, end_indices


def _compute_pit_values_hr_or_flux(target_matrix, prediction_matrix):
    """Computes PIT values for either heating rate or flux.

    PIT = probability integral transform

    E = number of examples
    H = number of heights (for heating rate) or number of variables (for flux)
    S = ensemble size

    :param target_matrix: E-by-H numpy array of actual values.
    :param prediction_matrix: E-by-H-by-S numpy array of predicted values.
    :return: pit_matrix: E-by-H numpy array of PIT values.
    """

    pit_matrix = numpy.full(target_matrix.shape, numpy.nan)

    for j in range(target_matrix.shape[0]):
        if numpy.mod(j, 100) == 0:
            print((
                'Have computed PIT values for {0:d} of {1:d} examples...'
            ).format(
                j, target_matrix.shape[0]
            ))

        for k in range(target_matrix.shape[1]):
            pit_matrix[j, k] = 0.01 * percentileofscore(
                a=prediction_matrix[j, k, :],
                score=target_matrix[j, k], kind='mean'
            )

    print('Have computed PIT values for all {0:d} examples!'.format(
        target_matrix.shape[0]
    ))
    return pit_matrix


def _plot_bar_graphs_hr_or_flux(
        mae_values, ssrel_values, ssrat_values, pitd_values, cef_values,
        for_heating_rate, model_description_strings, output_file_name):
    """Plots all bar graphs on the same axes, for either heating rate or flux.

    M = number of models

    :param mae_values: length-M numpy array of mean absolute errors.
    :param ssrel_values: length-M numpy array of spread-skill reliabilities.
    :param ssrat_values: length-M numpy array of spread-skill ratios.
    :param pitd_values: length-M numpy array of PIT deviations.
    :param cef_values: length-M numpy array of catastrophic-error frequencies.
    :param for_heating_rate: Boolean flag.  If True (False), errors are for
        heating rate (flux).
    :param model_description_strings: length-M list of model descriptions,
        which will be shown in the legend.
    :param output_file_name: Path to output file (figure will be saved here).
    """

    if for_heating_rate:
        metric_matrix = numpy.stack(
            [mae_values, ssrel_values, ssrat_values, 10 * pitd_values, cef_values],
            axis=1
        )
        metric_names = [
            r'MAE (K day$^{-1}$)',
            r'SSREL (K day$^{-1}$)',
            'SSRAT',
            r'10 $\times$ PITD',
            'CEF'
        ]
    else:
        metric_matrix = numpy.stack(
            [0.1 * mae_values, 0.1 * ssrel_values, ssrat_values, 10 * pitd_values, cef_values],
            axis=1
        )
        metric_names = [
            r'0.1 $\times$ MAE (W m$^{-2}$)',
            r'0.1 $\times$ SSREL (W m$^{-2}$)',
            'SSRAT',
            r'10 $\times$ PITD',
            'CEF'
        ]

    print(cef_values)

    num_models = len(model_description_strings)
    num_metrics = len(metric_names)

    x_tick_values = numpy.linspace(
        0, num_metrics - 1, num=num_metrics, dtype=float
    )
    bar_width = 0.8 / num_models

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    for i in range(num_models):
        print(metric_matrix[i, :])
        axes_object.bar(
            x_tick_values + i * bar_width,
            metric_matrix[i, :],
            bar_width,
            label=model_description_strings[i]
        )

    # Labeling
    axes_object.set_xticks(x_tick_values + 0.5 * (num_models - 1) * bar_width)
    axes_object.set_xticklabels(
        metric_names, fontsize=20, rotation=45, ha='right'
    )
    axes_object.set_ylabel('Error value')
    if for_heating_rate:
        axes_object.set_title('(a) Model comparison for heating rates')
    else:
        axes_object.set_title('(b) Model comparison for fluxes')

    axes_object.legend()
    pyplot.tight_layout()

    x_limits = axes_object.get_xlim()
    pyplot.plot(
        x_limits, numpy.full(2, 1.),
        linestyle='--', linewidth=2., color=numpy.full(3, 0.)
    )
    axes_object.set_xlim(x_limits)

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    imagemagick_utils.trim_whitespace(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        border_width_pixels=10
    )
    imagemagick_utils.resize_image(
        input_file_name=output_file_name,
        output_file_name=output_file_name,
        output_size_pixels=PANEL_SIZE_PX
    )


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
    heating_rate_ssrel_values_k_day01 = numpy.full(num_models, numpy.nan)
    flux_ssrel_values_w_m02 = numpy.full(num_models, numpy.nan)
    heating_rate_ssrat_values = numpy.full(num_models, numpy.nan)
    flux_ssrat_values = numpy.full(num_models, numpy.nan)
    heating_rate_pitd_values = numpy.full(num_models, numpy.nan)
    flux_pitd_values = numpy.full(num_models, numpy.nan)
    heating_rate_cat_error_freqs = numpy.full(num_models, numpy.nan)
    flux_cat_error_freqs = numpy.full(num_models, numpy.nan)

    for i in range(num_models):
        these_prediction_file_names = glob.glob(
            '{0:s}/predictions_part*.nc'.format(model_evaluation_dir_names[i])
        )
        these_prediction_file_names += glob.glob(
            '{0:s}/predictions.nc'.format(model_evaluation_dir_names[i])
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
        assert this_target_matrix.shape[3] == 1
        this_target_matrix = this_target_matrix[..., 0, 0]

        this_prediction_matrix = this_prediction_dict[
            prediction_io.VECTOR_PREDICTIONS_KEY
        ][..., 0, 0, :]
        this_mean_prediction_matrix = numpy.mean(
            this_prediction_matrix, axis=-1
        )

        print(SEPARATOR_STRING)

        start_indices, end_indices = __get_slices_for_multiprocessing(
            num_examples=this_target_matrix.shape[0]
        )
        argument_list = []
        for s, e in zip(start_indices, end_indices):
            argument_list.append((
                this_target_matrix[s:e, ...],
                this_prediction_matrix[s:e, ...]
            ))

        this_pit_matrix = numpy.full(this_target_matrix.shape, numpy.nan)
        with Pool() as pool_object:
            subarrays = pool_object.starmap(
                _compute_pit_values_hr_or_flux, argument_list
            )

            for k in range(len(start_indices)):
                s = start_indices[k]
                e = end_indices[k]
                this_pit_matrix[s:e, ...] = subarrays[k]

        print(SEPARATOR_STRING)

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
        assert this_target_matrix.shape[1] == 1
        assert this_target_matrix.shape[2] == 2

        this_target_matrix = this_target_matrix[:, 0, :]
        this_target_matrix = numpy.concatenate((
            this_target_matrix,
            this_target_matrix[:, [0]] - this_target_matrix[:, [1]]
        ), axis=1)

        this_prediction_matrix = this_prediction_dict[
            prediction_io.SCALAR_PREDICTIONS_KEY
        ]
        this_prediction_matrix = this_prediction_matrix[:, 0, ...]
        this_prediction_matrix = numpy.concatenate((
            this_prediction_matrix,
            this_prediction_matrix[:, [0], :] - this_prediction_matrix[:, [1], :]
        ), axis=1)

        this_mean_prediction_matrix = numpy.mean(
            this_prediction_matrix, axis=-1
        )

        print(SEPARATOR_STRING)

        start_indices, end_indices = __get_slices_for_multiprocessing(
            num_examples=this_target_matrix.shape[0]
        )
        argument_list = []
        for s, e in zip(start_indices, end_indices):
            argument_list.append((
                this_target_matrix[s:e, ...],
                this_prediction_matrix[s:e, ...]
            ))

        this_pit_matrix = numpy.full(this_target_matrix.shape, numpy.nan)
        with Pool() as pool_object:
            subarrays = pool_object.starmap(
                _compute_pit_values_hr_or_flux, argument_list
            )

            for k in range(len(start_indices)):
                s = start_indices[k]
                e = end_indices[k]
                this_pit_matrix[s:e, ...] = subarrays[k]

        print(SEPARATOR_STRING)

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
            this_eval_table_xarray[evaluation.SCALAR_MAE_KEY].values[0, ...],
            this_eval_table_xarray[evaluation.AUX_MAE_KEY].values[0, ...]
        ), axis=0)

        assert this_mae_matrix.shape[0] == 3
        flux_mae_values_w_m02[i] = numpy.mean(
            numpy.nanmean(this_mae_matrix, axis=1)
        )

        this_mae_matrix = (
            this_eval_table_xarray[evaluation.VECTOR_MAE_KEY].values[:, 0, ...]
        )
        assert this_mae_matrix.shape[1] == 1
        this_mae_matrix = this_mae_matrix[:, 0, :]
        heating_rate_mae_values_k_day01[i] = numpy.mean(
            numpy.nanmean(this_mae_matrix, axis=1)
        )

        this_file_name = '{0:s}/spread_vs_skill.nc'.format(
            model_evaluation_dir_names[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_ss_table_xarray = ss_utils.read_results(this_file_name)

        these_ssrel = numpy.concatenate((
            this_ss_table_xarray[ss_utils.SCALAR_SSREL_KEY].values[..., 0],
            this_ss_table_xarray[ss_utils.AUX_SSREL_KEY].values[..., 0]
        ))
        flux_ssrel_values_w_m02[i] = numpy.mean(these_ssrel)

        these_ssrat = numpy.concatenate((
            this_ss_table_xarray[ss_utils.SCALAR_SSRAT_KEY].values[..., 0],
            this_ss_table_xarray[ss_utils.AUX_SSRAT_KEY].values[..., 0]
        ))
        flux_ssrat_values[i] = numpy.mean(these_ssrat)

        heating_rate_ssrel_values_k_day01[i] = (
            this_ss_table_xarray[ss_utils.VECTOR_FLAT_SSREL_KEY].values[0, 0]
        )
        heating_rate_ssrat_values[i] = (
            this_ss_table_xarray[ss_utils.VECTOR_FLAT_SSRAT_KEY].values[0, 0]
        )

        this_file_name = '{0:s}/pit_histograms.nc'.format(
            model_evaluation_dir_names[i]
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_pit_table_xarray = pit_utils.read_results(this_file_name)

        these_pitd = numpy.concatenate((
            this_pit_table_xarray[pit_utils.SCALAR_PITD_KEY].values[..., 0],
            this_pit_table_xarray[pit_utils.AUX_PITD_KEY].values[..., 0]
        ))
        flux_pitd_values[i] = numpy.mean(these_pitd)

        heating_rate_pitd_values[i] = (
            this_pit_table_xarray[pit_utils.VECTOR_FLAT_PITD_KEY].values[0, 0]
        )

    panel_file_names = [
        '{0:s}/heating_rate_comparison.jpg'.format(output_dir_name)
    ]
    _plot_bar_graphs_hr_or_flux(
        mae_values=heating_rate_mae_values_k_day01,
        ssrel_values=heating_rate_ssrel_values_k_day01,
        ssrat_values=heating_rate_ssrat_values,
        pitd_values=heating_rate_pitd_values,
        cef_values=heating_rate_cat_error_freqs,
        for_heating_rate=True,
        model_description_strings=model_description_strings,
        output_file_name=panel_file_names[-1]
    )

    panel_file_names.append(
        '{0:s}/flux_comparison.jpg'.format(output_dir_name)
    )
    _plot_bar_graphs_hr_or_flux(
        mae_values=flux_mae_values_w_m02,
        ssrel_values=flux_ssrel_values_w_m02,
        ssrat_values=flux_ssrat_values,
        pitd_values=flux_pitd_values,
        cef_values=flux_cat_error_freqs,
        for_heating_rate=False,
        model_description_strings=model_description_strings,
        output_file_name=panel_file_names[-1]
    )

    concat_figure_file_name = '{0:s}/overall_model_comparison.jpg'.format(
        output_dir_name
    )
    print('Concatenating panels to: "{0:s}"...'.format(concat_figure_file_name))

    imagemagick_utils.concatenate_images(
        input_file_names=panel_file_names,
        output_file_name=concat_figure_file_name,
        num_panel_rows=2, num_panel_columns=1
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
