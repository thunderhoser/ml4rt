"""Creates figure showing permutation-test results for both models."""

import os
import sys
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import gg_permutation
import plotting_utils
import permutation_plotting
import example_utils
import permutation as ml4rt_permutation

CONFIDENCE_LEVEL = 0.99

PREDICTOR_NAME_TO_VERBOSE = {
    example_utils.ZENITH_ANGLE_NAME: 'Solar zenith angle',
    example_utils.ALBEDO_NAME: 'Surface albedo',
    example_utils.PRESSURE_NAME: 'Pressure',
    example_utils.TEMPERATURE_NAME: 'Temperature',
    example_utils.SPECIFIC_HUMIDITY_NAME: 'Specific humidity ',
    example_utils.RELATIVE_HUMIDITY_NAME: 'Relative humidity ',
    example_utils.LIQUID_WATER_CONTENT_NAME: 'Liquid-water content (LWC)',
    example_utils.ICE_WATER_CONTENT_NAME: 'Ice-water content (IWC)',
    example_utils.LIQUID_WATER_PATH_NAME: 'Downward liquid-water path (LWP)',
    example_utils.ICE_WATER_PATH_NAME: 'Downward ice-water path (IWP)',
    example_utils.WATER_VAPOUR_PATH_NAME: 'Downward water-vapour path (WVP)',
    example_utils.UPWARD_LIQUID_WATER_PATH_NAME:
        'Upward liquid-water path (LWP)',
    example_utils.UPWARD_ICE_WATER_PATH_NAME: 'Upward ice-water path (IWP)',
    example_utils.UPWARD_WATER_VAPOUR_PATH_NAME:
        'Upward water-vapour path (WVP)'
}

FONT_SIZE = 25
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)

FIGURE_RESOLUTION_DPI = 300

NUM_PANEL_ROWS = 2
NUM_PANEL_COLUMNS = 2
PANEL_SIZE_PX = int(5e6)
CONCAT_FIGURE_SIZE_PX = int(2e7)

EXPERIMENT1_DIR_ARG_NAME = 'input_exp1_permutation_dir_name'
EXPERIMENT2_DIR_ARG_NAME = 'input_exp2_permutation_dir_name'
USE_FORWARD_TEST_ARG_NAME = 'use_forward_test'
USE_MULTIPASS_TEST_ARG_NAME = 'use_multipass_test'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

EXPERIMENT1_DIR_HELP_STRING = (
    'Name of directory with permutation results for selected model from '
    'Experiment 1.'
)
EXPERIMENT2_DIR_HELP_STRING = 'Same as `{0:s}` but for Experiment 2.'.format(
    EXPERIMENT1_DIR_ARG_NAME
)
USE_FORWARD_TEST_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot forward (backwards) version of test.'
)
USE_MULTIPASS_TEST_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot multi-pass (single-pass) version of '
    'test.'
)
OUTPUT_FILE_HELP_STRING = 'Path to output file.  Figure will be saved here.'

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT1_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT1_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXPERIMENT2_DIR_ARG_NAME, type=str, required=True,
    help=EXPERIMENT2_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_FORWARD_TEST_ARG_NAME, type=int, required=True,
    help=USE_FORWARD_TEST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_MULTIPASS_TEST_ARG_NAME, type=int, required=True,
    help=USE_MULTIPASS_TEST_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _results_to_gg_format(permutation_dict):
    """Converts permutation results from ml4rt format to GewitterGefahr format.

    :param permutation_dict: Dictionary created by `run_forward_test` or
        `run_backwards_test` in `ml4rt.machine_learning.permutation`.
    :return: permutation_dict: Same but in format created by `run_forward_test`
        or `run_backwards_test` in `gewittergefahr.deep_learning.permutation`.
    """

    permutation_dict[gg_permutation.ORIGINAL_COST_ARRAY_KEY] = (
        permutation_dict[ml4rt_permutation.ORIGINAL_COST_KEY]
    )
    permutation_dict[gg_permutation.BACKWARDS_FLAG] = (
        permutation_dict[ml4rt_permutation.BACKWARDS_FLAG_KEY]
    )

    best_predictor_names = [
        PREDICTOR_NAME_TO_VERBOSE[s] for s in
        permutation_dict[ml4rt_permutation.BEST_PREDICTORS_KEY]
    ]
    best_heights_m_agl = permutation_dict[ml4rt_permutation.BEST_HEIGHTS_KEY]

    if best_heights_m_agl is not None:
        for k in range(len(best_predictor_names)):
            if numpy.isnan(best_heights_m_agl[k]):
                continue

            best_predictor_names[k] += ' at {0:d} m AGL'.format(
                int(numpy.round(best_heights_m_agl[k]))
            )

    step1_predictor_names = [
        PREDICTOR_NAME_TO_VERBOSE[s] for s in
        permutation_dict[ml4rt_permutation.STEP1_PREDICTORS_KEY]
    ]
    step1_heights_m_agl = permutation_dict[ml4rt_permutation.STEP1_HEIGHTS_KEY]

    if step1_heights_m_agl is not None:
        for k in range(len(step1_predictor_names)):
            if numpy.isnan(step1_heights_m_agl[k]):
                continue

            step1_predictor_names[k] += ' at {0:d} m AGL'.format(
                int(numpy.round(step1_heights_m_agl[k]))
            )

    permutation_dict[gg_permutation.BEST_PREDICTORS_KEY] = best_predictor_names
    permutation_dict[gg_permutation.STEP1_PREDICTORS_KEY] = (
        step1_predictor_names
    )

    return permutation_dict


def _run(exp1_permutation_dir_name, exp2_permutation_dir_name, use_forward_test,
         use_multipass_test, output_file_name):
    """Creates figure showing permutation-test results for both models.

    This is effectively the main method.

    :param exp1_permutation_dir_name: See documentation at top of file.
    :param exp2_permutation_dir_name: Same.
    :param use_forward_test: Same.
    :param use_multipass_test: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        file_name=output_file_name
    )

    exp1_flux_file_name = '{0:s}/{1:s}_perm_test_fluxes-only.nc'.format(
        exp1_permutation_dir_name,
        'forward' if use_forward_test else 'backwards'
    )
    exp1_heating_rate_file_name = '{0:s}/{1:s}_perm_test_hr-only.nc'.format(
        exp1_permutation_dir_name,
        'forward' if use_forward_test else 'backwards'
    )
    exp2_flux_file_name = '{0:s}/{1:s}_perm_test_fluxes-only.nc'.format(
        exp2_permutation_dir_name,
        'forward' if use_forward_test else 'backwards'
    )
    exp2_heating_rate_file_name = '{0:s}/{1:s}_perm_test_hr-only.nc'.format(
        exp2_permutation_dir_name,
        'forward' if use_forward_test else 'backwards'
    )

    print('Reading data from: "{0:s}"...'.format(exp1_heating_rate_file_name))
    exp1_heating_permutation_dict = ml4rt_permutation.read_file(
        exp1_heating_rate_file_name
    )
    exp1_heating_permutation_dict = _results_to_gg_format(
        exp1_heating_permutation_dict
    )

    figure_object, axes_object_matrix = plotting_utils.create_paneled_figure(
        num_rows=2, num_columns=2, shared_x_axis=False, shared_y_axis=True,
        keep_aspect_ratio=False, horizontal_spacing=0.25, vertical_spacing=0.25
    )

    if use_multipass_test:
        permutation_plotting.plot_multipass_test(
            permutation_dict=exp1_heating_permutation_dict,
            axes_object=axes_object_matrix[0, 0],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )
    else:
        permutation_plotting.plot_single_pass_test(
            permutation_dict=exp1_heating_permutation_dict,
            axes_object=axes_object_matrix[0, 0],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )

    plotting_utils.label_axes(
        axes_object=axes_object_matrix[0, 0], label_string='(a)',
        font_size=30, x_coord_normalized=0.1, y_coord_normalized=1.01
    )
    axes_object_matrix[0, 0].set_title('Exp 1, heating rates only')
    axes_object_matrix[0, 0].set_xlabel(r'Dual-weighted MSE (K$^3$ day$^{-3}$)')
    axes_object_matrix[0, 0].set_ylabel('')

    print('Reading data from: "{0:s}"...'.format(exp1_flux_file_name))
    exp1_flux_permutation_dict = ml4rt_permutation.read_file(
        exp1_flux_file_name
    )
    exp1_flux_permutation_dict = _results_to_gg_format(
        exp1_flux_permutation_dict
    )

    if use_multipass_test:
        permutation_plotting.plot_multipass_test(
            permutation_dict=exp1_flux_permutation_dict,
            axes_object=axes_object_matrix[0, 1],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )
    else:
        permutation_plotting.plot_single_pass_test(
            permutation_dict=exp1_flux_permutation_dict,
            axes_object=axes_object_matrix[0, 1],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )

    plotting_utils.label_axes(
        axes_object=axes_object_matrix[0, 1], label_string='(b)',
        font_size=30, x_coord_normalized=0.1, y_coord_normalized=1.01
    )
    axes_object_matrix[0, 1].set_title('Exp 1, fluxes only')
    axes_object_matrix[0, 1].set_xlabel(r'MSE (K day$^{-1}$)')
    axes_object_matrix[0, 1].set_ylabel('')

    print('Reading data from: "{0:s}"...'.format(exp2_heating_rate_file_name))
    exp2_heating_permutation_dict = ml4rt_permutation.read_file(
        exp2_heating_rate_file_name
    )
    exp2_heating_permutation_dict = _results_to_gg_format(
        exp2_heating_permutation_dict
    )

    if use_multipass_test:
        permutation_plotting.plot_multipass_test(
            permutation_dict=exp2_heating_permutation_dict,
            axes_object=axes_object_matrix[1, 0],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )
    else:
        permutation_plotting.plot_single_pass_test(
            permutation_dict=exp2_heating_permutation_dict,
            axes_object=axes_object_matrix[1, 0],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )

    plotting_utils.label_axes(
        axes_object=axes_object_matrix[1, 0], label_string='(c)',
        font_size=30, x_coord_normalized=0.1, y_coord_normalized=1.01
    )
    axes_object_matrix[1, 0].set_title('Exp 2, heating rates only')
    axes_object_matrix[1, 0].set_xlabel(r'Dual-weighted MSE (K$^3$ day$^{-3}$)')
    axes_object_matrix[1, 0].set_ylabel('')

    print('Reading data from: "{0:s}"...'.format(exp2_flux_file_name))
    exp2_flux_permutation_dict = ml4rt_permutation.read_file(
        exp2_flux_file_name
    )
    exp2_flux_permutation_dict = _results_to_gg_format(
        exp2_flux_permutation_dict
    )

    if use_multipass_test:
        permutation_plotting.plot_multipass_test(
            permutation_dict=exp2_flux_permutation_dict,
            axes_object=axes_object_matrix[1, 1],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )
    else:
        permutation_plotting.plot_single_pass_test(
            permutation_dict=exp2_flux_permutation_dict,
            axes_object=axes_object_matrix[1, 1],
            plot_percent_increase=False, confidence_level=CONFIDENCE_LEVEL
        )

    plotting_utils.label_axes(
        axes_object=axes_object_matrix[1, 1], label_string='(d)',
        font_size=30, x_coord_normalized=0.1, y_coord_normalized=1.01
    )
    axes_object_matrix[1, 1].set_title('Exp 2, fluxes only')
    axes_object_matrix[1, 1].set_xlabel(r'MSE (K day$^{-1}$)')
    axes_object_matrix[1, 1].set_ylabel('')

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        exp1_permutation_dir_name=getattr(
            INPUT_ARG_OBJECT, EXPERIMENT1_DIR_ARG_NAME
        ),
        exp2_permutation_dir_name=getattr(
            INPUT_ARG_OBJECT, EXPERIMENT2_DIR_ARG_NAME
        ),
        use_forward_test=bool(getattr(
            INPUT_ARG_OBJECT, USE_FORWARD_TEST_ARG_NAME
        )),
        use_multipass_test=bool(getattr(
            INPUT_ARG_OBJECT, USE_MULTIPASS_TEST_ARG_NAME
        )),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
