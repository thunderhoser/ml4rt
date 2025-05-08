"""Plots large- and catastrophic-error freqs for both heating rate and flux."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import evaluation_plotting as eval_plotting

METRES_TO_MICRONS = 1e6

FLUX_NAME_TO_FANCY_DICT = {
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'$F_{down}^{sfc}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'$F_{up}^{TOA}$',
    evaluation.SHORTWAVE_NET_FLUX_NAME: r'$F_{net}$'
}

LARGE_ERROR_FREQ_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
CATASTROPHIC_ERROR_FREQ_COLOUR = numpy.array([117, 112, 179], dtype=float) / 255
LINE_WIDTH = 4

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_prediction_file_names'
LARGE_HR_THRES_ARG_NAME = 'large_hr_error_threshold_k_day01'
LARGE_FLUX_THRES_ARG_NAME = 'large_flux_error_threshold_w_m02'
CATASTROPHIC_THRES_ARG_NAME = 'catastrophic_error_confidence_threshold'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to prediction files.  Each will be read by '
    '`prediction_io.read_file`.'
)
LARGE_HR_THRES_HELP_STRING = (
    'Large-error threshold for heating rates, applied independently to every '
    'data sample and height.  Any absolute error >= this value will be '
    'considered a "large error".'
)
LARGE_FLUX_THRES_HELP_STRING = (
    'Large-error threshold for fluxes, applied independently to every data '
    'sample and flux variable.  Any absolute error >= this value will be '
    'considered a "large error".'
)
CATASTROPHIC_THRES_HELP_STRING = (
    'Confidence threshold for catastrophic errors, applied independently to '
    'every atomic data sample (for heating rates, every pair of data sample '
    'and height; for fluxes, every pair of data sample and variable).  For '
    'example, if this value is 0.95, then any atomic data sample with a large '
    'error, and with the observation falling outside the 95% confidence '
    'interval, will be considered a "catastrophic error".'
)
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one set of plots for each.'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=INPUT_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LARGE_HR_THRES_ARG_NAME, type=float, required=True,
    help=LARGE_HR_THRES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LARGE_FLUX_THRES_ARG_NAME, type=float, required=True,
    help=LARGE_FLUX_THRES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CATASTROPHIC_THRES_ARG_NAME, type=float, required=True,
    help=CATASTROPHIC_THRES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WAVELENGTHS_ARG_NAME, type=float, nargs='+', required=False,
    default=[example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES],
    help=WAVELENGTHS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def wavelength_to_string(wavelength_metres):
    """Converts wavelength to string.

    :param wavelength_metres: Wavelength (scalar float).
    :return: wavelength_string_microns: Wavelength (string in microns).
    """

    if (
            wavelength_metres > 0.99 *
            example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
    ):
        return 'BB'

    return '{0:.2f}'.format(METRES_TO_MICRONS * wavelength_metres)


def _compute_large_error_freqs_1file(
        prediction_file_name, large_hr_error_threshold_k_day01,
        large_flux_error_threshold_w_m02,
        catastrophic_error_confidence_threshold,
        wavelengths_metres, flux_var_names):
    """Computes large-error frequencies for one prediction file.

    H = number of heights
    W = number of wavelengths
    F = number of flux variables

    :param prediction_file_name: Path to input file.
    :param large_hr_error_threshold_k_day01: See documentation at top of this
        file.
    :param large_flux_error_threshold_w_m02: Same.
    :param catastrophic_error_confidence_threshold: Same.
    :param wavelengths_metres: Same.
    :param flux_var_names: length-F list with names of flux variables.
    :return: hr_num_large_errors_matrix: H-by-W numpy array with large-error
        counts for heating rates.
    :return: flux_num_large_errors_matrix: F-by-W numpy array with large-error
        counts for flux variables.
    :return: hr_num_cat_errors_matrix: H-by-W numpy array with
        catastrophic-error counts for heating rates.
    :return: flux_num_cat_errors_matrix: F-by-W numpy array with
        catastrophic-error counts for flux variables.
    :return: num_examples: Number of data examples.
    """

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    pdict = prediction_dict

    wave_inds = numpy.array([
        example_utils.match_wavelengths(
            wavelengths_metres=pdict[prediction_io.TARGET_WAVELENGTHS_KEY],
            desired_wavelength_metres=w
        )
        for w in wavelengths_metres
    ], dtype=int)

    actual_hr_matrix_k_day01 = pdict[prediction_io.VECTOR_TARGETS_KEY]
    assert actual_hr_matrix_k_day01.shape[3] == 1
    actual_hr_matrix_k_day01 = actual_hr_matrix_k_day01[..., 0][..., wave_inds]

    predicted_hr_matrix_k_day01 = (
        pdict[prediction_io.VECTOR_PREDICTIONS_KEY][..., 0, :]
    )
    predicted_hr_matrix_k_day01 = predicted_hr_matrix_k_day01[..., wave_inds, :]
    mean_pred_hr_matrix_k_day01 = numpy.mean(
        predicted_hr_matrix_k_day01, axis=-1
    )

    hr_large_error_flag_matrix = numpy.absolute(
        actual_hr_matrix_k_day01 - mean_pred_hr_matrix_k_day01
    ) >= large_hr_error_threshold_k_day01

    hr_num_large_errors_matrix = numpy.sum(hr_large_error_flag_matrix, axis=0)

    num_examples = actual_hr_matrix_k_day01.shape[0]
    num_heights = actual_hr_matrix_k_day01.shape[1]
    num_wavelengths = actual_hr_matrix_k_day01.shape[2]
    hr_cat_error_flag_matrix = numpy.full(
        actual_hr_matrix_k_day01.shape, False, dtype=bool
    )

    for i in range(num_examples):
        for h in range(num_heights):
            for w in range(num_wavelengths):
                if not hr_large_error_flag_matrix[i, h, w]:
                    continue

                this_pit_value = 0.01 * percentileofscore(
                    a=predicted_hr_matrix_k_day01[i, h, w, :],
                    score=actual_hr_matrix_k_day01[i, h, w],
                    kind='mean'
                )

                cect = catastrophic_error_confidence_threshold

                if (
                        this_pit_value > 0.5 * (1 + cect) or
                        this_pit_value < 0.5 * (1 - cect)
                ):
                    hr_cat_error_flag_matrix[i, h, w] = True

    hr_num_cat_errors_matrix = numpy.sum(hr_cat_error_flag_matrix, axis=0)

    actual_flux_matrix_w_m02 = (
        pdict[prediction_io.SCALAR_TARGETS_KEY][:, wave_inds, :]
    )
    predicted_flux_matrix_w_m02 = (
        pdict[prediction_io.SCALAR_PREDICTIONS_KEY][:, wave_inds, ...]
    )

    down_index = flux_var_names.index(
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
    )
    up_index = flux_var_names.index(example_utils.SHORTWAVE_TOA_UP_FLUX_NAME)
    actual_net_flux_matrix_w_m02 = (
        actual_flux_matrix_w_m02[..., [down_index]] -
        actual_flux_matrix_w_m02[..., [up_index]]
    )
    actual_flux_matrix_w_m02 = numpy.concatenate(
        [actual_flux_matrix_w_m02, actual_net_flux_matrix_w_m02], axis=-1
    )
    predicted_net_flux_matrix_w_m02 = (
        predicted_flux_matrix_w_m02[..., [down_index], :] -
        predicted_flux_matrix_w_m02[..., [up_index], :]
    )
    predicted_flux_matrix_w_m02 = numpy.concatenate(
        [predicted_flux_matrix_w_m02, predicted_net_flux_matrix_w_m02], axis=-1
    )

    mean_pred_flux_matrix_w_m02 = numpy.mean(
        predicted_flux_matrix_w_m02, axis=-1
    )

    flux_large_error_flag_matrix = numpy.absolute(
        actual_flux_matrix_w_m02 - mean_pred_flux_matrix_w_m02
    ) >= large_flux_error_threshold_w_m02

    flux_num_large_errors_matrix = numpy.sum(
        flux_large_error_flag_matrix, axis=0
    )
    flux_num_large_errors_matrix = numpy.transpose(flux_num_large_errors_matrix)

    num_flux_vars = actual_flux_matrix_w_m02.shape[-1]
    flux_cat_error_flag_matrix = numpy.full(
        (num_examples, num_flux_vars, num_wavelengths), False, dtype=bool
    )

    for i in range(num_examples):
        for f in range(num_flux_vars):
            for w in range(num_wavelengths):
                if not flux_large_error_flag_matrix[i, w, f]:
                    continue

                this_pit_value = 0.01 * percentileofscore(
                    a=predicted_flux_matrix_w_m02[i, w, f, :],
                    score=actual_flux_matrix_w_m02[i, w, f],
                    kind='mean'
                )

                cect = catastrophic_error_confidence_threshold

                if (
                        this_pit_value > 0.5 * (1 + cect) or
                        this_pit_value < 0.5 * (1 - cect)
                ):
                    flux_cat_error_flag_matrix[i, f, w] = True

    flux_num_cat_errors_matrix = numpy.sum(flux_cat_error_flag_matrix, axis=0)

    return (
        hr_num_large_errors_matrix, flux_num_large_errors_matrix,
        hr_num_cat_errors_matrix, flux_num_cat_errors_matrix,
        num_examples
    )


def _plot_error_freqs_for_heating_rate(
        large_error_freqs, catastrophic_error_freqs, heights_m_agl,
        wavelength_metres, output_dir_name):
    """Plots large- and catastrophic-error frequencies for heating rate.

    These are vertical profiles (line graphs) plotted on the same axes.

    H = number of heights in grid

    :param large_error_freqs: length-H numpy array of large-error frequencies.
    :param catastrophic_error_freqs: length-H numpy array of catastrophic-error
        frequencies.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param wavelength_metres: Wavelength.
    :param output_dir_name: Path to output directory.  Figure will be saved
        here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )
    legend_handles = []
    legend_strings = []

    this_handle = eval_plotting.plot_score_profile(
        heights_m_agl=heights_m_agl,
        score_values=large_error_freqs,
        score_name=eval_plotting.MAE_NAME,
        line_colour=LARGE_ERROR_FREQ_COLOUR,
        line_width=4,
        line_style='solid',
        use_log_scale=True,
        axes_object=axes_object,
        are_axes_new=True
    )
    legend_handles.append(this_handle)
    legend_strings.append('Large-error freq')

    this_handle = eval_plotting.plot_score_profile(
        heights_m_agl=heights_m_agl,
        score_values=catastrophic_error_freqs,
        score_name=eval_plotting.MAE_NAME,
        line_colour=CATASTROPHIC_ERROR_FREQ_COLOUR,
        line_width=4,
        line_style='solid',
        use_log_scale=True,
        axes_object=axes_object,
        are_axes_new=False
    )
    legend_handles.append(this_handle)
    legend_strings.append('Catastrophic-error freq')

    axes_object.legend(
        legend_handles, legend_strings, loc='center right',
        bbox_to_anchor=(0.95, 0.5), fancybox=True, shadow=False,
        facecolor='white', edgecolor='k', framealpha=0.5, ncol=1
    )

    title_string = 'Error frequencies for SW heating rate'
    if wavelength_metres != example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES:
        title_string += ' at {0:s}'.format(
            wavelength_to_string(wavelength_metres)
        )
        title_string += r' $\mu$m'

    title_string += '\nMax LEF = {0:.2f}; max CEF = {1:.2f}'.format(
        numpy.nanmax(large_error_freqs),
        numpy.nanmax(catastrophic_error_freqs)
    )

    axes_object.set_xlabel('Frequency')
    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/{1:s}_{2:s}microns_error-freq_profile.jpg'.format(
        output_dir_name,
        example_utils.SHORTWAVE_HEATING_RATE_NAME.replace('_', '-'),
        wavelength_to_string(wavelength_metres)
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _plot_error_freqs_for_flux(
        large_error_freqs, catastrophic_error_freqs, flux_var_names,
        wavelength_metres, output_dir_name):
    """Plots large- and catastrophic-error frequencies for flux variables.

    These are bar graphs plotted on the same axes.

    F = number of flux variables

    :param large_error_freqs: length-F numpy array of large-error frequencies.
    :param catastrophic_error_freqs: length-F numpy array of catastrophic-error
        frequencies.
    :param flux_var_names: length-F list of variable names.
    :param wavelength_metres: Wavelength.
    :param output_dir_name: Path to output directory.  Figure will be saved
        here.
    """

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    num_flux_vars = len(flux_var_names)
    x_tick_values = numpy.linspace(
        0.5, num_flux_vars - 0.5, num=num_flux_vars, dtype=float
    )
    bar_width = 0.4

    lef_handle = axes_object.bar(
        x_tick_values - bar_width / 2,
        large_error_freqs,
        bar_width,
        color=LARGE_ERROR_FREQ_COLOUR
    )
    cef_handle = axes_object.bar(
        x_tick_values + bar_width / 2,
        catastrophic_error_freqs,
        bar_width,
        color=CATASTROPHIC_ERROR_FREQ_COLOUR
    )

    axes_object.legend(
        [lef_handle, cef_handle],
        ['Large-error freq', 'Catastrophic-error freq'],
        loc='lower left',
        bbox_to_anchor=(0.05, 0.1),
        fancybox=True,
        shadow=False,
        facecolor='white',
        edgecolor='k',
        framealpha=0.5,
        ncol=1
    )

    x_tick_labels = [FLUX_NAME_TO_FANCY_DICT[f] for f in flux_var_names]
    axes_object.set_xticks(x_tick_values)
    axes_object.set_xticklabels(x_tick_labels)
    axes_object.set_ylabel('LEF/CEF frequency')

    title_string = 'Error frequencies for SW fluxes'
    if wavelength_metres != example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES:
        title_string += ' at {0:s}'.format(
            wavelength_to_string(wavelength_metres)
        )
        title_string += r' $\mu$m'

    axes_object.set_title(title_string)

    figure_file_name = '{0:s}/fluxes_{1:s}microns_error-freq.jpg'.format(
        output_dir_name,
        wavelength_to_string(wavelength_metres)
    )

    print('Saving figure to: "{0:s}"...'.format(figure_file_name))
    figure_object.savefig(
        figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_names, large_hr_error_threshold_k_day01,
         large_flux_error_threshold_w_m02,
         catastrophic_error_confidence_threshold, wavelengths_metres,
         output_dir_name):
    """Plots large- and catastrophic-error freqs for both heating rate and flux.

    This is effectively the main method.

    :param prediction_file_names: See documentation at top of file.
    :param large_hr_error_threshold_k_day01: Same.
    :param large_flux_error_threshold_w_m02: Same.
    :param catastrophic_error_confidence_threshold: Same.
    :param wavelengths_metres: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_greater(large_hr_error_threshold_k_day01, 0.)
    error_checking.assert_is_greater(large_flux_error_threshold_w_m02, 0.)
    error_checking.assert_is_geq(catastrophic_error_confidence_threshold, 0.8)
    error_checking.assert_is_less_than(
        catastrophic_error_confidence_threshold, 1.
    )

    # Do actual stuff.
    print('Reading first file: "{0:s}"...'.format(prediction_file_names[0]))
    first_prediction_dict = prediction_io.read_file(prediction_file_names[0])
    heights_m_agl = first_prediction_dict[prediction_io.HEIGHTS_KEY]
    num_heights = len(heights_m_agl)
    num_wavelengths = len(wavelengths_metres)

    model_file_name = first_prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    flux_var_names = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    flux_var_names.append(evaluation.SHORTWAVE_NET_FLUX_NAME)

    num_flux_vars = (
        first_prediction_dict[prediction_io.SCALAR_TARGETS_KEY].shape[-1]
    )
    hr_num_large_errors_matrix = numpy.full(
        (num_heights, num_wavelengths), 0, dtype=int
    )
    hr_num_cat_errors_matrix = numpy.full(
        (num_heights, num_wavelengths), 0, dtype=int
    )
    flux_num_large_errors_matrix = numpy.full(
        (num_flux_vars, num_wavelengths), 0, dtype=int
    )
    flux_num_cat_errors_matrix = numpy.full(
        (num_flux_vars, num_wavelengths), 0, dtype=int
    )
    num_examples = 0

    for this_file_name in prediction_file_names:
        (
            this_hr_num_large_matrix, this_flux_num_large_matrix,
            this_hr_num_cat_matrix, this_flux_num_cat_matrix,
            this_num_examples
        ) = _compute_large_error_freqs_1file(
            prediction_file_name=this_file_name,
            large_hr_error_threshold_k_day01=large_hr_error_threshold_k_day01,
            large_flux_error_threshold_w_m02=large_flux_error_threshold_w_m02,
            catastrophic_error_confidence_threshold=
            catastrophic_error_confidence_threshold,
            wavelengths_metres=wavelengths_metres
        )

        hr_num_large_errors_matrix += this_hr_num_large_matrix
        hr_num_cat_errors_matrix += this_hr_num_cat_matrix
        flux_num_large_errors_matrix += this_flux_num_large_matrix
        flux_num_cat_errors_matrix += this_flux_num_cat_matrix
        num_examples += this_num_examples

    hr_large_error_freq_matrix = (
        hr_num_large_errors_matrix.astype(float) / num_examples
    )
    hr_cat_error_freq_matrix = (
        hr_num_cat_errors_matrix.astype(float) / num_examples
    )
    flux_large_error_freq_matrix = (
        flux_num_large_errors_matrix.astype(float) / num_examples
    )
    flux_cat_error_freq_matrix = (
        flux_num_cat_errors_matrix.astype(float) / num_examples
    )

    for w in range(num_wavelengths):
        _plot_error_freqs_for_heating_rate(
            large_error_freqs=hr_large_error_freq_matrix[:, w],
            catastrophic_error_freqs=hr_cat_error_freq_matrix[:, w],
            heights_m_agl=heights_m_agl,
            wavelength_metres=wavelengths_metres[w],
            output_dir_name=output_dir_name
        )
        _plot_error_freqs_for_flux(
            large_error_freqs=flux_large_error_freq_matrix[:, w],
            catastrophic_error_freqs=flux_cat_error_freq_matrix[:, w],
            flux_var_names=flux_var_names,
            wavelength_metres=wavelengths_metres[w],
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        large_hr_error_threshold_k_day01=getattr(
            INPUT_ARG_OBJECT, LARGE_HR_THRES_ARG_NAME
        ),
        large_flux_error_threshold_w_m02=getattr(
            INPUT_ARG_OBJECT, LARGE_FLUX_THRES_ARG_NAME
        ),
        catastrophic_error_confidence_threshold=getattr(
            INPUT_ARG_OBJECT, CATASTROPHIC_THRES_ARG_NAME
        ),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
