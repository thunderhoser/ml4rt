"""Plots height profiles of large and catastrophic HR errors."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.stats import percentileofscore
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.plotting import evaluation_plotting as eval_plotting

MAX_FREQUENCY_TO_PLOT = 0.5
METRES_TO_MICRONS = 1e6

LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255
LINE_WIDTH = 4

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILES_ARG_NAME = 'input_prediction_file_names'
LARGE_ERR_THRES_ARG_NAME = 'large_error_threshold_k_day01'
CATASTROPHIC_ERR_CONF_THRES_ARG_NAME = 'catastrophic_error_confidence_threshold'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILES_HELP_STRING = (
    'List of paths to prediction files.  Each will be read by '
    '`prediction_io.read_file`.'
)
LARGE_ERR_THRES_HELP_STRING = (
    'Large-error threshold, applied independently to each data sample and each '
    'height.  Any absolute error >= this value will be considered a "large '
    'error".'
)
CATASTROPHIC_ERR_CONF_THRES_HELP_STRING = (
    'Confidence threshold for catastrophic errors.  For example, if this value '
    'is 0.95 and {0:s} = 1.0, a "catastrophic error" will be any sample/height '
    'pair where both [a] the mean prediction has an absolute error >= {0:s} '
    'and [b] the observation falls outside the 95% confidence interval.'
).format(LARGE_ERR_THRES_ARG_NAME)

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
    '--' + LARGE_ERR_THRES_ARG_NAME, type=float, required=True,
    help=LARGE_ERR_THRES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CATASTROPHIC_ERR_CONF_THRES_ARG_NAME, type=float, required=True,
    help=CATASTROPHIC_ERR_CONF_THRES_HELP_STRING
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


def _compute_large_error_freqs_1file(
        prediction_file_name, large_error_threshold_k_day01,
        catastrophic_error_confidence_threshold, wavelengths_metres):
    """Computes large-error frequencies for one prediction file.

    H = number of heights
    W = number of wavelengths

    :param prediction_file_name: Path to input file.
    :param large_error_threshold_k_day01: See documentation at top of this file.
    :param catastrophic_error_confidence_threshold: Same.
    :param wavelengths_metres: Same.
    :return: num_large_errors_matrix: H-by-W numpy array with counts of large
        errors.
    :return: num_catastrophic_errors_matrix: H-by-W numpy array with counts of
        catastrophic errors.
    :return: num_examples: Number of examples (i.e., profiles).
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
    mean_predicted_hr_matrix_k_day01 = numpy.mean(
        predicted_hr_matrix_k_day01, axis=-1
    )

    large_error_flag_matrix = numpy.absolute(
        actual_hr_matrix_k_day01 - mean_predicted_hr_matrix_k_day01
    ) >= large_error_threshold_k_day01

    num_large_errors_matrix = numpy.sum(large_error_flag_matrix, axis=0)

    num_examples = actual_hr_matrix_k_day01.shape[0]
    num_heights = actual_hr_matrix_k_day01.shape[1]
    num_wavelengths = actual_hr_matrix_k_day01.shape[2]
    catastrophic_error_flag_matrix = numpy.full(
        actual_hr_matrix_k_day01.shape, False, dtype=bool
    )

    for i in range(num_examples):
        for h in range(num_heights):
            for w in range(num_wavelengths):
                if not large_error_flag_matrix[i, h, w]:
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
                    catastrophic_error_flag_matrix[i, h, w] = True

    num_catastrophic_errors_matrix = numpy.sum(
        catastrophic_error_flag_matrix, axis=0
    )

    return (
        num_large_errors_matrix, num_catastrophic_errors_matrix, num_examples
    )


def _run(prediction_file_names, large_error_threshold_k_day01,
         catastrophic_error_confidence_threshold, wavelengths_metres,
         output_dir_name):
    """Plots height profiles of large and catastrophic HR errors.

    This is effectively the main method.

    :param prediction_file_names: See documentation at top of file.
    :param large_error_threshold_k_day01: Same.
    :param catastrophic_error_confidence_threshold: Same.
    :param wavelengths_metres: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    error_checking.assert_is_greater(large_error_threshold_k_day01, 0.)
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

    num_large_errors_matrix = numpy.full(
        (num_heights, num_wavelengths), 0, dtype=int
    )
    num_catastrophic_errors_matrix = numpy.full(
        (num_heights, num_wavelengths), 0, dtype=int
    )
    num_examples = 0

    for this_file_name in prediction_file_names:
        (
            this_num_large_err_matrix,
            this_num_cat_err_matrix,
            this_num_examples
        ) = _compute_large_error_freqs_1file(
            prediction_file_name=this_file_name,
            large_error_threshold_k_day01=large_error_threshold_k_day01,
            catastrophic_error_confidence_threshold=
            catastrophic_error_confidence_threshold,
            wavelengths_metres=wavelengths_metres
        )

        num_large_errors_matrix += this_num_large_err_matrix
        num_catastrophic_errors_matrix += this_num_cat_err_matrix
        num_examples += this_num_examples

    large_error_freq_matrix = (
        num_large_errors_matrix.astype(float) / num_examples
    )
    catastrophic_error_freq_matrix = (
        num_catastrophic_errors_matrix.astype(float) / num_examples
    )

    for w in range(num_wavelengths):
        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        eval_plotting.plot_score_profile(
            heights_m_agl=heights_m_agl,
            score_values=large_error_freq_matrix[:, w],
            score_name=eval_plotting.PITD_NAME,
            line_colour=LINE_COLOUR, line_width=LINE_WIDTH,
            line_style='solid', use_log_scale=True,
            axes_object=axes_object, are_axes_new=True
        )

        axes_object.set_xlim([0, 0.5])
        axes_object.set_xlabel('Large-point-error frequency')

        title_string = (
            'Large-point-error freq for SW heating rate at {0:.2f}'
        ).format(
            METRES_TO_MICRONS * wavelengths_metres[w]
        )
        title_string += r' $\mu$m'
        title_string += '\nMax value = {0:.2f}'.format(
            numpy.max(large_error_freq_matrix[:, w])
        )
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/shortwave-heating-rate-k-day01_{1:.2f}microns_'
            'large_error_frequency.jpg'
        ).format(
            output_dir_name, METRES_TO_MICRONS * wavelengths_metres[w]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object, axes_object = pyplot.subplots(
            1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
        )
        eval_plotting.plot_score_profile(
            heights_m_agl=heights_m_agl,
            score_values=catastrophic_error_freq_matrix[:, w],
            score_name=eval_plotting.PITD_NAME,
            line_colour=LINE_COLOUR, line_width=LINE_WIDTH,
            line_style='solid', use_log_scale=True,
            axes_object=axes_object, are_axes_new=True
        )

        axes_object.set_xlim([0, 0.5])
        axes_object.set_xlabel('Catastrophic-error frequency')

        title_string = (
            'Catastrophic-error freq for SW heating rate at {0:.2f}'
        ).format(
            METRES_TO_MICRONS * wavelengths_metres[w]
        )
        title_string += r' $\mu$m'
        title_string += '\nMax value = {0:.2f}'.format(
            numpy.max(catastrophic_error_freq_matrix[:, w])
        )
        axes_object.set_title(title_string)

        figure_file_name = (
            '{0:s}/shortwave-heating-rate-k-day01_{1:.2f}microns_'
            'catastrophic_error_frequency.jpg'
        ).format(
            output_dir_name, METRES_TO_MICRONS * wavelengths_metres[w]
        )

        print('Saving figure to: "{0:s}"...'.format(figure_file_name))
        figure_object.savefig(
            figure_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_names=getattr(INPUT_ARG_OBJECT, INPUT_FILES_ARG_NAME),
        large_error_threshold_k_day01=getattr(
            INPUT_ARG_OBJECT, LARGE_ERR_THRES_ARG_NAME
        ),
        catastrophic_error_confidence_threshold=getattr(
            INPUT_ARG_OBJECT, CATASTROPHIC_ERR_CONF_THRES_ARG_NAME
        ),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
