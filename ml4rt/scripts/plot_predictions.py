"""Plots predictions and target values, one plot per data example."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
import matplotlib.colors
from matplotlib import pyplot
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.plotting import plotting_utils as gg_plotting_utils
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting

FONT_SIZE = 30
FIGURE_RESOLUTION_DPI = 300

TITLE_TIME_FORMAT = '%H00 UTC %b %-d %Y'

RANDOM_PRIORITY_STRING = 'random'
HIGH_FLUX_ERROR_PRIORITY_STRING = 'high_flux_error'
LOW_FLUX_ERROR_PRIORITY_STRING = 'low_flux_error'
HIGH_HR_ERROR_PRIORITY_STRING = 'high_heating_rate_error'
LOW_HR_ERROR_PRIORITY_STRING = 'low_heating_rate_error'
HIGH_HR_PRIORITY_STRING = 'high_heating_rate'
LOW_HR_PRIORITY_STRING = 'low_heating_rate'
ALL_PRIORITY_STRINGS = [
    RANDOM_PRIORITY_STRING,
    HIGH_FLUX_ERROR_PRIORITY_STRING, LOW_FLUX_ERROR_PRIORITY_STRING,
    HIGH_HR_ERROR_PRIORITY_STRING, LOW_HR_ERROR_PRIORITY_STRING,
    HIGH_HR_PRIORITY_STRING, LOW_HR_PRIORITY_STRING
]

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
NUM_EXAMPLES_ARG_NAME = 'num_examples_to_plot'
PLOTTING_PRIORITY_ARG_NAME = 'plotting_priority_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file.  Predictions and targets, for many examples, '
    'will be read from here by `prediction_io.read_file`.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to plot.  One figure will be created for every example.'
)
PLOTTING_PRIORITY_HELP_STRING = (
    'This string determines which examples to plot.  Must be in the following '
    'list:\n{0:s}'
).format(
    str(ALL_PRIORITY_STRINGS)
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOTTING_PRIORITY_ARG_NAME, type=str, required=True,
    help=PLOTTING_PRIORITY_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(prediction_file_name, num_examples_to_plot, plotting_priority_string,
         output_dir_name):
    """Plots predictions and target values, one plot per data example.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of this script.
    :param num_examples_to_plot: Same.
    :param plotting_priority_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    assert plotting_priority_string in ALL_PRIORITY_STRINGS

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)

    num_examples_total = len(prediction_dict[prediction_io.EXAMPLE_IDS_KEY])
    if num_examples_to_plot >= num_examples_total:
        plotting_priority_string = ''

    pdict = prediction_dict

    if plotting_priority_string == RANDOM_PRIORITY_STRING:
        good_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )
        good_indices = numpy.random.choice(
            good_indices, size=num_examples_to_plot, replace=False
        )
    elif plotting_priority_string == HIGH_FLUX_ERROR_PRIORITY_STRING:
        error_matrix = numpy.absolute(
            pdict[prediction_io.SCALAR_TARGETS_KEY] -
            numpy.mean(pdict[prediction_io.SCALAR_PREDICTIONS_KEY], axis=-1)
        )
        error_by_example = numpy.mean(error_matrix, axis=(1, 2))
        good_indices = numpy.argsort(-1 * error_by_example)[
            :num_examples_to_plot
        ]
    elif plotting_priority_string == LOW_FLUX_ERROR_PRIORITY_STRING:
        error_matrix = numpy.absolute(
            pdict[prediction_io.SCALAR_TARGETS_KEY] -
            numpy.mean(pdict[prediction_io.SCALAR_PREDICTIONS_KEY], axis=-1)
        )
        error_by_example = numpy.mean(error_matrix, axis=(1, 2))
        good_indices = numpy.argsort(error_by_example)[:num_examples_to_plot]
    elif plotting_priority_string == HIGH_HR_ERROR_PRIORITY_STRING:
        error_matrix = numpy.absolute(
            pdict[prediction_io.VECTOR_TARGETS_KEY] -
            numpy.mean(pdict[prediction_io.VECTOR_PREDICTIONS_KEY], axis=-1)
        )
        error_by_example = numpy.mean(error_matrix, axis=(1, 2, 3))
        good_indices = numpy.argsort(-1 * error_by_example)[
            :num_examples_to_plot
        ]
    elif plotting_priority_string == LOW_HR_ERROR_PRIORITY_STRING:
        error_matrix = numpy.absolute(
            pdict[prediction_io.VECTOR_TARGETS_KEY] -
            numpy.mean(pdict[prediction_io.VECTOR_PREDICTIONS_KEY], axis=-1)
        )
        error_by_example = numpy.mean(error_matrix, axis=(1, 2, 3))
        good_indices = numpy.argsort(error_by_example)[:num_examples_to_plot]
    elif plotting_priority_string == HIGH_HR_PRIORITY_STRING:
        max_hr_by_example_k_day01 = numpy.max(
            pdict[prediction_io.VECTOR_TARGETS_KEY], axis=(1, 2, 3)
        )
        good_indices = numpy.argsort(-1 * max_hr_by_example_k_day01)[
            :num_examples_to_plot
        ]
    elif plotting_priority_string == LOW_HR_PRIORITY_STRING:
        max_hr_by_example_k_day01 = numpy.max(
            pdict[prediction_io.VECTOR_TARGETS_KEY], axis=(1, 2, 3)
        )
        good_indices = numpy.argsort(max_hr_by_example_k_day01)[
            :num_examples_to_plot
        ]
    else:
        good_indices = numpy.linspace(
            0, num_examples_total - 1, num=num_examples_total, dtype=int
        )

    prediction_dict = prediction_io.subset_by_index(
        prediction_dict=prediction_dict, desired_indices=good_indices
    )
    pdict = prediction_dict
    num_examples = len(pdict[prediction_io.EXAMPLE_IDS_KEY])

    vector_target_matrix = pdict[prediction_io.VECTOR_TARGETS_KEY]
    assert vector_target_matrix.shape[-1] == 1

    heights_m_agl = pdict[prediction_io.HEIGHTS_KEY]
    wavelengths_metres = pdict[prediction_io.TARGET_WAVELENGTHS_KEY]
    example_id_strings = pdict[prediction_io.EXAMPLE_IDS_KEY]

    target_hr_matrix_k_day01 = vector_target_matrix[..., 0]
    predicted_hr_matrix_k_day01 = numpy.mean(
        pdict[prediction_io.VECTOR_PREDICTIONS_KEY], axis=-1
    )[..., 0]

    sort_indices = numpy.argsort(wavelengths_metres)[:-1]
    target_hr_matrix_k_day01 = target_hr_matrix_k_day01[..., sort_indices]
    predicted_hr_matrix_k_day01 = predicted_hr_matrix_k_day01[..., sort_indices]
    wavelengths_metres = wavelengths_metres[sort_indices]

    num_wavelengths = len(wavelengths_metres)
    wavelength_indices = 0.5 + numpy.linspace(
        0, num_wavelengths - 1, num=num_wavelengths, dtype=float
    )

    colour_map_object = pyplot.cm.get_cmap('viridis', num_wavelengths)
    colour_norm_object = pyplot.Normalize(vmin=0, vmax=num_wavelengths)
    wavelength_strings_microns = [
        '{0:.2f}'.format(w * 1e6) for w in wavelengths_metres
    ]

    for i in range(num_examples):
        metadata_dict = example_utils.parse_example_ids(
            [pdict[prediction_io.EXAMPLE_IDS_KEY][i]]
        )
        latitude_deg_n = metadata_dict[example_utils.LATITUDES_KEY][0]
        longitude_deg_e = lng_conversion.convert_lng_negative_in_west(
            metadata_dict[example_utils.LONGITUDES_KEY][0]
        )
        valid_time_string = time_conversion.unix_sec_to_string(
            metadata_dict[example_utils.VALID_TIMES_KEY][0],
            TITLE_TIME_FORMAT
        )

        example_id_string_fancy = 'for {0:s} at {1:.2f}'.format(
            valid_time_string, latitude_deg_n
        )
        example_id_string_fancy += r'$^{\circ}$N'
        example_id_string_fancy += ', {0:.2f}'.format(longitude_deg_e)
        example_id_string_fancy += r'$^{\circ}$E'

        figure_object = None
        axes_object = None

        for w in range(num_wavelengths):
            figure_object, axes_object = profile_plotting.plot_one_variable(
                values=target_hr_matrix_k_day01[i, :, w],
                heights_m_agl=heights_m_agl,
                use_log_scale=True,
                line_colour=matplotlib.colors.to_rgba(
                    c=colour_map_object(colour_norm_object(w + 0.5)), alpha=0.5
                ),
                line_width=4.5,
                line_style='solid',
                figure_object=figure_object
            )

            figure_object, axes_object = profile_plotting.plot_one_variable(
                values=predicted_hr_matrix_k_day01[i, :, w],
                heights_m_agl=heights_m_agl,
                use_log_scale=True,
                line_colour=colour_map_object(colour_norm_object(w + 0.5)),
                line_width=4.5,
                line_style='dashed',
                figure_object=figure_object
            )

        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=numpy.linspace(
                0, num_wavelengths - 1, num=num_wavelengths, dtype=float
            ),
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=False, extend_max=False
        )
        colour_bar_object.set_ticks(wavelength_indices)
        colour_bar_object.set_ticklabels(wavelength_strings_microns)
        colour_bar_object.set_label(r'Wavelength ($\mu$m)', fontsize=FONT_SIZE)
        axes_object.set_xlabel(r'Heating rate (K day$^{-1}$)')

        title_string = 'Actual (solid) and predicted (dashed) HR\n{0:s}'.format(
            example_id_string_fancy
        )
        axes_object.set_title(title_string, fontsize=FONT_SIZE)

        this_file_name = '{0:s}/{1:s}_comparison.jpg'.format(
            output_dir_name,
            example_id_strings[i].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

        figure_object = None

        for w in range(num_wavelengths):
            figure_object, axes_object = profile_plotting.plot_one_variable(
                values=(
                    predicted_hr_matrix_k_day01[i, :, w] -
                    target_hr_matrix_k_day01[i, :, w]
                ),
                heights_m_agl=heights_m_agl,
                use_log_scale=True,
                line_colour=colour_map_object(colour_norm_object(w + 0.5)),
                line_width=4.5,
                line_style='solid',
                figure_object=figure_object
            )

        colour_bar_object = gg_plotting_utils.plot_colour_bar(
            axes_object_or_matrix=axes_object,
            data_matrix=numpy.linspace(
                0, num_wavelengths - 1, num=num_wavelengths, dtype=float
            ),
            colour_map_object=colour_map_object,
            colour_norm_object=colour_norm_object,
            orientation_string='vertical',
            extend_min=False, extend_max=False
        )
        colour_bar_object.set_ticks(wavelength_indices)
        colour_bar_object.set_ticklabels(wavelength_strings_microns)
        colour_bar_object.set_label(r'Wavelength ($\mu$m)', fontsize=FONT_SIZE)
        axes_object.set_xlabel(r'Predicted minus actual (K day$^{-1}$)')
        title_string = 'HR errors\n{0:s}'.format(example_id_string_fancy)
        axes_object.set_title(title_string, fontsize=FONT_SIZE)

        this_file_name = '{0:s}/{1:s}_error.jpg'.format(
            output_dir_name,
            example_id_strings[i].replace('_', '-')
        )
        print('Saving figure to: "{0:s}"...'.format(this_file_name))
        figure_object.savefig(
            this_file_name, dpi=FIGURE_RESOLUTION_DPI,
            pad_inches=0, bbox_inches='tight'
        )
        pyplot.close(figure_object)

    example_id_array_string = ' '.join([
        '"{0:s}"'.format(e) for e in pdict[prediction_io.EXAMPLE_IDS_KEY]
    ])
    print('Example IDs as one array string:\n{0:s}'.format(
        example_id_array_string
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        num_examples_to_plot=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        plotting_priority_string=getattr(
            INPUT_ARG_OBJECT, PLOTTING_PRIORITY_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
