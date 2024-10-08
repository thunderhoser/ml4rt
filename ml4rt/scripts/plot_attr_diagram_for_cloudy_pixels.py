"""Plots height-agnostic attributes diagram for cloudy pixels."""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import example_io
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.utils import evaluation
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import evaluation_plotting

LINE_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

METRES_TO_MICRONS = 1e6

FIGURE_WIDTH_INCHES = 15
FIGURE_HEIGHT_INCHES = 15
FIGURE_RESOLUTION_DPI = 300

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
MIN_LWC_ARG_NAME = 'minimum_lwc_kg_m03'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
PLOT_SHORTWAVE_ARG_NAME = 'plot_shortwave'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing actual and predicted heating rates.  Will '
    'be read by `prediction_io.read_file`.'
)
MIN_LWC_HELP_STRING = (
    'Minimum liquid water content, used to define which pixels are "cloudy".'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with example files, to be found by '
    '`example_io.find_file` and read by `example_io.read_file`.  LWC values '
    'will be read from these files.'
)
PLOT_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot attributes diagram for shortwave '
    '(longwave) heating rate.'
)
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one attributes diagram for each.'
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
    '--' + MIN_LWC_ARG_NAME, type=float, required=True, help=MIN_LWC_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=True,
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PLOT_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=PLOT_SHORTWAVE_HELP_STRING
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


def _get_lwc_values(prediction_dict, example_dir_name):
    """Returns LWC profile for each predicted example.

    E = number of examples
    H = number of heights

    :param prediction_dict: Dictionary returned by `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of file.
    :return: lwc_matrix_kg_m03: E-by-H numpy array of liquid water contents.
    """

    valid_times_unix_sec = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    example_file_names = example_io.find_many_files(
        directory_name=example_dir_name,
        first_time_unix_sec=numpy.min(valid_times_unix_sec),
        last_time_unix_sec=numpy.max(valid_times_unix_sec),
        raise_error_if_any_missing=False
    )

    example_id_strings = []
    lwc_matrix_kg_m03 = numpy.array([])

    for this_file_name in example_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(
            netcdf_file_name=this_file_name,
            max_shortwave_heating_k_day01=numpy.inf,
            min_longwave_heating_k_day01=-1 * numpy.inf,
            max_longwave_heating_k_day01=numpy.inf
        )

        example_id_strings += this_example_dict[example_utils.EXAMPLE_IDS_KEY]

        this_matrix = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.LIQUID_WATER_CONTENT_NAME
        )

        if lwc_matrix_kg_m03.size == 0:
            lwc_matrix_kg_m03 = this_matrix + 0.
        else:
            lwc_matrix_kg_m03 = numpy.concatenate(
                (lwc_matrix_kg_m03, this_matrix), axis=0
            )

    desired_indices = example_utils.find_examples(
        all_id_strings=example_id_strings,
        desired_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
        allow_missing=False
    )

    return lwc_matrix_kg_m03[desired_indices, :]


def _plot_attr_diagram_1wavelength(
        training_example_dict, prediction_dict, example_dir_name,
        plot_shortwave, wavelength_metres, min_lwc_kg_m03, output_dir_name):
    """Plots attributes diagram for heating rate at one wavelength.

    :param training_example_dict: Dictionary with training examples for
        normalization, in format returned by `example_io.read_file`.
    :param prediction_dict: Dictionary with predictions, in format returned by
        `prediction_io.read_file`.
    :param example_dir_name: See documentation at top of this script.
    :param plot_shortwave: Same.
    :param wavelength_metres: Wavelength.
    :param min_lwc_kg_m03: See documentation at top of this script.
    :param output_dir_name: Same.
    """

    # Housekeeping.
    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    vector_target_names = (
        training_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
    )

    if plot_shortwave:
        t = vector_target_names.index(example_utils.SHORTWAVE_HEATING_RATE_NAME)
    else:
        t = vector_target_names.index(example_utils.LONGWAVE_HEATING_RATE_NAME)

    w = example_utils.match_wavelengths(
        wavelengths_metres=
        training_option_dict[neural_net.TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelength_metres
    )

    # Do actual stuff.
    training_hr_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=training_example_dict,
        field_name=(
            example_utils.SHORTWAVE_HEATING_RATE_NAME if plot_shortwave
            else example_utils.LONGWAVE_HEATING_RATE_NAME
        ),
        target_wavelength_metres=wavelength_metres
    )
    climo_heating_rate_k_day01 = numpy.mean(training_hr_matrix_k_day01)

    lwc_matrix_kg_m03 = _get_lwc_values(
        prediction_dict=prediction_dict, example_dir_name=example_dir_name
    )

    actual_heating_rates_k_day01 = prediction_dict[
        prediction_io.VECTOR_TARGETS_KEY
    ][..., w, t][lwc_matrix_kg_m03 >= min_lwc_kg_m03]

    predicted_heating_rates_k_day01 = prediction_dict[
        prediction_io.VECTOR_PREDICTIONS_KEY
    ][..., w, t][lwc_matrix_kg_m03 >= min_lwc_kg_m03]

    if plot_shortwave:
        (
            mean_predictions_k_day01,
            mean_observations_k_day01,
            prediction_counts
        ) = evaluation._get_rel_curve_one_scalar(
            target_values=actual_heating_rates_k_day01,
            predicted_values=predicted_heating_rates_k_day01,
            num_bins=41, min_bin_edge=0, max_bin_edge=41, invert=False
        )

        _, _, observation_counts = evaluation._get_rel_curve_one_scalar(
            target_values=actual_heating_rates_k_day01,
            predicted_values=predicted_heating_rates_k_day01,
            num_bins=41, min_bin_edge=0, max_bin_edge=41, invert=True
        )
    else:
        (
            mean_predictions_k_day01,
            mean_observations_k_day01,
            prediction_counts
        ) = evaluation._get_rel_curve_one_scalar(
            target_values=actual_heating_rates_k_day01,
            predicted_values=predicted_heating_rates_k_day01,
            num_bins=62, min_bin_edge=-51, max_bin_edge=11, invert=False
        )

        _, _, observation_counts = evaluation._get_rel_curve_one_scalar(
            target_values=actual_heating_rates_k_day01,
            predicted_values=predicted_heating_rates_k_day01,
            num_bins=62, min_bin_edge=-51, max_bin_edge=11, invert=True
        )

    concat_values = numpy.concatenate(
        (mean_predictions_k_day01, mean_observations_k_day01), axis=0
    )

    figure_object, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    evaluation_plotting.plot_attributes_diagram(
        figure_object=figure_object, axes_object=axes_object,
        mean_predictions=mean_predictions_k_day01,
        mean_observations=mean_observations_k_day01,
        mean_value_in_training=climo_heating_rate_k_day01,
        min_value_to_plot=numpy.nanmin(concat_values),
        max_value_to_plot=numpy.nanmax(concat_values),
        line_colour=LINE_COLOUR, line_style='solid', line_width=4
    )
    evaluation_plotting.plot_inset_histogram(
        figure_object=figure_object, bin_centers=mean_predictions_k_day01,
        bin_counts=prediction_counts, has_predictions=True,
        bar_colour=LINE_COLOUR
    )
    evaluation_plotting.plot_inset_histogram(
        figure_object=figure_object, bin_centers=mean_predictions_k_day01,
        bin_counts=observation_counts, has_predictions=False,
        bar_colour=LINE_COLOUR
    )

    axes_object.set_xlabel(r'Prediction (K day$^{-1}$)')
    axes_object.set_ylabel(r'Conditional mean observation (K day$^{-1}$)')

    title_string = (
        'Attributes diagram for {0:s} HR at {1:.2f} microns\n'
        'with LWC >= {2:.1f} g'
    ).format(
        'shortwave' if plot_shortwave else 'longwave',
        METRES_TO_MICRONS * wavelength_metres,
        1000 * min_lwc_kg_m03
    )

    title_string += r' m$^{-3}$'
    axes_object.set_title(title_string)

    output_file_name = (
        '{0:s}/attributes_diagram_cloudy_pixels_{1:.2f}microns.jpg'
    ).format(
        output_dir_name,
        METRES_TO_MICRONS * wavelength_metres
    )

    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_name, min_lwc_kg_m03, example_dir_name, plot_shortwave,
         wavelengths_metres, output_dir_name):
    """Plots height-agnostic attributes diagram for cloudy pixels.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param min_lwc_kg_m03: Same.
    :param example_dir_name: Same.
    :param plot_shortwave: Same.
    :param wavelengths_metres: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    error_checking.assert_is_greater(min_lwc_kg_m03, 0.)

    print('Reading data from: "{0:s}"...'.format(prediction_file_name))
    prediction_dict = prediction_io.read_file(prediction_file_name)
    prediction_dict = prediction_io.get_ensemble_mean(prediction_dict)

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    normalization_file_name = (
        prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
    )
    if normalization_file_name is None:
        normalization_file_name = (
            training_option_dict[neural_net.NORMALIZATION_FILE_KEY]
        )

    print((
        'Reading training examples (for climatology) from: "{0:s}"...'
    ).format(
        normalization_file_name
    ))
    training_example_dict = example_io.read_file(normalization_file_name)
    training_example_dict = example_utils.subset_by_height(
        example_dict=training_example_dict,
        heights_m_agl=training_option_dict[neural_net.HEIGHTS_KEY]
    )

    for this_wavelength_metres in wavelengths_metres:
        _plot_attr_diagram_1wavelength(
            training_example_dict=training_example_dict,
            prediction_dict=prediction_dict,
            example_dir_name=example_dir_name,
            plot_shortwave=plot_shortwave,
            wavelength_metres=this_wavelength_metres,
            min_lwc_kg_m03=min_lwc_kg_m03,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        min_lwc_kg_m03=getattr(INPUT_ARG_OBJECT, MIN_LWC_ARG_NAME),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        plot_shortwave=bool(getattr(INPUT_ARG_OBJECT, PLOT_SHORTWAVE_ARG_NAME)),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
