"""Plots predicted vs. actual target variables.

For every data example, this script creates one plot, showing the following:

- Predicted heating-rate profile
- Actual heating-rate profile
- Predicted boundary fluxes
- Actual boundary fluxes

The heating-rate profiles are plotted in a line graph, and the boundary fluxes
are plotted in an inset bar graph.
"""

import os
import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from gewittergefahr.gg_utils import file_system_utils
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.utils import evaluation
from ml4rt.machine_learning import neural_net
from ml4rt.plotting import profile_plotting

FLUX_NAME_TO_FANCY_DICT = {
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME: r'$F_{down}^{sfc}$',
    example_utils.SHORTWAVE_TOA_UP_FLUX_NAME: r'$F_{up}^{TOA}$',
    evaluation.SHORTWAVE_NET_FLUX_NAME: r'$F_{net}$'
}

FLUX_FONT_SIZE = 20
TARGET_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
PREDICTION_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

FIGURE_RESOLUTION_DPI = 300

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
MODEL_DESCRIPTION_ARG_NAME = 'model_description_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, containing predicted and actual target variables '
    'for one model.  Will be read by `prediction_io.read_file`.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (in range 0...1) for prediction uncertainty.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Will plot the first N examples, where N = `{0:s}`.  If you want to plot '
    'all examples, leave this alone.'
).format(
    NUM_EXAMPLES_ARG_NAME
)
MODEL_DESCRIPTION_HELP_STRING = 'Model description (will be plotted in title).'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.95,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=-1,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTION_ARG_NAME, type=str, required=True,
    help=MODEL_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_one_comparison(
        actual_heating_rates_k_day01, predicted_hr_matrix_k_day01,
        actual_fluxes_w_m02, predicted_flux_matrix_w_m02,
        example_id_string, model_metadata_dict,
        confidence_level, model_description_string, output_dir_name):
    """Plots predicted vs. actual for one data example.

    E = number of examples
    H = number of heights
    T_s = number of scalar targets
    S = number of ensemble members

    :param actual_heating_rates_k_day01: numpy array (length H) with actual
        heating rates in Kelvins per day.
    :param predicted_hr_matrix_k_day01: numpy array (H x S) with predicted
        heating rates in Kelvins per day.
    :param actual_fluxes_w_m02: numpy array (length T_s) with actual fluxes in
        Watts per square metre.
    :param predicted_flux_matrix_w_m02: numpy array (T_s x S) with predicted
        fluxes in Watts per square metre.
    :param example_id_string: Example ID.
    :param model_metadata_dict: Model metadata in format returned by
        `neural_net.read_metafile`.
    :param confidence_level: See documentation at top of this script.
    :param model_description_string: Same.
    :param output_dir_name: Same.
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    handle_dict = profile_plotting.plot_actual_and_predicted(
        actual_values=actual_heating_rates_k_day01,
        prediction_matrix=predicted_hr_matrix_k_day01,
        heights_m_agl=heights_m_agl,
        fancy_target_name=r'shortwave heating rate (K day$^{-1}$)',
        line_colours=[TARGET_COLOUR, PREDICTION_COLOUR],
        line_widths=numpy.full(2, 3.),
        line_styles=['solid', 'dashed'],
        use_log_scale=True,
        add_two_dummy_axes=False,
        plot_uncertainty_with_shading=True,
        confidence_level=confidence_level
    )

    figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
    axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

    x_min = min([
        axes_objects[0].get_xlim()[0],
        axes_objects[1].get_xlim()[0]
    ])
    x_min = max([x_min, 0.])

    axes_objects[0].set_xlim(left=x_min)
    axes_objects[1].set_xlim(left=x_min)

    mean_abs_error_k_day01 = evaluation._get_mae_one_scalar(
        target_values=actual_heating_rates_k_day01,
        predicted_values=numpy.mean(predicted_hr_matrix_k_day01, axis=-1)
    )
    title_string = '{0:s} (HR MAE = {1:.2f}'.format(
        model_description_string, mean_abs_error_k_day01
    )
    title_string += r' K day$^{-1}$)'
    axes_objects[0].set_title(title_string)

    inset_axes_object = axes_objects[0].inset_axes([0.45, 0.1, 0.5, 0.5])
    num_flux_vars = len(actual_fluxes_w_m02)
    x_tick_values = numpy.linspace(
        0.5, num_flux_vars - 0.5, num=num_flux_vars, dtype=float
    )
    bar_width = 0.4

    mean_flux_predictions_w_m02 = numpy.mean(
        predicted_flux_matrix_w_m02, axis=-1
    )
    lower_flux_predictions_w_m02 = numpy.percentile(
        predicted_flux_matrix_w_m02, 50 * (1. - confidence_level), axis=-1
    )
    upper_flux_predictions_w_m02 = numpy.percentile(
        predicted_flux_matrix_w_m02, 50 * (1. + confidence_level), axis=-1
    )
    flux_error_matrix_w_m02 = numpy.vstack([
        mean_flux_predictions_w_m02 - lower_flux_predictions_w_m02,
        upper_flux_predictions_w_m02 - mean_flux_predictions_w_m02
    ])
    flux_error_matrix_w_m02 = numpy.maximum(flux_error_matrix_w_m02, 0.)

    error_bar_dict = {
        'ecolor': numpy.full(3, 0.),
        'elinewidth': 3,
        'capsize': 7.5,
        'capthick': 2
    }

    inset_axes_object.bar(
        x_tick_values - bar_width / 2,
        actual_fluxes_w_m02,
        bar_width,
        color=TARGET_COLOUR
    )
    inset_axes_object.bar(
        x_tick_values + bar_width / 2,
        mean_flux_predictions_w_m02,
        bar_width,
        yerr=flux_error_matrix_w_m02,
        color=PREDICTION_COLOUR,
        error_kw=error_bar_dict
    )

    for j in range(num_flux_vars):
        this_y = 0.25 * (
            actual_fluxes_w_m02[j] + mean_flux_predictions_w_m02[j]
        )
        this_label = 'Err =\n{0:.1f}'.format(
            mean_flux_predictions_w_m02[j] - actual_fluxes_w_m02[j]
        )

        inset_axes_object.text(
            x_tick_values[j], this_y, this_label,
            ha='center',
            va='bottom',
            fontsize=FLUX_FONT_SIZE,
            fontweight='bold',
            color=numpy.full(3, 0.)
        )

    flux_names = generator_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY]
    x_tick_labels = [FLUX_NAME_TO_FANCY_DICT[f] for f in flux_names]
    inset_axes_object.set_xticks(x_tick_values)
    inset_axes_object.set_xticklabels(x_tick_labels, fontsize=FLUX_FONT_SIZE)

    inset_axes_object.tick_params(axis='y', labelsize=FLUX_FONT_SIZE)
    inset_axes_object.set_title(r'Fluxes (W m$^{-2}$)', fontsize=FLUX_FONT_SIZE)

    this_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string.replace('_', '-')
    )
    print('Saving figure to: "{0:s}"...'.format(this_file_name))

    figure_object.savefig(
        this_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)


def _run(prediction_file_name, confidence_level, num_examples,
         model_description_string, output_dir_name):
    """Plots predicted vs. actual target variables.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of this script.
    :param confidence_level: Same.
    :param num_examples: Same.
    :param model_description_string: Same.
    :param output_dir_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    if num_examples < 1:
        num_examples = None

    print((
        'Reading predicted and actual target variables from: "{0:s}"...'
    ).format(
        prediction_file_name
    ))

    prediction_dict = prediction_io.read_file(prediction_file_name)
    num_examples_orig = len(prediction_dict[prediction_io.EXAMPLE_IDS_KEY])

    if num_examples is not None and num_examples < num_examples_orig:
        desired_indices = numpy.linspace(
            0, num_examples - 1, num=num_examples, dtype=int
        )
        prediction_dict = prediction_io.subset_by_index(
            prediction_dict=prediction_dict, desired_indices=desired_indices
        )

    wavelengths_metres = prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY]
    assert len(wavelengths_metres) == 1
    example_utils.match_wavelengths(
        wavelengths_metres=wavelengths_metres,
        desired_wavelength_metres=example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES
    )

    model_file_name = prediction_dict[prediction_io.MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading model metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    generator_option_dict[neural_net.HEIGHTS_KEY] = prediction_dict[
        prediction_io.HEIGHTS_KEY
    ]
    generator_option_dict[neural_net.TARGET_WAVELENGTHS_KEY] = prediction_dict[
        prediction_io.TARGET_WAVELENGTHS_KEY
    ]
    model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY] = generator_option_dict

    num_examples = len(prediction_dict[prediction_io.EXAMPLE_IDS_KEY])
    hr_index = generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY].index(
        example_utils.SHORTWAVE_HEATING_RATE_NAME
    )
    pdict = prediction_dict

    for i in range(num_examples):
        _plot_one_comparison(
            actual_heating_rates_k_day01=
            pdict[prediction_io.VECTOR_TARGETS_KEY][i, :, 0, hr_index],
            predicted_hr_matrix_k_day01=
            pdict[prediction_io.VECTOR_PREDICTIONS_KEY][i, :, 0, hr_index, :],
            actual_fluxes_w_m02=
            pdict[prediction_io.SCALAR_TARGETS_KEY][i, 0, :],
            predicted_flux_matrix_w_m02=
            pdict[prediction_io.SCALAR_PREDICTIONS_KEY][i, 0, :, :],
            example_id_string=pdict[prediction_io.EXAMPLE_IDS_KEY][i],
            model_metadata_dict=model_metadata_dict,
            confidence_level=confidence_level,
            model_description_string=model_description_string,
            output_dir_name=output_dir_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        model_description_string=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTION_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
