"""Plots comparisons between predicted and actual (target) profiles."""

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
import prediction_io
import example_utils
import evaluation
import neural_net
import profile_plotting
import plot_errors_by_aod_and_sza as plot_errors_by_aod
import plot_errors_by_sfc_temp_and_moisture as plot_errors_by_lapse_rates

TITLE_FONT_SIZE = 15
FIGURE_RESOLUTION_DPI = 300

KG_TO_GRAMS = 1000.
METRES_TO_MICRONS = 1e6
RADIANS_TO_DEGREES = 180. / numpy.pi

SHORTWAVE_VECTOR_TARGET_NAMES = [
    example_utils.SHORTWAVE_HEATING_RATE_NAME,
    example_utils.SHORTWAVE_DOWN_FLUX_NAME, example_utils.SHORTWAVE_UP_FLUX_NAME
]
LONGWAVE_VECTOR_TARGET_NAMES = [
    example_utils.LONGWAVE_HEATING_RATE_NAME,
    example_utils.LONGWAVE_DOWN_FLUX_NAME, example_utils.LONGWAVE_UP_FLUX_NAME
]

TARGET_NAME_TO_VERBOSE = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: 'Downwelling shortwave flux',
    example_utils.SHORTWAVE_UP_FLUX_NAME: 'Upwelling shortwave flux',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: 'Shortwave heating rate',
    example_utils.LONGWAVE_DOWN_FLUX_NAME: 'Downwelling longwave flux',
    example_utils.LONGWAVE_UP_FLUX_NAME: 'Upwelling longwave flux',
    example_utils.LONGWAVE_HEATING_RATE_NAME: 'Longwave heating rate'
}

TARGET_NAME_TO_UNITS = {
    example_utils.SHORTWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_utils.SHORTWAVE_HEATING_RATE_NAME: r'K day$^{-1}$',
    example_utils.LONGWAVE_DOWN_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_UP_FLUX_NAME: r'W m$^{-2}$',
    example_utils.LONGWAVE_HEATING_RATE_NAME: r'K day$^{-1}$'
}

TARGET_COLOUR = numpy.array([31, 120, 180], dtype=float) / 255
PREDICTION_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255

LEGEND_BOUNDING_BOX_DICT = {
    'facecolor': 'white',
    'alpha': 0.5,
    'edgecolor': 'black',
    'linewidth': 1,
    'boxstyle': 'round'
}

PREDICTION_FILE_ARG_NAME = 'input_prediction_file_name'
PLOT_SHORTWAVE_ARG_NAME = 'plot_shortwave'
WAVELENGTHS_ARG_NAME = 'wavelengths_metres'
PLOT_UNCERTAINTY_ARG_NAME = 'plot_uncertainty'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
USE_LOG_SCALE_ARG_NAME = 'use_log_scale'
ADD_DUMMY_AXES_ARG_NAME = 'add_two_dummy_axes'
EXAMPLE_DIR_ARG_NAME = 'input_example_dir_name'
MODEL_DESCRIPTION_ARG_NAME = 'model_description_string'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

PREDICTION_FILE_HELP_STRING = (
    'Path to prediction file, containing both predicted and actual (target) '
    'profiles.  Will be read by `prediction_io.read_file`.'
)
PLOT_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot shortwave (longwave) values.'
)
WAVELENGTHS_HELP_STRING = (
    'List of wavelengths.  Will create one set of plots for each.'
)
PLOT_UNCERTAINTY_HELP_STRING = (
    'Boolean flag.  If 1 (0), will plot uncertainty in predictions.'
)
CONFIDENCE_LEVEL_HELP_STRING = (
    'Confidence level (in range 0...1) for prediction uncertainty.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Will plot the first N examples, where N = `{0:s}`.  If you want to plot '
    'all examples, leave this alone.'
).format(NUM_EXAMPLES_ARG_NAME)

USE_LOG_SCALE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use logarithmic (linear) scale for height '
    'axis.'
)
ADD_DUMMY_AXES_HELP_STRING = (
    'Boolean flag.  If 1, will add two dummy x-axes that correspond to nothing.'
    '  The only reason for doing this is to make the vertical scale of the '
    'figure match another figure with 4 variables plotted.'
)
EXAMPLE_DIR_HELP_STRING = (
    'Name of directory with full example files.  Will use these files to add '
    'metadata to legend.  If you do not want metadata in legend, leave this '
    'argument empty.'
)
MODEL_DESCRIPTION_HELP_STRING = (
    'Model description (will be plotted at the top of each legend).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + PREDICTION_FILE_ARG_NAME, type=str, required=True,
    help=PREDICTION_FILE_HELP_STRING
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
    '--' + PLOT_UNCERTAINTY_ARG_NAME, type=int, required=True,
    help=PLOT_UNCERTAINTY_HELP_STRING
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
    '--' + USE_LOG_SCALE_ARG_NAME, type=int, required=False, default=1,
    help=USE_LOG_SCALE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ADD_DUMMY_AXES_ARG_NAME, type=int, required=False, default=1,
    help=ADD_DUMMY_AXES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_DIR_ARG_NAME, type=str, required=False, default='',
    help=EXAMPLE_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_DESCRIPTION_ARG_NAME, type=str, required=False, default='',
    help=MODEL_DESCRIPTION_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _plot_comparisons_fancy(
        vector_target_matrix, vector_prediction_matrix, example_id_strings,
        model_metadata_dict, use_log_scale, plot_shortwave, wavelength_metres,
        output_dir_name):
    """Plots fancy comparisons (with all target variables in the same plot).

    E = number of examples
    H = number of heights
    T = number of target variables
    S = ensemble size

    :param vector_target_matrix: E-by-H-by-T numpy array of target (actual)
        values.
    :param vector_prediction_matrix: E-by-H-by-T-by-S numpy array of predicted
        values.
    :param example_id_strings: length-E list of example IDs.
    :param model_metadata_dict: Dictionary returned by
        `neural_net.read_metadata`.
    :param use_log_scale: See documentation at top of file.
    :param plot_shortwave: Same.
    :param wavelength_metres: Will plot for this wavelength only.
    :param output_dir_name: Path to output directory (figures will be saved
        here).
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    target_example_dict = {
        example_utils.HEIGHTS_KEY:
            generator_option_dict[neural_net.HEIGHTS_KEY],
        example_utils.TARGET_WAVELENGTHS_KEY:
            generator_option_dict[neural_net.TARGET_WAVELENGTHS_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_VALS_KEY: vector_target_matrix
    }

    prediction_example_dict = {
        example_utils.HEIGHTS_KEY:
            generator_option_dict[neural_net.HEIGHTS_KEY],
        example_utils.TARGET_WAVELENGTHS_KEY:
            generator_option_dict[neural_net.TARGET_WAVELENGTHS_KEY],
        example_utils.VECTOR_TARGET_NAMES_KEY:
            generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY],
        example_utils.VECTOR_TARGET_VALS_KEY:
            numpy.mean(vector_prediction_matrix, axis=-1)
    }

    num_examples = vector_target_matrix.shape[0]

    for i in range(num_examples):
        this_handle_dict = profile_plotting.plot_targets(
            example_dict=target_example_dict, example_index=i,
            for_shortwave=plot_shortwave, wavelength_metres=wavelength_metres,
            use_log_scale=use_log_scale, line_style='solid', handle_dict=None
        )
        profile_plotting.plot_targets(
            example_dict=prediction_example_dict, example_index=i,
            for_shortwave=plot_shortwave, wavelength_metres=wavelength_metres,
            use_log_scale=use_log_scale,
            line_style='dashed', handle_dict=this_handle_dict
        )

        this_file_name = '{0:s}/{1:s}_comparison_{2:.2f}microns.jpg'.format(
            output_dir_name,
            example_id_strings[i].replace('_', '-'),
            METRES_TO_MICRONS * wavelength_metres
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
        vector_target_matrix, vector_prediction_matrix, example_id_strings,
        model_metadata_dict, use_log_scale, add_two_dummy_axes, plot_shortwave,
        wavelength_metres, plot_uncertainty, confidence_level,
        annotation_strings, model_description_string, output_dir_name):
    """Plots simple comparisons (with each target var in a different plot).

    :param vector_target_matrix: See doc for `_plot_comparisons_fancy`.
    :param vector_prediction_matrix: Same.
    :param example_id_strings: Same.
    :param model_metadata_dict: Same.
    :param use_log_scale: Same.
    :param add_two_dummy_axes: Same.
    :param plot_shortwave: Same.
    :param wavelength_metres: Same.
    :param plot_uncertainty: Same.
    :param confidence_level: Same.
    :param annotation_strings: 1-D list of annotations, one per example.
    :param model_description_string: See documentation at top of file.
    :param output_dir_name: See doc for `_plot_comparisons_fancy`.
    """

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    heights_m_agl = generator_option_dict[neural_net.HEIGHTS_KEY]

    if plot_shortwave:
        target_names = [
            n for n in generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
            if n in SHORTWAVE_VECTOR_TARGET_NAMES
        ]
    else:
        target_names = [
            n for n in generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY]
            if n in LONGWAVE_VECTOR_TARGET_NAMES
        ]

    num_examples = vector_target_matrix.shape[0]
    num_target_vars = len(target_names)

    w = example_utils.match_wavelengths(
        wavelengths_metres=
        generator_option_dict[neural_net.TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelength_metres
    )

    for i in range(num_examples):
        for j in range(num_target_vars):
            t = generator_option_dict[neural_net.VECTOR_TARGET_NAMES_KEY].index(
                target_names[j]
            )

            fancy_target_name = '{0:s}{1:s} ({2:s})'.format(
                TARGET_NAME_TO_VERBOSE[target_names[j]][0].lower(),
                TARGET_NAME_TO_VERBOSE[target_names[j]][1:],
                TARGET_NAME_TO_UNITS[target_names[j]]
            )

            handle_dict = profile_plotting.plot_actual_and_predicted(
                actual_values=vector_target_matrix[i, :, w, t],
                prediction_matrix=vector_prediction_matrix[i, :, w, t, :],
                heights_m_agl=heights_m_agl,
                fancy_target_name=fancy_target_name,
                line_colours=[TARGET_COLOUR, PREDICTION_COLOUR],
                line_widths=numpy.full(2, 2.),
                line_styles=['solid', 'dashed'],
                use_log_scale=use_log_scale,
                add_two_dummy_axes=add_two_dummy_axes,
                plot_uncertainty_with_shading=plot_uncertainty,
                confidence_level=confidence_level
            )

            this_figure_object = handle_dict[profile_plotting.FIGURE_HANDLE_KEY]
            these_axes_objects = handle_dict[profile_plotting.AXES_OBJECTS_KEY]

            if plot_shortwave:
                x_min = min([
                    these_axes_objects[0].get_xlim()[0],
                    these_axes_objects[1].get_xlim()[0]
                ])
                x_min = max([x_min, 0.])

                these_axes_objects[0].set_xlim(left=x_min)
                these_axes_objects[1].set_xlim(left=x_min)

            this_mae = evaluation._get_mae_one_scalar(
                target_values=vector_target_matrix[i, :, w, t],
                predicted_values=
                numpy.mean(vector_prediction_matrix[i, :, w, t, :], axis=-1)
            )
            this_annotation_string = 'HR MAE = {0:.2f} {1:s}'.format(
                this_mae, TARGET_NAME_TO_UNITS[target_names[j]]
            )

            if model_description_string is not None:
                this_annotation_string = (
                    model_description_string + ':\n\n' +
                    this_annotation_string + '\n' +
                    annotation_strings[i]
                )
            elif annotation_strings[i] is not None:
                this_annotation_string += '\n' + annotation_strings[i]

            if plot_shortwave:
                these_axes_objects[1].text(
                    0.99, 0.1, this_annotation_string,
                    fontsize=TITLE_FONT_SIZE, color='k',
                    bbox=LEGEND_BOUNDING_BOX_DICT,
                    horizontalalignment='right', verticalalignment='bottom',
                    transform=these_axes_objects[1].transAxes, zorder=1e13
                )
            else:
                these_axes_objects[1].text(
                    0.01, 0.1, this_annotation_string,
                    fontsize=TITLE_FONT_SIZE, color='k',
                    bbox=LEGEND_BOUNDING_BOX_DICT,
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=these_axes_objects[1].transAxes, zorder=1e13
                )

            this_file_name = '{0:s}/{1:s}_{2:s}_{3:.2f}microns.jpg'.format(
                output_dir_name,
                example_id_strings[i].replace('_', '-'),
                target_names[j].replace('_', '-'),
                METRES_TO_MICRONS * wavelength_metres
            )
            print('Saving figure to: "{0:s}"...'.format(this_file_name))

            this_figure_object.savefig(
                this_file_name, dpi=FIGURE_RESOLUTION_DPI,
                pad_inches=0, bbox_inches='tight'
            )
            pyplot.close(this_figure_object)


def _run(prediction_file_name, plot_shortwave, wavelengths_metres,
         plot_uncertainty, confidence_level, num_examples, use_log_scale,
         add_two_dummy_axes, example_dir_name, model_description_string,
         output_dir_name):
    """Plots comparisons between predicted and actual (target) profiles.

    This is effectively the main method.

    :param prediction_file_name: See documentation at top of file.
    :param plot_shortwave: Same.
    :param wavelengths_metres: Same.
    :param plot_uncertainty: Same.
    :param confidence_level: Same.
    :param num_examples: Same.
    :param use_log_scale: Same.
    :param add_two_dummy_axes: Same.
    :param example_dir_name: Same.
    :param model_description_string: Same.
    :param output_dir_name: Same.
    """

    # Check input args.
    if example_dir_name in ['', 'None', 'none']:
        example_dir_name = None
    if model_description_string in ['', 'None', 'none']:
        model_description_string = None

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )
    if num_examples < 1:
        num_examples = None

    # Do actual stuff.
    print((
        'Reading predicted and actual (target) profiles from: "{0:s}"...'
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

    if example_dir_name is not None:
        if plot_shortwave:
            aod_values = plot_errors_by_aod._get_aerosol_optical_depths(
                prediction_dict=prediction_dict,
                example_dir_name=example_dir_name
            )
            zenith_angles_rad = example_utils.parse_example_ids(
                prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
            )[example_utils.ZENITH_ANGLES_KEY]

            zenith_angles_deg = RADIANS_TO_DEGREES * zenith_angles_rad

            annotation_strings = [
                'AOD = {0:.2f}\nSZA = {1:.1f} deg'.format(a, z)
                for a, z in zip(aod_values, zenith_angles_deg)
            ]
        else:
            temperature_lapse_rates_k_km01 = (
                plot_errors_by_lapse_rates._get_temperature_values(
                    prediction_dict=prediction_dict,
                    example_dir_name=example_dir_name,
                    get_lapse_rates=True
                )
            )

            humidity_lapse_rates_g_kg01_km01 = KG_TO_GRAMS * (
                plot_errors_by_lapse_rates._get_humidity_values(
                    prediction_dict=prediction_dict,
                    example_dir_name=example_dir_name,
                    get_lapse_rates=True
                )
            )

            annotation_strings = [
                'TT = {0:.2f} KK\nqq = {1:.1f} gg'.format(t, h) for t, h in
                zip(
                    temperature_lapse_rates_k_km01,
                    humidity_lapse_rates_g_kg01_km01
                )
            ]
            annotation_strings = [
                s.replace('TT', r'$\Gamma_T^{sfc}$') for s in annotation_strings
            ]
            annotation_strings = [
                s.replace('KK', r'K km$^{-1}$') for s in annotation_strings
            ]
            annotation_strings = [
                s.replace('qq', r'$\Gamma_q^{sfc}$') for s in annotation_strings
            ]
            annotation_strings = [
                s.replace('gg', r'g kg$^{-1}$ km$^{-1}$')
                for s in annotation_strings
            ]

    vector_target_matrix = prediction_dict[prediction_io.VECTOR_TARGETS_KEY]
    vector_prediction_matrix = (
        prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY]
    )
    scalar_target_matrix = prediction_dict[prediction_io.SCALAR_TARGETS_KEY]
    scalar_prediction_matrix = (
        prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY]
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

    generator_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
    vector_target_names = generator_option_dict[
        neural_net.VECTOR_TARGET_NAMES_KEY
    ]
    scalar_target_names = generator_option_dict[
        neural_net.SCALAR_TARGET_NAMES_KEY
    ]

    for this_wavelength_metres in wavelengths_metres:
        w = example_utils.match_wavelengths(
            wavelengths_metres=
            prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
            desired_wavelength_metres=this_wavelength_metres
        )

        d_idx = scalar_target_names.index(
            example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME if plot_shortwave
            else example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
        )
        down_flux_strings = [
            'Fdown error = {0:.1f} WW'.format(b) for b in (
                numpy.mean(scalar_prediction_matrix[:, w, d_idx, :], axis=-1)
                - scalar_target_matrix[:, w, d_idx]
            )
        ]
        down_flux_strings = [
            s.replace('Fdown', r'$F_{down}^{sfc}$')
            for s in down_flux_strings
        ]

        u_idx = scalar_target_names.index(
            example_utils.SHORTWAVE_TOA_UP_FLUX_NAME if plot_shortwave
            else example_utils.LONGWAVE_TOA_UP_FLUX_NAME
        )
        up_flux_strings = [
            'Fup error = {0:.1f} WW'.format(b) for b in (
                numpy.mean(scalar_prediction_matrix[:, w, u_idx, :], axis=-1)
                - scalar_target_matrix[:, w, u_idx]
            )
        ]
        up_flux_strings = [
            s.replace('Fup', r'$F_{up}^{TOA}$') for s in up_flux_strings
        ]

        actual_net_fluxes_w_m02 = (
            scalar_target_matrix[:, w, d_idx] -
            scalar_target_matrix[:, w, u_idx]
        )
        predicted_net_fluxes_w_m02 = (
            numpy.mean(scalar_prediction_matrix[:, w, d_idx, :], axis=-1) -
            numpy.mean(scalar_prediction_matrix[:, w, u_idx, :], axis=-1)
        )
        net_flux_strings = [
            'Fnet error = {0:.1f} WW'.format(b) for b in
            predicted_net_fluxes_w_m02 - actual_net_fluxes_w_m02
        ]
        net_flux_strings = [
            s.replace('Fnet', r'$F_{net}$') for s in net_flux_strings
        ]

        flux_strings = [
            '{0:s}\n{1:s}\n{2:s}'.format(d, u, n) for d, u, n in
            zip(down_flux_strings, up_flux_strings, net_flux_strings)
        ]
        flux_strings = [
            s.replace('WW', r'W m$^{-2}$') for s in flux_strings
        ]

        if example_dir_name is None:
            annotation_strings = flux_strings
        else:
            annotation_strings = [
                '{0:s}\n{1:s}'.format(f, a) for f, a in
                zip(flux_strings, annotation_strings)
            ]

        if plot_shortwave:
            plot_fancy = all([
                t in vector_target_names for t in SHORTWAVE_VECTOR_TARGET_NAMES
            ])
        else:
            plot_fancy = all([
                t in vector_target_names for t in LONGWAVE_VECTOR_TARGET_NAMES
            ])

        if plot_fancy:
            _plot_comparisons_fancy(
                vector_target_matrix=vector_target_matrix,
                vector_prediction_matrix=vector_prediction_matrix,
                example_id_strings=prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
                model_metadata_dict=model_metadata_dict,
                use_log_scale=use_log_scale,
                plot_shortwave=plot_shortwave,
                wavelength_metres=this_wavelength_metres,
                output_dir_name=output_dir_name
            )
        else:
            _plot_comparisons_simple(
                vector_target_matrix=vector_target_matrix,
                vector_prediction_matrix=vector_prediction_matrix,
                example_id_strings=
                prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
                model_metadata_dict=model_metadata_dict,
                use_log_scale=use_log_scale,
                add_two_dummy_axes=add_two_dummy_axes,
                plot_shortwave=plot_shortwave,
                wavelength_metres=this_wavelength_metres,
                plot_uncertainty=plot_uncertainty,
                confidence_level=confidence_level,
                annotation_strings=annotation_strings,
                model_description_string=model_description_string,
                output_dir_name=output_dir_name
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_file_name=getattr(
            INPUT_ARG_OBJECT, PREDICTION_FILE_ARG_NAME
        ),
        plot_shortwave=bool(getattr(INPUT_ARG_OBJECT, PLOT_SHORTWAVE_ARG_NAME)),
        wavelengths_metres=numpy.array(
            getattr(INPUT_ARG_OBJECT, WAVELENGTHS_ARG_NAME), dtype=float
        ),
        plot_uncertainty=bool(
            getattr(INPUT_ARG_OBJECT, PLOT_UNCERTAINTY_ARG_NAME)
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        use_log_scale=bool(getattr(INPUT_ARG_OBJECT, USE_LOG_SCALE_ARG_NAME)),
        add_two_dummy_axes=bool(
            getattr(INPUT_ARG_OBJECT, ADD_DUMMY_AXES_ARG_NAME)
        ),
        example_dir_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_DIR_ARG_NAME),
        model_description_string=getattr(
            INPUT_ARG_OBJECT, MODEL_DESCRIPTION_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
