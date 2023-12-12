"""Runs interpolation experiment."""

import argparse
import numpy
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from scipy.interpolate import interp1d
from gewittergefahr.gg_utils import geodetic_utils
from gewittergefahr.gg_utils import standard_atmosphere as standard_atmo
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import example_utils
from ml4rt.plotting import profile_plotting

TOLERANCE = 1e-6

LOW_RES_COLOUR = numpy.array([27, 158, 119], dtype=float) / 255
HIGH_RES_COLOUR = numpy.full(3, 0.)
FAKE_LOW_RES_COLOUR = numpy.array([217, 95, 2], dtype=float) / 255

LOW_RES_LINE_WIDTH = 6.
HIGH_RES_LINE_WIDTH = 1.
FAKE_LOW_RES_LINE_WIDTH = 3.

FIGURE_RESOLUTION_DPI = 300

EXAMPLE_FILE_ARG_NAME = 'input_example_file_name'
USE_SHORTWAVE_ARG_NAME = 'use_shortwave'
NUM_EXAMPLES_ARG_NAME = 'num_examples'
CHOOSE_MAX_HEATING_ARG_NAME = 'choose_max_heating_rate'
MAX_NOISE_ARG_NAME = 'max_noise_k_day01'
PRESSURE_CUTOFFS_ARG_NAME = 'pressure_cutoffs_pa'
PRESSURE_SPACINGS_ARG_NAME = 'pressure_spacings_pa'
FIRST_INTERP_METHOD_ARG_NAME = 'first_interp_method_name'
SECOND_INTERP_METHOD_ARG_NAME = 'second_interp_method_name'
INTERP_FLUXES_ARG_NAME = 'interp_fluxes'
OUTPUT_DIR_ARG_NAME = 'output_dir_name'

EXAMPLE_FILE_HELP_STRING = (
    'Path to example file.  Will be read by `example_io.read_file`.'
)
USE_SHORTWAVE_HELP_STRING = (
    'Boolean flag.  If 1 (0), will run experiment for shortwave (longwave) '
    'radiation.'
)
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples to experiment with.  Will pick the `{0:s}` examples '
    'with the largest column-max heating rate or absolute vertical gradient in '
    'heating rate.'
).format(NUM_EXAMPLES_ARG_NAME)

CHOOSE_MAX_HEATING_HELP_STRING = (
    'Boolean flag.  If 1, will pick the `{0:s}` examples with the largest '
    'column-max heating rate.  If 0, will pick the `{0:s}` examples with the '
    'largest absolute vertical gradient in heating rate.'
)
MAX_NOISE_HELP_STRING = (
    'Max noise to be added to high-resolution heating-rate profiles.  At each '
    'level in each profile, this script will add a random error ranging from '
    '-`{0:s}` to +`{0:s}`.'
).format(MAX_NOISE_ARG_NAME)

PRESSURE_CUTOFFS_HELP_STRING = (
    'List of pressure cutoffs (Pascals), indicating where different grid '
    'spacings will be used.'
)
PRESSURE_SPACINGS_HELP_STRING = (
    'List of grid spacings (Pascals).  The [i]th grid spacing in this list will'
    ' be used between the [i]th and [i + 1]th cutoffs in `{0:s}`.'
).format(PRESSURE_CUTOFFS_HELP_STRING)

FIRST_INTERP_METHOD_HELP_STRING = (
    'Method for first interpolation (the "kind" argument for '
    'scipy.interpolate.interp1d), from low-res grid to high-res grid.'
)
SECOND_INTERP_METHOD_HELP_STRING = (
    'Method for second interpolation (the "kind" argument for '
    'scipy.interpolate.interp1d), from high-res grid back to low-res grid.'
)
INTERP_FLUXES_HELP_STRING = (
    'Boolean flag.  If 1 (0), will interpolate fluxes and then convert to '
    'heating rates (interpolate heating rates directly).'
)
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Figures will be saved here.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + EXAMPLE_FILE_ARG_NAME, type=str, required=True,
    help=EXAMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_SHORTWAVE_ARG_NAME, type=int, required=True,
    help=USE_SHORTWAVE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=False, default=100,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CHOOSE_MAX_HEATING_ARG_NAME, type=int, required=False, default=1,
    help=CHOOSE_MAX_HEATING_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_NOISE_ARG_NAME, type=float, required=True,
    help=MAX_NOISE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_CUTOFFS_ARG_NAME, type=float, nargs='+', required=False,
    default=[0, 2500, 110000], help=PRESSURE_CUTOFFS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PRESSURE_SPACINGS_ARG_NAME, type=float, nargs='+', required=False,
    default=[10, 100], help=PRESSURE_SPACINGS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INTERP_METHOD_ARG_NAME, type=str, required=False,
    default='cubic', help=FIRST_INTERP_METHOD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SECOND_INTERP_METHOD_ARG_NAME, type=str, required=False,
    default='cubic', help=SECOND_INTERP_METHOD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INTERP_FLUXES_ARG_NAME, type=int, required=False, default=0,
    help=INTERP_FLUXES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _fluxes_to_heating_rate(down_fluxes_w_m02, up_fluxes_w_m02, pressures_pa):
    """Converts upwelling and downwelling fluxes to heating rate at each level.

    This is a light wrapper for `example_utils.fluxes_to_heating_rate`.

    L = number of levels in grid

    :param down_fluxes_w_m02: length-L numpy array of downwelling fluxes (Watts
        per square metre).
    :param up_fluxes_w_m02 : length-L numpy array of upwelling fluxes (Watts
        per square metre).
    :param pressures_pa: length-L numpy array of pressures (Pascals).
    :return: heating_rates_k_day01: length-L numpy array of heating rates
        (Kelvins per day).
    """

    target_matrix = numpy.vstack((down_fluxes_w_m02, up_fluxes_w_m02))
    target_matrix = numpy.expand_dims(numpy.transpose(target_matrix), axis=0)
    predictor_matrix = numpy.expand_dims(pressures_pa, axis=-1)
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)

    dummy_example_dict = {
        example_utils.VECTOR_TARGET_NAMES_KEY: [
            example_utils.SHORTWAVE_DOWN_FLUX_NAME,
            example_utils.SHORTWAVE_UP_FLUX_NAME
        ],
        example_utils.VECTOR_TARGET_VALS_KEY: target_matrix,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY: [example_utils.PRESSURE_NAME],
        example_utils.VECTOR_PREDICTOR_VALS_KEY: predictor_matrix
    }
    dummy_example_dict = example_utils.fluxes_to_heating_rate(
        dummy_example_dict
    )

    return example_utils.get_field_from_dict(
        example_dict=dummy_example_dict,
        field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
    )[0, :]


def _run_experiment_one_example(
        example_dict, example_index, max_noise_k_day01,
        high_res_pressures_pa, high_res_heights_m_asl, use_shortwave,
        first_interp_method_name, second_interp_method_name,
        interp_fluxes, output_dir_name):
    """Runs interpolation experiment for one example (one profile).

    H = number of levels in high-resolution grid

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param example_index: Will run experiment for [i]th example, where
        i = `example_index`.
    :param max_noise_k_day01: See documentation at top of file.
    :param high_res_pressures_pa: length-H numpy array of pressures (Pascals) in
        high-resolution grid.
    :param high_res_heights_m_asl: length-H numpy array of heights (metres above
        sea level) in high-resolution grid.
    :param use_shortwave: See documentation at top of file.
    :param first_interp_method_name: Same.
    :param second_interp_method_name: Same.
    :param interp_fluxes: Same.
    :param output_dir_name: Same.
    :return: max_difference_k_day01: Column-max difference between
        low-resolution and fake low-resolution heating rates.
    """

    example_id_string = (
        example_dict[example_utils.EXAMPLE_IDS_KEY][example_index]
    )
    metadata_dict = example_utils.parse_example_ids([example_id_string])
    surface_height_m_asl = geodetic_utils._get_elevation(
        latitude_deg=metadata_dict[example_utils.LATITUDES_KEY][0],
        longitude_deg=metadata_dict[example_utils.LONGITUDES_KEY][0]
    )[0]

    low_res_heights_m_agl = example_dict[example_utils.HEIGHTS_KEY]
    low_res_heights_m_asl = surface_height_m_asl + low_res_heights_m_agl
    low_res_pressures_pa = standard_atmo.height_to_pressure(
        low_res_heights_m_asl
    )
    low_res_heating_rates_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=
        example_utils.SHORTWAVE_HEATING_RATE_NAME if use_shortwave
        else example_utils.LONGWAVE_HEATING_RATE_NAME
    )[example_index, :]

    if interp_fluxes:
        low_res_down_fluxes_w_m02 = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=
            example_utils.SHORTWAVE_DOWN_FLUX_NAME if use_shortwave
            else example_utils.LONGWAVE_DOWN_FLUX_NAME
        )[example_index, :]

        interp_object = interp1d(
            x=low_res_pressures_pa[::-1], y=low_res_down_fluxes_w_m02[::-1],
            kind=first_interp_method_name, bounds_error=False,
            fill_value='extrapolate', assume_sorted=True
        )
        high_res_down_fluxes_w_m02 = interp_object(high_res_pressures_pa)

        low_res_up_fluxes_w_m02 = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=
            example_utils.SHORTWAVE_UP_FLUX_NAME if use_shortwave
            else example_utils.LONGWAVE_UP_FLUX_NAME
        )[example_index, :]

        interp_object = interp1d(
            x=low_res_pressures_pa[::-1], y=low_res_up_fluxes_w_m02[::-1],
            kind=first_interp_method_name, bounds_error=False,
            fill_value='extrapolate', assume_sorted=True
        )
        high_res_up_fluxes_w_m02 = interp_object(high_res_pressures_pa)

        high_res_heating_rates_k_day01 = _fluxes_to_heating_rate(
            down_fluxes_w_m02=high_res_down_fluxes_w_m02,
            up_fluxes_w_m02=high_res_up_fluxes_w_m02,
            pressures_pa=high_res_pressures_pa
        )

        interp_object = interp1d(
            x=high_res_pressures_pa[::-1],
            y=high_res_down_fluxes_w_m02[::-1],
            kind=second_interp_method_name, bounds_error=True,
            assume_sorted=True
        )
        fake_low_res_down_fluxes_w_m02 = interp_object(low_res_pressures_pa)

        interp_object = interp1d(
            x=high_res_pressures_pa[::-1],
            y=high_res_up_fluxes_w_m02[::-1],
            kind=second_interp_method_name, bounds_error=True,
            assume_sorted=True
        )
        fake_low_res_up_fluxes_w_m02 = interp_object(low_res_pressures_pa)

        fake_low_res_heating_rates_k_day01 = _fluxes_to_heating_rate(
            down_fluxes_w_m02=fake_low_res_down_fluxes_w_m02,
            up_fluxes_w_m02=fake_low_res_up_fluxes_w_m02,
            pressures_pa=low_res_pressures_pa
        )
    else:
        interp_object = interp1d(
            x=low_res_pressures_pa[::-1], y=low_res_heating_rates_k_day01[::-1],
            kind=first_interp_method_name, bounds_error=False,
            fill_value='extrapolate', assume_sorted=True
        )
        high_res_heating_rates_k_day01 = interp_object(high_res_pressures_pa)

        if max_noise_k_day01 > TOLERANCE:
            noise_values_k_day01 = numpy.random.uniform(
                low=-max_noise_k_day01, high=max_noise_k_day01,
                size=high_res_heating_rates_k_day01.shape
            )
            noise_values_k_day01 *= (
                max_noise_k_day01 /
                numpy.max(numpy.absolute(noise_values_k_day01))
            )
            high_res_heating_rates_k_day01 += noise_values_k_day01

        interp_object = interp1d(
            x=high_res_pressures_pa[::-1],
            y=high_res_heating_rates_k_day01[::-1],
            kind=second_interp_method_name, bounds_error=True,
            assume_sorted=True
        )
        fake_low_res_heating_rates_k_day01 = interp_object(low_res_pressures_pa)

    high_res_heights_m_agl = high_res_heights_m_asl - surface_height_m_asl

    figure_object, axes_object = profile_plotting.plot_one_variable(
        values=low_res_heating_rates_k_day01,
        heights_m_agl=low_res_heights_m_agl, use_log_scale=True,
        line_colour=LOW_RES_COLOUR, line_width=LOW_RES_LINE_WIDTH
    )
    profile_plotting.plot_one_variable(
        values=high_res_heating_rates_k_day01,
        heights_m_agl=high_res_heights_m_agl, use_log_scale=True,
        line_colour=HIGH_RES_COLOUR, line_width=HIGH_RES_LINE_WIDTH,
        figure_object=figure_object
    )
    profile_plotting.plot_one_variable(
        values=fake_low_res_heating_rates_k_day01,
        heights_m_agl=low_res_heights_m_agl, use_log_scale=True,
        line_colour=FAKE_LOW_RES_COLOUR, line_width=FAKE_LOW_RES_LINE_WIDTH,
        figure_object=figure_object
    )

    axes_object.set_xlim(left=-0.5)
    y_max = axes_object.get_ylim()[1]
    axes_object.set_ylim(top=y_max * 1.05)

    max_difference_k_day01 = numpy.max(numpy.absolute(
        low_res_heating_rates_k_day01 - fake_low_res_heating_rates_k_day01
    ))
    title_string = (
        'Max diff between low-res and reconstructed low-res = {0:.4f} K day'
    ).format(max_difference_k_day01)

    title_string = title_string + r'$^{-1}$'
    axes_object.set_title(title_string, fontsize=20)

    output_file_name = '{0:s}/{1:s}.jpg'.format(
        output_dir_name, example_id_string
    )
    print('Saving figure to: "{0:s}"...'.format(output_file_name))
    figure_object.savefig(
        output_file_name, dpi=FIGURE_RESOLUTION_DPI,
        pad_inches=0, bbox_inches='tight'
    )
    pyplot.close(figure_object)

    return max_difference_k_day01


def _run(example_file_name, use_shortwave, num_examples,
         choose_max_heating_rate, max_noise_k_day01, pressure_cutoffs_pa,
         pressure_spacings_pa, first_interp_method_name,
         second_interp_method_name, interp_fluxes, output_dir_name):
    """Runs interpolation experiment.

    This is effectively the main method.

    :param example_file_name: See documentation at top of file.
    :param use_shortwave: Same.
    :param num_examples: Same.
    :param choose_max_heating_rate: Same.
    :param max_noise_k_day01: Same.
    :param pressure_cutoffs_pa: Same.
    :param pressure_spacings_pa: Same.
    :param first_interp_method_name: Same.
    :param second_interp_method_name: Same.
    :param interp_fluxes: Same.
    :param output_dir_name: Same.
    """

    if interp_fluxes:
        max_noise_k_day01 = 0.

    error_checking.assert_is_greater(num_examples, 0)
    error_checking.assert_is_geq(max_noise_k_day01, 0.)

    error_checking.assert_is_geq_numpy_array(pressure_cutoffs_pa, 0.)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(pressure_cutoffs_pa), 0.
    )
    error_checking.assert_is_greater_numpy_array(pressure_spacings_pa, 0.)

    num_spacings = len(pressure_spacings_pa)
    expected_dim = numpy.array([num_spacings + 1], dtype=int)
    error_checking.assert_is_numpy_array(
        pressure_cutoffs_pa, exact_dimensions=expected_dim
    )

    high_res_pressures_pa = numpy.array([], dtype=float)

    for i in range(num_spacings):
        this_num_pressures = int(numpy.ceil(
            1 + (pressure_cutoffs_pa[i + 1] - pressure_cutoffs_pa[i]) /
            pressure_spacings_pa[i]
        ))
        these_pressures_pa = numpy.linspace(
            pressure_cutoffs_pa[i], pressure_cutoffs_pa[i + 1],
            num=this_num_pressures, dtype=float
        )

        if i != num_spacings - 1:
            these_pressures_pa = these_pressures_pa[:-1]

        high_res_pressures_pa = numpy.concatenate((
            high_res_pressures_pa, these_pressures_pa
        ))

    print('Number of levels in high-resolution grid = {0:d}'.format(
        len(high_res_pressures_pa)
    ))

    if high_res_pressures_pa[0] < TOLERANCE:
        high_res_pressures_pa[0] = 0.5 * high_res_pressures_pa[1]

    high_res_pressures_pa = high_res_pressures_pa[::-1]
    high_res_heights_m_asl = standard_atmo.pressure_to_height(
        high_res_pressures_pa
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    print('Reading data from: "{0:s}"...'.format(example_file_name))
    example_dict = example_io.read_file(example_file_name)

    heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=
        example_utils.SHORTWAVE_HEATING_RATE_NAME if use_shortwave
        else example_utils.LONGWAVE_HEATING_RATE_NAME
    )

    if choose_max_heating_rate:
        hr_criterion_by_example = numpy.max(heating_rate_matrix_k_day01, axis=1)
    else:
        abs_diff_matrix = numpy.absolute(
            numpy.diff(heating_rate_matrix_k_day01[:, :-1], axis=1)
        )
        hr_criterion_by_example = numpy.max(abs_diff_matrix, axis=1)

    good_indices = numpy.argsort(-1 * hr_criterion_by_example)
    good_indices = good_indices[:num_examples]
    example_dict = example_utils.subset_by_index(
        example_dict=example_dict, desired_indices=good_indices
    )

    num_examples = len(good_indices)
    max_differences_k_day01 = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        max_differences_k_day01[i] = _run_experiment_one_example(
            example_dict=example_dict, example_index=i,
            max_noise_k_day01=max_noise_k_day01,
            high_res_pressures_pa=high_res_pressures_pa,
            high_res_heights_m_asl=high_res_heights_m_asl,
            first_interp_method_name=first_interp_method_name,
            second_interp_method_name=second_interp_method_name,
            interp_fluxes=interp_fluxes, output_dir_name=output_dir_name
        )

    print('Average max difference = {0:.4f} K day^-1'.format(
        numpy.mean(max_differences_k_day01)
    ))
    print('Median max difference = {0:.4f} K day^-1'.format(
        numpy.median(max_differences_k_day01)
    ))
    print('Max max difference = {0:.4f} K day^-1'.format(
        numpy.max(max_differences_k_day01)
    ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        example_file_name=getattr(INPUT_ARG_OBJECT, EXAMPLE_FILE_ARG_NAME),
        use_shortwave=bool(getattr(INPUT_ARG_OBJECT, USE_SHORTWAVE_ARG_NAME)),
        num_examples=getattr(INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME),
        choose_max_heating_rate=bool(getattr(
            INPUT_ARG_OBJECT, CHOOSE_MAX_HEATING_ARG_NAME
        )),
        max_noise_k_day01=getattr(INPUT_ARG_OBJECT, MAX_NOISE_ARG_NAME),
        pressure_cutoffs_pa=numpy.array(
            getattr(INPUT_ARG_OBJECT, PRESSURE_CUTOFFS_ARG_NAME), dtype=float
        ),
        pressure_spacings_pa=numpy.array(
            getattr(INPUT_ARG_OBJECT, PRESSURE_SPACINGS_ARG_NAME), dtype=float
        ),
        first_interp_method_name=getattr(
            INPUT_ARG_OBJECT, FIRST_INTERP_METHOD_ARG_NAME
        ),
        second_interp_method_name=getattr(
            INPUT_ARG_OBJECT, SECOND_INTERP_METHOD_ARG_NAME
        ),
        interp_fluxes=bool(getattr(INPUT_ARG_OBJECT, INTERP_FLUXES_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
