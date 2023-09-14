"""Fits piecewise-linear model for uniformization."""

import os
import sys
import argparse
import numpy
import xarray
# import pwlf

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import example_io
import normalization
import example_utils
import piecewise_linear_fitting as pwlf

TOLERANCE = 1e-6

LINEAR_PIECE_DIM = 'linear_piece'
BREAK_POINT_DIM = 'break_point'

BREAK_POINT_KEY = 'break_point_physical_units'
SLOPE_KEY = 'slope'
INTERCEPT_KEY = 'intercept'

FIELD_KEY = 'field_name'
HEIGHT_KEY = 'height_m_agl'

INPUT_FILE_ARG_NAME = 'input_normalization_file_name'
FIELD_ARG_NAME = 'field_name'
HEIGHT_ARG_NAME = 'height_m_agl'
NUM_REFERENCE_VALUES_ARG_NAME = 'num_reference_values'
NUM_PIECES_ARG_NAME = 'num_linear_pieces'
MAX_ACCEPTABLE_ERROR_ARG_NAME = 'max_acceptable_error'
OUTPUT_FILE_ARG_NAME = 'output_model_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to normalization file, containing a large number of reference values '
    'for each field/height pair.  Will be read by `example_io.read_file`.'
)
FIELD_HELP_STRING = (
    'Will create piecewise-linear model to uniformize this field only.  The '
    'field name must be accepted by `example_utils.check_field_name`.'
)
HEIGHT_HELP_STRING = (
    'Will create piecewise-linear model to uniformize `{0:s}` only at this '
    'height (metres above ground).  If `{0:s}` is a scalar field (not a vector '
    'over height), leave this argument alone.'
).format(FIELD_ARG_NAME)

NUM_REFERENCE_VALUES_HELP_STRING = (
    'Number of reference values to use from the normalization file.'
)
NUM_PIECES_HELP_STRING = 'Number of pieces in piecewise-linear model.'
MAX_ACCEPTABLE_ERROR_HELP_STRING = 'Max acceptable model error.'
OUTPUT_FILE_HELP_STRING = (
    'Path to output (NetCDF) file.  The fitted model will be written here by '
    '`_write_model`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIELD_ARG_NAME, type=str, required=True, help=FIELD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HEIGHT_ARG_NAME, type=int, required=False, default=-1,
    help=HEIGHT_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_REFERENCE_VALUES_ARG_NAME, type=int, required=True,
    help=NUM_REFERENCE_VALUES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_PIECES_ARG_NAME, type=int, required=True,
    help=NUM_PIECES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_ACCEPTABLE_ERROR_ARG_NAME, type=float, required=True,
    help=MAX_ACCEPTABLE_ERROR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _apply_model_one_value_per_piece(
        model_break_points_physical, model_slopes, model_intercepts):
    """Applies trained model to one value per linear piece.

    P = number of linear pieces

    :param model_break_points_physical: length-(P + 1) numpy array of break
        points in physical units.
    :param model_slopes: length-P numpy array of slopes.
    :param model_intercepts: length-P numpy array of intercepts.
    :return: physical_values: length-(P + 2) numpy array of physical values.
    :return: estimated_normalized_values: length-(P + 2) numpy array of
        estimated normalized values.
    """

    physical_values = (
        model_break_points_physical[:-1] +
        0.5 * numpy.diff(model_break_points_physical)
    )

    # min_physical_value = (
    #     model_break_points_physical[0] -
    #     0.5 * numpy.diff(model_break_points_physical)[0]
    # )
    # max_physical_value = (
    #     model_break_points_physical[-1] +
    #     0.5 * numpy.diff(model_break_points_physical)[-1]
    # )
    # physical_values = numpy.concatenate((
    #     numpy.array([min_physical_value]),
    #     physical_values,
    #     numpy.array([max_physical_value])
    # ))

    num_linear_pieces = len(model_slopes)

    piece_indices = numpy.digitize(
        x=physical_values, bins=model_break_points_physical, right=False
    ) - 1
    piece_indices[piece_indices < 0] = 0
    piece_indices[piece_indices >= num_linear_pieces] = num_linear_pieces - 1

    num_examples = len(physical_values)
    estimated_normalized_values = numpy.full(num_examples, numpy.nan)

    for i in range(num_linear_pieces):
        example_indices = numpy.where(piece_indices == i)[0]
        estimated_normalized_values[example_indices] = (
            model_intercepts[i] +
            model_slopes[i] * physical_values[example_indices]
        )

    estimated_normalized_values[0] = max([
        estimated_normalized_values[0],
        0.
    ])
    estimated_normalized_values[-1] = min([
        estimated_normalized_values[-1],
        1.
    ])

    return physical_values, estimated_normalized_values


def _write_model(
        netcdf_file_name, model_break_points_physical, model_slopes,
        model_intercepts, field_name, height_m_agl):
    """Writes model to NetCDF file.

    P = number of linear pieces

    :param netcdf_file_name: Path to output file.
    :param model_break_points_physical: length-(P + 1) numpy array of break
        points in physical units.
    :param model_slopes: length-P numpy array of slopes.
    :param model_intercepts: length-P numpy array of intercepts.
    :param field_name: See documentation at top of script.
    :param height_m_agl: Same.
    """

    main_data_dict = {
        BREAK_POINT_KEY: ((BREAK_POINT_DIM,), model_break_points_physical),
        SLOPE_KEY: ((LINEAR_PIECE_DIM,), model_slopes),
        INTERCEPT_KEY: ((LINEAR_PIECE_DIM,), model_intercepts)
    }
    attribute_dict = {
        FIELD_KEY: field_name,
        HEIGHT_KEY: numpy.nan if height_m_agl is None else height_m_agl
    }

    model_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, attrs=attribute_dict
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    model_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def _run(normalization_file_name, field_name, height_m_agl,
         num_reference_values_to_use, num_linear_pieces, max_acceptable_error,
         output_file_name):
    """Fits piecewise-linear model for uniformization.

    This is effectively the main method.

    :param normalization_file_name: See documentation at top of file.
    :param field_name: Same.
    :param height_m_agl: Same.
    :param num_reference_values_to_use: Same.
    :param num_linear_pieces: Same.
    :param max_acceptable_error: Same.
    :param output_file_name: Same.
    """

    if height_m_agl < 0:
        height_m_agl = None

    error_checking.assert_is_geq(num_reference_values_to_use, 100)
    error_checking.assert_is_geq(num_linear_pieces, 10)
    error_checking.assert_is_greater(max_acceptable_error, 0.)
    error_checking.assert_is_leq(max_acceptable_error, 0.1)

    print('Reading reference values from: "{0:s}"...'.format(
        normalization_file_name
    ))
    reference_example_dict = example_io.read_file(normalization_file_name)

    physical_reference_values = example_utils.get_field_from_dict(
        example_dict=reference_example_dict,
        field_name=field_name, height_m_agl=height_m_agl
    )
    assert len(physical_reference_values.shape) == 1

    num_reference_values_total = len(physical_reference_values)
    num_reference_values_to_use = min([
        num_reference_values_to_use, num_reference_values_total
    ])
    take_every_nth_value = int(numpy.floor(
        float(num_reference_values_total) / num_reference_values_to_use
    ))

    physical_reference_values = numpy.sort(physical_reference_values)
    normalized_reference_values = normalization._orig_to_uniform_dist(
        orig_values_new=physical_reference_values + 0.,
        orig_values_training=physical_reference_values
    )

    max_physical_value = numpy.max(physical_reference_values)
    max_normalized_value = numpy.max(normalized_reference_values)

    percentile_levels = numpy.linspace(
        0, 100, num=num_linear_pieces + 1, dtype=float
    )
    percentile_levels[0] = 0.1
    percentile_levels[-1] = 99.9
    first_guess_break_points_physical = numpy.percentile(
        physical_reference_values, percentile_levels
    )

    print('Taking every {0:d}th of {1:d} reference values...'.format(
        take_every_nth_value, num_reference_values_total
    ))
    physical_reference_values = (
        physical_reference_values[::take_every_nth_value]
    )
    normalized_reference_values = (
        normalized_reference_values[::take_every_nth_value]
    )

    if not numpy.isclose(
            physical_reference_values[-1], max_physical_value, atol=TOLERANCE
    ):
        physical_reference_values = numpy.concatenate((
            physical_reference_values,
            numpy.array([max_physical_value])
        ))
        normalized_reference_values = numpy.concatenate((
            normalized_reference_values,
            numpy.array([max_normalized_value])
        ))

    print((
        'Fitting piecewise-linear model with {0:d} reference values...'
    ).format(
        len(physical_reference_values)
    ))

    model_object = pwlf.PiecewiseLinFit(
        physical_reference_values, normalized_reference_values
    )
    # model_break_points_physical = model_object.fitfast(
    #     n_segments=num_linear_pieces, pop=5
    # )
    # model_break_points_physical = model_object.fit(n_segments=num_linear_pieces)
    model_break_points_physical = model_object.fit_guess(
        guess_breakpoints=first_guess_break_points_physical
    )

    estimated_norm_reference_values = model_object.predict(
        physical_reference_values
    )
    absolute_errors = numpy.absolute(
        normalized_reference_values - estimated_norm_reference_values
    )

    if numpy.any(absolute_errors > max_acceptable_error):
        error_string = (
            '{0:d} of {1:d} predictions have an absolute error above {2:.4f}.  '
            'Absolute errors are sorted in descending order below:\n{3:s}'
        ).format(
            numpy.sum(absolute_errors > max_acceptable_error),
            len(absolute_errors),
            max_acceptable_error,
            str(numpy.sort(absolute_errors)[::-1])
        )

        raise ValueError(error_string)

    model_slopes = model_object.calc_slopes()
    model_intercepts = model_object.intercepts

    print('Writing model to: "{0:s}"...'.format(output_file_name))
    _write_model(
        netcdf_file_name=output_file_name,
        model_break_points_physical=model_break_points_physical,
        model_slopes=model_slopes,
        model_intercepts=model_intercepts,
        field_name=field_name,
        height_m_agl=height_m_agl
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        normalization_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        field_name=getattr(INPUT_ARG_OBJECT, FIELD_ARG_NAME),
        height_m_agl=getattr(INPUT_ARG_OBJECT, HEIGHT_ARG_NAME),
        num_reference_values_to_use=getattr(
            INPUT_ARG_OBJECT, NUM_REFERENCE_VALUES_ARG_NAME
        ),
        num_linear_pieces=getattr(INPUT_ARG_OBJECT, NUM_PIECES_ARG_NAME),
        max_acceptable_error=getattr(
            INPUT_ARG_OBJECT, MAX_ACCEPTABLE_ERROR_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
