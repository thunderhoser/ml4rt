"""Subsets GFS data from Jebb by site."""

import os
import sys
import argparse
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking

TOLERANCE = 1e-6

TIME_DIMENSION = 'time'
SITE_DIMENSION = 'site_index'
GRID_ROW_DIMENSION = 'grid_yt'
GRID_COLUMN_DIMENSION = 'grid_xt'
ROW_COL_TIME_DIMENSIONS = [
    GRID_ROW_DIMENSION, GRID_COLUMN_DIMENSION, TIME_DIMENSION
]

SURFACE_ALBEDO_KEY = 'albdo_ave'
FORECAST_HOUR_KEY = 'forecast_hour'

ATMOSPHERE_FILE_ARG_NAME = 'input_atmos_file_name'
SURFACE_FILE_ARG_NAME = 'input_surface_file_name'
SITE_ROWS_ARG_NAME = 'site_rows'
SITE_COLUMNS_ARG_NAME = 'site_columns'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ATMOSPHERE_FILE_HELP_STRING = (
    'Path to atmosphere file, containing GFS data for the whole 3-D atmosphere '
    'at one init time and one forecast time.'
)
SURFACE_FILE_HELP_STRING = (
    'Path to surface file, containing GFS data for only the surface at the same'
    ' init time and forecast time.'
)
SITE_ROWS_HELP_STRING = (
    'List of row indices (one per desired site).  Only data at these sites will'
    ' be written to the output file.'
)
SITE_COLUMNS_HELP_STRING = (
    'List of column indices (one per desired site).  Only data at these sites '
    'will be written to the output file.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Subset data will be written here in NetCDF format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + ATMOSPHERE_FILE_ARG_NAME, type=str, required=True,
    help=ATMOSPHERE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SURFACE_FILE_ARG_NAME, type=str, required=True,
    help=SURFACE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SITE_ROWS_ARG_NAME, type=int, nargs='+', required=True,
    help=SITE_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SITE_COLUMNS_ARG_NAME, type=int, nargs='+', required=True,
    help=SITE_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _file_name_to_forecast_hour(jebb_gfs_file_name):
    """Parses forecast hour from name of Jebb GFS file.

    :param jebb_gfs_file_name: Path to Jebb GFS file.
    :return: forecast_hour: Forecast hour (integer).
    """

    extensionless_file_name = os.path.splitext(jebb_gfs_file_name)[0]
    last_part = extensionless_file_name.split('.')[-1]
    assert last_part[-4] == 'f'

    return int(last_part[-3:])


def _run(atmosphere_file_name, surface_file_name, site_rows, site_columns,
         output_file_name):
    """Subsets GFS data from Jebb by site.

    This is effectively the main method.

    :param atmosphere_file_name: See documentation at top of file.
    :param surface_file_name: Same.
    :param site_rows: Same.
    :param site_columns: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    error_checking.assert_is_geq_numpy_array(site_rows, 0)
    error_checking.assert_is_geq_numpy_array(site_columns, 0)

    num_sites = len(site_rows)
    error_checking.assert_is_numpy_array(
        site_columns,
        exact_dimensions=numpy.array([num_sites], dtype=int)
    )

    forecast_hour = _file_name_to_forecast_hour(atmosphere_file_name)
    assert forecast_hour == _file_name_to_forecast_hour(surface_file_name)

    print('Reading data from: "{0:s}"...'.format(atmosphere_file_name))
    orig_table_xarray = xarray.open_dataset(atmosphere_file_name)

    num_grid_rows = len(orig_table_xarray.coords[GRID_COLUMN_DIMENSION].values)
    num_grid_columns = len(orig_table_xarray.coords[GRID_ROW_DIMENSION].values)
    error_checking.assert_is_less_than_numpy_array(site_rows, num_grid_rows)
    error_checking.assert_is_less_than_numpy_array(
        site_columns, num_grid_columns
    )

    print('Reading data from: "{0:s}"...'.format(surface_file_name))
    surface_table_xarray = xarray.open_dataset(surface_file_name)

    assert len(orig_table_xarray.coords[TIME_DIMENSION].values) == 1
    assert len(surface_table_xarray.coords[TIME_DIMENSION].values) == 1
    assert (
        orig_table_xarray.coords[TIME_DIMENSION].values[0] ==
        surface_table_xarray.coords[TIME_DIMENSION].values[0]
    )
    assert numpy.allclose(
        orig_table_xarray.coords[GRID_ROW_DIMENSION].values,
        surface_table_xarray.coords[GRID_ROW_DIMENSION].values,
        atol=TOLERANCE
    )
    assert numpy.allclose(
        orig_table_xarray.coords[GRID_COLUMN_DIMENSION].values,
        surface_table_xarray.coords[GRID_COLUMN_DIMENSION].values,
        atol=TOLERANCE
    )

    # Do actual stuff.
    print('\nCreating new metadata dictionary...')
    new_metadata_dict = dict()
    new_data_dict = dict()

    for this_key in orig_table_xarray.coords:
        if this_key in ROW_COL_TIME_DIMENSIONS:
            continue

        new_metadata_dict[this_key] = orig_table_xarray.coords[this_key].values

    new_metadata_dict[SITE_DIMENSION] = numpy.linspace(
        0, num_sites - 1, num=num_sites, dtype=int
    )

    for this_key in orig_table_xarray.variables:
        if this_key in orig_table_xarray.coords:
            continue

        print('Adding {0:s} to new data dictionary...'.format(this_key))

        orig_dimensions = orig_table_xarray[this_key].dims
        new_dimensions = [
            d for d in orig_dimensions if d not in ROW_COL_TIME_DIMENSIONS
        ]
        new_dimensions = tuple(new_dimensions + [SITE_DIMENSION])

        if TIME_DIMENSION in orig_dimensions:
            new_data_matrix = orig_table_xarray[this_key].values[0, ...][
                ..., site_rows, site_columns
            ]
        else:
            new_data_matrix = orig_table_xarray[this_key].values[
                ..., site_rows, site_columns
            ]

        print(new_data_matrix.shape)
        new_data_dict[this_key] = (new_dimensions, new_data_matrix)

    print('Adding {0:s} to new data dictionary...'.format(SURFACE_ALBEDO_KEY))

    orig_dimensions = surface_table_xarray[SURFACE_ALBEDO_KEY].dims
    new_dimensions = [
        d for d in orig_dimensions if d not in ROW_COL_TIME_DIMENSIONS
    ]
    new_dimensions = tuple(new_dimensions + [SITE_DIMENSION])

    new_data_dict[SURFACE_ALBEDO_KEY] = (
        new_dimensions,
        surface_table_xarray[SURFACE_ALBEDO_KEY].values[
            0, ..., site_rows, site_columns
        ]
    )

    orig_table_xarray.attrs[FORECAST_HOUR_KEY] = forecast_hour
    new_table_xarray = xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict,
        attrs=orig_table_xarray.attrs
    )

    print('\nWriting data to: "{0:s}"...'.format(output_file_name))
    new_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        atmosphere_file_name=getattr(
            INPUT_ARG_OBJECT, ATMOSPHERE_FILE_ARG_NAME
        ),
        surface_file_name=getattr(INPUT_ARG_OBJECT, SURFACE_FILE_ARG_NAME),
        site_rows=numpy.array(
            getattr(INPUT_ARG_OBJECT, SITE_ROWS_ARG_NAME), dtype=int
        ),
        site_columns=numpy.array(
            getattr(INPUT_ARG_OBJECT, SITE_COLUMNS_ARG_NAME), dtype=int
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
