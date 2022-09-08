"""Subsets GFS data from Jebb by site."""

import os
import argparse
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6

TIME_FORMAT = '%Y-%m-%d %H:%M:%S'

TIME_DIMENSION_ORIG = 'time'
SITE_DIMENSION = 'site_index'
GRID_ROW_DIMENSION = 'grid_yt'
GRID_COLUMN_DIMENSION = 'grid_xt'
TIME_DIMENSION = 'valid_time_unix_sec'
ROWCOL_DIMENSIONS = [GRID_ROW_DIMENSION, GRID_COLUMN_DIMENSION]

SURFACE_ALBEDO_KEY = 'albdo_ave'
SURFACE_TEMPERATURE_KEY = 'tmpsfc'
SURFACE_PRESSURE_KEY = 'pressfc'
SURFACE_UP_LONGWAVE_FLUX_KEY = 'ulwrf'

ATMOSPHERE_FILES_ARG_NAME = 'input_atmos_file_names'
SURFACE_FILES_ARG_NAME = 'input_surface_file_names'
SITE_ROWS_ARG_NAME = 'site_rows'
SITE_COLUMNS_ARG_NAME = 'site_columns'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

ATMOSPHERE_FILES_HELP_STRING = (
    'List of paths to atmosphere files, each containing GFS data for the whole '
    '3-D atmosphere at one init time and one forecast time.'
)
SURFACE_FILES_HELP_STRING = (
    'List of paths to surface files, each containing GFS data for only the '
    'surface at one init time and one forecast time.  Must have same length as '
    '`{0:s}`.'
).format(ATMOSPHERE_FILES_ARG_NAME)
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
    '--' + ATMOSPHERE_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=ATMOSPHERE_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SURFACE_FILES_ARG_NAME, type=str, nargs='+', required=True,
    help=SURFACE_FILES_HELP_STRING
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


def _subset_gfs_one_forecast_hour(atmosphere_file_name, surface_file_name,
                                  site_rows, site_columns):
    """Subsets GFS data for one forecast hour.

    :param atmosphere_file_name: See documentation at top of file.
    :param surface_file_name: Same.
    :param site_rows: Same.
    :param site_columns: Same.
    :return: subset_gfs_table_xarray: xarray table with subset GFS data.
    """

    # Read files and check input args.
    print('Reading data from: "{0:s}"...'.format(atmosphere_file_name))
    orig_table_xarray = xarray.open_dataset(atmosphere_file_name)

    num_grid_rows = len(orig_table_xarray.coords[GRID_ROW_DIMENSION].values)
    num_grid_columns = len(
        orig_table_xarray.coords[GRID_COLUMN_DIMENSION].values
    )
    error_checking.assert_is_less_than_numpy_array(site_rows, num_grid_rows)
    error_checking.assert_is_less_than_numpy_array(
        site_columns, num_grid_columns
    )

    print('Reading data from: "{0:s}"...'.format(surface_file_name))
    surface_table_xarray = xarray.open_dataset(surface_file_name)

    assert len(orig_table_xarray.coords[TIME_DIMENSION_ORIG].values) == 1
    assert len(surface_table_xarray.coords[TIME_DIMENSION_ORIG].values) == 1
    assert (
        orig_table_xarray.coords[TIME_DIMENSION_ORIG].values[0] ==
        surface_table_xarray.coords[TIME_DIMENSION_ORIG].values[0]
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
        if this_key in ROWCOL_DIMENSIONS + [TIME_DIMENSION_ORIG]:
            continue

        new_metadata_dict[this_key] = orig_table_xarray.coords[this_key].values

    valid_time_unix_sec = time_conversion.string_to_unix_sec(
        str(orig_table_xarray.coords[TIME_DIMENSION_ORIG].values[0]),
        TIME_FORMAT
    )
    new_metadata_dict[TIME_DIMENSION] = numpy.array(
        [valid_time_unix_sec], dtype=int
    )

    num_sites = len(site_rows)
    new_metadata_dict[SITE_DIMENSION] = numpy.linspace(
        0, num_sites - 1, num=num_sites, dtype=int
    )

    for this_key in orig_table_xarray.variables:
        if this_key in orig_table_xarray.coords:
            continue

        print('Adding {0:s} to new data dictionary...'.format(this_key))

        orig_dimensions = orig_table_xarray[this_key].dims
        new_dimensions = [
            d for d in orig_dimensions if d not in ROWCOL_DIMENSIONS
        ]
        new_dimensions = [
            TIME_DIMENSION if d == TIME_DIMENSION_ORIG else d
            for d in new_dimensions
        ]
        new_dimensions = tuple(new_dimensions + [SITE_DIMENSION])

        new_data_matrix = orig_table_xarray[this_key].values[
            ..., site_rows, site_columns
        ]
        new_data_dict[this_key] = (new_dimensions, new_data_matrix)

    for this_key in [
            SURFACE_ALBEDO_KEY, SURFACE_TEMPERATURE_KEY,
            SURFACE_PRESSURE_KEY, SURFACE_UP_LONGWAVE_FLUX_KEY
    ]:
        print('Adding {0:s} to new data dictionary...'.format(this_key))

        orig_dimensions = surface_table_xarray[this_key].dims
        new_dimensions = [
            d for d in orig_dimensions if d not in ROWCOL_DIMENSIONS
        ]
        new_dimensions = [
            TIME_DIMENSION if d == TIME_DIMENSION_ORIG else d
            for d in new_dimensions
        ]
        new_dimensions = tuple(new_dimensions + [SITE_DIMENSION])

        new_data_dict[this_key] = (
            new_dimensions,
            surface_table_xarray[this_key].values[..., site_rows, site_columns]
        )

    return xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict,
        attrs=orig_table_xarray.attrs
    )


def _run(atmosphere_file_names, surface_file_names, site_rows, site_columns,
         output_file_name):
    """Subsets GFS data from Jebb by site.

    This is effectively the main method.

    :param atmosphere_file_names: See documentation at top of file.
    :param surface_file_names: Same.
    :param site_rows: Same.
    :param site_columns: Same.
    :param output_file_name: Same.
    """

    # Check input args.
    error_checking.assert_is_geq_numpy_array(site_rows, 0)
    error_checking.assert_is_geq_numpy_array(site_columns, 0)

    num_sites = len(site_rows)
    error_checking.assert_is_numpy_array(
        site_columns,
        exact_dimensions=numpy.array([num_sites], dtype=int)
    )

    num_forecast_hours = len(atmosphere_file_names)
    error_checking.assert_is_numpy_array(
        numpy.array(surface_file_names),
        exact_dimensions=numpy.array([num_forecast_hours], dtype=int)
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    forecast_hours = numpy.array(
        [_file_name_to_forecast_hour(f) for f in atmosphere_file_names],
        dtype=int
    )
    surface_forecast_hours = numpy.array(
        [_file_name_to_forecast_hour(f) for f in surface_file_names],
        dtype=int
    )
    assert numpy.array_equal(forecast_hours, surface_forecast_hours)
    del surface_forecast_hours

    sort_indices = numpy.argsort(forecast_hours)
    del forecast_hours
    atmosphere_file_names = [atmosphere_file_names[k] for k in sort_indices]
    surface_file_names = [surface_file_names[k] for k in sort_indices]

    all_tables_xarray = [None] * num_forecast_hours

    for i in range(num_forecast_hours):
        all_tables_xarray[i] = _subset_gfs_one_forecast_hour(
            atmosphere_file_name=atmosphere_file_names[i],
            surface_file_name=surface_file_names[i],
            site_rows=site_rows, site_columns=site_columns
        )
        print('\n')

    subset_gfs_table_xarray = xarray.concat(
        objs=all_tables_xarray, dim=TIME_DIMENSION, data_vars='minimal'
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    subset_gfs_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        atmosphere_file_names=getattr(
            INPUT_ARG_OBJECT, ATMOSPHERE_FILES_ARG_NAME
        ),
        surface_file_names=getattr(INPUT_ARG_OBJECT, SURFACE_FILES_ARG_NAME),
        site_rows=numpy.array(
            getattr(INPUT_ARG_OBJECT, SITE_ROWS_ARG_NAME), dtype=int
        ),
        site_columns=numpy.array(
            getattr(INPUT_ARG_OBJECT, SITE_COLUMNS_ARG_NAME), dtype=int
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
