"""Subsets GFS file from Jebb by site."""

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

GRID_ROW_DIMENSION = 'grid_yt'
GRID_COLUMN_DIMENSION = 'grid_xt'
ROWCOL_DIMENSIONS = [GRID_ROW_DIMENSION, GRID_COLUMN_DIMENSION]

SITE_DIMENSION = 'site_index'

INPUT_FILE_ARG_NAME = 'input_file_name'
SITE_ROWS_ARG_NAME = 'site_rows'
SITE_COLUMNS_ARG_NAME = 'site_columns'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing GFS data for the whole grid at one init '
    'time and one forecast time.'
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
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
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


def _run(input_file_name, site_rows, site_columns, output_file_name):
    """Subsets GFS file from Jebb by site.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param site_rows: Same.
    :param site_columns: Same.
    :param output_file_name: Same.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)

    error_checking.assert_is_geq_numpy_array(site_rows, 0)
    error_checking.assert_is_geq_numpy_array(site_columns, 0)

    num_sites = len(site_rows)
    error_checking.assert_is_numpy_array(
        site_columns,
        exact_dimensions=numpy.array([num_sites], dtype=int)
    )

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    orig_table_xarray = xarray.open_dataset(input_file_name)

    num_grid_rows = len(orig_table_xarray.coords[GRID_COLUMN_DIMENSION].values)
    num_grid_columns = len(orig_table_xarray.coords[GRID_ROW_DIMENSION].values)
    error_checking.assert_is_less_than_numpy_array(site_rows, num_grid_rows)
    error_checking.assert_is_less_than_numpy_array(
        site_columns, num_grid_columns
    )

    orig_metadata_dict = orig_table_xarray.to_dict()['coords']
    new_metadata_dict = dict()

    for this_key in orig_metadata_dict:
        if this_key in ROWCOL_DIMENSIONS:
            continue

        new_metadata_dict[this_key] = (
            orig_metadata_dict[this_key]['data']
        )

    new_metadata_dict[SITE_DIMENSION] = numpy.linspace(
        0, num_sites - 1, num=num_sites, dtype=int
    )

    orig_data_dict = orig_table_xarray.to_dict()['data_vars']
    new_data_dict = dict()

    for this_key in orig_table_xarray.variables:
        these_dimensions = [
            d for d in orig_data_dict[this_key]['dims']
            if d not in ROWCOL_DIMENSIONS
        ]
        these_dimensions = tuple(these_dimensions + [SITE_DIMENSION])

        new_data_dict[this_key] = (
            these_dimensions,
            orig_data_dict[this_key]['data'][..., site_rows, site_columns]
        )

    new_table_xarray = xarray.Dataset(
        data_vars=new_data_dict, coords=new_metadata_dict,
        attrs=orig_table_xarray.attrs
    )

    print('Writing data to: "{0:s}"...'.format(output_file_name))
    new_table_xarray.to_netcdf(
        path=output_file_name, mode='w', format='NETCDF3_64BIT'
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        site_rows=numpy.array(
            getattr(INPUT_ARG_OBJECT, SITE_ROWS_ARG_NAME), dtype=int
        ),
        site_columns=numpy.array(
            getattr(INPUT_ARG_OBJECT, SITE_COLUMNS_ARG_NAME), dtype=int
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
