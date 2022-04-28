"""Computes surface emissivity (two forms) from raw GFS data.

'Raw GFS data' means the files originally sent by Jebb.
"""

import glob
import argparse
import numpy
import xarray
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils

DATE_FORMAT = '%Y%m%d'

VALID_TIME_FORMAT_ORIG = '%Y-%m-%d %H:%M:%S'
VALID_TIME_DIMENSION_ORIG = 'time'

SURFACE_TEMPERATURE_KEY = 'tmpsfc'
SURFACE_LONGWAVE_UP_FLUX_KEY = 'ulwrf'
SURFACE_LONGWAVE_DOWN_FLUX_KEY = 'dlwrf'

EXAMPLE_DIMENSION_KEY = 'example'
ROW_DIMENSION_KEY = 'row'
COLUMN_DIMENSION_KEY = 'column'

INIT_TIMES_KEY = 'init_times_unix_sec'
VALID_TIMES_KEY = 'valid_times_unix_sec'
FULL_EMISSIVITIES_KEY = 'full_emissivities'
APPROX_EMISSIVITIES_KEY = 'approx_emissivities'

STEFAN_BOLTZMANN_CONSTANT_W_M02_K04 = 5.67e-8

INPUT_DIR_ARG_NAME = 'input_dir_name'
FIRST_DATE_ARG_NAME = 'first_date_string'
LAST_DATE_ARG_NAME = 'last_date_string'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Name of top-level input directory.  GFS surface files (one per init time '
    'per valid time) will be found here.'
)
DATE_HELP_STRING = (
    'Date (format "yyyymmdd").  This script will operate on GFS files with '
    'initialization times from `{0:s}`...`{1:s}`.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Results (two emissivity grids per init time per '
    'valid time) will be saved here in NetCDF format.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True, help=DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, first_date_string, last_date_string, output_file_name):
    """Computes surface emissivity (two forms) from raw GFS data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param first_date_string: Same.
    :param last_date_string: Same.
    :param output_file_name: Same.
    :raises: ValueError: if no GFS surface files are found
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=output_file_name)
    unique_date_strings = time_conversion.get_spc_dates_in_range(
        first_date_string, last_date_string
    )

    gfs_file_names = []
    gfs_date_strings = []

    for this_date_string in unique_date_strings:
        this_file_pattern = '{0:s}/{1:s}/gfs.{1:s}/00/gfs.t00z.sfcf*.nc'.format(
            input_dir_name, this_date_string
        )
        these_file_names = glob.glob(this_file_pattern)

        if len(these_file_names) == 0:
            continue

        these_file_names.sort()
        gfs_file_names += these_file_names
        gfs_date_strings += [this_date_string] * len(these_file_names)

    if len(gfs_file_names) == 0:
        error_string = (
            'Cannot find any GFS surface files from dates {0:s} to {1:s} in '
            'directory: "{2:s}"'
        ).format(first_date_string, last_date_string, input_dir_name)

        raise ValueError(error_string)

    valid_times_unix_sec = []
    init_times_unix_sec = []
    full_emissivity_matrices = []
    approx_emissivity_matrices = []

    for i in range(len(gfs_file_names)):
        print('Reading data from: "{0:s}"...'.format(gfs_file_names[i]))
        gfs_table_xarray = xarray.open_dataset(gfs_file_names[i])

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            gfs_date_strings[i], DATE_FORMAT
        )
        init_times_unix_sec.append(this_time_unix_sec)

        this_time_unix_sec = time_conversion.string_to_unix_sec(
            str(gfs_table_xarray.coords[VALID_TIME_DIMENSION_ORIG].values[0]),
            VALID_TIME_FORMAT_ORIG
        )
        valid_times_unix_sec.append(this_time_unix_sec)

        upwelling_flux_matrix_w_m02 = (
            gfs_table_xarray[SURFACE_LONGWAVE_UP_FLUX_KEY].values[0, ...]
        )
        downwelling_flux_matrix_w_m02 = (
            gfs_table_xarray[SURFACE_LONGWAVE_DOWN_FLUX_KEY].values[0, ...]
        )
        blackbody_up_flux_matrix_w_m02 = STEFAN_BOLTZMANN_CONSTANT_W_M02_K04 * (
            gfs_table_xarray[SURFACE_TEMPERATURE_KEY].values[0, ...]
        ) ** 4

        this_emissivity_matrix = (
            (upwelling_flux_matrix_w_m02 - downwelling_flux_matrix_w_m02) /
            (blackbody_up_flux_matrix_w_m02 - downwelling_flux_matrix_w_m02)
        )
        full_emissivity_matrices.append(this_emissivity_matrix + 0.)

        this_emissivity_matrix = (
            upwelling_flux_matrix_w_m02 / blackbody_up_flux_matrix_w_m02
        )
        approx_emissivity_matrices.append(this_emissivity_matrix + 0.)

    valid_times_unix_sec = numpy.array(valid_times_unix_sec, dtype=int)
    init_times_unix_sec = numpy.array(init_times_unix_sec, dtype=int)
    full_emissivity_matrix = numpy.stack(full_emissivity_matrices, axis=0)
    del full_emissivity_matrices
    approx_emissivity_matrix = numpy.stack(approx_emissivity_matrices, axis=0)
    del approx_emissivity_matrices

    dataset_object = netCDF4.Dataset(
        output_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(
        EXAMPLE_DIMENSION_KEY, full_emissivity_matrix.shape[0]
    )
    dataset_object.createDimension(
        ROW_DIMENSION_KEY, full_emissivity_matrix.shape[1]
    )
    dataset_object.createDimension(
        COLUMN_DIMENSION_KEY, full_emissivity_matrix.shape[2]
    )

    dataset_object.createVariable(
        INIT_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[INIT_TIMES_KEY][:] = init_times_unix_sec

    dataset_object.createVariable(
        VALID_TIMES_KEY, datatype=numpy.int32, dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[VALID_TIMES_KEY][:] = valid_times_unix_sec

    these_dim = (EXAMPLE_DIMENSION_KEY, ROW_DIMENSION_KEY, COLUMN_DIMENSION_KEY)
    dataset_object.createVariable(
        FULL_EMISSIVITIES_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[FULL_EMISSIVITIES_KEY][:] = full_emissivity_matrix

    dataset_object.createVariable(
        APPROX_EMISSIVITIES_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[APPROX_EMISSIVITIES_KEY][:] = (
        approx_emissivity_matrix
    )

    dataset_object.close()


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
