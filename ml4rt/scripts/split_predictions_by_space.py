"""Splits predictions by spatial region."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import grids
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils
from ml4rt.utils import misc

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg'
LATITUDE_SPACING_ARG_NAME = 'latitude_spacing_deg'
LONGITUDE_SPACING_ARG_NAME = 'longitude_spacing_deg'
OUTPUT_DIR_ARG_NAME = 'output_prediction_dir_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing predictions for all spatial regions.  Will '
    'be read by `prediction_io.read_file`.'
)
MIN_LATITUDE_HELP_STRING = (
    'Minimum latitude (deg N) in equidistant grid.  To let min latitude be '
    'determined from data, leave this argument alone.'
)
MAX_LATITUDE_HELP_STRING = (
    'Max latitude (deg N) in equidistant grid.  To let max latitude be '
    'determined from data, leave this argument alone.'
)
MIN_LONGITUDE_HELP_STRING = (
    'Minimum longitude (deg E) in equidistant grid.  To let min longitude be '
    'determined from data, leave this argument alone.'
)
MAX_LONGITUDE_HELP_STRING = (
    'Max longitude (deg E) in equidistant grid.  To let max longitude be '
    'determined from data, leave this argument alone.'
)
LATITUDE_SPACING_HELP_SPACING = 'Meridional grid spacing (degrees).'
LONGITUDE_SPACING_HELP_SPACING = 'Zonal grid spacing (degrees).'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Spatially split predictions will be written '
    'here by `prediction_io.write_file`, and grid metadata will be written here'
    ' by `prediction_io.write_grid_metafile`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LATITUDE_ARG_NAME, type=float, required=False, default=numpy.nan,
    help=MIN_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LATITUDE_ARG_NAME, type=float, required=False, default=numpy.nan,
    help=MAX_LATITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MIN_LONGITUDE_ARG_NAME, type=float, required=False,
    default=numpy.nan, help=MIN_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_LONGITUDE_ARG_NAME, type=float, required=False,
    default=numpy.nan, help=MAX_LONGITUDE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LATITUDE_SPACING_ARG_NAME, type=float, required=False, default=1.,
    help=LATITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LONGITUDE_SPACING_ARG_NAME, type=float, required=False, default=1.,
    help=LONGITUDE_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, min_latitude_deg, max_latitude_deg, min_longitude_deg,
         max_longitude_deg, latitude_spacing_deg, longitude_spacing_deg,
         output_dir_name):
    """Splits predictions by spatial region.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param min_latitude_deg: Same.
    :param max_latitude_deg: Same.
    :param min_longitude_deg: Same.
    :param max_longitude_deg: Same.
    :param latitude_spacing_deg: Same.
    :param longitude_spacing_deg: Same.
    :param output_dir_name: Same.
    """

    # Read data.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)
    example_metadata_dict = example_utils.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )

    example_latitudes_deg = example_metadata_dict[example_utils.LATITUDES_KEY]
    example_longitudes_deg = example_metadata_dict[example_utils.LONGITUDES_KEY]

    these_limits_deg = numpy.array([
        min_latitude_deg, max_latitude_deg, min_longitude_deg, max_longitude_deg
    ])
    if numpy.any(numpy.isnan(these_limits_deg)):
        min_latitude_deg = numpy.min(example_latitudes_deg)
        max_latitude_deg = numpy.max(example_latitudes_deg)
        min_longitude_deg = numpy.min(example_longitudes_deg)
        max_longitude_deg = numpy.max(example_longitudes_deg)

    # Create grid.
    grid_point_latitudes_deg, grid_point_longitudes_deg = (
        misc.create_latlng_grid(
            min_latitude_deg=min_latitude_deg,
            max_latitude_deg=max_latitude_deg,
            latitude_spacing_deg=latitude_spacing_deg,
            min_longitude_deg=min_longitude_deg,
            max_longitude_deg=max_longitude_deg,
            longitude_spacing_deg=longitude_spacing_deg
        )
    )

    num_grid_rows = len(grid_point_latitudes_deg)
    num_grid_columns = len(grid_point_longitudes_deg)

    grid_edge_latitudes_deg, grid_edge_longitudes_deg = (
        grids.get_latlng_grid_cell_edges(
            min_latitude_deg=grid_point_latitudes_deg[0],
            min_longitude_deg=grid_point_longitudes_deg[0],
            lat_spacing_deg=numpy.diff(grid_point_latitudes_deg[:2])[0],
            lng_spacing_deg=numpy.diff(grid_point_longitudes_deg[:2])[0],
            num_rows=num_grid_rows, num_columns=num_grid_columns
        )
    )

    print(SEPARATOR_STRING)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            these_indices = grids.find_events_in_grid_cell(
                event_x_coords_metres=example_longitudes_deg,
                event_y_coords_metres=example_latitudes_deg,
                grid_edge_x_coords_metres=grid_edge_longitudes_deg,
                grid_edge_y_coords_metres=grid_edge_latitudes_deg,
                row_index=i, column_index=j, verbose=False
            )

            this_prediction_dict = prediction_io.subset_by_index(
                prediction_dict=copy.deepcopy(prediction_dict),
                desired_indices=these_indices
            )
            this_num_examples = len(
                this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
            )

            if this_num_examples == 0:
                continue

            this_output_file_name = prediction_io.find_file(
                directory_name=output_dir_name, grid_row=i, grid_column=j,
                raise_error_if_missing=False
            )
            print('Writing {0:d} examples to: "{1:s}"...'.format(
                len(this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]),
                this_output_file_name
            ))

            prediction_io.write_file(
                netcdf_file_name=this_output_file_name,
                scalar_target_matrix=
                this_prediction_dict[prediction_io.SCALAR_TARGETS_KEY],
                vector_target_matrix=
                this_prediction_dict[prediction_io.VECTOR_TARGETS_KEY],
                scalar_prediction_matrix=
                this_prediction_dict[prediction_io.SCALAR_PREDICTIONS_KEY],
                vector_prediction_matrix=
                this_prediction_dict[prediction_io.VECTOR_PREDICTIONS_KEY],
                heights_m_agl=this_prediction_dict[prediction_io.HEIGHTS_KEY],
                target_wavelengths_metres=
                this_prediction_dict[prediction_io.TARGET_WAVELENGTHS_KEY],
                example_id_strings=
                this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
                model_file_name=
                this_prediction_dict[prediction_io.MODEL_FILE_KEY],
                isotonic_model_file_name=
                this_prediction_dict[prediction_io.ISOTONIC_MODEL_FILE_KEY],
                uncertainty_calib_model_file_name=this_prediction_dict[
                    prediction_io.UNCERTAINTY_CALIB_MODEL_FILE_KEY
                ],
                normalization_file_name=
                this_prediction_dict[prediction_io.NORMALIZATION_FILE_KEY]
            )

    print(SEPARATOR_STRING)

    grid_metafile_name = prediction_io.find_grid_metafile(
        prediction_dir_name=output_dir_name, raise_error_if_missing=False
    )

    print('Writing grid metadata to: "{0:s}"...'.format(grid_metafile_name))
    prediction_io.write_grid_metafile(
        grid_point_latitudes_deg=grid_point_latitudes_deg,
        grid_point_longitudes_deg=grid_point_longitudes_deg,
        netcdf_file_name=grid_metafile_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        min_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        latitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LATITUDE_SPACING_ARG_NAME
        ),
        longitude_spacing_deg=getattr(
            INPUT_ARG_OBJECT, LONGITUDE_SPACING_ARG_NAME
        ),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
