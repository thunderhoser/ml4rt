"""Splits predictions by spatial region."""

import copy
import argparse
import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import projections
from ml4rt.io import example_io
from ml4rt.io import prediction_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_FILE_ARG_NAME = 'input_prediction_file_name'
MIN_LATITUDE_ARG_NAME = 'min_latitude_deg'
MAX_LATITUDE_ARG_NAME = 'max_latitude_deg'
MIN_LONGITUDE_ARG_NAME = 'min_longitude_deg'
MAX_LONGITUDE_ARG_NAME = 'max_longitude_deg'
GRID_SPACING_ARG_NAME = 'grid_spacing_metres'
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
GRID_SPACING_HELP_SPACING = 'Spacing for equidistant grid.'
OUTPUT_DIR_HELP_STRING = (
    'Name of output directory.  Spatially split predictions will be written '
    'here by `prediction_io.write_file`.'
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
    '--' + GRID_SPACING_ARG_NAME, type=float, required=False, default=1e5,
    help=GRID_SPACING_HELP_SPACING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_file_name, min_latitude_deg, max_latitude_deg,
         min_longitude_deg, max_longitude_deg, grid_spacing_metres,
         output_dir_name):
    """Splits predictions by spatial region.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param min_latitude_deg: Same.
    :param max_latitude_deg: Same.
    :param min_longitude_deg: Same.
    :param max_longitude_deg: Same.
    :param grid_spacing_metres: Same.
    :param output_dir_name: Same.
    """

    # Create grid.
    equidistant_grid_dict = grids.create_equidistant_grid(
        min_latitude_deg=min_latitude_deg, max_latitude_deg=max_latitude_deg,
        min_longitude_deg=min_longitude_deg,
        max_longitude_deg=max_longitude_deg,
        x_spacing_metres=grid_spacing_metres,
        y_spacing_metres=grid_spacing_metres, azimuthal=False
    )

    grid_metafile_name = grids.find_equidistant_metafile(
        directory_name=output_dir_name, raise_error_if_missing=False
    )

    print('Writing metadata for equidistant grid to: "{0:s}"...'.format(
        grid_metafile_name
    ))
    grids.write_equidistant_metafile(
        grid_dict=equidistant_grid_dict, pickle_file_name=grid_metafile_name
    )

    grid_points_x_metres = equidistant_grid_dict[grids.X_COORDS_KEY]
    grid_points_y_metres = equidistant_grid_dict[grids.Y_COORDS_KEY]
    projection_object = equidistant_grid_dict[grids.PROJECTION_KEY]

    grid_edges_x_metres = numpy.append(
        grid_points_x_metres - 0.5 * grid_spacing_metres,
        grid_points_x_metres[-1] + 0.5 * grid_spacing_metres
    )
    grid_edges_y_metres = numpy.append(
        grid_points_y_metres - 0.5 * grid_spacing_metres,
        grid_points_y_metres[-1] + 0.5 * grid_spacing_metres
    )

    # Read data.
    print('Reading data from: "{0:s}"...'.format(input_file_name))
    prediction_dict = prediction_io.read_file(input_file_name)
    example_metadata_dict = example_io.parse_example_ids(
        prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    )

    example_latitudes_deg = example_metadata_dict[example_io.LATITUDES_KEY]
    example_longitudes_deg = example_metadata_dict[example_io.LONGITUDES_KEY]

    examples_x_metres, examples_y_metres = projections.project_latlng_to_xy(
        latitudes_deg=example_latitudes_deg,
        longitudes_deg=example_longitudes_deg,
        projection_object=projection_object
    )

    num_grid_rows = len(grid_points_y_metres)
    num_grid_columns = len(grid_points_x_metres)
    print(SEPARATOR_STRING)

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            these_indices = grids.find_events_in_grid_cell(
                event_x_coords_metres=examples_x_metres,
                event_y_coords_metres=examples_y_metres,
                grid_edge_x_coords_metres=grid_edges_x_metres,
                grid_edge_y_coords_metres=grid_edges_y_metres,
                row_index=i, column_index=j, verbose=False
            )

            this_prediction_dict = prediction_io.subset_by_index(
                prediction_dict=copy.deepcopy(prediction_dict),
                desired_indices=these_indices
            )

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
                example_id_strings=
                this_prediction_dict[prediction_io.EXAMPLE_IDS_KEY],
                model_file_name=
                this_prediction_dict[prediction_io.MODEL_FILE_KEY]
            )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        min_latitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LATITUDE_ARG_NAME),
        max_latitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LATITUDE_ARG_NAME),
        min_longitude_deg=getattr(INPUT_ARG_OBJECT, MIN_LONGITUDE_ARG_NAME),
        max_longitude_deg=getattr(INPUT_ARG_OBJECT, MAX_LONGITUDE_ARG_NAME),
        grid_spacing_metres=getattr(INPUT_ARG_OBJECT, GRID_SPACING_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
