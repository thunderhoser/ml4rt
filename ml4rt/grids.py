"""Processing methods for gridded data.

DEFINITIONS

"Grid point" = center of grid cell (as opposed to edges of the grid cell).
"""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import longitude_conversion as lng_conversion
import error_checking

TOLERANCE = 1e-6


def get_xy_grid_points(x_min_metres=None, y_min_metres=None,
                       x_spacing_metres=None, y_spacing_metres=None,
                       num_rows=None, num_columns=None):
    """Generates unique x- and y-coords of grid points in regular x-y grid.

    M = number of rows in grid
    N = number of columns in grid

    :param x_min_metres: Minimum x-coordinate over all grid points.
    :param y_min_metres: Minimum y-coordinate over all grid points.
    :param x_spacing_metres: Spacing between adjacent grid points in x-
        direction.  Alternate interpretation: length of each grid cell in x-
        direction.
    :param y_spacing_metres: Spacing between adjacent grid points in y-
        direction.  Alternate interpretation: length of each grid cell in y-
        direction.
    :param num_rows: Number of rows (unique grid-point y-values) in grid.
    :param num_columns: Number of columns (unique grid-point x-values) in grid.
    :return: grid_point_x_metres: length-N numpy array with x-coordinates of
        grid points.
    :return: grid_point_y_metres: length-M numpy array with y-coordinates of
        grid points.
    """

    error_checking.assert_is_not_nan(x_min_metres)
    error_checking.assert_is_not_nan(y_min_metres)
    error_checking.assert_is_greater(x_spacing_metres, 0.)
    error_checking.assert_is_greater(y_spacing_metres, 0.)
    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_greater(num_rows, 0)
    error_checking.assert_is_integer(num_columns)
    error_checking.assert_is_greater(num_columns, 0)

    x_max_metres = x_min_metres + (num_columns - 1) * x_spacing_metres
    y_max_metres = y_min_metres + (num_rows - 1) * y_spacing_metres

    grid_point_x_metres = numpy.linspace(x_min_metres, x_max_metres,
                                         num=num_columns)
    grid_point_y_metres = numpy.linspace(y_min_metres, y_max_metres,
                                         num=num_rows)

    return grid_point_x_metres, grid_point_y_metres


def get_latlng_grid_points(min_latitude_deg=None, min_longitude_deg=None,
                           lat_spacing_deg=None, lng_spacing_deg=None,
                           num_rows=None, num_columns=None):
    """Generates unique lat and long of grid points in regular lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg: Minimum latitude over all grid points (deg N).
    :param min_longitude_deg: Minimum longitude over all grid points (deg E).
    :param lat_spacing_deg: Meridional spacing between adjacent grid points.
        Alternate interpretation: length of each grid cell in N-S direction.
    :param lng_spacing_deg: Zonal spacing between adjacent grid points.
        Alternate interpretation: length of each grid cell in E-W direction.
    :param num_rows: Number of rows (unique grid-point latitudes) in grid.
    :param num_columns: Number of columns (unique grid-point longitudes) in
        grid.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes of
        grid points (deg N).
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes of
        grid points (deg E).
    """

    error_checking.assert_is_valid_latitude(min_latitude_deg)
    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg, allow_nan=False)
    error_checking.assert_is_greater(lat_spacing_deg, 0.)
    error_checking.assert_is_greater(lng_spacing_deg, 0.)
    error_checking.assert_is_integer(num_rows)
    error_checking.assert_is_greater(num_rows, 0)
    error_checking.assert_is_integer(num_columns)
    error_checking.assert_is_greater(num_columns, 0)

    max_latitude_deg = min_latitude_deg + (num_rows - 1) * lat_spacing_deg
    max_longitude_deg = min_longitude_deg + (num_columns - 1) * lng_spacing_deg

    grid_point_latitudes_deg = numpy.linspace(min_latitude_deg,
                                              max_latitude_deg, num=num_rows)
    grid_point_longitudes_deg = numpy.linspace(min_longitude_deg,
                                               max_longitude_deg,
                                               num=num_columns)

    return grid_point_latitudes_deg, grid_point_longitudes_deg


def get_xy_grid_cell_edges(x_min_metres=None, y_min_metres=None,
                           x_spacing_metres=None, y_spacing_metres=None,
                           num_rows=None, num_columns=None):
    """Generates unique x- and y-coords of grid-cell edges in regular x-y grid.

    M = number of rows in grid
    N = number of columns in grid

    :param x_min_metres: See documentation for get_xy_grid_points.
    :param y_min_metres: See documentation for get_xy_grid_points.
    :param x_spacing_metres: See documentation for get_xy_grid_points.
    :param y_spacing_metres: See documentation for get_xy_grid_points.
    :param num_rows: See documentation for get_xy_grid_points.
    :param num_columns: See documentation for get_xy_grid_points.
    :return: grid_cell_edge_x_metres: length-(N + 1) numpy array with x-
        coordinates of grid points.
    :return: grid_cell_edge_y_metres: length-(M + 1) numpy array with y-
        coordinates of grid points.
    """

    grid_point_x_metres, grid_point_y_metres = get_xy_grid_points(
        x_min_metres=x_min_metres, y_min_metres=y_min_metres,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_rows, num_columns=num_columns)

    grid_cell_edge_x_metres = numpy.concatenate((
        grid_point_x_metres - x_spacing_metres / 2,
        grid_point_x_metres[[-1]] + x_spacing_metres / 2))
    grid_cell_edge_y_metres = numpy.concatenate((
        grid_point_y_metres - y_spacing_metres / 2,
        grid_point_y_metres[[-1]] + y_spacing_metres / 2))

    return grid_cell_edge_x_metres, grid_cell_edge_y_metres


def get_latlng_grid_cell_edges(min_latitude_deg=None, min_longitude_deg=None,
                               lat_spacing_deg=None, lng_spacing_deg=None,
                               num_rows=None, num_columns=None):
    """Generates unique lat and lng of grid-cell edges in regular lat-lng grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg: See documentation for get_latlng_grid_points.
    :param min_longitude_deg: See documentation for get_latlng_grid_points.
    :param lat_spacing_deg: See documentation for get_latlng_grid_points.
    :param lng_spacing_deg: See documentation for get_latlng_grid_points.
    :param num_rows: See documentation for get_latlng_grid_points.
    :param num_columns: See documentation for get_latlng_grid_points.
    :return: grid_cell_edge_latitudes_deg: length-(M + 1) numpy array with
        latitudes of grid-cell edges (deg N).
    :return: grid_cell_edge_longitudes_deg: length-(N + 1) numpy array with
        longitudes of grid-cell edges (deg E).
    """

    (grid_point_latitudes_deg,
     grid_point_longitudes_deg) = get_latlng_grid_points(
         min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
         lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg,
         num_rows=num_rows, num_columns=num_columns)

    grid_cell_edge_latitudes_deg = numpy.concatenate((
        grid_point_latitudes_deg - lat_spacing_deg / 2,
        grid_point_latitudes_deg[[-1]] + lat_spacing_deg / 2))
    grid_cell_edge_longitudes_deg = numpy.concatenate((
        grid_point_longitudes_deg - lng_spacing_deg / 2,
        grid_point_longitudes_deg[[-1]] + lng_spacing_deg / 2))

    return grid_cell_edge_latitudes_deg, grid_cell_edge_longitudes_deg


def xy_vectors_to_matrices(x_unique_metres, y_unique_metres):
    """For regular x-y grid, converts vectors of x- and y-coords to matrices.

    This method works for coordinates of either grid points or grid-cell edges.

    M = number of rows in grid
    N = number of columns in grid

    :param x_unique_metres: length-N numpy array with x-coordinates of either
        grid points or grid-cell edges.
    :param y_unique_metres: length-M numpy array with y-coordinates of either
        grid points or grid-cell edges.
    :return: x_matrix_metres: M-by-N numpy array, where x_matrix_metres[*, j] =
        x_unique_metres[j].  Each row in this matrix is the same.
    :return: y_matrix_metres: M-by-N numpy array, where y_matrix_metres[i, *] =
        y_unique_metres[i].  Each column in this matrix is the same.
    """

    error_checking.assert_is_numpy_array_without_nan(x_unique_metres)
    error_checking.assert_is_numpy_array(x_unique_metres, num_dimensions=1)
    error_checking.assert_is_numpy_array_without_nan(y_unique_metres)
    error_checking.assert_is_numpy_array(y_unique_metres, num_dimensions=1)

    return numpy.meshgrid(x_unique_metres, y_unique_metres)


def latlng_vectors_to_matrices(unique_latitudes_deg, unique_longitudes_deg):
    """Converts vectors of lat and long coordinates to matrices.

    This method works only for a regular lat-long grid.  Works for coordinates
    of either grid points or grid-cell edges.

    M = number of rows in grid
    N = number of columns in grid

    :param unique_latitudes_deg: length-M numpy array with latitudes (deg N) of
        either grid points or grid-cell edges.
    :param unique_longitudes_deg: length-N numpy array with longitudes (deg E)
        of either grid points or grid-cell edges.
    :return: latitude_matrix_deg: M-by-N numpy array, where
        latitude_matrix_deg[i, *] = unique_latitudes_deg[i].  Each column in
        this matrix is the same.
    :return: longitude_matrix_deg: M-by-N numpy array, where
        longitude_matrix_deg[*, j] = unique_longitudes_deg[j].  Each row in this
        matrix is the same.
    """

    error_checking.assert_is_valid_lat_numpy_array(unique_latitudes_deg)
    error_checking.assert_is_numpy_array(unique_latitudes_deg, num_dimensions=1)
    error_checking.assert_is_numpy_array(unique_longitudes_deg,
                                         num_dimensions=1)
    unique_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        unique_longitudes_deg, allow_nan=False)

    (longitude_matrix_deg, latitude_matrix_deg) = numpy.meshgrid(
        unique_longitudes_deg, unique_latitudes_deg)
    return latitude_matrix_deg, longitude_matrix_deg


def xy_field_grid_points_to_edges(field_matrix=None, x_min_metres=None,
                                  y_min_metres=None, x_spacing_metres=None,
                                  y_spacing_metres=None):
    """Re-references x-y field from grid points to edges.

    M = number of rows (unique grid-point x-coordinates)
    N = number of columns (unique grid-point y-coordinates)

    :param field_matrix: M-by-N numpy array with values of some variable
        (examples: temperature, radar reflectivity, etc.).  y should increase
        while traveling down a column, and x should increase while traveling
        right across a row.
    :param x_min_metres: Minimum x-coordinate over all grid points.
    :param y_min_metres: Minimum y-coordinate over all grid points.
    :param x_spacing_metres: Spacing between adjacent grid points in x-
        direction.
    :param y_spacing_metres: Spacing between adjacent grid points in y-
        direction.
    :return: field_matrix: Same as input, except that dimensions are now (M + 1)
        by (N + 1).  The last row and last column contain only NaN's.
    :return: grid_cell_edge_x_metres: length-(N + 1) numpy array with x-
        coordinates of grid-cell edges.
    :return: grid_cell_edge_y_metres: length-(M + 1) numpy array with y-
        coordinates of grid-cell edges.
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    num_rows = field_matrix.shape[0]
    num_columns = field_matrix.shape[1]

    grid_cell_edge_x_metres, grid_cell_edge_y_metres = get_xy_grid_cell_edges(
        x_min_metres=x_min_metres, y_min_metres=y_min_metres,
        x_spacing_metres=x_spacing_metres, y_spacing_metres=y_spacing_metres,
        num_rows=num_rows, num_columns=num_columns)

    nan_row = numpy.full((1, num_columns), numpy.nan)
    field_matrix = numpy.vstack((field_matrix, nan_row))

    nan_column = numpy.full((num_rows + 1, 1), numpy.nan)
    field_matrix = numpy.hstack((field_matrix, nan_column))

    return field_matrix, grid_cell_edge_x_metres, grid_cell_edge_y_metres


def latlng_field_grid_points_to_edges(
        field_matrix=None, min_latitude_deg=None, min_longitude_deg=None,
        lat_spacing_deg=None, lng_spacing_deg=None):
    """Re-references lat-long field from grid points to edges.

    M = number of rows (unique grid-point latitudes)
    N = number of columns (unique grid-point longitudes)

    :param field_matrix: M-by-N numpy array with values of some variable
        (examples: temperature, radar reflectivity, etc.).  Latitude should
        increase while traveling down a column, and longitude should increase
        while traveling right across a row.
    :param min_latitude_deg: See documentation for get_latlng_grid_points.
    :param min_longitude_deg: See documentation for get_latlng_grid_points.
    :param lat_spacing_deg: See documentation for get_latlng_grid_points.
    :param lng_spacing_deg: See documentation for get_latlng_grid_points.
    :param num_rows: See documentation for get_latlng_grid_points.
    :param num_columns: See documentation for get_latlng_grid_points.
    :return: field_matrix: Same as input, except that dimensions are now (M + 1)
        by (N + 1).  The last row and last column contain only NaN's.
    :return: grid_cell_edge_latitudes_deg: length-(M + 1) numpy array with
        latitudes of grid-cell edges (deg N).
    :return: grid_cell_edge_longitudes_deg: length-(N + 1) numpy array with
        longitudes of grid-cell edges (deg E).
    """

    error_checking.assert_is_real_numpy_array(field_matrix)
    error_checking.assert_is_numpy_array(field_matrix, num_dimensions=2)

    num_rows = field_matrix.shape[0]
    num_columns = field_matrix.shape[1]

    (grid_cell_edge_latitudes_deg,
     grid_cell_edge_longitudes_deg) = get_latlng_grid_cell_edges(
         min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
         lat_spacing_deg=lat_spacing_deg, lng_spacing_deg=lng_spacing_deg,
         num_rows=num_rows, num_columns=num_columns)

    nan_row = numpy.full((1, num_columns), numpy.nan)
    field_matrix = numpy.vstack((field_matrix, nan_row))

    nan_column = numpy.full((num_rows + 1, 1), numpy.nan)
    field_matrix = numpy.hstack((field_matrix, nan_column))

    return (field_matrix, grid_cell_edge_latitudes_deg,
            grid_cell_edge_longitudes_deg)


def find_events_in_grid_cell(
        event_x_coords_metres, event_y_coords_metres, grid_edge_x_coords_metres,
        grid_edge_y_coords_metres, row_index, column_index, verbose):
    """Finds events in a certain grid cell.

    E = number of events
    M = number of rows in grid
    N = number of columns in grid

    :param event_x_coords_metres: length-E numpy array of x-coordinates.
    :param event_y_coords_metres: length-E numpy array of y-coordinates.
    :param grid_edge_x_coords_metres: length-(N + 1) numpy array with
        x-coordinates at edges of grid cells.
    :param grid_edge_y_coords_metres: length-(M + 1) numpy array with
        y-coordinates at edges of grid cells.
    :param row_index: Will find events in [i]th row of grid, where
        i = `row_index.`
    :param column_index: Will find events in [j]th column of grid, where
        j = `column_index.`
    :param verbose: Boolean flag.  If True, messages will be printed to command
        window.
    :return: desired_indices: 1-D numpy array with indices of events in desired
        grid cell.
    """

    error_checking.assert_is_numpy_array_without_nan(event_x_coords_metres)
    error_checking.assert_is_numpy_array(
        event_x_coords_metres, num_dimensions=1)

    num_events = len(event_x_coords_metres)
    these_expected_dim = numpy.array([num_events], dtype=int)

    error_checking.assert_is_numpy_array_without_nan(event_y_coords_metres)
    error_checking.assert_is_numpy_array(
        event_y_coords_metres, exact_dimensions=these_expected_dim)

    error_checking.assert_is_numpy_array(
        grid_edge_x_coords_metres, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_edge_x_coords_metres), 0
    )

    error_checking.assert_is_numpy_array(
        grid_edge_y_coords_metres, num_dimensions=1)
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(grid_edge_y_coords_metres), 0
    )

    error_checking.assert_is_integer(row_index)
    error_checking.assert_is_geq(row_index, 0)
    error_checking.assert_is_integer(column_index)
    error_checking.assert_is_geq(column_index, 0)
    error_checking.assert_is_boolean(verbose)

    x_min_metres = grid_edge_x_coords_metres[column_index]
    x_max_metres = grid_edge_x_coords_metres[column_index + 1]
    y_min_metres = grid_edge_y_coords_metres[row_index]
    y_max_metres = grid_edge_y_coords_metres[row_index + 1]

    if row_index == len(grid_edge_y_coords_metres) - 2:
        y_max_metres += TOLERANCE
    if column_index == len(grid_edge_x_coords_metres) - 2:
        x_max_metres += TOLERANCE

    # TODO(thunderhoser): If need be, I could speed this up by computing
    # `row_flags` only once per row and `column_flags` only once per column.
    row_flags = numpy.logical_and(
        event_y_coords_metres >= y_min_metres,
        event_y_coords_metres < y_max_metres
    )

    if not numpy.any(row_flags):
        if verbose:
            print('0 of {0:d} events are in grid cell ({1:d}, {2:d})!'.format(
                num_events, row_index, column_index
            ))

        return numpy.array([], dtype=int)

    column_flags = numpy.logical_and(
        event_x_coords_metres >= x_min_metres,
        event_x_coords_metres < x_max_metres
    )

    if not numpy.any(column_flags):
        if verbose:
            print('0 of {0:d} events are in grid cell ({1:d}, {2:d})!'.format(
                num_events, row_index, column_index
            ))

        return numpy.array([], dtype=int)

    desired_indices = numpy.where(numpy.logical_and(
        row_flags, column_flags
    ))[0]

    if verbose:
        print('{0:d} of {1:d} events are in grid cell ({2:d}, {3:d})!'.format(
            len(desired_indices), num_events, row_index, column_index
        ))

    return desired_indices
