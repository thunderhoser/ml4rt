"""Methods for handling polygons.

In general, x- and y- coordinates may be in one of three formats:

[1] Metres.
[2] Longitude (deg E) and latitude (deg N), respectively.
[3] Columns and rows in a grid, respectively.
"""

import os
import sys
import numpy
import shapely.geometry

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking


def _check_vertex_arrays(x_coordinates, y_coordinates, allow_nan=True):
    """Checks vertex arrays for errors.

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).

    V = number of vertices

    :param x_coordinates: length-V numpy array with x-coordinates of vertices.
        The first NaN separates the exterior from the first hole, and the [i]th
        NaN separates the [i - 1]th hole from the [i]th hole.
    :param y_coordinates: Same as above, except for y-coordinates.
    :param allow_nan: Boolean flag.  If True, input arrays may contain NaN's
        (however, NaN's must occur at the exact same positions in the two
        arrays).
    :raises: ValueError: if allow_nan = True but NaN's do not occur at the same
        positions in the two arrays.
    """

    error_checking.assert_is_boolean(allow_nan)

    if allow_nan:
        error_checking.assert_is_real_numpy_array(x_coordinates)
        error_checking.assert_is_real_numpy_array(y_coordinates)
    else:
        error_checking.assert_is_numpy_array_without_nan(x_coordinates)
        error_checking.assert_is_numpy_array_without_nan(y_coordinates)

    error_checking.assert_is_numpy_array(x_coordinates, num_dimensions=1)
    num_vertices = len(x_coordinates)
    error_checking.assert_is_numpy_array(
        y_coordinates, exact_dimensions=numpy.array([num_vertices]))

    x_nan_indices = numpy.where(numpy.isnan(x_coordinates))[0]
    y_nan_indices = numpy.where(numpy.isnan(y_coordinates))[0]
    if not numpy.array_equal(x_nan_indices, y_nan_indices):
        error_string = (
            '\nNaN''s occur at the following positions in `x_coordinates`:\n' +
            str(x_nan_indices) +
            '\nand the following positions in `y_coordinates`:\n' +
            str(y_nan_indices) +
            '\nNaN''s should occur at the same positions in the two arrays.')
        raise ValueError(error_string)


def _vertex_arrays_to_list(vertex_x_coords, vertex_y_coords):
    """Converts vertices of simple polygon from two arrays to one list.

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).

    V = number of vertices

    :param vertex_x_coords: See documentation for _check_vertex_arrays.
    :param vertex_y_coords: See documentation for _check_vertex_arrays.
    :return: vertex_coords_as_list: length-V list, where each element is an
        (x, y) tuple.
    """

    _check_vertex_arrays(vertex_x_coords, vertex_y_coords, allow_nan=False)

    num_vertices = len(vertex_x_coords)
    vertex_coords_as_list = []
    for i in range(num_vertices):
        vertex_coords_as_list.append((vertex_x_coords[i], vertex_y_coords[i]))

    return vertex_coords_as_list


def vertex_arrays_to_polygon_object(
        exterior_x_coords, exterior_y_coords, hole_x_coords_list=None,
        hole_y_coords_list=None):
    """Converts polygon from vertex arrays to `shapely.geometry.Polygon` object.

    V_e = number of exterior vertices
    H = number of holes
    V_hi = number of vertices in [i]th hole

    :param exterior_x_coords: numpy array (length V_e) with x-coordinates of
        exterior vertices.
    :param exterior_y_coords: numpy array (length V_e) with y-coordinates of
        exterior vertices.
    :param hole_x_coords_list: length-H list, where the [i]th item is a numpy
        array (length V_hi) with x-coordinates of interior vertices.
    :param hole_y_coords_list: Same as above, except for y-coordinates.
    :return: polygon_object: `shapely.geometry.Polygon` object.
    :raises: ValueError: if the polygon is invalid.
    """

    exterior_coords_as_list = _vertex_arrays_to_list(
        exterior_x_coords, exterior_y_coords)
    if hole_x_coords_list is None:
        return shapely.geometry.Polygon(shell=exterior_coords_as_list)

    num_holes = len(hole_x_coords_list)
    outer_list_of_hole_coords = []
    for i in range(num_holes):
        outer_list_of_hole_coords.append(_vertex_arrays_to_list(
            hole_x_coords_list[i], hole_y_coords_list[i]))

    polygon_object = shapely.geometry.Polygon(
        shell=exterior_coords_as_list, holes=tuple(outer_list_of_hole_coords))

    if not polygon_object.is_valid:
        raise ValueError('Resulting polygon is invalid.')

    return polygon_object


def point_in_or_on_polygon(
        polygon_object, query_x_coordinate, query_y_coordinate):
    """Returns True if point is inside/touching the polygon, False otherwise.

    x- and y-coordinates may be in one of three formats (see docstring at top of
    file).  However, the 3 input arguments must have coordinates in the same
    format.

    :param polygon_object: `shapely.geometry.Polygon` object.
    :param query_x_coordinate: x-coordinate of query point.
    :param query_y_coordinate: y-coordinate of query point.
    :return: result: Boolean flag.  True if point is inside/touching the
        polygon, False otherwise.
    """

    error_checking.assert_is_not_nan(query_x_coordinate)
    error_checking.assert_is_not_nan(query_y_coordinate)

    point_object = shapely.geometry.Point(
        query_x_coordinate, query_y_coordinate)
    if polygon_object.contains(point_object):
        return True

    return polygon_object.touches(point_object)
