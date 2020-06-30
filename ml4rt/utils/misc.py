"""Miscellaneous helper methods."""

import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion


def create_latlng_grid(
        min_latitude_deg, max_latitude_deg, latitude_spacing_deg,
        min_longitude_deg, max_longitude_deg, longitude_spacing_deg):
    """Creates lat-long grid.

    M = number of rows in grid
    N = number of columns in grid

    :param min_latitude_deg: Minimum latitude (deg N) in grid.
    :param max_latitude_deg: Max latitude (deg N) in grid.
    :param latitude_spacing_deg: Spacing (deg N) between grid points in adjacent
        rows.
    :param min_longitude_deg: Minimum longitude (deg E) in grid.
    :param max_longitude_deg: Max longitude (deg E) in grid.
    :param longitude_spacing_deg: Spacing (deg E) between grid points in
        adjacent columns.
    :return: grid_point_latitudes_deg: length-M numpy array with latitudes
        (deg N) of grid points.
    :return: grid_point_longitudes_deg: length-N numpy array with longitudes
        (deg E) of grid points.
    """

    # TODO(thunderhosder): Make this handle wrap-around issues.

    min_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        min_longitude_deg
    )
    max_longitude_deg = lng_conversion.convert_lng_positive_in_west(
        max_longitude_deg
    )

    min_latitude_deg = number_rounding.floor_to_nearest(
        min_latitude_deg, latitude_spacing_deg
    )
    max_latitude_deg = number_rounding.ceiling_to_nearest(
        max_latitude_deg, latitude_spacing_deg
    )
    min_longitude_deg = number_rounding.floor_to_nearest(
        min_longitude_deg, longitude_spacing_deg
    )
    max_longitude_deg = number_rounding.ceiling_to_nearest(
        max_longitude_deg, longitude_spacing_deg
    )

    num_grid_rows = 1 + int(numpy.round(
        (max_latitude_deg - min_latitude_deg) / latitude_spacing_deg
    ))
    num_grid_columns = 1 + int(numpy.round(
        (max_longitude_deg - min_longitude_deg) / longitude_spacing_deg
    ))

    return grids.get_latlng_grid_points(
        min_latitude_deg=min_latitude_deg, min_longitude_deg=min_longitude_deg,
        lat_spacing_deg=latitude_spacing_deg,
        lng_spacing_deg=longitude_spacing_deg,
        num_rows=num_grid_rows, num_columns=num_grid_columns
    )
