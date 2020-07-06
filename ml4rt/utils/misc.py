"""Miscellaneous helper methods."""

import numpy
from gewittergefahr.gg_utils import grids
from gewittergefahr.gg_utils import number_rounding
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking


def subset_examples(indices_to_keep, num_examples_to_keep, num_examples_total):
    """Subsets examples.

    :param indices_to_keep: 1-D numpy array with indices to keep.  If None, will
        use `num_examples_to_keep` instead.
    :param num_examples_to_keep: Number of examples to keep.  If None, will use
        `indices_to_keep` instead.
    :param num_examples_total: Total number of examples available.
    :return: indices_to_keep: See input doc.
    :raises: ValueError: if both `indices_to_keep` and `num_examples_to_keep`
        are None.
    """

    if len(indices_to_keep) == 1 and indices_to_keep[0] < 0:
        indices_to_keep = None
    if indices_to_keep is not None:
        num_examples_to_keep = None
    if num_examples_to_keep < 1:
        num_examples_to_keep = None

    if indices_to_keep is None and num_examples_to_keep is None:
        raise ValueError(
            'Input args indices_to_keep and num_examples_to_keep cannot both be'
            ' empty.'
        )

    if indices_to_keep is not None:
        error_checking.assert_is_geq_numpy_array(indices_to_keep, 0)
        error_checking.assert_is_less_than_numpy_array(
            indices_to_keep, num_examples_total
        )

        return indices_to_keep

    indices_to_keep = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )

    if num_examples_to_keep >= num_examples_total:
        return indices_to_keep

    return numpy.random.choice(
        indices_to_keep, size=num_examples_to_keep, replace=False
    )


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
