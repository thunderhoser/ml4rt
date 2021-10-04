"""Handles profiles of aerosol-related quantities."""

import os
import sys
import pickle
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import polygons
import longitude_conversion as lng_conversion
import error_checking
import land_ocean_mask

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))

POLAR_REGION_NAME = 'polar'
FIRST_URBAN_REGION_NAME = 'urban1'
SECOND_URBAN_REGION_NAME = 'urban2'
FIRST_DESERT_DUST_REGION_NAME = 'desert_dust1'
SECOND_DESERT_DUST_REGION_NAME = 'desert_dust2'
BIOMASS_BURNING_REGION_NAME = 'biomass_burning'
LAND_REGION_NAME = 'land'
OCEAN_REGION_NAME = 'ocean'

ALL_REGION_NAMES = [
    POLAR_REGION_NAME, FIRST_URBAN_REGION_NAME, SECOND_URBAN_REGION_NAME,
    FIRST_DESERT_DUST_REGION_NAME, SECOND_DESERT_DUST_REGION_NAME,
    BIOMASS_BURNING_REGION_NAME, LAND_REGION_NAME, OCEAN_REGION_NAME
]

# Mean aerosol optical depth.  This value is unitless.
REGION_TO_OPTICAL_DEPTH_MEAN = {
    POLAR_REGION_NAME: 0.03,
    FIRST_URBAN_REGION_NAME: 0.15,
    SECOND_URBAN_REGION_NAME: 0.2,
    FIRST_DESERT_DUST_REGION_NAME: 0.2,
    SECOND_DESERT_DUST_REGION_NAME: 0.15,
    BIOMASS_BURNING_REGION_NAME: 0.2,
    LAND_REGION_NAME: 0.1,
    OCEAN_REGION_NAME: 0.07
}

# Standard deviation of aerosol optical depth.  This value is unitless.
REGION_TO_OPTICAL_DEPTH_STDEV = {
    POLAR_REGION_NAME: 0.2,
    FIRST_URBAN_REGION_NAME: 0.2,
    SECOND_URBAN_REGION_NAME: 0.3,
    FIRST_DESERT_DUST_REGION_NAME: 0.3,
    SECOND_DESERT_DUST_REGION_NAME: 0.3,
    BIOMASS_BURNING_REGION_NAME: 0.3,
    LAND_REGION_NAME: 0.2,
    OCEAN_REGION_NAME: 0.1
}

REGION_TO_OPTICAL_DEPTH_SHAPE_PARAM = dict()
REGION_TO_OPTICAL_DEPTH_SCALE_PARAM = dict()

for j in REGION_TO_OPTICAL_DEPTH_MEAN:
    REGION_TO_OPTICAL_DEPTH_SHAPE_PARAM[j] = (
        REGION_TO_OPTICAL_DEPTH_MEAN[j] / REGION_TO_OPTICAL_DEPTH_STDEV[j]
    ) ** 2

    REGION_TO_OPTICAL_DEPTH_SCALE_PARAM[j] = (
        (REGION_TO_OPTICAL_DEPTH_STDEV[j] ** 2) /
        REGION_TO_OPTICAL_DEPTH_MEAN[j]
    )

# Mean single-scattering albedo.  This value is unitless.
REGION_TO_ALBEDO_MEAN = {
    POLAR_REGION_NAME: 0.95,
    FIRST_URBAN_REGION_NAME: 0.94,
    SECOND_URBAN_REGION_NAME: 0.91,
    FIRST_DESERT_DUST_REGION_NAME: 0.95,
    SECOND_DESERT_DUST_REGION_NAME: 0.95,
    BIOMASS_BURNING_REGION_NAME: 0.91,
    LAND_REGION_NAME: 0.95,
    OCEAN_REGION_NAME: 0.96
}

# Standard deviation of single-scattering albedo.  This value is unitless.
REGION_TO_ALBEDO_STDEV = {
    POLAR_REGION_NAME: 0.02,
    FIRST_URBAN_REGION_NAME: 0.02,
    SECOND_URBAN_REGION_NAME: 0.04,
    FIRST_DESERT_DUST_REGION_NAME: 0.02,
    SECOND_DESERT_DUST_REGION_NAME: 0.02,
    BIOMASS_BURNING_REGION_NAME: 0.05,
    LAND_REGION_NAME: 0.02,
    OCEAN_REGION_NAME: 0.02
}

# Mean asymmetry parameter.  This value is unitless.
REGION_TO_ASYMMETRY_PARAM_MEAN = {
    POLAR_REGION_NAME: 0.72,
    FIRST_URBAN_REGION_NAME: 0.7,
    SECOND_URBAN_REGION_NAME: 0.7,
    FIRST_DESERT_DUST_REGION_NAME: 0.78,
    SECOND_DESERT_DUST_REGION_NAME: 0.78,
    BIOMASS_BURNING_REGION_NAME: 0.72,
    LAND_REGION_NAME: 0.7,
    OCEAN_REGION_NAME: 0.75
}

# Standard deviation of asymmetry parameter.  This value is unitless.
REGION_TO_ASYMMETRY_PARAM_STDEV = {
    POLAR_REGION_NAME: 0.03,
    FIRST_URBAN_REGION_NAME: 0.03,
    SECOND_URBAN_REGION_NAME: 0.03,
    FIRST_DESERT_DUST_REGION_NAME: 0.05,
    SECOND_DESERT_DUST_REGION_NAME: 0.03,
    BIOMASS_BURNING_REGION_NAME: 0.03,
    LAND_REGION_NAME: 0.03,
    OCEAN_REGION_NAME: 0.03
}

# Mean scale height for aerosol optical depth.
REGION_TO_SCALE_HEIGHT_MEAN_METRES = {
    POLAR_REGION_NAME: 500.,
    FIRST_URBAN_REGION_NAME: 1500.,
    SECOND_URBAN_REGION_NAME: 1500.,
    FIRST_DESERT_DUST_REGION_NAME: 1500.,
    SECOND_DESERT_DUST_REGION_NAME: 1500.,
    BIOMASS_BURNING_REGION_NAME: 2000.,
    LAND_REGION_NAME: 1500.,
    OCEAN_REGION_NAME: 1000.
}

# Standard deviation of scale height for aerosol optical depth.
REGION_TO_SCALE_HEIGHT_STDEV_METRES = {
    POLAR_REGION_NAME: 100.,
    FIRST_URBAN_REGION_NAME: 300.,
    SECOND_URBAN_REGION_NAME: 100.,
    FIRST_DESERT_DUST_REGION_NAME: 200.,
    SECOND_DESERT_DUST_REGION_NAME: 200.,
    BIOMASS_BURNING_REGION_NAME: 300.,
    LAND_REGION_NAME: 300.,
    OCEAN_REGION_NAME: 100.
}


def _read_region_coords(pickle_file_name=None):
    """Reads lat-long coordinates for each aerosol region.

    :param pickle_file_name: Path to input file.  This file should contain one
        dictionary, formatted like the output variable from this method.
    :return: region_dict: Dictionary, where each key is a region name in
        `ALL_REGION_NAMES` and the corresponding value is a tuple of two 1-D
        numpy arrays -- the first containing latitudes in deg N, the second
        containing longitudes in deg E.
    """

    if pickle_file_name is None:
        pickle_file_name = '{0:s}/aerosol_regions.p'.format(THIS_DIRECTORY_NAME)

    error_checking.assert_file_exists(pickle_file_name)

    print('Reading aerosol regions from: "{0:s}"...'.format(pickle_file_name))
    pickle_file_handle = open(pickle_file_name, 'rb')
    region_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    for this_key in region_dict:
        assert this_key in ALL_REGION_NAMES

        these_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
            region_dict[this_key][1]
        )
        region_dict[this_key] = (
            region_dict[this_key][0], these_longitudes_deg_e
        )

    return region_dict


def _split_array_by_nan(input_array):
    """Splits numpy array into list of contiguous subarrays without NaN.

    :param input_array: 1-D numpy array.
    :return: list_of_arrays: 1-D list of 1-D numpy arrays.  Each numpy array is
        without NaN.
    """

    error_checking.assert_is_real_numpy_array(input_array)
    error_checking.assert_is_numpy_array(input_array, num_dimensions=1)

    return [
        input_array[i] for i in
        numpy.ma.clump_unmasked(numpy.ma.masked_invalid(input_array))
    ]


def _find_examples_in_one_polygon(
        polygon_latitudes_deg_n, polygon_longitudes_deg_e,
        example_latitudes_deg_n, example_longitudes_deg_e,
        region_name_by_example, polygon_region_name):
    """Finds examples in one polygon.

    V = number of vertices in polygon
    E = number of examples

    :param polygon_latitudes_deg_n: length-V numpy array of latitudes at polygon
        vertices (deg north).
    :param polygon_longitudes_deg_e: length-V numpy array of longitudes at
        polygon vertices (deg east).
    :param example_latitudes_deg_n: length-E numpy array of example latitudes
        (deg north).
    :param example_longitudes_deg_e: length-E numpy array of example longitudes
        (deg east).
    :param region_name_by_example: length-E numpy array of region names, with
        empty string for examples as yet unassigned.
    :param polygon_region_name: Region name for this polygon.
    :return: region_name_by_example: Same as input but maybe with fewer empty
        strings.
    """

    unassigned_indices = numpy.where(region_name_by_example == '')[0]
    if len(unassigned_indices) == 0:
        return region_name_by_example

    latitude_flags = numpy.logical_and(
        example_latitudes_deg_n[unassigned_indices] >=
        numpy.min(polygon_latitudes_deg_n),
        example_latitudes_deg_n[unassigned_indices] <=
        numpy.max(polygon_latitudes_deg_n)
    )

    longitude_flags = numpy.logical_and(
        example_longitudes_deg_e[unassigned_indices] >=
        numpy.min(polygon_longitudes_deg_e),
        example_longitudes_deg_e[unassigned_indices] <=
        numpy.max(polygon_longitudes_deg_e)
    )

    in_bbox_subindices = numpy.where(numpy.logical_and(
        latitude_flags, longitude_flags
    ))[0]

    if len(in_bbox_subindices) == 0:
        return region_name_by_example

    unassigned_indices = unassigned_indices[in_bbox_subindices]

    polygon_object = polygons.vertex_arrays_to_polygon_object(
        exterior_x_coords=polygon_longitudes_deg_e,
        exterior_y_coords=polygon_latitudes_deg_n
    )

    for i in unassigned_indices:
        this_flag = polygons.point_in_or_on_polygon(
            polygon_object=polygon_object,
            query_x_coordinate=example_longitudes_deg_e[i],
            query_y_coordinate=example_latitudes_deg_n[i]
        )

        if not this_flag:
            continue

        region_name_by_example[i] = polygon_region_name

    return region_name_by_example


def assign_examples_to_regions(example_latitudes_deg_n,
                               example_longitudes_deg_e):
    """Assigns each example to an aerosol region.

    E = number of examples

    :param example_latitudes_deg_n: length-E numpy array of latitudes (deg
        north).
    :param example_longitudes_deg_e: length-E numpy array of longitudes (deg
        east).
    :return: region_names: length-E list with names of aerosol regions.
    """

    region_dict = _read_region_coords()

    example_longitudes_deg_e = lng_conversion.convert_lng_negative_in_west(
        example_longitudes_deg_e, allow_nan=False
    )
    num_examples = len(example_longitudes_deg_e)

    error_checking.assert_is_valid_lat_numpy_array(
        example_latitudes_deg_n, allow_nan=False
    )
    error_checking.assert_is_numpy_array(
        example_latitudes_deg_n,
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )

    region_name_by_example = numpy.full(num_examples, '', dtype=object)
    polar_indices = numpy.where(
        numpy.absolute(example_latitudes_deg_n) >= 60.
    )[0]
    region_name_by_example[polar_indices] = POLAR_REGION_NAME

    for this_region_name in region_dict:
        polygon_latitude_arrays_deg_n = _split_array_by_nan(
            region_dict[this_region_name][0]
        )
        polygon_longitude_arrays_deg_e = _split_array_by_nan(
            region_dict[this_region_name][1]
        )
        num_polygons = len(polygon_latitude_arrays_deg_n)

        for k in range(num_polygons):
            region_name_by_example = _find_examples_in_one_polygon(
                polygon_latitudes_deg_n=polygon_latitude_arrays_deg_n[k],
                polygon_longitudes_deg_e=polygon_longitude_arrays_deg_e[k],
                example_latitudes_deg_n=example_latitudes_deg_n,
                example_longitudes_deg_e=example_longitudes_deg_e,
                region_name_by_example=region_name_by_example,
                polygon_region_name=this_region_name
            )

    unassigned_indices = numpy.where(region_name_by_example == '')[0]
    if len(unassigned_indices) == 0:
        return region_name_by_example.tolist()

    land_flags = numpy.array([
        land_ocean_mask.is_land(lat=y, lon=x)
        for y, x in zip(
            example_latitudes_deg_n[unassigned_indices],
            example_longitudes_deg_e[unassigned_indices]
        )
    ], dtype=bool)

    land_indices = unassigned_indices[land_flags]
    region_name_by_example[land_indices] = LAND_REGION_NAME

    ocean_indices = unassigned_indices[land_flags == False]
    region_name_by_example[ocean_indices] = OCEAN_REGION_NAME

    unassigned_indices = numpy.where(
        region_name_by_example == ''
    )[0]

    assert len(unassigned_indices) == 0
    return region_name_by_example.tolist()
