"""Unit tests for example_io.py."""

import copy
import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io

TOLERANCE = 1e-6

# The following constants are used to test get_grid_cell_edges,
# get_grid_cell_widths, _get_water_content_profiles, and
# _get_water_path_profiles.
CENTER_HEIGHTS_M_AGL = numpy.array([
    10, 20, 40, 60, 80, 100, 30000, 33000, 36000, 39000, 42000, 46000, 50000
], dtype=float)

EDGE_HEIGHTS_M_AGL = numpy.array([
    5, 15, 30, 50, 70, 90, 15050, 31500, 34500, 37500, 40500, 44000, 48000,
    52000
], dtype=float)

GRID_CELL_WIDTHS_METRES = numpy.array([
    10, 15, 20, 20, 20, 14960, 16450, 3000, 3000, 3000, 3500, 4000, 4000
], dtype=float)

LAYERWISE_PATH_MATRIX_KG_M02 = numpy.array([
    [1, 1, 1, 1, 1, 1000, 1000, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 1000, 1000, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 1000, 1000, 3, 3, 3, 3, 3, 3]
], dtype=float)

WATER_CONTENT_MATRIX_KG03 = numpy.array([
    [0.1, 1. / 15, 0.05, 0.05, 0.05, 1. / 14.96, 1. / 16.45,
     1. / 3000, 1. / 3000, 1. / 3000, 1. / 3500, 1. / 4000, 1. / 4000],
    [0.2, 2. / 15, 0.1, 0.1, 0.1, 1. / 14.96, 1. / 16.45,
     2. / 3000, 2. / 3000, 2. / 3000, 2. / 3500, 2. / 4000, 2. / 4000],
    [0.3, 3. / 15, 0.15, 0.15, 0.15, 1. / 14.96, 1. / 16.45,
     3. / 3000, 3. / 3000, 3. / 3000, 3. / 3500, 3. / 4000, 3. / 4000]
])

DOWNWARD_PATH_MATRIX_KG_M02 = numpy.array([
    [2011, 2010, 2009, 2008, 2007, 2006, 1006, 6, 5, 4, 3, 2, 1],
    [2022, 2020, 2018, 2016, 2014, 2012, 1012, 12, 10, 8, 6, 4, 2],
    [2033, 2030, 2027, 2024, 2021, 2018, 1018, 18, 15, 12, 9, 6, 3]
], dtype=float)

UPWARD_PATH_MATRIX_KG_M02 = numpy.array([
    [1, 2, 3, 4, 5, 1005, 2005, 2006, 2007, 2008, 2009, 2010, 2011],
    [2, 4, 6, 8, 10, 1010, 2010, 2012, 2014, 2016, 2018, 2020, 2022],
    [3, 6, 9, 12, 15, 1015, 2015, 2018, 2021, 2024, 2027, 2030, 2033]
], dtype=float)

ORIG_VECTOR_PREDICTOR_NAMES = [
    example_io.LIQUID_WATER_CONTENT_NAME, example_io.ICE_WATER_CONTENT_NAME
]
ORIG_VECTOR_PREDICTOR_MATRIX = numpy.stack(
    (WATER_CONTENT_MATRIX_KG03, WATER_CONTENT_MATRIX_KG03 / 1000), axis=-1
)
THESE_TIMES_UNIX_SEC = numpy.array([300, 600, 900], dtype=int)

EXAMPLE_DICT_WITHOUT_PATHS = {
    example_io.VECTOR_PREDICTOR_NAMES_KEY: ORIG_VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: ORIG_VECTOR_PREDICTOR_MATRIX,
    example_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    example_io.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

THESE_VECTOR_PREDICTOR_NAMES = ORIG_VECTOR_PREDICTOR_NAMES + [
    example_io.LIQUID_WATER_PATH_NAME, example_io.ICE_WATER_PATH_NAME
]
NEW_PREDICTOR_MATRIX = numpy.stack(
    (DOWNWARD_PATH_MATRIX_KG_M02, DOWNWARD_PATH_MATRIX_KG_M02 / 1000), axis=-1
)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.concatenate(
    (ORIG_VECTOR_PREDICTOR_MATRIX, NEW_PREDICTOR_MATRIX), axis=-1
)

EXAMPLE_DICT_WITH_DOWNWARD_PATHS = {
    example_io.VECTOR_PREDICTOR_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_PREDICTOR_NAMES),
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX + 0.,
    example_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    example_io.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

THESE_VECTOR_PREDICTOR_NAMES = ORIG_VECTOR_PREDICTOR_NAMES + [
    example_io.UPWARD_LIQUID_WATER_PATH_NAME,
    example_io.UPWARD_ICE_WATER_PATH_NAME
]
NEW_PREDICTOR_MATRIX = numpy.stack(
    (UPWARD_PATH_MATRIX_KG_M02, UPWARD_PATH_MATRIX_KG_M02 / 1000), axis=-1
)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.concatenate(
    (ORIG_VECTOR_PREDICTOR_MATRIX, NEW_PREDICTOR_MATRIX), axis=-1
)

EXAMPLE_DICT_WITH_UPWARD_PATHS = {
    example_io.VECTOR_PREDICTOR_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_PREDICTOR_NAMES),
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX + 0.,
    example_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    example_io.HEIGHTS_KEY: CENTER_HEIGHTS_M_AGL
}

# The following constants are used to test fluxes_to_heating_rate.
THIS_UP_FLUX_MATRIX_W_M02 = numpy.array([
    [100, 150, 200, 250, 300, 350],
    [400, 500, 600, 700, 800, 900],
    [0, 0, 0, 0, 0, 0]
], dtype=float)

THIS_DOWN_FLUX_MATRIX_W_M02 = numpy.array([
    [50, 125, 200, 275, 350, 425],
    [500, 550, 600, 650, 700, 750],
    [1000, 1000, 1000, 1000, 1000, 1000]
], dtype=float)

THIS_PRESSURE_MATRIX_PASCALS = 100 * numpy.array([
    [1000, 950, 900, 850, 800, 750],
    [1000, 900, 800, 700, 600, 500],
    [1000, 950, 900, 850, 800, 750]
], dtype=float)

# THIS_NET_FLUX_MATRIX_W_M02 = numpy.array([
#     [-50, -25, 0, 25, 50, 75],
#     [100, 50, 0, -50, -100, -150],
#     [1000, 1000, 1000, 1000, 1000, 1000]
# ], dtype=float)

THIS_NET_FLUX_DIFF_MATRIX_W02 = numpy.array([
    [25, 25, 25, 25, 25, 25],
    [-50, -50, -50, -50, -50, -50],
    [0, 0, 0, 0, 0, 0]
], dtype=float)

THIS_PRESSURE_DIFF_MATRIX_PASCALS = 100 * numpy.array([
    [-50, -50, -50, -50, -50, -50],
    [-100, -100, -100, -100, -100, -100],
    [-50, -50, -50, -50, -50, -50]
], dtype=float)

THIS_COEFF = example_io.DAYS_TO_SECONDS * (
    example_io.GRAVITY_CONSTANT_M_S02 /
    example_io.DRY_AIR_SPECIFIC_HEAT_J_KG01_K01
)

THIS_HEATING_RATE_MATRIX_K_DAY01 = THIS_COEFF * (
    THIS_NET_FLUX_DIFF_MATRIX_W02 /
    numpy.absolute(THIS_PRESSURE_DIFF_MATRIX_PASCALS)
)

THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_PRESSURE_MATRIX_PASCALS, axis=-1
)
THESE_VECTOR_PREDICTOR_NAMES = [example_io.PRESSURE_NAME]
THESE_HEIGHTS_M_AGL = numpy.array(
    [0, 500, 1000, 1500, 2000, 2500, 3000], dtype=float
)

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_UP_FLUX_MATRIX_W_M02, THIS_DOWN_FLUX_MATRIX_W_M02), axis=-1
)
THESE_VECTOR_TARGET_NAMES = [
    example_io.SHORTWAVE_UP_FLUX_NAME, example_io.SHORTWAVE_DOWN_FLUX_NAME
]

EXAMPLE_DICT_SANS_HEATING_RATE = {
    example_io.VECTOR_PREDICTOR_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_PREDICTOR_NAMES),
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX + 0.,
    example_io.VECTOR_TARGET_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_TARGET_NAMES),
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX + 0.,
    example_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    example_io.HEIGHTS_KEY: THESE_HEIGHTS_M_AGL
}

THIS_VECTOR_TARGET_MATRIX = numpy.stack((
    THIS_UP_FLUX_MATRIX_W_M02, THIS_DOWN_FLUX_MATRIX_W_M02,
    THIS_HEATING_RATE_MATRIX_K_DAY01
), axis=-1)
THESE_VECTOR_TARGET_NAMES = [
    example_io.SHORTWAVE_UP_FLUX_NAME, example_io.SHORTWAVE_DOWN_FLUX_NAME,
    example_io.SHORTWAVE_HEATING_RATE_NAME
]

EXAMPLE_DICT_WITH_HEATING_RATE = {
    example_io.VECTOR_PREDICTOR_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_PREDICTOR_NAMES),
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX + 0.,
    example_io.VECTOR_TARGET_NAMES_KEY:
        copy.deepcopy(THESE_VECTOR_TARGET_NAMES),
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX + 0.,
    example_io.VALID_TIMES_KEY: THESE_TIMES_UNIX_SEC,
    example_io.HEIGHTS_KEY: THESE_HEIGHTS_M_AGL
}

# The following constants are used to test find_file and file_name_to_year.
EXAMPLE_DIR_NAME = 'foo'
YEAR = 2018
EXAMPLE_FILE_NAME = 'foo/radiative_transfer_examples_2018.nc'

# The following constants are used to test find_many_files.
FIRST_FILE_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1999-12-31-235959', '%Y-%m-%d-%H%M%S'
)
LAST_FILE_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2005-01-01-000000', '%Y-%m-%d-%H%M%S'
)
EXAMPLE_FILE_NAMES = [
    'foo/radiative_transfer_examples_1999.nc',
    'foo/radiative_transfer_examples_2000.nc',
    'foo/radiative_transfer_examples_2001.nc',
    'foo/radiative_transfer_examples_2002.nc',
    'foo/radiative_transfer_examples_2003.nc',
    'foo/radiative_transfer_examples_2004.nc',
    'foo/radiative_transfer_examples_2005.nc'
]

# The following constants are used to test concat_examples.
FIRST_TIMES_UNIX_SEC = numpy.array([0, 300, 600, 1200], dtype=int)
FIRST_STANDARD_ATMO_FLAGS = numpy.array([0, 1, 2, 3], dtype=int)

SCALAR_PREDICTOR_NAMES = [
    example_io.ZENITH_ANGLE_NAME, example_io.LATITUDE_NAME
]

FIRST_ZENITH_ANGLES_RADIANS = numpy.array([0, 1, 2, 3], dtype=float)
FIRST_LATITUDES_DEG_N = numpy.array([40.02, 40.02, 40.02, 40.02])
FIRST_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (FIRST_ZENITH_ANGLES_RADIANS, FIRST_LATITUDES_DEG_N)
))

VECTOR_PREDICTOR_NAMES = [example_io.TEMPERATURE_NAME]
HEIGHTS_M_AGL = numpy.array([100, 500], dtype=float)

FIRST_TEMP_MATRIX_KELVINS = numpy.array([
    [290, 295],
    [289, 294],
    [288, 293],
    [287, 292.5]
])
FIRST_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    FIRST_TEMP_MATRIX_KELVINS, axis=-1
)

SCALAR_TARGET_NAMES = [example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME]

FIRST_SURFACE_DOWN_FLUXES_W_M02 = numpy.array(
    [200, 200, 200, 200], dtype=float
)
FIRST_SCALAR_TARGET_MATRIX = numpy.reshape(
    FIRST_SURFACE_DOWN_FLUXES_W_M02,
    (len(FIRST_SURFACE_DOWN_FLUXES_W_M02), 1)
)

VECTOR_TARGET_NAMES = [
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME
]

FIRST_DOWN_FLUX_MATRIX_W_M02 = numpy.array([
    [300, 200],
    [500, 300],
    [450, 450],
    [200, 100]
], dtype=float)

FIRST_UP_FLUX_MATRIX_W_M02 = numpy.array([
    [150, 150],
    [200, 150],
    [300, 350],
    [400, 100]
], dtype=float)

FIRST_VECTOR_TARGET_MATRIX = numpy.stack(
    (FIRST_DOWN_FLUX_MATRIX_W_M02, FIRST_UP_FLUX_MATRIX_W_M02), axis=-1
)

FIRST_EXAMPLE_DICT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: FIRST_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: FIRST_VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC,
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS
}

SECOND_EXAMPLE_DICT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: FIRST_SCALAR_PREDICTOR_MATRIX * 2,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: FIRST_VECTOR_PREDICTOR_MATRIX * 3,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX * 4,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX * 5,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC * 6,
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS + 1
}

CONCAT_EXAMPLE_DICT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: numpy.concatenate(
        (FIRST_SCALAR_PREDICTOR_MATRIX, FIRST_SCALAR_PREDICTOR_MATRIX * 2),
        axis=0
    ),
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: numpy.concatenate(
        (FIRST_VECTOR_PREDICTOR_MATRIX, FIRST_VECTOR_PREDICTOR_MATRIX * 3),
        axis=0
    ),
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: numpy.concatenate(
        (FIRST_SCALAR_TARGET_MATRIX, FIRST_SCALAR_TARGET_MATRIX * 4),
        axis=0
    ),
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: numpy.concatenate(
        (FIRST_VECTOR_TARGET_MATRIX, FIRST_VECTOR_TARGET_MATRIX * 5),
        axis=0
    ),
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: numpy.concatenate(
        (FIRST_TIMES_UNIX_SEC, FIRST_TIMES_UNIX_SEC * 6),
        axis=0
    ),
    example_io.STANDARD_ATMO_FLAGS_KEY: numpy.concatenate(
        (FIRST_STANDARD_ATMO_FLAGS, FIRST_STANDARD_ATMO_FLAGS + 1),
        axis=0
    )
}

# The following constants are used to test reduce_sample_size.
NUM_EXAMPLES_MEDIUM = 2
FIRST_EXAMPLE_INDEX_MEDIUM = 1
GOOD_EXAMPLE_INDICES_MEDIUM = numpy.array([1, 2], dtype=int)

FIRST_EXAMPLE_DICT_MEDIUM = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[1:3, ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[1:3, ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[1:3, ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[1:3, ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[1:3, ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[1:3, ...]
}

NUM_EXAMPLES_SMALL = 2
FIRST_EXAMPLE_INDEX_SMALL = 3
GOOD_EXAMPLE_INDICES_SMALL = numpy.array([3], dtype=int)

FIRST_EXAMPLE_DICT_SMALL = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[[3], ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[[3], ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[[3], ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[[3], ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[[3], ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[[3], ...]
}

NUM_EXAMPLES_EMPTY = 2
FIRST_EXAMPLE_INDEX_EMPTY = 4
GOOD_EXAMPLE_INDICES_EMPTY = numpy.array([], dtype=int)

FIRST_EXAMPLE_DICT_EMPTY = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[[], ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[[], ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[[], ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[[], ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[[], ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[[], ...]
}

# The following constants are used to test subset_by_time.
FIRST_SUBSET_TIME_UNIX_SEC = 1
LAST_SUBSET_TIME_UNIX_SEC = 600
GOOD_INDICES_SELECT_TIMES = numpy.array([1, 2], dtype=int)

FIRST_EXAMPLE_DICT_SELECT_TIMES = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[1:3, ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[1:3, ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[1:3, ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[1:3, ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[1:3, ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[1:3, ...]
}

# The following constants are used to test subset_by_standard_atmo.
STANDARD_ATMO_ENUM = 2
GOOD_INDICES_STANDARD_ATMO = numpy.array([2], dtype=int)

FIRST_EXAMPLE_DICT_SELECT_ATMO_TYPES = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[[2], ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[[2], ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[[2], ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[[2], ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[[2], ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[[2], ...]
}

# The following constants are used to test subset_by_field.
FIELD_NAMES_TO_KEEP = [
    example_io.SHORTWAVE_UP_FLUX_NAME, example_io.LATITUDE_NAME,
    example_io.SHORTWAVE_DOWN_FLUX_NAME
]

FIRST_EXAMPLE_DICT_SELECT_FIELDS = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: [SCALAR_PREDICTOR_NAMES[1]],
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[..., [1]],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: [],
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[..., []],
    example_io.SCALAR_TARGET_NAMES_KEY: [],
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[..., []],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES[::-1],
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[..., ::-1],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC,
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS
}

# The following constants are used to test subset_by_height.
HEIGHTS_TO_KEEP_M_AGL = numpy.array([500, 100], dtype=float)

FIRST_EXAMPLE_DICT_SELECT_HEIGHTS = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: FIRST_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[:, ::-1, :],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[:, ::-1, :],
    example_io.HEIGHTS_KEY: HEIGHTS_TO_KEEP_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC,
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS
}

# The following constants are used to test find_examples.
ALL_ID_STRINGS = ['south_boulder', 'bear', 'green', 'flagstaff', 'sanitas']
DESIRED_ID_STRINGS_0MISSING = ['green', 'bear']
RELEVANT_INDICES_0MISSING = numpy.array([2, 1], dtype=int)

DESIRED_ID_STRINGS_2MISSING = ['green', 'paiute', 'bear', 'audubon']
RELEVANT_INDICES_2MISSING = numpy.array([2, -1, 1, -1], dtype=int)

# The following constants are used to test subset_by_index.
FIRST_EXAMPLE_DICT_SELECT_INDICES = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[[2, 1], ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[[2, 1], ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[[2, 1], ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[[2, 1], ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[[2, 1], ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[[2, 1], ...]
}

# The following constants are used to test average_examples.
THIS_SCALAR_PREDICTOR_MATRIX = numpy.array([[1.5, 40.02]])
THIS_VECTOR_PREDICTOR_MATRIX = numpy.array([[288.5, 293.625]])
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_VECTOR_PREDICTOR_MATRIX, axis=-1
)
THIS_SCALAR_TARGET_MATRIX = numpy.array([[200]], dtype=float)
THIS_VECTOR_TARGET_MATRIX = numpy.array([
    [362.5, 262.5],
    [262.5, 187.5]
])

FIRST_EXAMPLE_DICT_AVERAGE = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

# The following constants are used to test create_example_ids and
# parse_example_ids.
LATITUDES_FOR_IDS_DEG_N = numpy.array([40, 40.04, 53.5, 40.0381113])
LONGITUDES_FOR_IDS_DEG_E = numpy.array([255, 254.74, 246.5, 254.7440276])
ZENITH_ANGLES_FOR_IDS_RAD = numpy.array([0.5, 0.666, 0.7777777, 1])

TIMES_FOR_IDS_UNIX_SEC = numpy.array([
    0, int(1e7), int(1e8), int(1e9)
], dtype=int)

STANDARD_ATMO_FLAGS_FOR_IDS = numpy.array([
    example_io.MIDLATITUDE_WINTER_ENUM, example_io.MIDLATITUDE_WINTER_ENUM,
    example_io.SUBARCTIC_WINTER_ENUM, example_io.MIDLATITUDE_WINTER_ENUM
], dtype=int)

THIS_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack((
    LATITUDES_FOR_IDS_DEG_N, LONGITUDES_FOR_IDS_DEG_E, ZENITH_ANGLES_FOR_IDS_RAD
)))

EXAMPLE_DICT_FOR_IDS = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: [
        example_io.LATITUDE_NAME, example_io.LONGITUDE_NAME,
        example_io.ZENITH_ANGLE_NAME
    ],
    example_io.SCALAR_PREDICTOR_VALS_KEY: THIS_PREDICTOR_MATRIX,
    example_io.VALID_TIMES_KEY: TIMES_FOR_IDS_UNIX_SEC,
    example_io.STANDARD_ATMO_FLAGS_KEY: STANDARD_ATMO_FLAGS_FOR_IDS
}

EXAMPLE_ID_STRINGS = [
    'lat=40.000000_long=255.000000_zenith-angle-rad=0.500000_'
    'time=0000000000_atmo={0:d}'.format(
        example_io.MIDLATITUDE_WINTER_ENUM
    ),
    'lat=40.040000_long=254.740000_zenith-angle-rad=0.666000_'
    'time=0010000000_atmo={0:d}'.format(
        example_io.MIDLATITUDE_WINTER_ENUM
    ),
    'lat=53.500000_long=246.500000_zenith-angle-rad=0.777778_'
    'time=0100000000_atmo={0:d}'.format(
        example_io.SUBARCTIC_WINTER_ENUM
    ),
    'lat=40.038111_long=254.744028_zenith-angle-rad=1.000000_'
    'time=1000000000_atmo={0:d}'.format(
        example_io.MIDLATITUDE_WINTER_ENUM
    )
]


def _compare_example_dicts(first_example_dict, second_example_dict):
    """Compares two dictionaries with learning examples.

    :param first_example_dict: See doc for `example_io.read_file`.
    :param second_example_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_example_dict.keys())
    second_keys = list(first_example_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    keys_to_compare = [
        example_io.SCALAR_PREDICTOR_VALS_KEY,
        example_io.VECTOR_PREDICTOR_VALS_KEY,
        example_io.SCALAR_TARGET_VALS_KEY, example_io.VECTOR_TARGET_VALS_KEY,
        example_io.HEIGHTS_KEY
    ]

    for this_key in keys_to_compare:
        if this_key not in first_example_dict:
            continue

        if not numpy.allclose(
                first_example_dict[this_key], second_example_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    try:
        if not numpy.array_equal(
                first_example_dict[example_io.VALID_TIMES_KEY],
                second_example_dict[example_io.VALID_TIMES_KEY]
        ):
            return False
    except KeyError:
        pass

    keys_to_compare = [
        example_io.SCALAR_PREDICTOR_NAMES_KEY,
        example_io.VECTOR_PREDICTOR_NAMES_KEY,
        example_io.SCALAR_TARGET_NAMES_KEY, example_io.VECTOR_TARGET_NAMES_KEY
    ]

    for this_key in keys_to_compare:
        if this_key not in first_example_dict:
            continue

        if first_example_dict[this_key] != second_example_dict[this_key]:
            return False

    return True


class ExampleIoTests(unittest.TestCase):
    """Each method is a unit test for example_io.py."""

    def test_get_grid_cell_edges(self):
        """Ensures correct output from get_grid_cell_edges."""

        these_edge_heights_m_agl = (
            example_io.get_grid_cell_edges(CENTER_HEIGHTS_M_AGL)
        )
        self.assertTrue(numpy.allclose(
            these_edge_heights_m_agl, EDGE_HEIGHTS_M_AGL, atol=TOLERANCE
        ))

    def test_get_grid_cell_widths(self):
        """Ensures correct output from get_grid_cell_widths."""

        these_widths_metres = (
            example_io.get_grid_cell_widths(EDGE_HEIGHTS_M_AGL)
        )
        self.assertTrue(numpy.allclose(
            these_widths_metres, GRID_CELL_WIDTHS_METRES, atol=TOLERANCE
        ))

    def test_get_water_content_profiles(self):
        """Ensures correct output from _get_water_content_profiles."""

        this_content_matrix_kg_m03 = example_io._get_water_content_profiles(
            layerwise_path_matrix_kg_m02=LAYERWISE_PATH_MATRIX_KG_M02,
            heights_m_agl=CENTER_HEIGHTS_M_AGL
        )

        self.assertTrue(numpy.allclose(
            this_content_matrix_kg_m03, WATER_CONTENT_MATRIX_KG03,
            atol=TOLERANCE
        ))

    def test_get_water_path_profiles_downward(self):
        """Ensures correct output from _get_water_path_profiles.

        In this case, paths are integrated downward from top of atmosphere.
        """

        this_example_dict = example_io._get_water_path_profiles(
            example_dict=copy.deepcopy(EXAMPLE_DICT_WITHOUT_PATHS),
            get_lwp=True, get_iwp=True, integrate_upward=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_WITH_DOWNWARD_PATHS
        ))

    def test_get_water_path_profiles_upward(self):
        """Ensures correct output from _get_water_path_profiles.

        In this case, paths are integrated upward from surface.
        """

        this_example_dict = example_io._get_water_path_profiles(
            example_dict=copy.deepcopy(EXAMPLE_DICT_WITHOUT_PATHS),
            get_lwp=True, get_iwp=True, integrate_upward=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_WITH_UPWARD_PATHS
        ))

    def test_fluxes_to_heating_rate(self):
        """Ensures correct output from fluxes_to_heating_rate."""

        this_example_dict = example_io.fluxes_to_heating_rate(
            copy.deepcopy(EXAMPLE_DICT_SANS_HEATING_RATE)
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_WITH_HEATING_RATE
        ))

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = example_io.find_file(
            example_dir_name=EXAMPLE_DIR_NAME, year=YEAR,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == EXAMPLE_FILE_NAME)

    def test_file_name_to_year(self):
        """Ensures correct output from file_name_to_year."""

        this_year = example_io.file_name_to_year(EXAMPLE_FILE_NAME)
        self.assertTrue(this_year == YEAR)

    def test_find_many_files(self):
        """Ensures correct output from find_many_files."""

        these_file_names = example_io.find_many_files(
            example_dir_name=EXAMPLE_DIR_NAME,
            first_time_unix_sec=FIRST_FILE_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_FILE_TIME_UNIX_SEC,
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == EXAMPLE_FILE_NAMES)

    def test_concat_examples_good(self):
        """Ensures correct output from concat_examples.

        In this case, not expecting an error.
        """

        this_example_dict = example_io.concat_examples(
            [FIRST_EXAMPLE_DICT, SECOND_EXAMPLE_DICT]
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, CONCAT_EXAMPLE_DICT
        ))

    def test_concat_examples_bad_heights(self):
        """Ensures correct output from concat_examples.

        In this case, expecting an error due to mismatched heights.
        """

        this_second_example_dict = copy.deepcopy(SECOND_EXAMPLE_DICT)
        this_second_example_dict[example_io.HEIGHTS_KEY] += 1

        with self.assertRaises(ValueError):
            example_io.concat_examples(
                [FIRST_EXAMPLE_DICT, this_second_example_dict]
            )

    def test_concat_examples_bad_fields(self):
        """Ensures correct output from concat_examples.

        In this case, expecting an error due to mismatched fields.
        """

        this_second_example_dict = copy.deepcopy(SECOND_EXAMPLE_DICT)
        this_second_example_dict[example_io.SCALAR_PREDICTOR_NAMES_KEY].append(
            example_io.ALBEDO_NAME
        )

        with self.assertRaises(ValueError):
            example_io.concat_examples(
                [FIRST_EXAMPLE_DICT, this_second_example_dict]
            )

    def test_get_field_zenith_no_height(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for zenith angle at no particular height.
        """

        this_vector = example_io.get_field_from_dict(
            example_dict=FIRST_EXAMPLE_DICT,
            field_name=example_io.ZENITH_ANGLE_NAME, height_m_agl=None
        )

        self.assertTrue(numpy.allclose(
            this_vector, FIRST_ZENITH_ANGLES_RADIANS
        ))

    def test_get_field_zenith_with_height(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for zenith angle at particular height.
        """

        this_vector = example_io.get_field_from_dict(
            example_dict=FIRST_EXAMPLE_DICT,
            field_name=example_io.ZENITH_ANGLE_NAME, height_m_agl=10.
        )

        self.assertTrue(numpy.allclose(
            this_vector, FIRST_ZENITH_ANGLES_RADIANS
        ))

    def test_get_field_temperature_no_height(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for temperature at no particular height.
        """

        this_matrix = example_io.get_field_from_dict(
            example_dict=FIRST_EXAMPLE_DICT,
            field_name=example_io.TEMPERATURE_NAME, height_m_agl=None
        )

        self.assertTrue(numpy.allclose(this_matrix, FIRST_TEMP_MATRIX_KELVINS))

    def test_get_field_temperature_100m(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for temperature at 100 m AGL.
        """

        this_vector = example_io.get_field_from_dict(
            example_dict=FIRST_EXAMPLE_DICT,
            field_name=example_io.TEMPERATURE_NAME, height_m_agl=100.
        )

        self.assertTrue(numpy.allclose(
            this_vector, FIRST_TEMP_MATRIX_KELVINS[:, 0]
        ))

    def test_get_field_temperature_500m(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for temperature at 500 m AGL.
        """

        this_vector = example_io.get_field_from_dict(
            example_dict=FIRST_EXAMPLE_DICT,
            field_name=example_io.TEMPERATURE_NAME, height_m_agl=500.
        )

        self.assertTrue(numpy.allclose(
            this_vector, FIRST_TEMP_MATRIX_KELVINS[:, 1]
        ))

    def test_get_field_temperature_600m(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for temperature at 600 m AGL (unavailable).
        """

        with self.assertRaises(ValueError):
            example_io.get_field_from_dict(
                example_dict=FIRST_EXAMPLE_DICT,
                field_name=example_io.TEMPERATURE_NAME, height_m_agl=600.
            )

    def test_get_field_lwp(self):
        """Ensures correct output from get_field_from_dict.

        In this case, looking for liquid-water path (unavailable).
        """

        with self.assertRaises(ValueError):
            example_io.get_field_from_dict(
                example_dict=FIRST_EXAMPLE_DICT,
                field_name=example_io.LIQUID_WATER_PATH_NAME, height_m_agl=None
            )

    def test_reduce_sample_size_medium(self):
        """Ensures correct output from reduce_sample_size.

        In this case, reducing to medium sample size.
        """

        this_example_dict, these_indices = example_io.reduce_sample_size(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            num_examples_to_keep=NUM_EXAMPLES_MEDIUM,
            first_example_to_keep=FIRST_EXAMPLE_INDEX_MEDIUM
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_MEDIUM
        ))
        self.assertTrue(numpy.array_equal(
            these_indices, GOOD_EXAMPLE_INDICES_MEDIUM
        ))

    def test_reduce_sample_size_small(self):
        """Ensures correct output from reduce_sample_size.

        In this case, reducing to small sample size.
        """

        this_example_dict, these_indices = example_io.reduce_sample_size(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            num_examples_to_keep=NUM_EXAMPLES_SMALL,
            first_example_to_keep=FIRST_EXAMPLE_INDEX_SMALL
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SMALL
        ))
        self.assertTrue(numpy.array_equal(
            these_indices, GOOD_EXAMPLE_INDICES_SMALL
        ))

    def test_reduce_sample_size_empty(self):
        """Ensures correct output from reduce_sample_size.

        In this case, reducing to zero sample size.
        """

        this_example_dict, these_indices = example_io.reduce_sample_size(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            num_examples_to_keep=NUM_EXAMPLES_EMPTY,
            first_example_to_keep=FIRST_EXAMPLE_INDEX_EMPTY
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_EMPTY
        ))
        self.assertTrue(numpy.array_equal(
            these_indices, GOOD_EXAMPLE_INDICES_EMPTY
        ))

    def test_subset_by_time(self):
        """Ensures correct output from subset_by_time."""

        this_example_dict, these_indices = example_io.subset_by_time(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            first_time_unix_sec=FIRST_SUBSET_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_SUBSET_TIME_UNIX_SEC
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SELECT_TIMES
        ))
        self.assertTrue(numpy.array_equal(
            these_indices, GOOD_INDICES_SELECT_TIMES
        ))

    def test_subset_by_standard_atmo(self):
        """Ensures correct output from subset_by_standard_atmo."""

        this_example_dict, these_indices = example_io.subset_by_standard_atmo(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            standard_atmo_enum=STANDARD_ATMO_ENUM
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SELECT_ATMO_TYPES
        ))
        self.assertTrue(numpy.array_equal(
            these_indices, GOOD_INDICES_STANDARD_ATMO
        ))

    def test_subset_by_field(self):
        """Ensures correct output from subset_by_field."""

        this_example_dict = example_io.subset_by_field(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            field_names=FIELD_NAMES_TO_KEEP
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SELECT_FIELDS
        ))

    def test_subset_by_height(self):
        """Ensures correct output from subset_by_height."""

        this_example_dict = example_io.subset_by_height(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            heights_m_agl=HEIGHTS_TO_KEEP_M_AGL
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SELECT_HEIGHTS
        ))

    def test_find_examples_0missing(self):
        """Ensures correct output from find_examples.

        In this case, no desired examples are missing.
        """

        these_indices = example_io.find_examples(
            all_id_strings=ALL_ID_STRINGS,
            desired_id_strings=DESIRED_ID_STRINGS_0MISSING, allow_missing=False
        )
        self.assertTrue(numpy.array_equal(
            these_indices, RELEVANT_INDICES_0MISSING
        ))

    def test_find_examples_2missing_allowed(self):
        """Ensures correct output from find_examples.

        In this case, 2 desired examples are missing but this is allowed.
        """

        these_indices = example_io.find_examples(
            all_id_strings=ALL_ID_STRINGS,
            desired_id_strings=DESIRED_ID_STRINGS_2MISSING, allow_missing=True
        )
        self.assertTrue(numpy.array_equal(
            these_indices, RELEVANT_INDICES_2MISSING
        ))

    def test_find_examples_2missing_disallowed(self):
        """Ensures correct output from find_examples.

        In this case, 2 desired examples are missing and this is *not* allowed.
        """

        with self.assertRaises(ValueError):
            example_io.find_examples(
                all_id_strings=ALL_ID_STRINGS,
                desired_id_strings=DESIRED_ID_STRINGS_2MISSING,
                allow_missing=False
            )

    def test_subset_by_index(self):
        """Ensures correct output from subset_by_index."""

        this_example_dict = example_io.subset_by_index(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            desired_indices=RELEVANT_INDICES_0MISSING
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SELECT_INDICES
        ))

    def test_average_examples(self):
        """Ensures correct output from average_examples."""

        this_example_dict = example_io.average_examples(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT), use_pmm=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_AVERAGE
        ))

    def test_create_example_ids(self):
        """Ensures correct output from create_example_ids."""

        these_id_strings = example_io.create_example_ids(EXAMPLE_DICT_FOR_IDS)
        self.assertTrue(these_id_strings == EXAMPLE_ID_STRINGS)

    def test_parse_example_ids(self):
        """Ensures correct output from parse_example_ids."""

        metadata_dict = example_io.parse_example_ids(EXAMPLE_ID_STRINGS)
        these_latitudes_deg_n = metadata_dict[example_io.LATITUDES_KEY]
        these_longitudes_deg_e = metadata_dict[example_io.LONGITUDES_KEY]
        these_zenith_angles_rad = metadata_dict[example_io.ZENITH_ANGLES_KEY]
        these_times_unix_sec = metadata_dict[example_io.VALID_TIMES_KEY]
        these_standard_atmo_flags = (
            metadata_dict[example_io.STANDARD_ATMO_FLAGS_KEY]
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg_n, LATITUDES_FOR_IDS_DEG_N, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg_e, LONGITUDES_FOR_IDS_DEG_E, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_zenith_angles_rad, ZENITH_ANGLES_FOR_IDS_RAD, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            these_times_unix_sec, TIMES_FOR_IDS_UNIX_SEC
        ))
        self.assertTrue(numpy.array_equal(
            these_standard_atmo_flags, STANDARD_ATMO_FLAGS_FOR_IDS
        ))


if __name__ == '__main__':
    unittest.main()
