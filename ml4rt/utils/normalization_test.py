"""Unit tests for normalization.py"""

import copy
import unittest
import numpy
import scipy.stats
from ml4rt.utils import example_utils
from ml4rt.utils import normalization

TOLERANCE = 1e-6

# The following constants are used to test normalize_data and denormalize_data,
# with min-max normalization and *no* separation by height.
MIN_NORMALIZED_VALUE = 0.
MAX_NORMALIZED_VALUE = 1.

SCALAR_PREDICTOR_NAMES = [
    example_utils.ZENITH_ANGLE_NAME, example_utils.LATITUDE_NAME
]
SCALAR_TARGET_NAMES = [example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME]
VECTOR_PREDICTOR_NAMES = [example_utils.TEMPERATURE_NAME]
VECTOR_TARGET_NAMES = [
    example_utils.SHORTWAVE_DOWN_FLUX_NAME, example_utils.SHORTWAVE_UP_FLUX_NAME
]
HEIGHTS_M_AGL = numpy.array([100, 500], dtype=float)

ZENITH_ANGLES_RADIANS = numpy.array([0, 1, 2, 3], dtype=float)
LATITUDES_DEG_N = numpy.array([40.02, 40.02, 40.02, 40.02])
SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (ZENITH_ANGLES_RADIANS, LATITUDES_DEG_N)
))

SHORTWAVE_SURFACE_DOWN_FLUXES_W_M02 = numpy.array(
    [200, 200, 200, 200], dtype=float
)
SCALAR_TARGET_MATRIX = numpy.reshape(
    SHORTWAVE_SURFACE_DOWN_FLUXES_W_M02,
    (len(SHORTWAVE_SURFACE_DOWN_FLUXES_W_M02), 1)
)

TEMPERATURE_MATRIX_KELVINS = numpy.array([
    [290, 295],
    [289, 294],
    [288, 293],
    [287, 292.5]
])
VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(TEMPERATURE_MATRIX_KELVINS, axis=-1)

SHORTWAVE_DOWN_FLUX_MATRIX_W_M02 = numpy.array([
    [300, 200],
    [500, 300],
    [450, 450],
    [200, 100]
], dtype=float)

SHORTWAVE_UP_FLUX_MATRIX_W_M02 = numpy.array([
    [150, 150],
    [200, 150],
    [300, 350],
    [400, 100]
], dtype=float)

VECTOR_TARGET_MATRIX = numpy.stack(
    (SHORTWAVE_DOWN_FLUX_MATRIX_W_M02, SHORTWAVE_UP_FLUX_MATRIX_W_M02), axis=-1
)

EXAMPLE_DICT_ORIG = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

TRAINING_ZENITH_ANGLES_RADIANS = numpy.linspace(1, 3, num=11, dtype=float)
TRAINING_LATITUDES_DEG_N = numpy.linspace(40, 50, num=11, dtype=float)
TRAINING_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (TRAINING_ZENITH_ANGLES_RADIANS, TRAINING_LATITUDES_DEG_N)
))

TRAINING_SURFACE_DOWN_FLUXES_W_M02 = numpy.linspace(
    0, 1000, num=11, dtype=float
)
TRAINING_SCALAR_TARGET_MATRIX = numpy.reshape(
    TRAINING_SURFACE_DOWN_FLUXES_W_M02,
    (len(TRAINING_SURFACE_DOWN_FLUXES_W_M02), 1)
)

THESE_TEMPS_HEIGHT1_KELVINS = numpy.linspace(280, 290, num=11, dtype=float)
THESE_TEMPS_HEIGHT2_KELVINS = numpy.linspace(290, 300, num=11, dtype=float)
TRAINING_TEMP_MATRIX_KELVINS = numpy.transpose(numpy.vstack((
    THESE_TEMPS_HEIGHT1_KELVINS, THESE_TEMPS_HEIGHT2_KELVINS
)))

TRAINING_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    TRAINING_TEMP_MATRIX_KELVINS, axis=-1
)

THESE_FLUXES_HEIGHT1_W_M02 = numpy.linspace(0, 500, num=11, dtype=float)
THESE_FLUXES_HEIGHT2_W_M02 = numpy.linspace(500, 1000, num=11, dtype=float)
TRAINING_DOWN_FLUX_MATRIX_W_M02 = numpy.transpose(numpy.vstack((
    THESE_FLUXES_HEIGHT1_W_M02, THESE_FLUXES_HEIGHT2_W_M02
)))

THESE_FLUXES_HEIGHT1_W_M02 = numpy.linspace(0, 500, num=11, dtype=float)
THESE_FLUXES_HEIGHT2_W_M02 = numpy.linspace(600, 1100, num=11, dtype=float)
TRAINING_UP_FLUX_MATRIX_W_M02 = numpy.transpose(numpy.vstack((
    THESE_FLUXES_HEIGHT1_W_M02, THESE_FLUXES_HEIGHT2_W_M02
)))

TRAINING_VECTOR_TARGET_MATRIX = numpy.stack(
    (TRAINING_DOWN_FLUX_MATRIX_W_M02, TRAINING_UP_FLUX_MATRIX_W_M02), axis=-1
)

TRAINING_EXAMPLE_DICT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: TRAINING_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: TRAINING_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: TRAINING_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: TRAINING_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

THESE_ZENITH_ANGLES = numpy.array([0, 0, 5, 10], dtype=float) / 10
THESE_LATITUDES = numpy.array([0.1, 0.1, 0.1, 0.1])
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES)
))

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([0.2, 0.2, 0.2, 0.2])
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

# TEMPERATURE_MATRIX_KELVINS = numpy.array([
#     [290, 295],
#     [289, 294],
#     [288, 293],
#     [287, 292.5]
# ])

THIS_TEMPERATURE_MATRIX = numpy.array([
    [10, 16],
    [9, 15],
    [8, 14],
    [7, 14]
], dtype=float) / 21

THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [6, 4],
    [10, 6],
    [9, 9],
    [4, 2]
], dtype=float) / 21

THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [3, 3],
    [4, 3],
    [6, 7],
    [8, 2]
], dtype=float) / 21

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)

EXAMPLE_DICT_MINMAX_PREDICTORS_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_TARGETS_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_BOTH_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

# The following constants are used to test normalize_data and denormalize_data,
# with z-score normalization and *no* separation by height.
THIS_SCALAR_TARGET_MATRIX = numpy.maximum(
    THIS_SCALAR_TARGET_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_SCALAR_TARGET_MATRIX = numpy.minimum(
    THIS_SCALAR_TARGET_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_SCALAR_TARGET_MATRIX = scipy.stats.norm.ppf(
    THIS_SCALAR_TARGET_MATRIX, loc=0., scale=1.
)

THIS_SCALAR_PREDICTOR_MATRIX = numpy.maximum(
    THIS_SCALAR_PREDICTOR_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_SCALAR_PREDICTOR_MATRIX = numpy.minimum(
    THIS_SCALAR_PREDICTOR_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_SCALAR_PREDICTOR_MATRIX = scipy.stats.norm.ppf(
    THIS_SCALAR_PREDICTOR_MATRIX, loc=0., scale=1.
)

THIS_VECTOR_TARGET_MATRIX = numpy.maximum(
    THIS_VECTOR_TARGET_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_VECTOR_TARGET_MATRIX = numpy.minimum(
    THIS_VECTOR_TARGET_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_VECTOR_TARGET_MATRIX = scipy.stats.norm.ppf(
    THIS_VECTOR_TARGET_MATRIX, loc=0., scale=1.
)

THIS_VECTOR_PREDICTOR_MATRIX = numpy.maximum(
    THIS_VECTOR_PREDICTOR_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.minimum(
    THIS_VECTOR_PREDICTOR_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_VECTOR_PREDICTOR_MATRIX = scipy.stats.norm.ppf(
    THIS_VECTOR_PREDICTOR_MATRIX, loc=0., scale=1.
)

EXAMPLE_DICT_Z_PREDICTORS_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_TARGETS_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_BOTH_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

THESE_ZENITH_ANGLES_RADIANS = numpy.array([1, 1, 2, 3], dtype=float)
THESE_LATITUDES_DEG_N = numpy.array([41, 41, 41, 41], dtype=float)
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES_RADIANS, THESE_LATITUDES_DEG_N)
))

THIS_SCALAR_TARGET_MATRIX = SCALAR_TARGET_MATRIX + 0.

THIS_TEMPERATURE_MATRIX_KELVINS = numpy.array([
    [289.5, 295],
    [289, 294],
    [288, 292.5],
    [286.5, 292.5]
])
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX_KELVINS, axis=-1
)

THIS_DOWN_FLUX_MATRIX_W_M02 = numpy.array([
    [300, 200],
    [475, 300],
    [450, 450],
    [200, 100]
], dtype=float)

THIS_UP_FLUX_MATRIX_W_M02 = numpy.array([
    [150, 150],
    [200, 150],
    [300, 325],
    [400, 100]
], dtype=float)

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_DOWN_FLUX_MATRIX_W_M02, THIS_UP_FLUX_MATRIX_W_M02), axis=-1
)

EXAMPLE_DICT_DENORM_BOTH_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_DENORM_PREDICTORS_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_DENORM_TARGETS_NO_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

# The following constants are used to test normalize_data and denormalize_data,
# with min-max normalization and separation by height.
THESE_ZENITH_ANGLES = numpy.array([0, 0, 5, 10], dtype=float) / 10
THESE_LATITUDES = numpy.array([0.1, 0.1, 0.1, 0.1])
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES)
))

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([0.2, 0.2, 0.2, 0.2])
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

THIS_TEMPERATURE_MATRIX = numpy.array([
    [10, 5],
    [9, 4],
    [8, 3],
    [7, 3]
], dtype=float) / 10

THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [6, 0],
    [10, 0],
    [9, 0],
    [4, 0]
], dtype=float) / 10

THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [3, 0],
    [4, 0],
    [6, 0],
    [8, 0]
], dtype=float) / 10

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)

EXAMPLE_DICT_MINMAX_PREDICTORS_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_TARGETS_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_BOTH_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

# The following constants are used to test normalize_data and denormalize_data,
# with z-score normalization and separation by height.
THIS_SCALAR_TARGET_MATRIX = numpy.maximum(
    THIS_SCALAR_TARGET_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_SCALAR_TARGET_MATRIX = numpy.minimum(
    THIS_SCALAR_TARGET_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_SCALAR_TARGET_MATRIX = scipy.stats.norm.ppf(
    THIS_SCALAR_TARGET_MATRIX, loc=0., scale=1.
)

THIS_SCALAR_PREDICTOR_MATRIX = numpy.maximum(
    THIS_SCALAR_PREDICTOR_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_SCALAR_PREDICTOR_MATRIX = numpy.minimum(
    THIS_SCALAR_PREDICTOR_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_SCALAR_PREDICTOR_MATRIX = scipy.stats.norm.ppf(
    THIS_SCALAR_PREDICTOR_MATRIX, loc=0., scale=1.
)

THIS_VECTOR_TARGET_MATRIX = numpy.maximum(
    THIS_VECTOR_TARGET_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_VECTOR_TARGET_MATRIX = numpy.minimum(
    THIS_VECTOR_TARGET_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_VECTOR_TARGET_MATRIX = scipy.stats.norm.ppf(
    THIS_VECTOR_TARGET_MATRIX, loc=0., scale=1.
)

THIS_VECTOR_PREDICTOR_MATRIX = numpy.maximum(
    THIS_VECTOR_PREDICTOR_MATRIX, normalization.MIN_CUMULATIVE_DENSITY
)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.minimum(
    THIS_VECTOR_PREDICTOR_MATRIX, normalization.MAX_CUMULATIVE_DENSITY
)
THIS_VECTOR_PREDICTOR_MATRIX = scipy.stats.norm.ppf(
    THIS_VECTOR_PREDICTOR_MATRIX, loc=0., scale=1.
)

EXAMPLE_DICT_Z_PREDICTORS_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_TARGETS_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_BOTH_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

THESE_ZENITH_ANGLES_RADIANS = numpy.array([1, 1, 2, 3], dtype=float)
THESE_LATITUDES_DEG_N = numpy.array([41, 41, 41, 41], dtype=float)
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES_RADIANS, THESE_LATITUDES_DEG_N)
))

THIS_SCALAR_TARGET_MATRIX = SCALAR_TARGET_MATRIX + 0.

THIS_TEMPERATURE_MATRIX_KELVINS = numpy.array([
    [290, 295],
    [289, 294],
    [288, 293],
    [287, 293]
], dtype=float)
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX_KELVINS, axis=-1
)

THIS_DOWN_FLUX_MATRIX_W_M02 = numpy.array([
    [300, 500],
    [500, 500],
    [450, 500],
    [200, 500]
], dtype=float)

THIS_UP_FLUX_MATRIX_W_M02 = numpy.array([
    [150, 600],
    [200, 600],
    [300, 600],
    [400, 600]
], dtype=float)

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_DOWN_FLUX_MATRIX_W_M02, THIS_UP_FLUX_MATRIX_W_M02), axis=-1
)

EXAMPLE_DICT_DENORM_BOTH_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_DENORM_PREDICTORS_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_DENORM_TARGETS_WITH_HEIGHT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL
}

# The following constants are used to test create_mean_example.
MEAN_SCALAR_PREDICTOR_MATRIX = numpy.array([[2, 45]], dtype=float)
MEAN_VECTOR_PREDICTOR_MATRIX = numpy.array([285, 295], dtype=float)
MEAN_SCALAR_TARGET_MATRIX = numpy.array([[500]], dtype=float)
MEAN_VECTOR_TARGET_MATRIX = numpy.array([
    [250, 250],
    [750, 850]
], dtype=float)

MEAN_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    MEAN_VECTOR_PREDICTOR_MATRIX, axis=0
)
MEAN_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    MEAN_VECTOR_PREDICTOR_MATRIX, axis=-1
)
MEAN_VECTOR_TARGET_MATRIX = numpy.expand_dims(MEAN_VECTOR_TARGET_MATRIX, axis=0)

MEAN_EXAMPLE_DICT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: MEAN_SCALAR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_VALS_KEY: MEAN_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: MEAN_VECTOR_PREDICTOR_MATRIX,
    example_utils.VECTOR_TARGET_VALS_KEY: MEAN_VECTOR_TARGET_MATRIX
}


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
        example_utils.SCALAR_PREDICTOR_VALS_KEY,
        example_utils.VECTOR_PREDICTOR_VALS_KEY,
        example_utils.SCALAR_TARGET_VALS_KEY,
        example_utils.VECTOR_TARGET_VALS_KEY
    ]

    for this_key in keys_to_compare:
        if not numpy.allclose(
                first_example_dict[this_key], second_example_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    return True


class NormalizationTests(unittest.TestCase):
    """Each method is a unit test for normalization.py."""

    def test_normalize_data_minmax_both_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for both predictors and
        targets, with *no* separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=False, apply_to_predictors=True,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_BOTH_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_MINMAX_BOTH_NO_HEIGHT),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=False, apply_to_predictors=True,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM_BOTH_NO_HEIGHT
        ))

    def test_normalize_data_minmax_predictors_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for predictors only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=False, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_PREDICTORS_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            new_example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_PREDICTORS_NO_HEIGHT),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=False, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM_PREDICTORS_NO_HEIGHT
        ))

    def test_normalize_data_minmax_targets_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for targets only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=False, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_TARGETS_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            new_example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_TARGETS_NO_HEIGHT),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=False, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM_TARGETS_NO_HEIGHT
        ))

    def test_normalize_data_z_both_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for both predictors and
        targets, with *no* separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            uniformize=True,
            separate_heights=False, apply_to_predictors=True,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_BOTH_NO_HEIGHT
        ))

    def test_normalize_data_z_predictors_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for predictors only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            uniformize=True,
            separate_heights=False, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_PREDICTORS_NO_HEIGHT
        ))

    def test_normalize_data_z_targets_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for targets only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            uniformize=True,
            separate_heights=False, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_TARGETS_NO_HEIGHT
        ))

    def test_normalize_data_minmax_both_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for both predictors and
        targets, with separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_BOTH_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            new_example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_BOTH_WITH_HEIGHT),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM_BOTH_WITH_HEIGHT
        ))

    def test_normalize_data_minmax_predictors_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for predictors only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_PREDICTORS_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            new_example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_PREDICTORS_WITH_HEIGHT),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM_PREDICTORS_WITH_HEIGHT
        ))

    def test_normalize_data_minmax_targets_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for targets only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_TARGETS_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            new_example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_TARGETS_WITH_HEIGHT),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            uniformize=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM_TARGETS_WITH_HEIGHT
        ))

    def test_normalize_data_z_both_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for both predictors and
        targets, with separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            uniformize=True,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_BOTH_WITH_HEIGHT
        ))

    def test_normalize_data_z_predictors_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for predictors only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            uniformize=True,
            separate_heights=True, apply_to_predictors=True,
            apply_to_vector_targets=False, apply_to_scalar_targets=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_PREDICTORS_WITH_HEIGHT
        ))

    def test_normalize_data_z_targets_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for targets only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT,
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            uniformize=True,
            separate_heights=True, apply_to_predictors=False,
            apply_to_vector_targets=True, apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_TARGETS_WITH_HEIGHT
        ))

    def test_create_mean_example(self):
        """Ensures correct output from create_mean_example."""

        this_example_dict = normalization.create_mean_example(
            new_example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            training_example_dict=TRAINING_EXAMPLE_DICT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, MEAN_EXAMPLE_DICT
        ))


if __name__ == '__main__':
    unittest.main()
