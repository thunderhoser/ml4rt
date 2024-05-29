"""Unit tests for normalization.py"""

import copy
import unittest
import numpy
import xarray
import scipy.stats
from ml4rt.utils import example_utils
from ml4rt.utils import normalization

# Define helpful constants.
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-5

SCALAR_PREDICTOR_NAMES = [
    example_utils.ZENITH_ANGLE_NAME, example_utils.LATITUDE_NAME
]
SCALAR_TARGET_NAMES = [example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME]
VECTOR_PREDICTOR_NAMES = [example_utils.TEMPERATURE_NAME]
VECTOR_TARGET_NAMES = [
    example_utils.SHORTWAVE_DOWN_FLUX_NAME, example_utils.SHORTWAVE_UP_FLUX_NAME
]
HEIGHTS_M_AGL = numpy.array([100, 500], dtype=float)
TARGET_WAVELENGTHS_METRES = numpy.array([1e-6])

# Create unnormalized examples.
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
SCALAR_TARGET_MATRIX = numpy.expand_dims(SCALAR_TARGET_MATRIX, axis=-2)
VECTOR_TARGET_MATRIX = numpy.expand_dims(VECTOR_TARGET_MATRIX, axis=-2)

EXAMPLE_DICT_ORIG = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_utils.TARGET_WAVELENGTHS_KEY: TARGET_WAVELENGTHS_METRES
}

# Create normalization params.
TRAINING_ZENITH_ANGLES_RADIANS = numpy.linspace(1, 3, num=11, dtype=float)
TRAINING_LATITUDES_DEG_N = numpy.linspace(40, 50, num=11, dtype=float)
SCALAR_PREDICTOR_QUANTILE_MATRIX = numpy.vstack([
    TRAINING_ZENITH_ANGLES_RADIANS, TRAINING_LATITUDES_DEG_N
])

TRAINING_SURFACE_DOWN_FLUXES_W_M02 = numpy.linspace(
    0, 1000, num=11, dtype=float
)
SCALAR_TARGET_QUANTILE_MATRIX = numpy.reshape(
    TRAINING_SURFACE_DOWN_FLUXES_W_M02, (1, 1, 11)
)

THESE_TEMPS_HEIGHT1_KELVINS = numpy.linspace(280, 290, num=11, dtype=float)
THESE_TEMPS_HEIGHT2_KELVINS = numpy.linspace(290, 300, num=11, dtype=float)
TRAINING_TEMP_MATRIX_KELVINS = numpy.vstack([
    THESE_TEMPS_HEIGHT1_KELVINS, THESE_TEMPS_HEIGHT2_KELVINS
])
VECTOR_PREDICTOR_QUANTILE_MATRIX = numpy.expand_dims(
    TRAINING_TEMP_MATRIX_KELVINS, axis=-2
)

THESE_FLUXES_HEIGHT1_W_M02 = numpy.linspace(0, 500, num=11, dtype=float)
THESE_FLUXES_HEIGHT2_W_M02 = numpy.linspace(500, 1000, num=11, dtype=float)
TRAINING_DOWN_FLUX_MATRIX_W_M02 = numpy.vstack([
    THESE_FLUXES_HEIGHT1_W_M02, THESE_FLUXES_HEIGHT2_W_M02
])

THESE_FLUXES_HEIGHT1_W_M02 = numpy.linspace(0, 500, num=11, dtype=float)
THESE_FLUXES_HEIGHT2_W_M02 = numpy.linspace(600, 1100, num=11, dtype=float)
TRAINING_UP_FLUX_MATRIX_W_M02 = numpy.vstack([
    THESE_FLUXES_HEIGHT1_W_M02, THESE_FLUXES_HEIGHT2_W_M02
])

VECTOR_TARGET_QUANTILE_MATRIX = numpy.stack(
    [TRAINING_DOWN_FLUX_MATRIX_W_M02, TRAINING_UP_FLUX_MATRIX_W_M02], axis=-2
)
VECTOR_TARGET_QUANTILE_MATRIX = numpy.expand_dims(
    VECTOR_TARGET_QUANTILE_MATRIX, axis=-3
)

COORD_DICT = {
    normalization.VECTOR_PREDICTOR_DIM: VECTOR_PREDICTOR_NAMES,
    normalization.VECTOR_TARGET_DIM: VECTOR_TARGET_NAMES,
    normalization.SCALAR_PREDICTOR_DIM: SCALAR_PREDICTOR_NAMES,
    normalization.SCALAR_TARGET_DIM: SCALAR_TARGET_NAMES,
    normalization.WAVELENGTH_DIM: TARGET_WAVELENGTHS_METRES,
    normalization.HEIGHT_DIM: HEIGHTS_M_AGL,
    normalization.QUANTILE_LEVEL_DIM: numpy.linspace(0, 1, num=11, dtype=float)
}

MAIN_DATA_DICT = {
    normalization.VECTOR_PREDICTOR_QUANTILE_KEY: (
        (normalization.HEIGHT_DIM, normalization.VECTOR_PREDICTOR_DIM,
         normalization.QUANTILE_LEVEL_DIM),
        VECTOR_PREDICTOR_QUANTILE_MATRIX
    ),
    normalization.VECTOR_TARGET_QUANTILE_KEY: (
        (normalization.HEIGHT_DIM, normalization.WAVELENGTH_DIM,
         normalization.VECTOR_TARGET_DIM, normalization.QUANTILE_LEVEL_DIM),
        VECTOR_TARGET_QUANTILE_MATRIX
    ),
    normalization.SCALAR_PREDICTOR_QUANTILE_KEY: (
        (normalization.SCALAR_PREDICTOR_DIM, normalization.QUANTILE_LEVEL_DIM),
        SCALAR_PREDICTOR_QUANTILE_MATRIX
    ),
    normalization.SCALAR_TARGET_QUANTILE_KEY: (
        (normalization.WAVELENGTH_DIM, normalization.SCALAR_TARGET_DIM,
         normalization.QUANTILE_LEVEL_DIM),
        SCALAR_TARGET_QUANTILE_MATRIX
    ),
}

NORM_PARAM_TABLE_XARRAY = xarray.Dataset(
    data_vars=MAIN_DATA_DICT, coords=COORD_DICT
)

THESE_ZENITH_ANGLES = numpy.array([0.0, 0.0, 0.5, 1.0])
THESE_LATITUDES = numpy.array([0.002, 0.002, 0.002, 0.002])
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES)
))

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([0.2, 0.2, 0.2, 0.2])
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

THIS_TEMPERATURE_MATRIX = numpy.array([
    [1.0, 0.5],
    [0.9, 0.4],
    [0.8, 0.3],
    [0.7, 0.25]
])
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [0.6, 0.0],
    [1.0, 0.0],
    [0.9, 0.0],
    [0.4, 0.0]
])
THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [0.3, 0.0],
    [0.4, 0.0],
    [0.6, 0.0],
    [0.8, 0.0]
])

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)
THIS_SCALAR_TARGET_MATRIX = numpy.expand_dims(
    THIS_SCALAR_TARGET_MATRIX, axis=-2
)
THIS_VECTOR_TARGET_MATRIX = numpy.expand_dims(
    THIS_VECTOR_TARGET_MATRIX, axis=-2
)

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

EXAMPLE_DICT_NORMALIZED = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_utils.TARGET_WAVELENGTHS_KEY: TARGET_WAVELENGTHS_METRES
}

THESE_ZENITH_ANGLES_RADIANS = numpy.array([1, 1, 2, 3], dtype=float)
THESE_LATITUDES_DEG_N = numpy.array([40.02, 40.02, 40.02, 40.02], dtype=float)
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES_RADIANS, THESE_LATITUDES_DEG_N)
))

THIS_SCALAR_TARGET_MATRIX = SCALAR_TARGET_MATRIX + 0.

THIS_TEMPERATURE_MATRIX_KELVINS = numpy.array([
    [290, 295],
    [289, 294],
    [288, 293],
    [287, 292.5]
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
THIS_VECTOR_TARGET_MATRIX = numpy.expand_dims(
    THIS_VECTOR_TARGET_MATRIX, axis=-2
)

EXAMPLE_DICT_DENORMALIZED = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_utils.TARGET_WAVELENGTHS_KEY: TARGET_WAVELENGTHS_METRES
}


def _compare_example_dicts(
        first_example_dict, second_example_dict,
        absolute_error_tolerance=None, relative_error_tolerance=None):
    """Compares two dictionaries with learning examples.

    :param first_example_dict: See doc for `example_io.read_file`.
    :param second_example_dict: Same.
    :param absolute_error_tolerance: Absolute error tolerance.  If you want to
        use relative tolerance, make this None.
    :param relative_error_tolerance: Relative error tolerance.  If you want to
        use absolute tolerance, make this None.
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
        if absolute_error_tolerance is None:
            if not numpy.allclose(
                    first_example_dict[this_key], second_example_dict[this_key],
                    rtol=relative_error_tolerance
            ):
                return False
        else:
            if not numpy.allclose(
                    first_example_dict[this_key], second_example_dict[this_key],
                    atol=absolute_error_tolerance
            ):
                return False

    return True


class NormalizationTests(unittest.TestCase):
    """Each method is a unit test for normalization.py."""

    def test_normalize_data(self):
        """Ensures correct output from normalize_data."""

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_ORIG),
            normalization_param_table_xarray=NORM_PARAM_TABLE_XARRAY,
            apply_to_predictors=True,
            apply_to_vector_targets=True,
            apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            first_example_dict=this_example_dict,
            second_example_dict=EXAMPLE_DICT_NORMALIZED,
            absolute_error_tolerance=ABSOLUTE_TOLERANCE
        ))

    def test_denormalize_data(self):
        """Ensures correct output from denormalize_data."""

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_NORMALIZED),
            normalization_param_table_xarray=NORM_PARAM_TABLE_XARRAY,
            apply_to_predictors=True,
            apply_to_vector_targets=True,
            apply_to_scalar_targets=True
        )

        self.assertTrue(_compare_example_dicts(
            first_example_dict=this_example_dict,
            second_example_dict=EXAMPLE_DICT_DENORMALIZED,
            relative_error_tolerance=RELATIVE_TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
