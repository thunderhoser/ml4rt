"""Unit tests for normalization.py"""

import copy
import unittest
import numpy
import pandas
from ml4rt.io import example_io
from ml4rt.utils import normalization

TOLERANCE = 1e-6
MIN_NORMALIZED_VALUE = 0.
MAX_NORMALIZED_VALUE = 1.

ZENITH_ANGLES_RADIANS = numpy.array([0, 1, 2, 3], dtype=float)
LATITUDES_DEG_N = numpy.array([40.02, 40.02, 40.02, 40.02])
SCALAR_PREDICTOR_NAMES = [
    example_io.ZENITH_ANGLE_NAME, example_io.LATITUDE_NAME
]
SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (ZENITH_ANGLES_RADIANS, LATITUDES_DEG_N)
))

TEMPERATURE_MATRIX_KELVINS = numpy.array([
    [290, 295],
    [289, 294],
    [288, 293],
    [287, 292.5]
])
VECTOR_PREDICTOR_NAMES = [example_io.TEMPERATURE_NAME]
HEIGHTS_M_AGL = numpy.array([100, 500], dtype=float)
VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(TEMPERATURE_MATRIX_KELVINS, axis=-1)

SHORTWAVE_SURFACE_DOWN_FLUXES_W_M02 = numpy.array(
    [200, 200, 200, 200], dtype=float
)
SCALAR_TARGET_NAMES = [example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME]
SCALAR_TARGET_MATRIX = numpy.reshape(
    SHORTWAVE_SURFACE_DOWN_FLUXES_W_M02,
    (len(SHORTWAVE_SURFACE_DOWN_FLUXES_W_M02), 1)
)

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

VECTOR_TARGET_NAMES = [
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME
]
VECTOR_TARGET_MATRIX = numpy.stack(
    (SHORTWAVE_DOWN_FLUX_MATRIX_W_M02, SHORTWAVE_UP_FLUX_MATRIX_W_M02), axis=-1
)

EXAMPLE_DICT_DENORM = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

THIS_DICT = {
    example_io.ZENITH_ANGLE_NAME: numpy.array([0.75, 0.25, 0, 1.5]),
    example_io.LATITUDE_NAME: numpy.array([45, 10, -90, 90], dtype=float),
    example_io.TEMPERATURE_NAME: numpy.array([270, 10, 200, 310], dtype=float),
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME:
        numpy.array([300, 50, 0, 1000], dtype=float),
    example_io.SHORTWAVE_DOWN_FLUX_NAME:
        numpy.array([200, 100, 0, 1000], dtype=float),
    example_io.SHORTWAVE_UP_FLUX_NAME:
        numpy.array([150, 75, 0, 1000], dtype=float)
}
NORM_TABLE_NO_HEIGHT = pandas.DataFrame.from_dict(THIS_DICT, orient='index')

COLUMN_DICT_OLD_TO_NEW = {
    0: normalization.MEAN_VALUE_COLUMN,
    1: normalization.STANDARD_DEVIATION_COLUMN,
    2: normalization.MIN_VALUE_COLUMN,
    3: normalization.MAX_VALUE_COLUMN
}
NORM_TABLE_NO_HEIGHT.rename(columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

THESE_ZENITH_ANGLES = numpy.array([-3, 1, 5, 9], dtype=float)
THESE_LATITUDES_DEG_N = numpy.array([-0.498, -0.498, -0.498, -0.498])
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES_DEG_N)
))

THIS_TEMPERATURE_MATRIX = numpy.array([
    [2, 2.5],
    [1.9, 2.4],
    [1.8, 2.3],
    [1.7, 2.25]
])
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([-2, -2, -2, -2], dtype=float)
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [1, 0],
    [3, 1],
    [2.5, 2.5],
    [0, -1]
])

THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [0, 0],
    [2, 0],
    [6, 8],
    [10, -2]
], dtype=float) / 3

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)

EXAMPLE_DICT_Z_PREDICTORS_NO_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_TARGETS_NO_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_BOTH_NO_HEIGHT = {
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

THESE_ZENITH_ANGLES = numpy.array([0, 1, 2, 3], dtype=float) / 1.5
THESE_LATITUDES_DEG_N = numpy.array([130.02, 130.02, 130.02, 130.02]) / 180
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES_DEG_N)
))

THIS_TEMPERATURE_MATRIX = numpy.array([
    [90, 95],
    [89, 94],
    [88, 93],
    [87, 92.5]
]) / 110
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([0.2, 0.2, 0.2, 0.2])
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [0.3, 0.2],
    [0.5, 0.3],
    [0.45, 0.45],
    [0.2, 0.1]
])

THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [0.15, 0.15],
    [0.2, 0.15],
    [0.3, 0.35],
    [0.4, 0.1]
])

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)

EXAMPLE_DICT_MINMAX_PREDICTORS_NO_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_TARGETS_NO_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_BOTH_NO_HEIGHT = {
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

DUMMY_HEIGHT_M_AGL = normalization.DUMMY_HEIGHT_M_AGL

THIS_DICT = {
    (example_io.ZENITH_ANGLE_NAME, DUMMY_HEIGHT_M_AGL):
        numpy.array([0.75, 0.25, 0, 1.5]),
    (example_io.LATITUDE_NAME, DUMMY_HEIGHT_M_AGL):
        numpy.array([45, 10, -90, 90], dtype=float),
    (example_io.TEMPERATURE_NAME, 100):
        numpy.array([295, 15, 220, 310], dtype=float),
    (example_io.TEMPERATURE_NAME, 500): numpy.array([290, 12.5, 215, 305]),
    (example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME, DUMMY_HEIGHT_M_AGL):
        numpy.array([300, 50, 0, 1000], dtype=float),
    (example_io.SHORTWAVE_DOWN_FLUX_NAME, 100):
        numpy.array([300, 150, 0, 1000], dtype=float),
    (example_io.SHORTWAVE_DOWN_FLUX_NAME, 500):
        numpy.array([275, 125, 0, 800], dtype=float),
    (example_io.SHORTWAVE_UP_FLUX_NAME, 100):
        numpy.array([200, 100, 0, 1000], dtype=float),
    (example_io.SHORTWAVE_UP_FLUX_NAME, 500):
        numpy.array([175, 75, 0, 900], dtype=float)
}
NORM_TABLE_WITH_HEIGHT = pandas.DataFrame.from_dict(THIS_DICT, orient='index')
NORM_TABLE_WITH_HEIGHT.rename(columns=COLUMN_DICT_OLD_TO_NEW, inplace=True)

THESE_ZENITH_ANGLES = numpy.array([-3, 1, 5, 9], dtype=float)
THESE_LATITUDES_DEG_N = numpy.array([-0.498, -0.498, -0.498, -0.498])
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES_DEG_N)
))

THIS_TEMPERATURE_MATRIX = numpy.array([
    [-5. / 15, 5. / 12.5],
    [-6. / 15, 4. / 12.5],
    [-7. / 15, 3. / 12.5],
    [-8. / 15, 2.5 / 12.5]
])

THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([-2, -2, -2, -2], dtype=float)
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [0, -75. / 125],
    [4. / 3, 25. / 125],
    [1, 175. / 125],
    [-2. / 3, -175. / 125]
])

THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [-0.5, -1. / 3],
    [0, -1. / 3],
    [1, 7. / 3],
    [2, -1]
], dtype=float)

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)

EXAMPLE_DICT_Z_PREDICTORS_WITH_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_TARGETS_WITH_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_Z_BOTH_WITH_HEIGHT = {
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

THESE_ZENITH_ANGLES = numpy.array([0, 1, 2, 3], dtype=float) / 1.5
THESE_LATITUDES_DEG_N = numpy.array([130.02, 130.02, 130.02, 130.02]) / 180
THIS_SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (THESE_ZENITH_ANGLES, THESE_LATITUDES_DEG_N)
))

THIS_TEMPERATURE_MATRIX = numpy.array([
    [70, 80],
    [69, 79],
    [68, 78],
    [67, 77.5]
]) / 90
THIS_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    THIS_TEMPERATURE_MATRIX, axis=-1
)

THESE_SHORTWAVE_SURFACE_DOWN_FLUXES = numpy.array([0.2, 0.2, 0.2, 0.2])
THIS_SCALAR_TARGET_MATRIX = numpy.reshape(
    THESE_SHORTWAVE_SURFACE_DOWN_FLUXES,
    (len(THESE_SHORTWAVE_SURFACE_DOWN_FLUXES), 1)
)

THIS_SHORTWAVE_DOWN_FLUX_MATRIX = numpy.array([
    [0.3, 0.25],
    [0.5, 0.375],
    [0.45, 4.5 / 8],
    [0.2, 0.125]
])

THIS_SHORTWAVE_UP_FLUX_MATRIX = numpy.array([
    [0.15, 1.5 / 9],
    [0.2, 1.5 / 9],
    [0.3, 3.5 / 9],
    [0.4, 1. / 9]
], dtype=float)

THIS_VECTOR_TARGET_MATRIX = numpy.stack(
    (THIS_SHORTWAVE_DOWN_FLUX_MATRIX, THIS_SHORTWAVE_UP_FLUX_MATRIX), axis=-1
)

EXAMPLE_DICT_MINMAX_PREDICTORS_WITH_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: THIS_SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: THIS_VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_TARGETS_WITH_HEIGHT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: THIS_SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: THIS_VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL
}

EXAMPLE_DICT_MINMAX_BOTH_WITH_HEIGHT = {
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
        example_io.SCALAR_TARGET_VALS_KEY, example_io.VECTOR_TARGET_VALS_KEY
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

    def test_normalize_data_z_both_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for both predictors and
        targets, with *no* separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_BOTH_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_Z_BOTH_NO_HEIGHT),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_z_predictors_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for predictors only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_PREDICTORS_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_Z_PREDICTORS_NO_HEIGHT),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_z_targets_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for targets only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_TARGETS_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_Z_TARGETS_NO_HEIGHT),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_minmax_both_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for both predictors and
        targets, with *no* separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_BOTH_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_MINMAX_BOTH_NO_HEIGHT),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_minmax_predictors_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for predictors only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_PREDICTORS_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_PREDICTORS_NO_HEIGHT),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_minmax_targets_no_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for targets only, with *no*
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_TARGETS_NO_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_MINMAX_TARGETS_NO_HEIGHT),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=False,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_NO_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_z_both_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for both predictors and
        targets, with separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_BOTH_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_Z_BOTH_WITH_HEIGHT),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_z_predictors_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for predictors only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_PREDICTORS_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_Z_PREDICTORS_WITH_HEIGHT),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_z_targets_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using z-score normalization for targets only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_Z_TARGETS_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_Z_TARGETS_WITH_HEIGHT),
            normalization_type_string=normalization.Z_SCORE_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_minmax_both_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for both predictors and
        targets, with separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_BOTH_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_MINMAX_BOTH_WITH_HEIGHT),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_minmax_predictors_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for predictors only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_PREDICTORS_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=
            copy.deepcopy(EXAMPLE_DICT_MINMAX_PREDICTORS_WITH_HEIGHT),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=True, apply_to_targets=False,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))

    def test_normalize_data_minmax_targets_with_height(self):
        """Ensures correct output from normalize_data.

        In this case, using min-max normalization for targets only, with
        separation by height.
        """

        this_example_dict = normalization.normalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_DENORM),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_MINMAX_TARGETS_WITH_HEIGHT
        ))

        this_example_dict = normalization.denormalize_data(
            example_dict=copy.deepcopy(EXAMPLE_DICT_MINMAX_TARGETS_WITH_HEIGHT),
            normalization_type_string=normalization.MINMAX_NORM_STRING,
            normalization_file_name=None, separate_heights=True,
            min_normalized_value=MIN_NORMALIZED_VALUE,
            max_normalized_value=MAX_NORMALIZED_VALUE,
            apply_to_predictors=False, apply_to_targets=True,
            test_mode=True, normalization_table=NORM_TABLE_WITH_HEIGHT
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, EXAMPLE_DICT_DENORM
        ))


if __name__ == '__main__':
    unittest.main()
