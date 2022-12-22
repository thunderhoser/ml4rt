"""Unit tests for prediction_io.py."""

import copy
import unittest
import numpy
from ml4rt.io import prediction_io
from ml4rt.utils import example_utils

TOLERANCE = 1e-6

# The following constants are used for many unit tests.
SCALAR_TARGET_MATRIX = numpy.array([
    [0, 100],
    [100, 200],
    [300, 300],
    [400, 150],
    [25, 500]
], dtype=float)

SCALAR_PREDICTION_MATRIX = numpy.array([
    [50, 75],
    [75, 100],
    [250, 350],
    [400, 200],
    [500, 900]
], dtype=float)

VECTOR_TARGET_MATRIX_FIELD1 = numpy.array([
    [1, 2, 3],
    [3, 4, 5],
    [5, 6, 7],
    [7, 8, 9],
    [9, 10, 11]
], dtype=float)

VECTOR_TARGET_MATRIX_FIELD2 = numpy.array([
    [100, 125, 100],
    [150, 125, 150],
    [300, 100, 75],
    [400, 200, 200],
    [300, 350, 350]
], dtype=float)

HEIGHTS_M_AGL = numpy.array([10, 500, 1000], dtype=float)

VECTOR_TARGET_MATRIX = numpy.stack(
    (VECTOR_TARGET_MATRIX_FIELD1, VECTOR_TARGET_MATRIX_FIELD2), axis=-1
)

VECTOR_PREDICTION_MATRIX_FIELD1 = numpy.array([
    [3, 2, 1],
    [3, 3, 3],
    [6, 6, 6],
    [8, 7, 9],
    [12, 12, 12]
], dtype=float)

VECTOR_PREDICTION_MATRIX_FIELD2 = numpy.array([
    [100, 100, 100],
    [150, 150, 150],
    [250, 150, 100],
    [350, 300, 250],
    [400, 500, 450]
], dtype=float)

VECTOR_PREDICTION_MATRIX = numpy.stack(
    (VECTOR_PREDICTION_MATRIX_FIELD1, VECTOR_PREDICTION_MATRIX_FIELD2), axis=-1
)

DUMMY_SCALAR_PREDICTOR_MATRIX = numpy.array([
    [0, 1, 0, 0.5],
    [2, 3, 0.25, 0.6],
    [4, 5, 0.5, 0.7],
    [6, 7, 0.75, 0.8],
    [8, 9, 1, 0.9]
])

DUMMY_STANDARD_ATMO_ENUMS = numpy.array([
    example_utils.MIDLATITUDE_WINTER_ENUM,
    example_utils.MIDLATITUDE_SUMMER_ENUM,
    example_utils.TROPICS_ENUM, example_utils.MIDLATITUDE_WINTER_ENUM,
    example_utils.SUBARCTIC_SUMMER_ENUM
])

DUMMY_VECTOR_PREDICTOR_MATRIX = numpy.array([0, 1, 2, 3, 4], dtype=float)
DUMMY_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    DUMMY_VECTOR_PREDICTOR_MATRIX, axis=-1
)
DUMMY_VECTOR_PREDICTOR_MATRIX = numpy.expand_dims(
    DUMMY_VECTOR_PREDICTOR_MATRIX, axis=-1
)

DUMMY_EXAMPLE_DICT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: [
        example_utils.LATITUDE_NAME, example_utils.LONGITUDE_NAME,
        example_utils.ALBEDO_NAME, example_utils.ZENITH_ANGLE_NAME
    ],
    example_utils.SCALAR_PREDICTOR_VALS_KEY: DUMMY_SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: [example_utils.TEMPERATURE_NAME],
    example_utils.VECTOR_PREDICTOR_VALS_KEY: DUMMY_VECTOR_PREDICTOR_MATRIX,
    example_utils.HEIGHTS_KEY: numpy.array([10.]),
    example_utils.VALID_TIMES_KEY:
        numpy.array([1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9], dtype=int),
    example_utils.STANDARD_ATMO_FLAGS_KEY: DUMMY_STANDARD_ATMO_ENUMS
}

EXAMPLE_ID_STRINGS = example_utils.create_example_ids(DUMMY_EXAMPLE_DICT)
MODEL_FILE_NAME = 'foo'
ISOTONIC_MODEL_FILE_NAME = 'bar'
NORMALIZATION_FILE_NAME = 'moo'

PREDICTION_DICT = {
    prediction_io.SCALAR_TARGETS_KEY: SCALAR_TARGET_MATRIX,
    prediction_io.SCALAR_PREDICTIONS_KEY: SCALAR_PREDICTION_MATRIX,
    prediction_io.VECTOR_TARGETS_KEY: VECTOR_TARGET_MATRIX,
    prediction_io.VECTOR_PREDICTIONS_KEY: VECTOR_PREDICTION_MATRIX,
    prediction_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.ISOTONIC_MODEL_FILE_KEY: ISOTONIC_MODEL_FILE_NAME,
    prediction_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: EXAMPLE_ID_STRINGS
}

# The following constants are used to test subset_by_standard_atmo.
STANDARD_ATMO_ENUM = example_utils.MIDLATITUDE_WINTER_ENUM + 0
THESE_INDICES = numpy.array([0, 3], dtype=int)

PREDICTION_DICT_SUBSET_BY_ATMO = {
    prediction_io.SCALAR_TARGETS_KEY:
        SCALAR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.SCALAR_PREDICTIONS_KEY:
        SCALAR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_TARGETS_KEY:
        VECTOR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_PREDICTIONS_KEY:
        VECTOR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.ISOTONIC_MODEL_FILE_KEY: ISOTONIC_MODEL_FILE_NAME,
    prediction_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: [
        EXAMPLE_ID_STRINGS[k] for k in THESE_INDICES
    ]
}

# The following constants are used to test subset_by_zenith_angle.
MIN_ZENITH_ANGLE_RAD = 0.666
MAX_ZENITH_ANGLE_RAD = 0.9
THESE_INDICES = numpy.array([2, 3], dtype=int)

PREDICTION_DICT_SUBSET_BY_ANGLE = {
    prediction_io.SCALAR_TARGETS_KEY:
        SCALAR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.SCALAR_PREDICTIONS_KEY:
        SCALAR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_TARGETS_KEY:
        VECTOR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_PREDICTIONS_KEY:
        VECTOR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.ISOTONIC_MODEL_FILE_KEY: ISOTONIC_MODEL_FILE_NAME,
    prediction_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: [
        EXAMPLE_ID_STRINGS[k] for k in THESE_INDICES
    ]
}

# The following constants are used to test subset_by_albedo.
MIN_ALBEDO = 0.5
MAX_ALBEDO = 1
THESE_INDICES = numpy.array([2, 3, 4], dtype=int)

PREDICTION_DICT_SUBSET_BY_ALBEDO = {
    prediction_io.SCALAR_TARGETS_KEY:
        SCALAR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.SCALAR_PREDICTIONS_KEY:
        SCALAR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_TARGETS_KEY:
        VECTOR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_PREDICTIONS_KEY:
        VECTOR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.ISOTONIC_MODEL_FILE_KEY: ISOTONIC_MODEL_FILE_NAME,
    prediction_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: [
        EXAMPLE_ID_STRINGS[k] for k in THESE_INDICES
    ]
}

# The following constants are used to test subset_by_month.
DESIRED_MONTH = 11
THESE_INDICES = numpy.array([1], dtype=int)

PREDICTION_DICT_SUBSET_BY_MONTH = {
    prediction_io.SCALAR_TARGETS_KEY:
        SCALAR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.SCALAR_PREDICTIONS_KEY:
        SCALAR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_TARGETS_KEY:
        VECTOR_TARGET_MATRIX[THESE_INDICES, ...],
    prediction_io.VECTOR_PREDICTIONS_KEY:
        VECTOR_PREDICTION_MATRIX[THESE_INDICES, ...],
    prediction_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.ISOTONIC_MODEL_FILE_KEY: ISOTONIC_MODEL_FILE_NAME,
    prediction_io.NORMALIZATION_FILE_KEY: NORMALIZATION_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: [
        EXAMPLE_ID_STRINGS[k] for k in THESE_INDICES
    ]
}

# The following constants are used to test average_predictions_many_examples.
MEAN_SCALAR_TARGET_MATRIX = numpy.array([[165, 250]], dtype=float)
MEAN_SCALAR_PREDICTION_MATRIX = numpy.array([[255, 325]], dtype=float)

MEAN_TARGET_MATRIX_FIELD1 = numpy.array([[5, 6, 7]], dtype=float)
MEAN_TARGET_MATRIX_FIELD2 = numpy.array([[250, 180, 175]], dtype=float)
MEAN_VECTOR_TARGET_MATRIX = numpy.stack(
    (MEAN_TARGET_MATRIX_FIELD1, MEAN_TARGET_MATRIX_FIELD2), axis=-1
)

MEAN_PREDICTION_MATRIX_FIELD1 = numpy.array([[6.4, 6, 6.2]])
MEAN_PREDICTION_MATRIX_FIELD2 = numpy.array([[250, 240, 210]], dtype=float)
MEAN_VECTOR_PREDICTION_MATRIX = numpy.stack(
    (MEAN_PREDICTION_MATRIX_FIELD1, MEAN_PREDICTION_MATRIX_FIELD2), axis=-1
)

MEAN_PREDICTION_DICT = {
    prediction_io.SCALAR_TARGETS_KEY: MEAN_SCALAR_TARGET_MATRIX,
    prediction_io.SCALAR_PREDICTIONS_KEY: MEAN_SCALAR_PREDICTION_MATRIX,
    prediction_io.VECTOR_TARGETS_KEY: MEAN_VECTOR_TARGET_MATRIX,
    prediction_io.VECTOR_PREDICTIONS_KEY: MEAN_VECTOR_PREDICTION_MATRIX,
    prediction_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME
}

# The following constants are used to test find_file and file_name_to_metadata.
DIRECTORY_NAME = 'foo'
GRID_ROW = 0
GRID_COLUMN = 666
ZENITH_ANGLE_BIN = 99
ALBEDO_BIN = 1
FILE_MONTH = 12

FILE_NAME_DEFAULT = 'foo/predictions.nc'
FILE_NAME_SPATIAL = (
    'foo/grid-row=000/predictions_grid-row=000_grid-column=666.nc'
)
FILE_NAME_ANGULAR = 'foo/predictions_zenith-angle-bin=099.nc'
FILE_NAME_ALBEDO = 'foo/predictions_albedo-bin=001.nc'
FILE_NAME_MONTHLY = 'foo/predictions_month=12.nc'

METADATA_DICT_DEFAULT = {
    prediction_io.ZENITH_ANGLE_BIN_KEY: None,
    prediction_io.ALBEDO_BIN_KEY: None,
    prediction_io.SURFACE_TEMP_BIN_KEY: None,
    prediction_io.SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_TOA_UP_FLUX_BIN_KEY: None,
    prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY: None,
    prediction_io.MONTH_KEY: None,
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}

METADATA_DICT_SPATIAL = {
    prediction_io.ZENITH_ANGLE_BIN_KEY: None,
    prediction_io.ALBEDO_BIN_KEY: None,
    prediction_io.SURFACE_TEMP_BIN_KEY: None,
    prediction_io.SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_TOA_UP_FLUX_BIN_KEY: None,
    prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY: None,
    prediction_io.MONTH_KEY: None,
    prediction_io.GRID_ROW_KEY: 0,
    prediction_io.GRID_COLUMN_KEY: 666
}

METADATA_DICT_ANGULAR = {
    prediction_io.ZENITH_ANGLE_BIN_KEY: 99,
    prediction_io.ALBEDO_BIN_KEY: None,
    prediction_io.SURFACE_TEMP_BIN_KEY: None,
    prediction_io.SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_TOA_UP_FLUX_BIN_KEY: None,
    prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY: None,
    prediction_io.MONTH_KEY: None,
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}

METADATA_DICT_ALBEDO = {
    prediction_io.ZENITH_ANGLE_BIN_KEY: None,
    prediction_io.ALBEDO_BIN_KEY: 1,
    prediction_io.SURFACE_TEMP_BIN_KEY: None,
    prediction_io.SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_TOA_UP_FLUX_BIN_KEY: None,
    prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY: None,
    prediction_io.MONTH_KEY: None,
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}

METADATA_DICT_MONTHLY = {
    prediction_io.ZENITH_ANGLE_BIN_KEY: None,
    prediction_io.ALBEDO_BIN_KEY: None,
    prediction_io.SURFACE_TEMP_BIN_KEY: None,
    prediction_io.SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_SFC_DOWN_FLUX_BIN_KEY: None,
    prediction_io.LONGWAVE_TOA_UP_FLUX_BIN_KEY: None,
    prediction_io.AEROSOL_OPTICAL_DEPTH_BIN_KEY: None,
    prediction_io.MONTH_KEY: 12,
    prediction_io.GRID_ROW_KEY: None,
    prediction_io.GRID_COLUMN_KEY: None
}


def _compare_prediction_dicts(first_prediction_dict, second_prediction_dict):
    """Compares two dictionaries with predicted and actual target values.

    :param first_prediction_dict: See doc for `prediction_io.read_file`.
    :param second_prediction_dict: Same.
    :return: are_dicts_equal: Boolean flag.
    """

    first_keys = list(first_prediction_dict.keys())
    second_keys = list(first_prediction_dict.keys())
    if set(first_keys) != set(second_keys):
        return False

    keys_to_compare = [
        prediction_io.SCALAR_TARGETS_KEY, prediction_io.SCALAR_PREDICTIONS_KEY,
        prediction_io.VECTOR_TARGETS_KEY, prediction_io.VECTOR_PREDICTIONS_KEY,
        prediction_io.HEIGHTS_KEY
    ]

    for this_key in keys_to_compare:
        if not numpy.allclose(
                first_prediction_dict[this_key],
                second_prediction_dict[this_key],
                atol=TOLERANCE
        ):
            return False

    if (
            first_prediction_dict[prediction_io.MODEL_FILE_KEY] !=
            second_prediction_dict[prediction_io.MODEL_FILE_KEY]
    ):
        return False

    if prediction_io.EXAMPLE_IDS_KEY not in first_keys:
        return True

    if (
            first_prediction_dict[prediction_io.EXAMPLE_IDS_KEY] !=
            second_prediction_dict[prediction_io.EXAMPLE_IDS_KEY]
    ):
        return False

    return True


class PredictionIoTests(unittest.TestCase):
    """Each method is a unit test for prediction_io.py."""

    def test_average_predictions_many_examples(self):
        """Ensures correct output from average_predictions_many_examples."""

        this_prediction_dict = prediction_io.average_predictions_many_examples(
            prediction_dict=copy.deepcopy(PREDICTION_DICT), use_pmm=False,
            test_mode=True
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, MEAN_PREDICTION_DICT
        ))

    def test_subset_by_standard_atmo(self):
        """Ensures correct output from subset_by_standard_atmo."""

        this_prediction_dict = prediction_io.subset_by_standard_atmo(
            prediction_dict=copy.deepcopy(PREDICTION_DICT),
            standard_atmo_enum=STANDARD_ATMO_ENUM
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_ATMO
        ))

    def test_subset_by_zenith_angle(self):
        """Ensures correct output from subset_by_zenith_angle."""

        this_prediction_dict = prediction_io.subset_by_zenith_angle(
            prediction_dict=copy.deepcopy(PREDICTION_DICT),
            min_zenith_angle_rad=MIN_ZENITH_ANGLE_RAD,
            max_zenith_angle_rad=MAX_ZENITH_ANGLE_RAD
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_ANGLE
        ))

    def test_subset_by_albedo(self):
        """Ensures correct output from subset_by_albedo."""

        this_prediction_dict = prediction_io.subset_by_albedo(
            prediction_dict=copy.deepcopy(PREDICTION_DICT),
            min_albedo=MIN_ALBEDO, max_albedo=MAX_ALBEDO
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_ALBEDO
        ))

    def test_subset_by_month(self):
        """Ensures correct output from subset_by_month."""

        this_prediction_dict = prediction_io.subset_by_month(
            prediction_dict=copy.deepcopy(PREDICTION_DICT),
            desired_month=DESIRED_MONTH
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_MONTH
        ))

    def test_find_file_default(self):
        """Ensures correct output from find_file.

        In this case, using default metadata (no splitting by time or space).
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_DEFAULT)

    def test_find_file_spatial(self):
        """Ensures correct output from find_file.

        In this case, splitting by space.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, grid_row=GRID_ROW,
            grid_column=GRID_COLUMN, raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_SPATIAL)

    def test_find_file_angular(self):
        """Ensures correct output from find_file.

        In this case, splitting by solar zenith angle.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, zenith_angle_bin=ZENITH_ANGLE_BIN,
            raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_ANGULAR)

    def test_find_file_albedo(self):
        """Ensures correct output from find_file.

        In this case, splitting by albedo.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, albedo_bin=ALBEDO_BIN,
            raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_ALBEDO)

    def test_find_file_monthly(self):
        """Ensures correct output from find_file.

        In this case, splitting by month.
        """

        this_file_name = prediction_io.find_file(
            directory_name=DIRECTORY_NAME, month=FILE_MONTH,
            raise_error_if_missing=False
        )
        self.assertTrue(this_file_name == FILE_NAME_MONTHLY)

    def test_file_name_to_metadata_default(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, using default metadata (no splitting by time or space).
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_DEFAULT
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_DEFAULT)

    def test_file_name_to_metadata_spatial(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by space.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_SPATIAL
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_SPATIAL)

    def test_file_name_to_metadata_angular(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by solar zenith angle.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_ANGULAR
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_ANGULAR)

    def test_file_name_to_metadata_albedo(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by albedo.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_ALBEDO
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_ALBEDO)

    def test_file_name_to_metadata_monthly(self):
        """Ensures correct output from file_name_to_metadata.

        In this case, splitting by month.
        """

        this_metadata_dict = prediction_io.file_name_to_metadata(
            FILE_NAME_MONTHLY
        )
        self.assertTrue(this_metadata_dict == METADATA_DICT_MONTHLY)


if __name__ == '__main__':
    unittest.main()
