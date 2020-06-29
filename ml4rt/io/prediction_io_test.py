"""Unit tests for example_io.py."""

import copy
import unittest
import numpy
from ml4rt.io import example_io
from ml4rt.io import prediction_io

TOLERANCE = 1e-6

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

DUMMY_PREDICTOR_MATRIX = numpy.array([
    [0, 1, 0.5],
    [2, 3, 0.6],
    [4, 5, 0.7],
    [6, 7, 0.8],
    [8, 9, 0.9]
])

DUMMY_STANDARD_ATMO_ENUMS = numpy.array([
    example_io.MIDLATITUDE_WINTER_ENUM, example_io.MIDLATITUDE_SUMMER_ENUM,
    example_io.TROPICS_ENUM, example_io.MIDLATITUDE_WINTER_ENUM,
    example_io.SUBARCTIC_SUMMER_ENUM
])

DUMMY_EXAMPLE_DICT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: [
        example_io.LATITUDE_NAME, example_io.LONGITUDE_NAME,
        example_io.ZENITH_ANGLE_NAME
    ],
    example_io.SCALAR_PREDICTOR_VALS_KEY: DUMMY_PREDICTOR_MATRIX,
    example_io.VALID_TIMES_KEY:
        numpy.array([1e9, 1.1e9, 1.2e9, 1.3e9, 1.4e9], dtype=int),
    example_io.STANDARD_ATMO_FLAGS_KEY: DUMMY_STANDARD_ATMO_ENUMS
}

EXAMPLE_ID_STRINGS = example_io.create_example_ids(DUMMY_EXAMPLE_DICT)
MODEL_FILE_NAME = 'foo'

PREDICTION_DICT = {
    prediction_io.SCALAR_TARGETS_KEY: SCALAR_TARGET_MATRIX,
    prediction_io.SCALAR_PREDICTIONS_KEY: SCALAR_PREDICTION_MATRIX,
    prediction_io.VECTOR_TARGETS_KEY: VECTOR_TARGET_MATRIX,
    prediction_io.VECTOR_PREDICTIONS_KEY: VECTOR_PREDICTION_MATRIX,
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: EXAMPLE_ID_STRINGS
}

# The following constants are used to test subset_by_standard_atmo.
STANDARD_ATMO_ENUM = example_io.MIDLATITUDE_WINTER_ENUM + 0
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
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
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
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
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
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME,
    prediction_io.EXAMPLE_IDS_KEY: [
        EXAMPLE_ID_STRINGS[k] for k in THESE_INDICES
    ]
}

# The following constants are used to test average_predictions.
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
    prediction_io.MODEL_FILE_KEY: MODEL_FILE_NAME
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
        prediction_io.VECTOR_TARGETS_KEY, prediction_io.VECTOR_PREDICTIONS_KEY
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

    def test_average_predictions(self):
        """Ensures correct output from average_predictions."""

        this_prediction_dict = prediction_io.average_predictions(
            prediction_dict=copy.deepcopy(PREDICTION_DICT), use_pmm=False
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

    def test_subset_by_month(self):
        """Ensures correct output from subset_by_month."""

        this_prediction_dict = prediction_io.subset_by_month(
            prediction_dict=copy.deepcopy(PREDICTION_DICT),
            desired_month=DESIRED_MONTH
        )
        self.assertTrue(_compare_prediction_dicts(
            this_prediction_dict, PREDICTION_DICT_SUBSET_BY_MONTH
        ))


if __name__ == '__main__':
    unittest.main()
