"""Unit tests for example_io.py."""

import copy
import unittest
import numpy
from ml4rt.io import example_io

TOLERANCE = 1e-6

# The following constants are used to test find_file.
EXAMPLE_DIR_NAME = 'foo'
YEAR = 2018
EXAMPLE_FILE_NAME = 'foo/radiative_transfer_examples_2018.nc'

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
NUM_EXAMPLES_TO_KEEP = 2

FIRST_EXAMPLE_DICT_REDUCED = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY:
        FIRST_SCALAR_PREDICTOR_MATRIX[:2, ...],
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY:
        FIRST_VECTOR_PREDICTOR_MATRIX[:2, ...],
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: FIRST_SCALAR_TARGET_MATRIX[:2, ...],
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: FIRST_VECTOR_TARGET_MATRIX[:2, ...],
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: FIRST_TIMES_UNIX_SEC[:2, ...],
    example_io.STANDARD_ATMO_FLAGS_KEY: FIRST_STANDARD_ATMO_FLAGS[:2, ...]
}

# The following constants are used to test subset_by_time.
FIRST_TIME_UNIX_SEC = 1
LAST_TIME_UNIX_SEC = 600

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
        if first_example_dict[this_key] != second_example_dict[this_key]:
            return False

    return True


class ExampleIoTests(unittest.TestCase):
    """Each method is a unit test for example_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = example_io.find_file(
            example_dir_name=EXAMPLE_DIR_NAME, year=YEAR,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == EXAMPLE_FILE_NAME)

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

    def test_reduce_sample_size(self):
        """Ensures correct output from reduce_sample_size."""

        this_example_dict = example_io.reduce_sample_size(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            num_examples_to_keep=NUM_EXAMPLES_TO_KEEP, test_mode=True
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_REDUCED
        ))

    def test_subset_by_time(self):
        """Ensures correct output from subset_by_time."""

        this_example_dict = example_io.subset_by_time(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT),
            first_time_unix_sec=FIRST_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_TIME_UNIX_SEC
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_SELECT_TIMES
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

    def test_average_examples(self):
        """Ensures correct output from average_examples."""

        this_example_dict = example_io.average_examples(
            example_dict=copy.deepcopy(FIRST_EXAMPLE_DICT), use_pmm=False
        )

        self.assertTrue(_compare_example_dicts(
            this_example_dict, FIRST_EXAMPLE_DICT_AVERAGE
        ))


if __name__ == '__main__':
    unittest.main()
