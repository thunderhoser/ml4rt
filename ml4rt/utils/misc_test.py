"""Unit tests for misc.py."""

import unittest
import numpy
from ml4rt.utils import misc

TOLERANCE = 1e-6

# The following constants are used to test create_latlng_grid.
MIN_GRID_LATITUDE_DEG = 49.123
MAX_GRID_LATITUDE_DEG = 59.321
MIN_GRID_LONGITUDE_DEG = 240.567
MAX_GRID_LONGITUDE_DEG = -101.789
LATITUDE_SPACING_DEG = 1.
LONGITUDE_SPACING_DEG = 2.

GRID_POINT_LATITUDES_DEG = numpy.array(
    [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60], dtype=float
)
GRID_POINT_LONGITUDES_DEG = numpy.array(
    [240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260], dtype=float
)

# The following constants are used to test find_best_and_worst_predictions.
BIAS_MATRIX = numpy.array([
    [-6, 1, 4, 1, 10],
    [9, 6, 7, -8, 7],
    [-7, -5, -10, 3, -5],
    [4, -9, 2, -4, 5],
    [-8, -6, -4, -1, -2],
    [-4, -8, -9, 2, 0],
    [-2, 8, -8, 0, -8],
    [5, 7, -8, 6, -8],
    [5, 5, -10, 1, -3],
    [-3, 2, 0, 3, 6],
    [-2, -6, -6, -8, 5],
    [-1, 3, 4, 4, -7],
    [8, -9, 4, 6, -5],
    [-1, -5, 2, 4, -3],
    [-9, 1, 2, 8, 9]
], dtype=float)

NUM_EXAMPLES_PER_SET = 5
HIGH_BIAS_INDICES = numpy.array([0, 1, 14, 6, 12], dtype=int)
LOW_BIAS_INDICES = numpy.array([2, 8, 3, 5, 12], dtype=int)
LOW_ABS_ERROR_INDICES = numpy.array([13, 9, 11, 4, 6], dtype=int)


class MiscTests(unittest.TestCase):
    """Each method is a unit test for misc.py."""

    def test_create_latlng_grid(self):
        """Ensures correct output from create_latlng_grid."""

        these_latitudes_deg, these_longitudes_deg = misc.create_latlng_grid(
            min_latitude_deg=MIN_GRID_LATITUDE_DEG,
            max_latitude_deg=MAX_GRID_LATITUDE_DEG,
            latitude_spacing_deg=LATITUDE_SPACING_DEG,
            min_longitude_deg=MIN_GRID_LONGITUDE_DEG,
            max_longitude_deg=MAX_GRID_LONGITUDE_DEG,
            longitude_spacing_deg=LONGITUDE_SPACING_DEG
        )

        self.assertTrue(numpy.allclose(
            these_latitudes_deg, GRID_POINT_LATITUDES_DEG, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_longitudes_deg, GRID_POINT_LONGITUDES_DEG, atol=TOLERANCE
        ))

    def test_find_best_and_worst_predictions(self):
        """Ensures correct output from find_best_and_worst_predictions."""

        (
            these_high_bias_indices,
            these_low_bias_indices,
            these_low_abs_error_indices
        ) = misc.find_best_and_worst_predictions(
            bias_matrix=BIAS_MATRIX,
            absolute_error_matrix=numpy.absolute(BIAS_MATRIX),
            num_examples_per_set=NUM_EXAMPLES_PER_SET
        )

        self.assertTrue(numpy.array_equal(
            these_high_bias_indices, HIGH_BIAS_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_low_bias_indices, LOW_BIAS_INDICES
        ))
        self.assertTrue(numpy.array_equal(
            these_low_abs_error_indices, LOW_ABS_ERROR_INDICES
        ))


if __name__ == '__main__':
    unittest.main()
