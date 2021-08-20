"""Unit tests for heating_rate_interp.py"""

import unittest
import numpy
from ml4rt.utils import heating_rate_interp

TOLERANCE = 1e-6
HALF_WINDOW_SIZE_PX = 2

ORIG_HEATING_RATE_MATRIX_K_DAY01 = numpy.array([
    [0, 0, 0],
    [1, 10, 9],
    [2, 10, 0],
    [3, 10, 0],
    [4, 20, 0],
    [5, 20, 0],
    [5, 20, 0],
    [4, 30, 16],
    [3, 30, 0],
    [2, 30, 0]
], dtype=float)

MAX_HEATING_RATE_MATRIX_K_DAY01 = numpy.array([
    [2, 10, 9],
    [3, 10, 9],
    [4, 20, 9],
    [5, 20, 9],
    [5, 20, 0],
    [5, 30, 16],
    [5, 30, 16],
    [5, 30, 16],
    [5, 30, 16],
    [4, 30, 16]
], dtype=float)

ORIG_HEIGHTS_METRES = numpy.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float
)
FIRST_NEW_HEIGHTS_METRES = ORIG_HEIGHTS_METRES + 0.
FIRST_NEW_MASK_FLAG_MATRIX = numpy.array([
    [0, 0, 0],
    [0, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 1, 0],
    [0, 1, 0]
], dtype=bool)

ORIG_HEATING_RATE_MATRIX_K_DAY01 = numpy.transpose(
    ORIG_HEATING_RATE_MATRIX_K_DAY01
)
MAX_HEATING_RATE_MATRIX_K_DAY01 = numpy.transpose(
    MAX_HEATING_RATE_MATRIX_K_DAY01
)
FIRST_NEW_MASK_FLAG_MATRIX = numpy.transpose(FIRST_NEW_MASK_FLAG_MATRIX)

SECOND_NEW_HEIGHTS_METRES = numpy.array([0, 2.5, 5, 7.5, 10])
SECOND_NEW_MASK_FLAG_MATRIX = numpy.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0, 1, 0]
], dtype=bool)

SECOND_NEW_MASK_FLAG_MATRIX = numpy.transpose(SECOND_NEW_MASK_FLAG_MATRIX)


class HeatingRateInterpTests(unittest.TestCase):
    """Each method is a unit test for heating_rate_interp.py."""

    def test_find_local_maxima_first(self):
        """Ensures correct output from _find_local_maxima.

        In this case, using first set of inputs.
        """

        this_mask_matrix, this_heating_rate_matrix_k_day01 = (
            heating_rate_interp._find_local_maxima(
                orig_heating_rate_matrix_k_day01=
                ORIG_HEATING_RATE_MATRIX_K_DAY01,
                orig_heights_m_agl=ORIG_HEIGHTS_METRES,
                new_heights_m_agl=FIRST_NEW_HEIGHTS_METRES,
                half_window_size_px=HALF_WINDOW_SIZE_PX
            )
        )

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, FIRST_NEW_MASK_FLAG_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_heating_rate_matrix_k_day01, MAX_HEATING_RATE_MATRIX_K_DAY01,
            atol=TOLERANCE
        ))

    def test_find_local_maxima_second(self):
        """Ensures correct output from _find_local_maxima.

        In this case, using second set of inputs.
        """

        this_mask_matrix, this_heating_rate_matrix_k_day01 = (
            heating_rate_interp._find_local_maxima(
                orig_heating_rate_matrix_k_day01=
                ORIG_HEATING_RATE_MATRIX_K_DAY01,
                orig_heights_m_agl=ORIG_HEIGHTS_METRES,
                new_heights_m_agl=SECOND_NEW_HEIGHTS_METRES,
                half_window_size_px=HALF_WINDOW_SIZE_PX
            )
        )

        self.assertTrue(numpy.array_equal(
            this_mask_matrix, SECOND_NEW_MASK_FLAG_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_heating_rate_matrix_k_day01, MAX_HEATING_RATE_MATRIX_K_DAY01,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
