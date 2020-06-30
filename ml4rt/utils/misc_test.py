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


if __name__ == '__main__':
    unittest.main()
