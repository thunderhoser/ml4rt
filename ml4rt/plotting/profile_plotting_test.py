"""Unit tests for profile_plotting.py."""

import unittest
import numpy
from ml4rt.plotting import profile_plotting

TICK_VALUES_KM_AGL = numpy.array([
    0.0001, 0.0025, 0.005, 0.001, 0.07, 0.1, 0.5, 0.999, 1.1,
    5.5, 8.8, 12.2, 50, 100, 1000
])

TICK_STRINGS = [
    '0.0001', '0.003', '0.005', '0.001', '0.07', '0.1', '0.5', '1.0', '1.1',
    '5.5', '8.8', '12.2', '50.0', '100.0', '1000.0'
]


class ProfilePlottingTests(unittest.TestCase):
    """Each method is a unit test for profile_plotting.py."""

    def test_create_log_height_labels(self):
        """Ensures correct output from create_log_height_labels."""

        these_tick_strings = profile_plotting.create_log_height_labels(
            TICK_VALUES_KM_AGL
        )
        self.assertTrue(these_tick_strings == TICK_STRINGS)


if __name__ == '__main__':
    unittest.main()
