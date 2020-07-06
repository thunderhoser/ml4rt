"""Unit tests for profile_plotting.py."""

import unittest
import numpy
from ml4rt.plotting import profile_plotting

TICK_VALUES_KM_AGL = numpy.array([
    0.0001, 0.0025, 0.005, 0.001, 0.07, 0.1, 0.5, 0.999, 1.1,
    5.5, 8.8, 12.2, 50, 100, 1000
])

TICK_STRINGS_LOG_SCALE = [
    '0.0001', '0.003', '0.005', '0.001', '0.07', '0.1', '0.5', '1.0', '1.1',
    '5.5', '8.8', '12.2', '50.0', '100.0', '1000.0'
]

TICK_STRINGS_LINEAR_SCALE = [
    '0.00010', '0.0025', '0.0050', '0.0010', '0.070', '0.10', '0.50', '1.00',
    '1.1', '5.5', '8.8', '12.2', '50.0', '100.0', '1000.0'
]


class ProfilePlottingTests(unittest.TestCase):
    """Each method is a unit test for profile_plotting.py."""

    def test_create_height_labels_log(self):
        """Ensures correct output from create_height_labels.

        In this case, assuming that height axis is logarithmic.
        """

        these_tick_strings = profile_plotting.create_height_labels(
            tick_values_km_agl=TICK_VALUES_KM_AGL, use_log_scale=True
        )
        self.assertTrue(these_tick_strings == TICK_STRINGS_LOG_SCALE)

    def test_create_height_labels_linear(self):
        """Ensures correct output from create_height_labels.

        In this case, assuming that height axis is linear.
        """

        these_tick_strings = profile_plotting.create_height_labels(
            tick_values_km_agl=TICK_VALUES_KM_AGL, use_log_scale=False
        )
        self.assertTrue(these_tick_strings == TICK_STRINGS_LINEAR_SCALE)


if __name__ == '__main__':
    unittest.main()
