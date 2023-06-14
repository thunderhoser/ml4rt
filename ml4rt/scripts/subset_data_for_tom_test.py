"""Unit tests for subset_data_for_tom.py."""

import unittest
import numpy
from gewittergefahr.gg_utils import time_conversion
from ml4rt.scripts import subset_data_for_tom

VALID_TIME_STRINGS = [
    '2024-01-01-000000', '2024-02-29-235959', '2024-03-01-000000',
    '2024-04-04-040404', '2024-05-31-235959', '2024-06-01-000000',
    '2024-07-07-070707', '2024-08-31-235959', '2024-09-01-000000',
    '2024-10-10-101010', '2024-11-30-235959', '2024-12-01-000000'
]

VALID_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y-%m-%d-%H%M%S')
    for t in VALID_TIME_STRINGS
], dtype=int)

SEASON_INDICES = numpy.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0], dtype=int)


class SubsetDataForTomTests(unittest.TestCase):
    """Each method is a unit test for subset_data_for_tom.py."""

    def test_time_to_season(self):
        """Ensures correct output from _time_to_season."""

        for i in range(len(VALID_TIMES_UNIX_SEC)):
            this_season_index = subset_data_for_tom._time_to_season(
                VALID_TIMES_UNIX_SEC[i]
            )
            self.assertTrue(this_season_index == SEASON_INDICES[i])


if __name__ == '__main__':
    unittest.main()
