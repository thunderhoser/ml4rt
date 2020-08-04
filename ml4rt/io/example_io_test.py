"""Unit tests for example_io.py."""

import unittest
from gewittergefahr.gg_utils import time_conversion
from ml4rt.io import example_io

# The following constants are used to test find_file and file_name_to_year.
EXAMPLE_DIR_NAME = 'foo'
YEAR = 2018
EXAMPLE_FILE_NAME = 'foo/learning_examples_2018.nc'

# The following constants are used to test find_many_files.
FIRST_FILE_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '1999-12-31-235959', '%Y-%m-%d-%H%M%S'
)
LAST_FILE_TIME_UNIX_SEC = time_conversion.string_to_unix_sec(
    '2005-01-01-000000', '%Y-%m-%d-%H%M%S'
)
EXAMPLE_FILE_NAMES = [
    'foo/learning_examples_1999.nc', 'foo/learning_examples_2000.nc',
    'foo/learning_examples_2001.nc', 'foo/learning_examples_2002.nc',
    'foo/learning_examples_2003.nc', 'foo/learning_examples_2004.nc',
    'foo/learning_examples_2005.nc'
]


class ExampleIoTests(unittest.TestCase):
    """Each method is a unit test for example_io.py."""

    def test_find_file(self):
        """Ensures correct output from find_file."""

        this_file_name = example_io.find_file(
            directory_name=EXAMPLE_DIR_NAME, year=YEAR,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == EXAMPLE_FILE_NAME)

    def test_file_name_to_year(self):
        """Ensures correct output from file_name_to_year."""

        this_year = example_io.file_name_to_year(EXAMPLE_FILE_NAME)
        self.assertTrue(this_year == YEAR)

    def test_find_many_files(self):
        """Ensures correct output from find_many_files."""

        these_file_names = example_io.find_many_files(
            directory_name=EXAMPLE_DIR_NAME,
            first_time_unix_sec=FIRST_FILE_TIME_UNIX_SEC,
            last_time_unix_sec=LAST_FILE_TIME_UNIX_SEC,
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=False, test_mode=True
        )

        self.assertTrue(these_file_names == EXAMPLE_FILE_NAMES)


if __name__ == '__main__':
    unittest.main()
