"""Unit tests for example_io.py."""

import unittest
from ml4rt.io import example_io

FIRST_EXAMPLE_DIR_NAME = 'foo'
FIRST_YEAR = 2018
FIRST_YEAR_PART_NUMBER = None
FIRST_EXAMPLE_FILE_NAME = 'foo/learning_examples_2018.nc'

SECOND_EXAMPLE_DIR_NAME = 'bar'
SECOND_YEAR = 2019
SECOND_YEAR_PART_NUMBER = 3
SECOND_EXAMPLE_FILE_NAME = 'bar/learning_examples_2019_part03.nc'


class ExampleIoTests(unittest.TestCase):
    """Each method is a unit test for example_io.py."""

    def test_find_file_first(self):
        """Ensures correct output from find_file.

        In this case, using first set of input args.
        """

        this_file_name = example_io.find_file(
            directory_name=FIRST_EXAMPLE_DIR_NAME, year=FIRST_YEAR,
            year_part_number=FIRST_YEAR_PART_NUMBER,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == FIRST_EXAMPLE_FILE_NAME)

    def test_file_name_to_year_first(self):
        """Ensures correct output from file_name_to_year.

        In this case, using first set of input args.
        """

        this_year = example_io.file_name_to_year(FIRST_EXAMPLE_FILE_NAME)
        self.assertTrue(this_year == FIRST_YEAR)

    def test_file_name_to_year_part_first(self):
        """Ensures correct output from file_name_to_year_part.

        In this case, using first set of input args.
        """

        this_part_number = example_io.file_name_to_year_part(
            FIRST_EXAMPLE_FILE_NAME
        )

        if FIRST_YEAR_PART_NUMBER is None:
            self.assertTrue(this_part_number is None)
        else:
            self.assertTrue(this_part_number == FIRST_YEAR_PART_NUMBER)

    def test_find_file_second(self):
        """Ensures correct output from find_file.

        In this case, using second set of input args.
        """

        this_file_name = example_io.find_file(
            directory_name=SECOND_EXAMPLE_DIR_NAME, year=SECOND_YEAR,
            year_part_number=SECOND_YEAR_PART_NUMBER,
            raise_error_if_missing=False
        )

        self.assertTrue(this_file_name == SECOND_EXAMPLE_FILE_NAME)

    def test_file_name_to_year_second(self):
        """Ensures correct output from file_name_to_year.

        In this case, using second set of input args.
        """

        this_year = example_io.file_name_to_year(SECOND_EXAMPLE_FILE_NAME)
        self.assertTrue(this_year == SECOND_YEAR)

    def test_file_name_to_year_part_second(self):
        """Ensures correct output from file_name_to_year_part.

        In this case, using second set of input args.
        """

        this_part_number = example_io.file_name_to_year_part(
            SECOND_EXAMPLE_FILE_NAME
        )

        if SECOND_YEAR_PART_NUMBER is None:
            self.assertTrue(this_part_number is None)
        else:
            self.assertTrue(this_part_number == SECOND_YEAR_PART_NUMBER)


if __name__ == '__main__':
    unittest.main()
