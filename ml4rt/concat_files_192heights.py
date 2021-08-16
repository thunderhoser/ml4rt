"""Concatenates example files with 192 heights."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import example_utils

# YEARS = numpy.array([2018, 2019], dtype=int)
YEARS = numpy.array([2019], dtype=int)

EXAMPLE_DIR_NAME = (
    '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist/ml4rt_project/examples/'
    '192heights/non_tropical_sites'
)


def _run():
    """Concatenates example files with 192 heights.

    This is effectively the main method.
    """

    for this_year in YEARS:
        input_file_names = [
            '{0:s}/learning_examples_{1:d}{2:d}.nc'.format(
                EXAMPLE_DIR_NAME, this_year, i
            ) for i in range(4)
        ]

        output_file_name = '{0:s}/learning_examples_{1:d}.nc'.format(
            EXAMPLE_DIR_NAME, this_year
        )

        example_dicts = []

        for this_file_name in input_file_names:
            print('Reading data from: "{0:s}"...'.format(this_file_name))
            example_dicts.append(
                example_io.read_file(this_file_name)
            )

        print('Concatenating dictionaries...')
        example_dict = example_utils.concat_examples(example_dicts)
        del example_dicts

        print('Writing data to file: "{0:s}"...'.format(output_file_name))
        example_io.write_file(
            example_dict=example_dict, netcdf_file_name=output_file_name
        )


if __name__ == '__main__':
    _run()
