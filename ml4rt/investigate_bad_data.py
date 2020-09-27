"""Investigates bad data."""

import os
import sys
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import example_io
import example_utils

HOME_DIR_NAME = '/scratch1/RDARCH/rda-ghpcs/Ryan.Lagerquist'
EXAMPLE_DIR_NAME = '{0:s}/ml4rt_project/examples/new_locations'.format(
    HOME_DIR_NAME
)

YEARS = numpy.array([2017, 2018, 2019, 2020], dtype=int)


def _run():
    """Investigates bad data (this is effectively the main method)."""

    for this_year in YEARS:
        this_file_name = example_io.find_file(
            directory_name=EXAMPLE_DIR_NAME, year=this_year
        )
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_example_dict = example_io.read_file(this_file_name)

        these_heating_rates_k_day01 = example_utils.get_field_from_dict(
            example_dict=this_example_dict,
            field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
        )
        these_heating_rates_k_day01 = numpy.sort(numpy.ravel(
            these_heating_rates_k_day01
        ))
        print(these_heating_rates_k_day01)


if __name__ == '__main__':
    _run()
