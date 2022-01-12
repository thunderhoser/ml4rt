"""Subsets low-aerosol examples and writes them to a new file."""

import os
import sys
import argparse
import numpy
from scipy.integrate import simps

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_io
import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
YEAR_ARG_NAME = 'year'
MAX_AOD_ARG_NAME = 'max_aerosol_optical_depth'
OUTPUT_DIR_ARG_NAME = 'output_example_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing examples to be subset.  Files therein '
    'will be found by `example_io.find_files_one_year` and read by '
    '`example_io.read_file`.'
)
YEAR_HELP_STRING = 'Will subset examples for this year.'
MAX_AOD_HELP_STRING = (
    'Maximum aerosol optical depth (AOD).  Only examples with <= this threshold'
    ' will be considered "low-aerosol".'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory, where subset examples will be written.  Files '
    'will be written here by `example_io.write_file`, to exact locations '
    'determined by `example_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_AOD_ARG_NAME, type=float, required=True, help=MAX_AOD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_example_dir_name, year, max_aerosol_optical_depth,
         output_example_dir_name):
    """Subsets low-aerosol examples and writes them to a new file.
    
    :param input_example_dir_name: See documentation at top of file.
    :param year: Same.
    :param max_aerosol_optical_depth: Same.
    :param output_example_dir_name: Same.
    """
    
    error_checking.assert_is_greater(max_aerosol_optical_depth, 0.)
    
    input_example_file_names = example_io.find_files_one_year(
        directory_name=input_example_dir_name, year=year,
        raise_error_if_missing=True
    )
    
    for i in range(len(input_example_file_names)):
        print('Reading data from: "{0:s}"...'.format(
            input_example_file_names[i]
        ))
        example_dict = example_io.read_file(
            netcdf_file_name=input_example_file_names[i],
            exclude_summit_greenland=True, max_heating_rate_k_day=numpy.inf
        )

        aerosol_extinction_matrix_metres01 = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_utils.AEROSOL_EXTINCTION_NAME
        )

        if (
                example_utils.HEIGHT_NAME in
                example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
        ):
            height_matrix_m_agl = example_utils.get_field_from_dict(
                example_dict=example_dict, field_name=example_utils.HEIGHT_NAME
            )
        else:
            height_matrix_m_agl = example_dict[example_utils.HEIGHTS_KEY]

        num_examples = aerosol_extinction_matrix_metres01.shape[0]

        if len(height_matrix_m_agl.shape) == 2:
            aerosol_optical_depths = numpy.full(num_examples, numpy.nan)
            print('\n')

            for k in range(num_examples):
                if numpy.mod(k, 1000) == 0:
                    print((
                        'Have computed aerosol optical depth for {0:d} of {1:d}'
                        ' profiles...'
                    ).format(
                        k, num_examples
                    ))

                aerosol_optical_depths[k] = simps(
                    y=aerosol_extinction_matrix_metres01[k, :],
                    x=height_matrix_m_agl[k, :],
                    even='avg'
                )

            print((
                'Have computed aerosol optical depth for all {0:d} profiles!\n'
            ).format(
                num_examples
            ))
        else:
            print((
                'Computing aerosol optical depth for {0:d} profiles...'
            ).format(
                num_examples
            ))
            aerosol_optical_depths = simps(
                y=aerosol_extinction_matrix_metres01, x=height_matrix_m_agl,
                axis=-1, even='avg'
            )

        good_indices = numpy.where(
            aerosol_optical_depths <= max_aerosol_optical_depth
        )[0]
        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_indices
        )

        this_output_file_name = '{0:s}/{1:s}'.format(
            output_example_dir_name,
            os.path.split(input_example_file_names[i])[1]
        )

        print((
            'Writing data for {0:d} low-aerosol examples (AOD <= {1:.4f}) to: '
            '"{2:s}"...'
        ).format(
            len(good_indices), max_aerosol_optical_depth, this_output_file_name
        ))

        example_io.write_file(
            example_dict=example_dict, netcdf_file_name=this_output_file_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_example_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        max_aerosol_optical_depth=getattr(INPUT_ARG_OBJECT, MAX_AOD_ARG_NAME),
        output_example_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
