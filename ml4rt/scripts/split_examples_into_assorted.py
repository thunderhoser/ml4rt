"""Splits examples into two sets of sites:

- Assorted1 (10 Arctic, 9 mid-latitude, 5 tropical)
- Assorted2 (2 Arctic, 2 mid-latitude, 2 tropical)
"""

import copy
import argparse
import numpy
from sklearn.metrics.pairwise import euclidean_distances
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from ml4rt.io import example_io
from ml4rt.utils import example_utils

TOLERANCE_DEG2 = 1e-6

ASSORTED2_LATITUDES_DEG_N = numpy.array([
    89.9983, 71.6260, 12.049037, 26.133038, 39.0827, 36.5693
])
ASSORTED2_LONGITUDES_DEG_E = numpy.array([
    164.0000, 129.7669, 298.225193, 265.124474, -28.0923, 262.5687
])

TROPICAL_DIR_ARG_NAME = 'input_tropical_dir_name'
NON_TROPICAL_DIR_ARG_NAME = 'input_non_tropical_dir_name'
YEAR_ARG_NAME = 'year'
ASSORTED1_DIR_ARG_NAME = 'output_assorted1_dir_name'
ASSORTED2_DIR_ARG_NAME = 'output_assorted2_dir_name'

TROPICAL_DIR_HELP_STRING = (
    'Name of top-level directory with tropical examples.  Files therein will be'
    ' found by `example_io.find_file` and read by `example_io.read_file`.'
)
NON_TROPICAL_DIR_HELP_STRING = (
    'Same as `{0:s}` but for non-tropical examples.'
).format(TROPICAL_DIR_ARG_NAME)

YEAR_HELP_STRING = 'Year (integer).  Will re-split data only for this year.'
ASSORTED1_DIR_HELP_STRING = (
    'Name of top-level output directory for examples at Assorted1 sites.  Files'
    ' will be written here by `example_io.write_file`, to exact locations '
    'determined by `example_io.find_file`.'
)
ASSORTED2_DIR_HELP_STRING = (
    'Same as `{0:s}` but for Assorted2 examples.'
).format(ASSORTED1_DIR_ARG_NAME)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + TROPICAL_DIR_ARG_NAME, type=str, required=True,
    help=TROPICAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NON_TROPICAL_DIR_ARG_NAME, type=str, required=True,
    help=NON_TROPICAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + YEAR_ARG_NAME, type=int, required=True, help=YEAR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ASSORTED1_DIR_ARG_NAME, type=str, required=True,
    help=ASSORTED1_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ASSORTED2_DIR_ARG_NAME, type=str, required=True,
    help=ASSORTED2_DIR_HELP_STRING
)


def _run(tropical_example_dir_name, non_tropical_example_dir_name, year,
         assorted1_example_dir_name, assorted2_example_dir_name):
    """Splits examples into Assorted1 and Assorted2 sites.

    This is effectively the main method.

    :param tropical_example_dir_name: See documentation at top of file.
    :param non_tropical_example_dir_name: Same.
    :param year: Same.
    :param assorted1_example_dir_name: Same.
    :param assorted2_example_dir_name: Same.
    """

    tropical_example_file_name = example_io.find_file(
        directory_name=tropical_example_dir_name, year=year,
        raise_error_if_missing=True
    )
    non_tropical_example_file_name = example_io.find_file(
        directory_name=non_tropical_example_dir_name, year=year,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(tropical_example_file_name))
    tropical_example_dict = example_io.read_file(tropical_example_file_name)

    print('Reading data from: "{0:s}"...'.format(
        non_tropical_example_file_name
    ))
    non_tropical_example_dict = example_io.read_file(
        non_tropical_example_file_name
    )

    example_dict = example_utils.concat_examples([
        tropical_example_dict, non_tropical_example_dict
    ])
    del tropical_example_dict, non_tropical_example_dict

    example_metadata_dict = example_utils.parse_example_ids(
        example_dict[example_utils.EXAMPLE_IDS_KEY]
    )
    example_latitudes_deg_n = example_metadata_dict[example_utils.LATITUDES_KEY]
    example_longitudes_deg_e = lng_conversion.convert_lng_positive_in_west(
        example_metadata_dict[example_utils.LONGITUDES_KEY]
    )

    example_coord_matrix = numpy.transpose(numpy.vstack((
        example_latitudes_deg_n, example_longitudes_deg_e
    )))
    assorted2_coord_matrix = numpy.transpose(numpy.vstack((
        ASSORTED2_LATITUDES_DEG_N, ASSORTED2_LONGITUDES_DEG_E
    )))
    distance_matrix_deg2 = euclidean_distances(
        X=example_coord_matrix, Y=assorted2_coord_matrix, squared=True
    )

    assorted2_flags = numpy.any(distance_matrix_deg2 <= TOLERANCE_DEG2, axis=1)
    assorted2_example_dict = example_utils.subset_by_index(
        example_dict=copy.deepcopy(example_dict),
        desired_indices=numpy.where(assorted2_flags)[0]
    )
    assorted2_example_file_name = example_io.find_file(
        directory_name=assorted2_example_dir_name, year=year,
        raise_error_if_missing=False
    )

    print('Writing {0:d} examples in set Assorted2 to: "{1:s}"...'.format(
        len(assorted2_example_dict[example_utils.VALID_TIMES_KEY]),
        assorted2_example_file_name
    ))
    example_io.write_file(
        example_dict=assorted2_example_dict,
        netcdf_file_name=assorted2_example_file_name
    )

    assorted1_example_dict = example_utils.subset_by_index(
        example_dict=example_dict,
        desired_indices=numpy.where(numpy.invert(assorted2_flags))[0]
    )
    assorted1_example_file_name = example_io.find_file(
        directory_name=assorted1_example_dir_name, year=year,
        raise_error_if_missing=False
    )

    print('Writing {0:d} examples in set Assorted1 to: "{1:s}"...'.format(
        len(assorted1_example_dict[example_utils.VALID_TIMES_KEY]),
        assorted1_example_file_name
    ))
    example_io.write_file(
        example_dict=assorted1_example_dict,
        netcdf_file_name=assorted1_example_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        tropical_example_dir_name=getattr(
            INPUT_ARG_OBJECT, TROPICAL_DIR_ARG_NAME
        ),
        non_tropical_example_dir_name=getattr(
            INPUT_ARG_OBJECT, NON_TROPICAL_DIR_ARG_NAME
        ),
        year=getattr(INPUT_ARG_OBJECT, YEAR_ARG_NAME),
        assorted1_example_dir_name=getattr(
            INPUT_ARG_OBJECT, ASSORTED1_DIR_ARG_NAME
        ),
        assorted2_example_dir_name=getattr(
            INPUT_ARG_OBJECT, ASSORTED2_DIR_ARG_NAME
        )
    )
