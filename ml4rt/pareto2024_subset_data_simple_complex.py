"""Subsets data into simple and complex datasets.

Complex = multiple layers of mixed-phase cloud
Simple = no cloud, low humidity, low solar elevation angle (high zenith angle)
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import example_io
import example_utils

TOLERANCE = 1e-6

KG_TO_GRAMS = 1000.
RADIANS_TO_DEGREES = 180. / numpy.pi

DUMMY_FIRST_TIME_UNIX_SEC = 0
DUMMY_LAST_TIME_UNIX_SEC = int(3e9)

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
CLOUD_WATER_THRESHOLD_ARG_NAME = 'cloud_water_path_threshold_kg_m02'
HUMIDITY_THRESHOLD_ARG_NAME = 'specific_humidity_threshold_kg_kg01'
ZENITH_ANGLE_THRESHOLD_ARG_NAME = 'zenith_angle_threshold_deg'
SIMPLE_FILE_ARG_NAME = 'output_simple_example_file_name'
COMPLEX_FILE_ARG_NAME = 'output_complex_example_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing examples to subset.  Files therein '
    'will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)

CLOUD_WATER_THRESHOLD_HELP_STRING = (
    'Minimum path for cloud layer.  Note that only mixed-phase cloud layers '
    'will be counted.  Thus, for an atmospheric layer to be considered a cloud '
    'layer, it must have (1) a minimum total ice/liquid path of {0:s}; '
    '(2) non-zero total ice/liquid water content at every level; '
    '(3) total ice path > 0; '
    '(4) total liquid path > 0.'
).format(CLOUD_WATER_THRESHOLD_ARG_NAME)

HUMIDITY_THRESHOLD_HELP_STRING = (
    'Maximum specific humidity.  For a profile to be considered "simple," '
    'every level in the profile must have specific humidity <= {0:s}.'
).format(HUMIDITY_THRESHOLD_ARG_NAME)

ZENITH_ANGLE_THRESHOLD_HELP_STRING = (
    'Minimum solar zenith angle.  For a profile to be considered "simple," the '
    'zenith angle must be >= {0:s}.'
).format(ZENITH_ANGLE_THRESHOLD_ARG_NAME)

SIMPLE_FILE_HELP_STRING = (
    'Path to output file for simple cases.  The simple cases will be written '
    'to this location by `example_io.write_file`.'
)
COMPLEX_FILE_HELP_STRING = (
    'Path to output file for complex cases.  The complex cases will be written '
    'to this location by `example_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CLOUD_WATER_THRESHOLD_ARG_NAME, type=float, required=False,
    default=0.025, help=CLOUD_WATER_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HUMIDITY_THRESHOLD_ARG_NAME, type=float, required=True,
    help=HUMIDITY_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + ZENITH_ANGLE_THRESHOLD_ARG_NAME, type=float, required=True,
    help=ZENITH_ANGLE_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + SIMPLE_FILE_ARG_NAME, type=str, required=True,
    help=SIMPLE_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + COMPLEX_FILE_ARG_NAME, type=str, required=True,
    help=COMPLEX_FILE_HELP_STRING
)


def _run(input_dir_name, cloud_water_path_threshold_kg_m02,
         specific_humidity_threshold_kg_kg01, zenith_angle_threshold_deg,
         simple_example_file_name, complex_example_file_name):
    """Subsets data into simple and complex datasets.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param cloud_water_path_threshold_kg_m02: Same.
    :param specific_humidity_threshold_kg_kg01: Same.
    :param zenith_angle_threshold_deg: Same.
    :param simple_example_file_name: Same.
    :param complex_example_file_name: Same.
    """

    # Check input args.
    error_checking.assert_is_greater(cloud_water_path_threshold_kg_m02, 0.)
    error_checking.assert_is_greater(specific_humidity_threshold_kg_kg01, 0.)
    error_checking.assert_is_greater(zenith_angle_threshold_deg, 0.)
    error_checking.assert_is_leq(zenith_angle_threshold_deg, 90.)

    # Read input files.
    input_file_names = example_io.find_many_files(
        directory_name=input_dir_name,
        first_time_unix_sec=DUMMY_FIRST_TIME_UNIX_SEC,
        last_time_unix_sec=DUMMY_LAST_TIME_UNIX_SEC,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    num_files = len(input_file_names)
    example_dicts = [dict()] * num_files

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(input_file_names[i]))
        example_dicts[i] = example_io.read_file(input_file_names[i])

    example_dict = example_utils.concat_examples(example_dicts)
    del example_dicts

    # Find complex cases.
    _, num_clouds_by_example = example_utils.find_cloud_layers(
        example_dict=example_dict,
        min_path_kg_m02=cloud_water_path_threshold_kg_m02,
        cloud_type_string=example_utils.MIXED_PHASE_CLOUD_TYPE_STRING,
        fog_only=False
    )

    complex_indices = numpy.where(num_clouds_by_example > 1)[0]
    complex_example_dict = example_utils.subset_by_index(
        example_dict=example_dict,
        desired_indices=complex_indices
    )
    print((
        'Number of complex cases (with multiple mixed-phase cloud layers of '
        'path >= {0:f} g m^-2) = {1:d} of {2:d}'
    ).format(
        KG_TO_GRAMS * cloud_water_path_threshold_kg_m02,
        len(complex_indices),
        len(num_clouds_by_example)
    ))

    print('Writing all {0:d} complex cases to file: "{1:s}"...'.format(
        len(complex_indices), complex_example_file_name
    ))
    example_io.write_file(
        example_dict=complex_example_dict,
        netcdf_file_name=complex_example_file_name
    )

    # Find simple cases.
    liquid_water_content_matrix_kg_m03 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.LIQUID_WATER_CONTENT_NAME
    )
    ice_water_content_matrix_kg_m03 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.ICE_WATER_CONTENT_NAME
    )
    water_content_matrix_kg_m03 = (
        liquid_water_content_matrix_kg_m03 + ice_water_content_matrix_kg_m03
    )
    column_integ_water_contents_kg_m03 = numpy.sum(
        water_content_matrix_kg_m03, axis=1
    )

    simple_indices = numpy.where(
        column_integ_water_contents_kg_m03 < TOLERANCE
    )[0]
    simple_example_dict = example_utils.subset_by_index(
        example_dict=example_dict,
        desired_indices=simple_indices
    )
    print('Number of cases with no cloud at all = {0:d} of {1:d}'.format(
        len(simple_indices),
        len(column_integ_water_contents_kg_m03)
    ))

    del example_dict

    specific_humidity_matrix_kg_kg01 = example_utils.get_field_from_dict(
        example_dict=simple_example_dict,
        field_name=example_utils.SPECIFIC_HUMIDITY_NAME
    )
    simple_flags = numpy.all(
        specific_humidity_matrix_kg_kg01 <= specific_humidity_threshold_kg_kg01,
        axis=1
    )

    simple_indices = numpy.where(simple_flags)[0]
    simple_example_dict = example_utils.subset_by_index(
        example_dict=simple_example_dict,
        desired_indices=simple_indices
    )
    print((
        'Number of cases with no specific humidity above {0:f} g kg^-1 = '
        '{1:d} of {2:d}'
    ).format(
        KG_TO_GRAMS * specific_humidity_threshold_kg_kg01,
        len(simple_indices),
        len(simple_flags)
    ))

    zenith_angles_deg = RADIANS_TO_DEGREES * example_utils.get_field_from_dict(
        example_dict=simple_example_dict,
        field_name=example_utils.ZENITH_ANGLE_NAME
    )

    simple_indices = numpy.where(
        zenith_angles_deg >= zenith_angle_threshold_deg
    )[0]
    simple_example_dict = example_utils.subset_by_index(
        example_dict=simple_example_dict,
        desired_indices=simple_indices
    )
    print((
        'Number of cases with zenith angle at least {0:f} deg = {1:d} of {2:d}'
    ).format(
        zenith_angle_threshold_deg,
        len(simple_indices),
        len(simple_flags)
    ))

    print('Writing all {0:d} simple cases to file: "{1:s}"...'.format(
        len(simple_indices), simple_example_file_name
    ))
    example_io.write_file(
        example_dict=simple_example_dict,
        netcdf_file_name=simple_example_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        cloud_water_path_threshold_kg_m02=getattr(
            INPUT_ARG_OBJECT, CLOUD_WATER_THRESHOLD_ARG_NAME
        ),
        specific_humidity_threshold_kg_kg01=getattr(
            INPUT_ARG_OBJECT, HUMIDITY_THRESHOLD_ARG_NAME
        ),
        zenith_angle_threshold_deg=getattr(
            INPUT_ARG_OBJECT, ZENITH_ANGLE_THRESHOLD_ARG_NAME
        ),
        simple_example_file_name=getattr(
            INPUT_ARG_OBJECT, SIMPLE_FILE_ARG_NAME
        ),
        complex_example_file_name=getattr(
            INPUT_ARG_OBJECT, COMPLEX_FILE_ARG_NAME
        )
    )
