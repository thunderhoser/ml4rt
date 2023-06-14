"""Subsets data for Tom Beucler."""

import argparse
import numpy
from scipy.integrate import simps
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io
from ml4rt.utils import example_utils

TOLERANCE = 1e-6

DUMMY_FIRST_TIME_UNIX_SEC = 0
DUMMY_LAST_TIME_UNIX_SEC = int(3e9)

NUM_SEASONS = 4
MIN_WATER_PATH_FOR_CLOUD_LAYER_KG_M02 = 0.025

INPUT_DIR_ARG_NAME = 'input_example_dir_name'
NUM_ROWS_ARG_NAME = 'num_grid_rows'
NUM_COLUMNS_ARG_NAME = 'num_grid_columns'
NUM_EXAMPLES_ARG_NAME = 'num_examples_per_grid_cell_per_season'
HIGH_AOD_THRESHOLD_ARG_NAME = 'high_aod_threshold_unitless'
LOW_ZENITH_ANGLE_THRESHOLD_ARG_NAME = 'low_zenith_angle_threshold_deg'
OUTPUT_FILE_ARG_NAME = 'output_example_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing examples to subset.  Files therein '
    'will be found by `example_io.find_file` and read by '
    '`example_io.read_file`.'
)
NUM_ROWS_HELP_STRING = 'Number of rows in global grid.'
NUM_COLUMNS_HELP_STRING = 'Number of columns in global grid.'
NUM_EXAMPLES_HELP_STRING = (
    'Number of examples in subset, for each pair of grid cell and season.'
)
HIGH_AOD_THRESHOLD_HELP_STRING = (
    'Threshold for high aerosol optical depth.  Each pair of grid cell and '
    'season, if possible, will include at least one high-AOD/low-zenith-angle '
    'example.'
)
LOW_ZENITH_ANGLE_THRESHOLD_HELP_STRING = (
    'Threshold for low solar zenith angle.  Each pair of grid cell and season, '
    'if possible, will include at least one high-AOD/low-zenith-angle example.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file (with subset data).  Will be written by '
    '`example_io.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_ROWS_ARG_NAME, type=int, required=True, help=NUM_ROWS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_COLUMNS_ARG_NAME, type=int, required=True,
    help=NUM_COLUMNS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_EXAMPLES_ARG_NAME, type=int, required=True,
    help=NUM_EXAMPLES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + HIGH_AOD_THRESHOLD_ARG_NAME, type=float, required=True,
    help=HIGH_AOD_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LOW_ZENITH_ANGLE_THRESHOLD_ARG_NAME, type=float, required=True,
    help=LOW_ZENITH_ANGLE_THRESHOLD_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _create_filtered_index_array(max_index, indices_to_exclude):
    """Creates filtered numpy array of indices.

    :param max_index: Max index.  Letting this be N, the default array will
        contain all integers in 0...N.
    :param indices_to_exclude: 1-D numpy array of indices to exclude.
    :return: filtered_indices: 1-D numpy array of indices.
    """

    all_indices = numpy.linspace(0, max_index, num=max_index + 1, dtype=int)
    return numpy.setdiff1d(all_indices, indices_to_exclude)


def _time_to_season(valid_time_unix_sec):
    """Converts time to season.

    :param valid_time_unix_sec: Time.
    :return: season_index: Integer (0 for DJF, 1 for MAM, 2 for JJA, 3 for SON).
    """

    valid_time_string = time_conversion.unix_sec_to_string(
        valid_time_unix_sec, '%Y-%m'
    )
    month_index = int(valid_time_string.split('-')[1])

    return int(numpy.floor(
        (1. / 3) *
        numpy.mod(month_index, 12)
    ))


def _get_aerosol_optical_depths(example_dict):
    """Computes AOD for each profile.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :return: aerosol_optical_depths_unitless: length-E numpy array of AOD
        values, where E is the number of examples.
    """

    num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])

    height_matrix_m_agl = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.HEIGHT_NAME
    )
    aerosol_extinction_matrix_metres01 = example_utils.get_field_from_dict(
        example_dict=example_dict,
        field_name=example_utils.AEROSOL_EXTINCTION_NAME
    )
    aerosol_optical_depths_unitless = numpy.full(num_examples, numpy.nan)

    for i in range(num_examples):
        # if numpy.mod(i, 1000) == 0:
        #     print('Have computed AOD for {0:d} of {1:d} profiles...'.format(
        #         i, num_examples
        #     ))

        aerosol_optical_depths_unitless[i] = simps(
            y=aerosol_extinction_matrix_metres01[i, :],
            x=height_matrix_m_agl[i, :],
            even='avg'
        )

    # print('Have computed AOD for all {0:d} profiles!'.format(num_examples))
    return aerosol_optical_depths_unitless


def _get_diverse_cloud_examples(example_dict, for_ice):
    """Finds diverse examples with respect to either liquid or ice cloud.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`.
    :param for_ice: Boolean flag, indicating whether to do this for ice or
        liquid.
    :return: diverse_cloud_indices: 1-D numpy array of indices.
    """

    num_examples = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    total_water_path_field_name = (
        example_utils.COLUMN_ICE_WATER_PATH_NAME if for_ice
        else example_utils.COLUMN_LIQUID_WATER_PATH_NAME
    )

    _, num_clouds_by_example = example_utils.find_cloud_layers(
        example_dict=example_dict,
        min_path_kg_m02=MIN_WATER_PATH_FOR_CLOUD_LAYER_KG_M02,
        for_ice=for_ice
    )

    no_cloud_indices = numpy.where(num_clouds_by_example == 0)[0]
    single_cloud_indices = numpy.where(num_clouds_by_example == 1)[0]
    multi_cloud_indices = numpy.where(num_clouds_by_example >= 2)[0]

    if len(no_cloud_indices) == 0:
        total_path_by_example_kg_m02 = example_utils.get_field_from_dict(
            example_dict=example_dict, field_name=total_water_path_field_name
        )
        no_cloud_index = numpy.argmin(total_path_by_example_kg_m02)
    else:
        no_cloud_index = numpy.random.choice(
            no_cloud_indices, size=1, replace=False
        )[0]

    if len(single_cloud_indices) == 0:
        total_path_by_example_kg_m02 = example_utils.get_field_from_dict(
            example_dict=example_dict, field_name=total_water_path_field_name
        )
        nonzero_water_indices = numpy.where(
            total_path_by_example_kg_m02 >= TOLERANCE
        )[0]

        if len(nonzero_water_indices) == 0:
            all_example_indices = numpy.linspace(
                0, num_examples - 1, num=num_examples, dtype=int
            )
            single_cloud_index = numpy.random.choice(
                all_example_indices, size=1, replace=False
            )[0]
        else:
            single_cloud_index = numpy.random.choice(
                nonzero_water_indices, size=1, replace=False
            )[0]
    else:
        single_cloud_index = numpy.random.choice(
            single_cloud_indices, size=1, replace=False
        )[0]

    if len(multi_cloud_indices) == 0:
        total_path_by_example_kg_m02 = example_utils.get_field_from_dict(
            example_dict=example_dict, field_name=total_water_path_field_name
        )
        multi_cloud_index = numpy.argmax(total_path_by_example_kg_m02)
    else:
        multi_cloud_index = numpy.random.choice(
            multi_cloud_indices, size=1, replace=False
        )[0]

    return numpy.array(
        [no_cloud_index, single_cloud_index, multi_cloud_index], dtype=int
    )


def _subset_examples_one_grid_cell_and_season(
        example_dict, high_aod_threshold_unitless,
        low_zenith_angle_threshold_deg, num_examples_to_keep):
    """Subsets examples for one pair of grid cell season.

    :param example_dict: Dictionary in format returned by
        `example_io.read_file`, containing only examples in one grid cell.
    :param high_aod_threshold_unitless: See documentation at top of file.
    :param low_zenith_angle_threshold_deg: Same.
    :param num_examples_to_keep: Number of examples to keep.
    :return: example_dict: Same as input but maybe with fewer examples.
    """

    num_examples_total = len(example_dict[example_utils.EXAMPLE_IDS_KEY])
    if num_examples_to_keep >= num_examples_total:
        return example_dict

    print('Finding diverse ice-cloud examples...')
    diverse_ice_cloud_indices = _get_diverse_cloud_examples(
        example_dict=example_dict, for_ice=True
    )

    print('Finding diverse liquid-cloud examples...')
    diverse_liquid_cloud_indices = _get_diverse_cloud_examples(
        example_dict=example_dict, for_ice=False
    )

    print('Finding high-AOD/low-zenith-angle example...')
    zenith_angle_by_example_deg = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.ZENITH_ANGLE_NAME
    )
    aod_by_example_unitless = _get_aerosol_optical_depths(example_dict)

    high_aod_low_sza_indices = numpy.where(numpy.logical_and(
        zenith_angle_by_example_deg <= low_zenith_angle_threshold_deg,
        aod_by_example_unitless >= high_aod_threshold_unitless
    ))[0]

    if len(high_aod_low_sza_indices) == 0:
        zad = zenith_angle_by_example_deg
        zenith_angle_by_example_norm = (
            (zad - numpy.mean(zad)) / numpy.std(zad, ddof=1)
        )

        aodu = aod_by_example_unitless
        aod_by_example_norm = (
            (aodu - numpy.mean(aodu)) / numpy.std(aodu, ddof=1)
        )

        high_aod_low_sza_indices = numpy.array(
            [numpy.argmax(aod_by_example_norm - zenith_angle_by_example_norm)],
            dtype=int
        )
    else:
        high_aod_low_sza_indices = numpy.random.choice(
            high_aod_low_sza_indices, size=1, replace=False
        )

    special_indices = numpy.concatenate((
        diverse_ice_cloud_indices, diverse_liquid_cloud_indices,
        high_aod_low_sza_indices
    ))
    special_indices = numpy.unique(special_indices)

    num_examples_remaining = num_examples_to_keep - len(special_indices)
    if num_examples_remaining <= 0:
        return example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=special_indices
        )

    non_special_indices = _create_filtered_index_array(
        max_index=num_examples_total - 1, indices_to_exclude=special_indices
    )
    non_special_indices = numpy.random.choice(
        non_special_indices, size=num_examples_remaining, replace=False
    )

    return example_utils.subset_by_index(
        example_dict=example_dict,
        desired_indices=numpy.concatenate((
            special_indices, non_special_indices
        ))
    )


def _run(input_dir_name, num_grid_rows, num_grid_columns,
         num_examples_per_grid_cell_per_season, high_aod_threshold_unitless,
         low_zenith_angle_threshold_deg, output_file_name):
    """Subsets data for Tom Beucler.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of file.
    :param num_grid_rows: Same.
    :param num_grid_columns: Same.
    :param num_examples_per_grid_cell_per_season: Same.
    :param high_aod_threshold_unitless: Same.
    :param low_zenith_angle_threshold_deg: Same.
    :param output_file_name: Same.
    """

    error_checking.assert_is_greater(num_grid_rows, 0)
    error_checking.assert_is_greater(num_grid_columns, 0)
    error_checking.assert_is_greater(num_examples_per_grid_cell_per_season, 0)
    error_checking.assert_is_greater(high_aod_threshold_unitless, 0.)
    error_checking.assert_is_greater(low_zenith_angle_threshold_deg, 0.)
    error_checking.assert_is_less_than(low_zenith_angle_threshold_deg, 90.)

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

    edge_latitudes_deg_n = numpy.linspace(
        -90, 90, num=num_grid_rows + 1, dtype=float
    )
    grid_latitudes_deg_n = (
        edge_latitudes_deg_n[:-1] + numpy.diff(edge_latitudes_deg_n) / 2
    )

    edge_longitudes_deg_e = numpy.linspace(
        0, 360, num=num_grid_columns + 1, dtype=float
    )
    grid_longitudes_deg_e = (
        edge_longitudes_deg_e[:-1] + numpy.diff(edge_longitudes_deg_e) / 2
    )

    latitude_by_example_deg_n = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.LATITUDE_NAME
    )
    longitude_by_example_deg_e = example_utils.get_field_from_dict(
        example_dict=example_dict, field_name=example_utils.LONGITUDE_NAME
    )
    longitude_by_example_deg_e = lng_conversion.convert_lng_positive_in_west(
        longitude_by_example_deg_e, allow_nan=False
    )

    row_index_by_example = numpy.array([
        numpy.argmin(numpy.absolute(grid_latitudes_deg_n - this_lat))
        for this_lat in latitude_by_example_deg_n
    ], dtype=int)

    column_index_by_example = numpy.array([
        numpy.argmin(numpy.absolute(grid_longitudes_deg_e - this_lng))
        for this_lng in longitude_by_example_deg_e
    ], dtype=int)

    valid_time_by_example_unix_sec = example_utils.parse_example_ids(
        example_dict[example_utils.EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    season_index_by_example = numpy.array(
        [_time_to_season(t) for t in valid_time_by_example_unix_sec], dtype=int
    )

    subset_example_dicts = []

    for i in range(num_grid_rows):
        for j in range(num_grid_columns):
            for k in range(NUM_SEASONS):
                these_spatial_flags = numpy.logical_and(
                    row_index_by_example == i,
                    column_index_by_example == j
                )
                these_time_flags = season_index_by_example == k
                these_example_indices = numpy.where(numpy.logical_and(
                    these_spatial_flags, these_time_flags
                ))[0]

                if len(these_example_indices) == 0:
                    continue

                this_example_dict = example_utils.subset_by_index(
                    example_dict=example_dict,
                    desired_indices=these_example_indices
                )

                print((
                    '\nSubsetting examples for grid row {0:d} of {1:d}, '
                    'column {2:d} of {3:d}, season {4:d} of {5:d}...'
                ).format(
                    i + 1, num_grid_rows,
                    j + 1, num_grid_columns,
                    k + 1, NUM_SEASONS
                ))

                this_example_dict = _subset_examples_one_grid_cell_and_season(
                    example_dict=this_example_dict,
                    high_aod_threshold_unitless=high_aod_threshold_unitless,
                    low_zenith_angle_threshold_deg=
                    low_zenith_angle_threshold_deg,
                    num_examples_to_keep=num_examples_per_grid_cell_per_season
                )

                subset_example_dicts.append(this_example_dict)

    subset_example_dict = example_utils.concat_examples(subset_example_dicts)

    print('\nWriting {0:d} subset examples to: "{1:s}"...'.format(
        len(subset_example_dict[example_utils.EXAMPLE_IDS_KEY]),
        output_file_name
    ))
    example_io.write_file(
        example_dict=subset_example_dict, netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        num_grid_rows=getattr(INPUT_ARG_OBJECT, NUM_ROWS_ARG_NAME),
        num_grid_columns=getattr(INPUT_ARG_OBJECT, NUM_COLUMNS_ARG_NAME),
        num_examples_per_grid_cell_per_season=getattr(
            INPUT_ARG_OBJECT, NUM_EXAMPLES_ARG_NAME
        ),
        high_aod_threshold_unitless=getattr(
            INPUT_ARG_OBJECT, HIGH_AOD_THRESHOLD_ARG_NAME
        ),
        low_zenith_angle_threshold_deg=getattr(
            INPUT_ARG_OBJECT, LOW_ZENITH_ANGLE_THRESHOLD_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
