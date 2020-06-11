"""Input/output methods for learning examples."""

import copy
import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import longitude_conversion as longitude_conv
from gewittergefahr.gg_utils import error_checking

TOLERANCE = 1e-6
KM_TO_METRES = 1000.
DEG_TO_RADIANS = numpy.pi / 180

DEFAULT_MAX_PMM_PERCENTILE_LEVEL = 99.

SCALAR_PREDICTOR_VALS_KEY = 'scalar_predictor_matrix'
SCALAR_PREDICTOR_NAMES_KEY = 'scalar_predictor_names'
VECTOR_PREDICTOR_VALS_KEY = 'vector_predictor_matrix'
VECTOR_PREDICTOR_NAMES_KEY = 'vector_predictor_names'
SCALAR_TARGET_VALS_KEY = 'scalar_target_matrix'
SCALAR_TARGET_NAMES_KEY = 'scalar_target_names'
VECTOR_TARGET_VALS_KEY = 'vector_target_matrix'
VECTOR_TARGET_NAMES_KEY = 'vector_target_names'
VALID_TIMES_KEY = 'valid_times_unix_sec'
HEIGHTS_KEY = 'heights_m_agl'
STANDARD_ATMO_FLAGS_KEY = 'standard_atmo_flags'

DICTIONARY_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, SCALAR_PREDICTOR_NAMES_KEY,
    VECTOR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_NAMES_KEY,
    SCALAR_TARGET_VALS_KEY, SCALAR_TARGET_NAMES_KEY,
    VECTOR_TARGET_VALS_KEY, VECTOR_TARGET_NAMES_KEY,
    VALID_TIMES_KEY, HEIGHTS_KEY, STANDARD_ATMO_FLAGS_KEY
]
ONE_PER_EXAMPLE_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_VALS_KEY,
    SCALAR_TARGET_VALS_KEY, VECTOR_TARGET_VALS_KEY,
    VALID_TIMES_KEY, STANDARD_ATMO_FLAGS_KEY
]
ONE_PER_FIELD_KEYS = [
    SCALAR_PREDICTOR_VALS_KEY, SCALAR_PREDICTOR_NAMES_KEY,
    VECTOR_PREDICTOR_VALS_KEY, VECTOR_PREDICTOR_NAMES_KEY,
    SCALAR_TARGET_VALS_KEY, SCALAR_TARGET_NAMES_KEY,
    VECTOR_TARGET_VALS_KEY, VECTOR_TARGET_NAMES_KEY
]

VALID_TIMES_KEY_ORIG = 'time'
HEIGHTS_KEY_ORIG = 'height'
STANDARD_ATMO_FLAGS_KEY_ORIG = 'stdatmos'

TROPICS_ENUM = 1
MIDLATITUDE_SUMMER_ENUM = 2
MIDLATITUDE_WINTER_ENUM = 3
SUBARCTIC_SUMMER_ENUM = 4
SUBARCTIC_WINTER_ENUM = 5
US_STANDARD_ATMO_ENUM = 6
STANDARD_ATMO_ENUMS = [
    TROPICS_ENUM, MIDLATITUDE_SUMMER_ENUM, MIDLATITUDE_WINTER_ENUM,
    SUBARCTIC_SUMMER_ENUM, SUBARCTIC_WINTER_ENUM, US_STANDARD_ATMO_ENUM
]

ZENITH_ANGLE_NAME = 'zenith_angle_radians'
LATITUDE_NAME = 'latitude_deg_n'
LONGITUDE_NAME = 'longitude_deg_e'
ALBEDO_NAME = 'albedo'
LIQUID_WATER_PATH_NAME = 'liquid_water_path_kg_m02'
ICE_WATER_PATH_NAME = 'ice_water_path_kg_m02'
PRESSURE_NAME = 'pressure_pascals'
TEMPERATURE_NAME = 'temperature_kelvins'
SPECIFIC_HUMIDITY_NAME = 'specific_humidity_kg_kg01'
LIQUID_WATER_CONTENT_NAME = 'liquid_water_content_kg_m02'
ICE_WATER_CONTENT_NAME = 'ice_water_content_kg_m02'

SCALAR_PREDICTOR_NAMES = [
    ZENITH_ANGLE_NAME, LATITUDE_NAME, LONGITUDE_NAME, ALBEDO_NAME,
    LIQUID_WATER_PATH_NAME, ICE_WATER_PATH_NAME
]
VECTOR_PREDICTOR_NAMES = [
    PRESSURE_NAME, TEMPERATURE_NAME, SPECIFIC_HUMIDITY_NAME,
    LIQUID_WATER_CONTENT_NAME, ICE_WATER_CONTENT_NAME
]
PREDICTOR_NAMES = SCALAR_PREDICTOR_NAMES + VECTOR_PREDICTOR_NAMES

PREDICTOR_NAME_TO_ORIG = {
    ZENITH_ANGLE_NAME: 'sza',
    LATITUDE_NAME: 'lat',
    LONGITUDE_NAME: 'lon',
    ALBEDO_NAME: 'albedo',
    LIQUID_WATER_PATH_NAME: 'lwp',
    ICE_WATER_PATH_NAME: 'iwp',
    PRESSURE_NAME: 'p',
    TEMPERATURE_NAME: 't',
    SPECIFIC_HUMIDITY_NAME: 'q',
    LIQUID_WATER_CONTENT_NAME: 'lwc',
    ICE_WATER_CONTENT_NAME: 'iwc'
}

PREDICTOR_NAME_TO_CONV_FACTOR = {
    ZENITH_ANGLE_NAME: DEG_TO_RADIANS,
    LATITUDE_NAME: 1.,
    LONGITUDE_NAME: 1.,
    ALBEDO_NAME: 1.,
    LIQUID_WATER_PATH_NAME: 0.001,
    ICE_WATER_PATH_NAME: 0.001,
    PRESSURE_NAME: 100.,
    TEMPERATURE_NAME: 1.,
    SPECIFIC_HUMIDITY_NAME: 0.001,
    LIQUID_WATER_CONTENT_NAME: 0.001,
    ICE_WATER_CONTENT_NAME: 0.001
}

SHORTWAVE_HEATING_RATE_NAME = 'shortwave_heating_rate_K_s01'
SHORTWAVE_DOWN_FLUX_NAME = 'shortwave_down_flux_W_m02'
SHORTWAVE_UP_FLUX_NAME = 'shortwave_up_flux_W_m02'
SHORTWAVE_SURFACE_DOWN_FLUX_NAME = 'shortwave_surface_down_flux_W_m02'
SHORTWAVE_TOA_UP_FLUX_NAME = 'shortwave_toa_up_flux_W_m02'

SCALAR_TARGET_NAMES = [
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME, SHORTWAVE_TOA_UP_FLUX_NAME
]
VECTOR_TARGET_NAMES = [
    SHORTWAVE_DOWN_FLUX_NAME, SHORTWAVE_UP_FLUX_NAME,
    SHORTWAVE_HEATING_RATE_NAME
]
TARGET_NAMES = SCALAR_TARGET_NAMES + VECTOR_TARGET_NAMES

TARGET_NAME_TO_ORIG = {
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 'sfcflux',
    SHORTWAVE_TOA_UP_FLUX_NAME: 'toaflux',
    SHORTWAVE_DOWN_FLUX_NAME: 'fluxd',
    SHORTWAVE_UP_FLUX_NAME: 'fluxu',
    SHORTWAVE_HEATING_RATE_NAME: 'hr'
}

TARGET_NAME_TO_CONV_FACTOR = {
    SHORTWAVE_SURFACE_DOWN_FLUX_NAME: 1.,
    SHORTWAVE_TOA_UP_FLUX_NAME: 1.,
    SHORTWAVE_DOWN_FLUX_NAME: 1.,
    SHORTWAVE_UP_FLUX_NAME: 1.,
    SHORTWAVE_HEATING_RATE_NAME: 1. / 86400
}


def _check_field_name(field_name):
    """Ensures that field name is valid (either predictor or target variable).

    :param field_name: Field name.
    :raises: ValueError: if `field_name not in PREDICTOR_NAMES + TARGET_NAMES`.
    """

    error_checking.assert_is_string(field_name)
    if field_name in PREDICTOR_NAMES + TARGET_NAMES:
        return

    error_string = (
        '\nField "{0:s}" is not valid predictor or target variable.  Valid '
        'options listed below:\n{1:s}'
    ).format(field_name, str(PREDICTOR_NAMES + TARGET_NAMES))

    raise ValueError(error_string)


def _match_heights(heights_m_agl, desired_height_m_agl):
    """Finds nearest available height to desired height.

    :param heights_m_agl: 1-D numpy array of available heights (metres above
        ground level).
    :param desired_height_m_agl: Desired height (metres above ground level).
    :return: matching_index: Index of desired height in array.  If
        `matching_index == k`, then `heights_m_agl[k]` is the desired height.
    :raises: ValueError: if there is no available height within 0.5 metres of
        the desired height.
    """

    error_checking.assert_is_geq_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(heights_m_agl, num_dimensions=1)
    error_checking.assert_is_geq(desired_height_m_agl, 0.)

    height_diffs_metres = numpy.absolute(heights_m_agl - desired_height_m_agl)
    matching_index = numpy.argmin(height_diffs_metres)

    if height_diffs_metres[matching_index] <= 0.5:
        return matching_index

    error_string = (
        'Cannot find available height within 0.5 metres of desired height '
        '({0:.1f} m AGL).  Nearest available height is {1:.1f} m AGL.'
    ).format(desired_height_m_agl, heights_m_agl[matching_index])

    raise ValueError(error_string)


def check_standard_atmo_type(standard_atmo_enum):
    """Ensures that standard-atmosphere type is valid.

    :param standard_atmo_enum: Integer (must be in `STANDARD_ATMO_ENUMS`).
    :raises: ValueError: if `standard_atmo_enum not in STANDARD_ATMO_ENUMS`.
    """

    error_checking.assert_is_integer(standard_atmo_enum)

    if standard_atmo_enum not in STANDARD_ATMO_ENUMS:
        error_string = (
            'Standard-atmosphere type {0:d} is invalid.  Must be in the '
            'following list:\n{1:s}'
        ).format(standard_atmo_enum, str(STANDARD_ATMO_ENUMS))

        raise ValueError(error_string)


def find_file(example_dir_name, year, raise_error_if_missing=True):
    """Finds NetCDF file with learning examples.

    :param example_dir_name: Name of directory where file is expected.
    :param year: Year (integer).
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: example_file_name: File path.
    """

    error_checking.assert_is_string(example_dir_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_missing)

    example_file_name = '{0:s}/radiative_transfer_examples_{1:04d}.nc'.format(
        example_dir_name, year
    )

    if raise_error_if_missing and not os.path.isfile(example_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            example_file_name
        )
        raise ValueError(error_string)

    return example_file_name


def find_many_files(
        example_dir_name, first_time_unix_sec, last_time_unix_sec,
        raise_error_if_any_missing=True, raise_error_if_all_missing=True,
        test_mode=False):
    """Finds many NetCDF files with learning examples.

    :param example_dir_name: Name of directory where files are expected.
    :param first_time_unix_sec: First time at which examples are desired.
    :param last_time_unix_sec: Last time at which examples are desired.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :param test_mode: Leave this alone.
    :return: example_file_names: 1-D list of paths to example files.  This list
        does *not* contain expected paths to non-existent files.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)
    error_checking.assert_is_boolean(test_mode)

    start_year = int(
        time_conversion.unix_sec_to_string(first_time_unix_sec, '%Y')
    )
    end_year = int(
        time_conversion.unix_sec_to_string(last_time_unix_sec, '%Y')
    )
    years = numpy.linspace(
        start_year, end_year, num=end_year - start_year + 1, dtype=int
    )

    example_file_names = []

    for this_year in years:
        this_file_name = find_file(
            example_dir_name=example_dir_name, year=this_year,
            raise_error_if_missing=raise_error_if_any_missing
        )

        print(this_file_name)

        if test_mode or os.path.isfile(this_file_name):
            example_file_names.append(this_file_name)

    if raise_error_if_all_missing and len(example_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from years {1:d}-{2:d}.'
        ).format(
            example_dir_name, start_year, end_year
        )
        raise ValueError(error_string)

    return example_file_names


def read_file(example_file_name):
    """Reads NetCDF file with learning examples.

    T = number of times
    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors
    T_s = number of scalar targets
    T_v = number of vector targets

    :param example_file_name: Path to NetCDF file with learning examples.
    :return: example_dict: Dictionary with the following keys.
    example_dict['scalar_predictor_matrix']: numpy array (T x P_s) with values
        of scalar predictors.
    example_dict['scalar_predictor_names']: list (length P_s) with names of
        scalar predictors.
    example_dict['vector_predictor_matrix']: numpy array (T x H x P_v) with
        values of vector predictors.
    example_dict['vector_predictor_names']: list (length P_v) with names of
        vector predictors.
    example_dict['scalar_target_matrix']: numpy array (T x T_s) with values of
        scalar targets.
    example_dict['scalar_predictor_names']: list (length T_s) with names of
        scalar targets.
    example_dict['vector_target_matrix']: numpy array (T x H x T_v) with values
        of vector targets.
    example_dict['vector_predictor_names']: list (length T_v) with names of
        vector targets.
    example_dict['valid_times_unix_sec']: length-T numpy array of valid times
        (Unix seconds).
    example_dict['heights_m_agl']: length-H numpy array of heights (metres above
        ground level).
    example_dict['standard_atmo_flags']: length-T numpy array of flags (each in
        the list `STANDARD_ATMO_ENUMS`).
    """

    dataset_object = netCDF4.Dataset(example_file_name)

    example_dict = {
        SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
        VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
        SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
        VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
        VALID_TIMES_KEY: numpy.array(
            dataset_object.variables[VALID_TIMES_KEY_ORIG][:], dtype=int
        ),
        HEIGHTS_KEY: KM_TO_METRES * numpy.array(
            dataset_object.variables[HEIGHTS_KEY_ORIG][:], dtype=float
        ),
        STANDARD_ATMO_FLAGS_KEY: numpy.array(
            numpy.round(
                dataset_object.variables[STANDARD_ATMO_FLAGS_KEY_ORIG][:]
            ), dtype=int
        )
    }

    num_times = len(example_dict[VALID_TIMES_KEY])
    num_heights = len(example_dict[HEIGHTS_KEY])
    num_scalar_predictors = len(SCALAR_PREDICTOR_NAMES)
    num_vector_predictors = len(VECTOR_PREDICTOR_NAMES)
    num_scalar_targets = len(SCALAR_TARGET_NAMES)
    num_vector_targets = len(VECTOR_TARGET_NAMES)

    scalar_predictor_matrix = numpy.full(
        (num_times, num_scalar_predictors), numpy.nan
    )
    vector_predictor_matrix = numpy.full(
        (num_times, num_heights, num_vector_predictors), numpy.nan
    )
    scalar_target_matrix = numpy.full(
        (num_times, num_scalar_targets), numpy.nan
    )
    vector_target_matrix = numpy.full(
        (num_times, num_heights, num_vector_targets), numpy.nan
    )

    for k in range(num_scalar_predictors):
        this_predictor_name_orig = (
            PREDICTOR_NAME_TO_ORIG[SCALAR_PREDICTOR_NAMES[k]]
        )
        this_conversion_factor = (
            PREDICTOR_NAME_TO_CONV_FACTOR[SCALAR_PREDICTOR_NAMES[k]]
        )
        scalar_predictor_matrix[:, k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_predictor_name_orig][:], dtype=float
        )

    for k in range(num_vector_predictors):
        this_predictor_name_orig = (
            PREDICTOR_NAME_TO_ORIG[VECTOR_PREDICTOR_NAMES[k]]
        )
        this_conversion_factor = (
            PREDICTOR_NAME_TO_CONV_FACTOR[VECTOR_PREDICTOR_NAMES[k]]
        )
        vector_predictor_matrix[..., k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_predictor_name_orig][:], dtype=float
        )

    for k in range(num_scalar_targets):
        this_target_name_orig = TARGET_NAME_TO_ORIG[SCALAR_TARGET_NAMES[k]]
        this_conversion_factor = (
            TARGET_NAME_TO_CONV_FACTOR[SCALAR_TARGET_NAMES[k]]
        )
        scalar_target_matrix[:, k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_target_name_orig][:], dtype=float
        )

    for k in range(num_vector_targets):
        this_target_name_orig = TARGET_NAME_TO_ORIG[VECTOR_TARGET_NAMES[k]]
        this_conversion_factor = (
            TARGET_NAME_TO_CONV_FACTOR[VECTOR_TARGET_NAMES[k]]
        )
        vector_target_matrix[..., k] = this_conversion_factor * numpy.array(
            dataset_object.variables[this_target_name_orig][:], dtype=float
        )

    longitude_index = SCALAR_PREDICTOR_NAMES.index(LONGITUDE_NAME)
    scalar_predictor_matrix[:, longitude_index] = (
        longitude_conv.convert_lng_positive_in_west(
            longitudes_deg=scalar_predictor_matrix[:, longitude_index],
            allow_nan=False
        )
    )

    example_dict.update({
        SCALAR_PREDICTOR_VALS_KEY: scalar_predictor_matrix,
        VECTOR_PREDICTOR_VALS_KEY: vector_predictor_matrix,
        SCALAR_TARGET_VALS_KEY: scalar_target_matrix,
        VECTOR_TARGET_VALS_KEY: vector_target_matrix
    })

    dataset_object.close()
    return example_dict


def concat_examples(example_dicts):
    """Concatenates many dictionaries with examples into one.

    :param example_dicts: List of dictionaries, each in the format returned by
        `read_file`.
    :return: example_dict: Single dictionary, also in the format returned by
        `read_file`.
    :raises: ValueError: if any two dictionaries have different predictor
        variables, target variables, or height coordinates.
    """

    example_dict = copy.deepcopy(example_dicts[0])

    keys_to_match = [
        SCALAR_PREDICTOR_NAMES_KEY, VECTOR_PREDICTOR_NAMES_KEY,
        SCALAR_TARGET_NAMES_KEY, VECTOR_TARGET_NAMES_KEY, HEIGHTS_KEY
    ]

    for i in range(1, len(example_dicts)):
        if not numpy.allclose(
                example_dict[HEIGHTS_KEY], example_dicts[i][HEIGHTS_KEY],
                atol=TOLERANCE
        ):
            error_string = (
                '1st and {0:d}th dictionaries have different height coords '
                '(units are m AGL).  1st dictionary:\n{1:s}\n\n'
                '{0:d}th dictionary:\n{2:s}'
            ).format(
                i + 1, str(example_dict[HEIGHTS_KEY]),
                str(example_dicts[i][HEIGHTS_KEY])
            )

            raise ValueError(error_string)

        for this_key in keys_to_match:
            if this_key == HEIGHTS_KEY:
                continue

            if example_dict[this_key] == example_dicts[i][this_key]:
                continue

            error_string = (
                '1st and {0:d}th dictionaries have different values for '
                '"{1:s}".  1st dictionary:\n{2:s}\n\n'
                '{0:d}th dictionary:\n{3:s}'
            ).format(
                i + 1, this_key, str(example_dict[this_key]),
                str(example_dicts[i][this_key])
            )

            raise ValueError(error_string)

        for this_key in DICTIONARY_KEYS:
            if this_key in keys_to_match:
                continue

            example_dict[this_key] = numpy.concatenate((
                example_dict[this_key], example_dicts[i][this_key]
            ), axis=0)

    return example_dict


def get_field_from_dict(example_dict, field_name, height_m_agl=None):
    """Returns field from dictionary of examples.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param field_name: Name of field (may be predictor or target variable).
    :param height_m_agl: Height (metres above ground level).  For scalar field,
        `height_m_agl` will not be used.  For vector field, `height_m_agl` will
        be used only if `height_m_agl is not None`.
    :return: data_matrix: numpy array with data values for given field.
    """

    _check_field_name(field_name)

    if field_name in SCALAR_PREDICTOR_NAMES:
        height_m_agl = None
        field_index = example_dict[SCALAR_PREDICTOR_NAMES_KEY].index(field_name)
        data_matrix = example_dict[SCALAR_PREDICTOR_VALS_KEY][..., field_index]
    elif field_name in SCALAR_TARGET_NAMES:
        height_m_agl = None
        field_index = example_dict[SCALAR_TARGET_NAMES_KEY].index(field_name)
        data_matrix = example_dict[SCALAR_TARGET_VALS_KEY][..., field_index]
    elif field_name in VECTOR_PREDICTOR_NAMES:
        field_index = example_dict[VECTOR_PREDICTOR_NAMES_KEY].index(field_name)
        data_matrix = example_dict[VECTOR_PREDICTOR_VALS_KEY][..., field_index]
    else:
        field_index = example_dict[VECTOR_TARGET_NAMES_KEY].index(field_name)
        data_matrix = example_dict[VECTOR_TARGET_VALS_KEY][..., field_index]

    if height_m_agl is None:
        return data_matrix

    height_index = _match_heights(
        heights_m_agl=example_dict[HEIGHTS_KEY],
        desired_height_m_agl=height_m_agl
    )

    return data_matrix[..., height_index]


def reduce_sample_size(example_dict, num_examples_to_keep, test_mode=False):
    """Reduces sample size by randomly removing examples.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param num_examples_to_keep: Number of examples to keep.
    :param test_mode: Leave this alone.
    :return: example_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_integer(num_examples_to_keep)
    error_checking.assert_is_greater(num_examples_to_keep, 0)
    error_checking.assert_is_boolean(test_mode)

    num_examples_total = len(example_dict[VALID_TIMES_KEY])
    if num_examples_total <= num_examples_to_keep:
        return example_dict

    all_indices = numpy.linspace(
        0, num_examples_total - 1, num=num_examples_total, dtype=int
    )

    if test_mode:
        indices_to_keep = all_indices[:num_examples_to_keep]
    else:
        indices_to_keep = numpy.random.choice(
            all_indices, size=num_examples_to_keep, replace=False
        )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][indices_to_keep, ...]

    return example_dict


def subset_by_time(example_dict, first_time_unix_sec, last_time_unix_sec):
    """Subsets examples by time.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param first_time_unix_sec: Earliest time to keep.
    :param last_time_unix_sec: Latest time to keep.
    :return: example_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_integer(first_time_unix_sec)
    error_checking.assert_is_integer(last_time_unix_sec)
    error_checking.assert_is_geq(last_time_unix_sec, first_time_unix_sec)

    good_indices = numpy.where(numpy.logical_and(
        example_dict[VALID_TIMES_KEY] >= first_time_unix_sec,
        example_dict[VALID_TIMES_KEY] <= last_time_unix_sec
    ))[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][good_indices, ...]

    return example_dict


def subset_by_standard_atmo(example_dict, standard_atmo_enum):
    """Subsets examples by standard-atmosphere type.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param standard_atmo_enum: See doc for `check_standard_atmo_type`.
    :return: example_dict: Same as input but with fewer examples.
    """

    check_standard_atmo_type(standard_atmo_enum)

    good_indices = numpy.where(
        example_dict[STANDARD_ATMO_FLAGS_KEY] == standard_atmo_enum,
    )[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        example_dict[this_key] = example_dict[this_key][good_indices, ...]

    return example_dict


def subset_by_field(example_dict, field_names):
    """Subsets examples by field.

    :param example_dict: Dictionary of examples (in the format returned by
        `read_file`).
    :param field_names: 1-D list of field names to keep (each must be accepted
        by `_check_field_name`).
    :return: example_dict: Same as input but with fewer fields.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(field_names), num_dimensions=1
    )

    scalar_predictor_indices = []
    scalar_target_indices = []
    vector_predictor_indices = []
    vector_target_indices = []

    for this_field_name in field_names:
        _check_field_name(this_field_name)

        if this_field_name in SCALAR_PREDICTOR_NAMES:
            scalar_predictor_indices.append(
                example_dict[SCALAR_PREDICTOR_NAMES_KEY].index(this_field_name)
            )
        elif this_field_name in SCALAR_TARGET_NAMES:
            scalar_target_indices.append(
                example_dict[SCALAR_TARGET_NAMES_KEY].index(this_field_name)
            )
        elif this_field_name in VECTOR_PREDICTOR_NAMES:
            vector_predictor_indices.append(
                example_dict[VECTOR_PREDICTOR_NAMES_KEY].index(this_field_name)
            )
        else:
            vector_target_indices.append(
                example_dict[VECTOR_TARGET_NAMES_KEY].index(this_field_name)
            )

    scalar_predictor_indices = numpy.array(scalar_predictor_indices, dtype=int)
    scalar_target_indices = numpy.array(scalar_target_indices, dtype=int)
    vector_predictor_indices = numpy.array(vector_predictor_indices, dtype=int)
    vector_target_indices = numpy.array(vector_target_indices, dtype=int)

    example_dict[SCALAR_PREDICTOR_NAMES_KEY] = [
        example_dict[SCALAR_PREDICTOR_NAMES_KEY][k]
        for k in scalar_predictor_indices
    ]
    example_dict[SCALAR_TARGET_NAMES_KEY] = [
        example_dict[SCALAR_TARGET_NAMES_KEY][k] for k in scalar_target_indices
    ]
    example_dict[VECTOR_PREDICTOR_NAMES_KEY] = [
        example_dict[VECTOR_PREDICTOR_NAMES_KEY][k]
        for k in vector_predictor_indices
    ]
    example_dict[VECTOR_TARGET_NAMES_KEY] = [
        example_dict[VECTOR_TARGET_NAMES_KEY][k] for k in vector_target_indices
    ]

    example_dict[SCALAR_PREDICTOR_VALS_KEY] = (
        example_dict[SCALAR_PREDICTOR_VALS_KEY][..., scalar_predictor_indices]
    )
    example_dict[SCALAR_TARGET_VALS_KEY] = (
        example_dict[SCALAR_TARGET_VALS_KEY][..., scalar_target_indices]
    )
    example_dict[VECTOR_PREDICTOR_VALS_KEY] = (
        example_dict[VECTOR_PREDICTOR_VALS_KEY][..., vector_predictor_indices]
    )
    example_dict[VECTOR_TARGET_VALS_KEY] = (
        example_dict[VECTOR_TARGET_VALS_KEY][..., vector_target_indices]
    )

    return example_dict


def average_examples(
        example_dict, use_pmm,
        max_pmm_percentile_level=DEFAULT_MAX_PMM_PERCENTILE_LEVEL):
    """Averages predictor and target fields over many examples.

    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors
    T_s = number of scalar targets
    T_v = number of vector targets

    :param example_dict: See doc for `read_file`.
    :param use_pmm: Boolean flag.  If True, will use probability-matched means
        for vector fields (vertical profiles).  If False, will use arithmetic
        means for vector fields.
    :param max_pmm_percentile_level: [used only if `use_pmm == True`]
        Max percentile level for probability-matched means.
    :return: mean_example_dict: Dictionary with the following keys.
    mean_example_dict['scalar_predictor_matrix']: numpy array (1 x P_s) with
        values of scalar predictors.
    mean_example_dict['scalar_predictor_names']: Same as input.
    mean_example_dict['vector_predictor_matrix']: numpy array (1 x H x P_v) with
        values of vector predictors.
    mean_example_dict['vector_predictor_names']: Same as input.
    mean_example_dict['scalar_target_matrix']: numpy array (1 x T_s) with values
        of scalar targets.
    mean_example_dict['scalar_predictor_names']: Same as input.
    mean_example_dict['vector_target_matrix']: numpy array (1 x H x T_v) with
        values of vector targets.
    mean_example_dict['vector_predictor_names']: Same as input.
    mean_example_dict['heights_m_agl']: length-H numpy array of heights (metres
        above ground level).
    """

    error_checking.assert_is_boolean(use_pmm)
    error_checking.assert_is_geq(max_pmm_percentile_level, 90.)
    error_checking.assert_is_leq(max_pmm_percentile_level, 100.)

    mean_scalar_predictor_matrix = numpy.mean(
        example_dict[SCALAR_PREDICTOR_VALS_KEY], axis=0
    )
    mean_scalar_predictor_matrix = numpy.expand_dims(
        mean_scalar_predictor_matrix, axis=0
    )

    mean_scalar_target_matrix = numpy.mean(
        example_dict[SCALAR_TARGET_VALS_KEY], axis=0
    )
    mean_scalar_target_matrix = numpy.expand_dims(
        mean_scalar_target_matrix, axis=0
    )

    if use_pmm:
        mean_vector_predictor_matrix = pmm.run_pmm_many_variables(
            input_matrix=example_dict[VECTOR_PREDICTOR_VALS_KEY],
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        mean_vector_predictor_matrix = numpy.mean(
            example_dict[VECTOR_PREDICTOR_VALS_KEY], axis=0
        )

    mean_vector_predictor_matrix = numpy.expand_dims(
        mean_vector_predictor_matrix, axis=0
    )

    if use_pmm:
        mean_vector_target_matrix = pmm.run_pmm_many_variables(
            input_matrix=example_dict[VECTOR_TARGET_VALS_KEY],
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        mean_vector_target_matrix = numpy.mean(
            example_dict[VECTOR_TARGET_VALS_KEY], axis=0
        )

    mean_vector_target_matrix = numpy.expand_dims(
        mean_vector_target_matrix, axis=0
    )

    return {
        SCALAR_PREDICTOR_NAMES_KEY: example_dict[SCALAR_PREDICTOR_NAMES_KEY],
        SCALAR_PREDICTOR_VALS_KEY: mean_scalar_predictor_matrix,
        SCALAR_TARGET_NAMES_KEY: example_dict[SCALAR_TARGET_NAMES_KEY],
        SCALAR_TARGET_VALS_KEY: mean_scalar_target_matrix,
        VECTOR_PREDICTOR_NAMES_KEY: example_dict[VECTOR_PREDICTOR_NAMES_KEY],
        VECTOR_PREDICTOR_VALS_KEY: mean_vector_predictor_matrix,
        VECTOR_TARGET_NAMES_KEY: example_dict[VECTOR_TARGET_NAMES_KEY],
        VECTOR_TARGET_VALS_KEY: mean_vector_target_matrix,
        HEIGHTS_KEY: example_dict[HEIGHTS_KEY]
    }
