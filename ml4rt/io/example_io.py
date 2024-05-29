"""Helper methods for learning examples."""

import os
import glob
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import example_utils

TOLERANCE = 1e-6

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
TARGET_WAVELENGTH_DIM_KEY = 'target_wavelength'
SCALAR_PREDICTOR_DIM_KEY = 'scalar_predictor'
VECTOR_PREDICTOR_DIM_KEY = 'vector_predictor'
SCALAR_TARGET_DIM_KEY = 'scalar_target'
VECTOR_TARGET_DIM_KEY = 'vector_target'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'
FIELD_NAME_CHAR_DIM_KEY = 'field_name_char'

NORMALIZATION_FILE_KEY = 'normalization_file_name'
NORMALIZE_PREDICTORS_KEY = 'normalize_predictors'
NORMALIZE_SCALAR_TARGETS_KEY = 'normalize_scalar_targets'
NORMALIZE_VECTOR_TARGETS_KEY = 'normalize_vector_targets'

NORM_METADATA_STRING_KEYS = [NORMALIZATION_FILE_KEY]
NORM_METADATA_BOOLEAN_KEYS = [
    NORMALIZE_PREDICTORS_KEY,
    NORMALIZE_SCALAR_TARGETS_KEY,
    NORMALIZE_VECTOR_TARGETS_KEY
]

DEFAULT_NORM_METADATA_DICT = {
    NORMALIZATION_FILE_KEY: None,
    NORMALIZE_PREDICTORS_KEY: False,
    NORMALIZE_SCALAR_TARGETS_KEY: False,
    NORMALIZE_VECTOR_TARGETS_KEY: False
}


def _check_normalization_metadata(normalization_metadata_dict):
    """Error-checks metadata for normalization.

    :param normalization_metadata_dict: Dictionary with the following keys.
    normalization_metadata_dict["normalization_file_name"]: Path to
        normalization file, readable by `normalization.read_params`.
    normalization_metadata_dict["normalize_predictors"]: Boolean flag,
        indicating whether or not predictor variables are normalized.
    normalization_metadata_dict["normalize_scalar_targets"]: Boolean flag,
        indicating whether or not scalar target variables (fluxes) are
        normalized.
    normalization_metadata_dict["normalize_vector_targets"]: Boolean flag,
        indicating whether or not vector target variables (heating rates) are
        normalized.

    :return: normalization_metadata_dict: Same as input, but some values may
        have been replaced with defaults.
    """

    orig_metadata_dict = normalization_metadata_dict.copy()
    normalization_metadata_dict = DEFAULT_NORM_METADATA_DICT.copy()
    normalization_metadata_dict.update(orig_metadata_dict)

    if normalization_metadata_dict[NORMALIZATION_FILE_KEY] is None:
        return DEFAULT_NORM_METADATA_DICT

    nmd = normalization_metadata_dict
    error_checking.assert_is_boolean(nmd[NORMALIZE_PREDICTORS_KEY])
    error_checking.assert_is_boolean(nmd[NORMALIZE_SCALAR_TARGETS_KEY])
    error_checking.assert_is_boolean(nmd[NORMALIZE_VECTOR_TARGETS_KEY])

    normalize_any = (
        nmd[NORMALIZE_PREDICTORS_KEY]
        or nmd[NORMALIZE_SCALAR_TARGETS_KEY]
        or nmd[NORMALIZE_VECTOR_TARGETS_KEY]
    )
    assert normalize_any

    return normalization_metadata_dict


def are_normalization_metadata_same(first_metadata_dict, second_metadata_dict):
    """Determines whether two sets of normalization metadata are the same.

    :param first_metadata_dict: See doc for `_check_normalization_metadata`.
    :param second_metadata_dict: Same.
    :return: are_metadata_same: Boolean flag.
    """

    first_metadata_dict = _check_normalization_metadata(first_metadata_dict)
    second_metadata_dict = _check_normalization_metadata(second_metadata_dict)

    first_norm_file_name = first_metadata_dict[NORMALIZATION_FILE_KEY]
    second_norm_file_name = second_metadata_dict[NORMALIZATION_FILE_KEY]

    if first_norm_file_name is not None:
        first_norm_file_name = first_norm_file_name.replace(
            '/training_perturbed_for_uq/', '/training_all_perturbed_for_uq/'
        )
    if second_norm_file_name is not None:
        second_norm_file_name = second_norm_file_name.replace(
            '/training_perturbed_for_uq/', '/training_all_perturbed_for_uq/'
        )

    if first_norm_file_name != second_norm_file_name:
        return False

    for this_key in NORM_METADATA_STRING_KEYS + NORM_METADATA_BOOLEAN_KEYS:
        if this_key == NORMALIZATION_FILE_KEY:
            continue

        if first_metadata_dict[this_key] != second_metadata_dict[this_key]:
            return False

    return True


def find_file(directory_name, year, year_part_number=None,
              raise_error_if_missing=True):
    """Finds NetCDF file with learning examples.

    :param directory_name: Name of directory where file is expected.
    :param year: Year (integer).
    :param year_part_number: Part of year.  If you are looking for a file
        containing the whole year, make this None.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: example_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)
    error_checking.assert_is_integer(year)
    error_checking.assert_is_boolean(raise_error_if_missing)

    if year_part_number is not None:
        error_checking.assert_is_integer(year_part_number)
        error_checking.assert_is_greater(year_part_number, 0)

    example_file_name = '{0:s}/learning_examples_{1:04d}{2:s}.nc'.format(
        directory_name, year,
        '' if year_part_number is None
        else '_part{0:02d}'.format(year_part_number)
    )

    if raise_error_if_missing and not os.path.isfile(example_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            example_file_name
        )
        raise ValueError(error_string)

    return example_file_name


def find_files_one_year(directory_name, year, raise_error_if_missing=True):
    """Finds NetCDF files with learning examples for one year.

    :param directory_name: Name of directory where file is expected.
    :param year: Year (integer).
    :param raise_error_if_missing: Boolean flag.  If any file is missing and
        `raise_error_if_missing == True`, will throw error.  If any file is
        missing and `raise_error_if_missing == False`, will return empty list.
    :return: example_file_names: List of file paths.
    :raises: ValueError: if any file is missing
        and `raise_error_if_missing == True`.
    """

    example_file_name = find_file(
        directory_name=directory_name, year=year, year_part_number=None,
        raise_error_if_missing=False
    )

    if os.path.isfile(example_file_name):
        return [example_file_name]

    example_file_pattern = (
        '{0:s}/learning_examples_{1:04d}_part[0-9][0-9].nc'
    ).format(directory_name, year)

    example_file_names = glob.glob(example_file_pattern)

    if len(example_file_names) == 0:
        if raise_error_if_missing:
            error_string = (
                'Cannot find any file with the following pattern: "{0:s}"'
            ).format(example_file_pattern)

            raise ValueError(error_string)

        return []

    year_part_numbers = numpy.array(
        [file_name_to_year_part(f) for f in example_file_names], dtype=int
    )

    sort_indices = numpy.argsort(year_part_numbers)
    year_part_numbers = year_part_numbers[sort_indices]
    example_file_names = [example_file_names[k] for k in sort_indices]

    if raise_error_if_missing:
        assert numpy.all(numpy.diff(year_part_numbers) == 1)

    return example_file_names


def file_name_to_year(example_file_name):
    """Parses year from file name.

    :param example_file_name: Path to example file (readable by `read_file`).
    :return: year: Year (integer).
    """

    error_checking.assert_is_string(example_file_name)
    pathless_file_name = os.path.split(example_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    words = extensionless_file_name.split('_')

    if words[-1].startswith('part'):
        return int(words[-2])

    return int(words[-1])


def file_name_to_year_part(example_file_name):
    """Parses year part from file name.

    :param example_file_name: Path to example file (readable by `read_file`).
    :return: year_part_number: Year part (integer).
    """

    error_checking.assert_is_string(example_file_name)
    pathless_file_name = os.path.split(example_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    words = extensionless_file_name.split('_')

    if not words[-1].startswith('part'):
        return None

    return int(words[-1].replace('part', ''))


def find_many_files(
        directory_name, first_time_unix_sec, last_time_unix_sec,
        raise_error_if_any_missing=True, raise_error_if_all_missing=True):
    """Finds many NetCDF files with learning examples.

    :param directory_name: Name of directory where files are expected.
    :param first_time_unix_sec: First time at which examples are desired.
    :param last_time_unix_sec: Last time at which examples are desired.
    :param raise_error_if_any_missing: Boolean flag.  If any file is missing and
        `raise_error_if_any_missing == True`, will throw error.
    :param raise_error_if_all_missing: Boolean flag.  If all files are missing
        and `raise_error_if_all_missing == True`, will throw error.
    :return: example_file_names: 1-D list of paths to example files.  This list
        does *not* contain expected paths to non-existent files.
    """

    error_checking.assert_is_boolean(raise_error_if_any_missing)
    error_checking.assert_is_boolean(raise_error_if_all_missing)

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
        example_file_names += find_files_one_year(
            directory_name=directory_name, year=this_year,
            raise_error_if_missing=raise_error_if_any_missing
        )

    if raise_error_if_all_missing and len(example_file_names) == 0:
        error_string = (
            'Cannot find any file in directory "{0:s}" from years {1:d}-{2:d}.'
        ).format(
            directory_name, start_year, end_year
        )
        raise ValueError(error_string)

    return example_file_names


def read_file(
        netcdf_file_name, max_shortwave_heating_k_day01=41.5,
        min_longwave_heating_k_day01=-90., max_longwave_heating_k_day01=50.):
    """Reads learning examples from NetCDF file.

    E = number of examples
    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors
    T_s = number of scalar targets
    T_v = number of vector targets
    W = number of wavelengths

    :param netcdf_file_name: Path to input file.
    :param max_shortwave_heating_k_day01: Max shortwave heating rate.  Will not
        read any examples with greater heating rate anywhere in profile.
    :param min_longwave_heating_k_day01: Same but for minimum longwave heating.
    :param max_longwave_heating_k_day01: Same but for max longwave heating.

    :return: example_dict: Dictionary with the following keys.
    example_dict['scalar_predictor_matrix']: numpy array (E x P_s) with values
        of scalar predictors.
    example_dict['scalar_predictor_names']: list (length P_s) with names of
        scalar predictors.
    example_dict['vector_predictor_matrix']: numpy array (E x H x P_v) with
        values of vector predictors.
    example_dict['vector_predictor_names']: list (length P_v) with names of
        vector predictors.
    example_dict['scalar_target_matrix']: numpy array (E x W x T_s) with values
        of scalar targets.
    example_dict['scalar_target_names']: list (length T_s) with names of scalar
        targets.
    example_dict['target_wavelengths_metres']: length-W numpy array of
        wavelengths.
    example_dict['vector_target_matrix']: numpy array (E x H x W x T_v) with
        values of vector targets.
    example_dict['vector_target_names']: list (length T_v) with names of vector
        targets.
    example_dict['valid_times_unix_sec']: length-E numpy array of valid times
        (Unix seconds).
    example_dict['heights_m_agl']: length-H numpy array of heights (metres above
        ground level).
    example_dict['standard_atmo_flags']: length-E numpy array of flags (each in
        the list `STANDARD_ATMO_ENUMS`).
    example_dict['example_id_strings']: length-E list of example IDs.
    example_dict['normalization_metadata_dict']: See doc for
        `_check_normalization_metadata`.
    """

    error_checking.assert_is_not_nan(max_shortwave_heating_k_day01)
    error_checking.assert_is_greater(
        max_longwave_heating_k_day01, min_longwave_heating_k_day01
    )

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    nmd = dict()

    if hasattr(dataset_object, NORMALIZE_PREDICTORS_KEY):
        for this_key in NORM_METADATA_STRING_KEYS:
            nmd[this_key] = str(getattr(dataset_object, this_key))
            if nmd[this_key] == 'None':
                nmd[this_key] = None

        for this_key in NORM_METADATA_BOOLEAN_KEYS:
            nmd[this_key] = bool(getattr(dataset_object, this_key))

    elif hasattr(dataset_object, NORMALIZATION_FILE_KEY):
        nmd[NORMALIZATION_FILE_KEY] = str(
            getattr(dataset_object, NORMALIZATION_FILE_KEY)
        )

        predictor_norm_type_string = str(
            getattr(dataset_object, 'predictor_norm_type_string')
        )
        nmd[NORMALIZE_PREDICTORS_KEY] = predictor_norm_type_string != 'None'

        scalar_target_norm_type_string = str(
            getattr(dataset_object, 'scalar_target_norm_type_string')
        )
        nmd[NORMALIZE_SCALAR_TARGETS_KEY] = (
            scalar_target_norm_type_string != 'None'
        )

        vector_target_norm_type_string = str(
            getattr(dataset_object, 'vector_target_norm_type_string')
        )
        nmd[NORMALIZE_VECTOR_TARGETS_KEY] = (
            vector_target_norm_type_string != 'None'
        )
    else:
        nmd = _check_normalization_metadata(nmd)

    normalization_metadata_dict = nmd

    example_id_strings = [
        str(id) for id in netCDF4.chartostring(
            dataset_object.variables[example_utils.EXAMPLE_IDS_KEY][:]
        )
    ]

    found_wavelengths = (
        example_utils.TARGET_WAVELENGTHS_KEY in dataset_object.variables
    )
    if found_wavelengths:
        target_wavelengths_metres = numpy.array(
            dataset_object.variables[example_utils.TARGET_WAVELENGTHS_KEY][:],
            dtype=float
        )
    else:
        target_wavelengths_metres = numpy.array(
            [example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]
        )

    example_dict = {
        example_utils.EXAMPLE_IDS_KEY: example_id_strings,
        example_utils.NORMALIZATION_METADATA_KEY: normalization_metadata_dict,
        example_utils.TARGET_WAVELENGTHS_KEY: target_wavelengths_metres
    }

    string_keys = [
        example_utils.SCALAR_PREDICTOR_NAMES_KEY,
        example_utils.VECTOR_PREDICTOR_NAMES_KEY,
        example_utils.SCALAR_TARGET_NAMES_KEY,
        example_utils.VECTOR_TARGET_NAMES_KEY
    ]
    main_data_keys = [
        example_utils.SCALAR_PREDICTOR_VALS_KEY,
        example_utils.VECTOR_PREDICTOR_VALS_KEY,
        example_utils.SCALAR_TARGET_VALS_KEY,
        example_utils.VECTOR_TARGET_VALS_KEY
    ]
    integer_keys = [
        example_utils.VALID_TIMES_KEY, example_utils.STANDARD_ATMO_FLAGS_KEY
    ]

    for this_key in string_keys:
        example_dict[this_key] = [
            str(n) for n in
            netCDF4.chartostring(dataset_object.variables[this_key][:])
        ]

    for this_key in main_data_keys:
        example_dict[this_key] = numpy.array(
            dataset_object.variables[this_key][:], dtype=numpy.float32
        )

    if not found_wavelengths:
        for this_key in [
                example_utils.SCALAR_TARGET_VALS_KEY,
                example_utils.VECTOR_TARGET_VALS_KEY
        ]:
            example_dict[this_key] = numpy.expand_dims(
                example_dict[this_key], axis=-2
            )

    for this_key in integer_keys:
        example_dict[this_key] = numpy.array(
            numpy.round(dataset_object.variables[this_key][:]), dtype=int
        )

    example_dict[example_utils.HEIGHTS_KEY] = numpy.array(
        dataset_object.variables[example_utils.HEIGHTS_KEY][:], dtype=float
    )

    dataset_object.close()

    if (
            example_utils.SHORTWAVE_HEATING_RATE_NAME not in
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    ):
        max_shortwave_heating_k_day01 = numpy.inf

    if (
            example_utils.LONGWAVE_HEATING_RATE_NAME not in
            example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    ):
        min_longwave_heating_k_day01 = -numpy.inf
        max_longwave_heating_k_day01 = numpy.inf

    if not numpy.isinf(max_shortwave_heating_k_day01):
        heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_utils.SHORTWAVE_HEATING_RATE_NAME
        )
        good_example_flags = numpy.all(
            heating_rate_matrix_k_day01 <= max_shortwave_heating_k_day01,
            axis=(1, 2)
        )
        good_example_indices = numpy.where(good_example_flags)[0]

        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_example_indices
        )

    if not numpy.isinf(min_longwave_heating_k_day01):
        heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_utils.LONGWAVE_HEATING_RATE_NAME
        )
        good_example_flags = numpy.all(
            heating_rate_matrix_k_day01 >= min_longwave_heating_k_day01,
            axis=(1, 2)
        )
        good_example_indices = numpy.where(good_example_flags)[0]

        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_example_indices
        )

    if not numpy.isinf(max_longwave_heating_k_day01):
        heating_rate_matrix_k_day01 = example_utils.get_field_from_dict(
            example_dict=example_dict,
            field_name=example_utils.LONGWAVE_HEATING_RATE_NAME
        )
        good_example_flags = numpy.all(
            heating_rate_matrix_k_day01 <= max_longwave_heating_k_day01,
            axis=(1, 2)
        )
        good_example_indices = numpy.where(good_example_flags)[0]

        example_dict = example_utils.subset_by_index(
            example_dict=example_dict, desired_indices=good_example_indices
        )

    return example_dict


def write_file(example_dict, netcdf_file_name):
    """Writes learning examples to NetCDF file.

    :param example_dict: See doc for `read_processed_file`.
    :param netcdf_file_name: Path to output file.
    """

    if example_utils.NORMALIZATION_METADATA_KEY in example_dict:
        normalization_metadata_dict = _check_normalization_metadata(
            example_dict[example_utils.NORMALIZATION_METADATA_KEY]
        )
    else:
        normalization_metadata_dict = _check_normalization_metadata(
            dict()
        )

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(netcdf_file_name, 'w', format='NETCDF4')

    for this_key in NORM_METADATA_STRING_KEYS:
        dataset_object.setncattr(
            this_key, str(normalization_metadata_dict[this_key])
        )
    for this_key in NORM_METADATA_BOOLEAN_KEYS:
        dataset_object.setncattr(
            this_key, int(normalization_metadata_dict[this_key])
        )

    num_examples = len(example_dict[example_utils.VALID_TIMES_KEY])
    num_heights = len(example_dict[example_utils.HEIGHTS_KEY])
    num_scalar_predictors = len(
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY]
    )
    num_vector_predictors = len(
        example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY]
    )
    num_scalar_targets = len(
        example_dict[example_utils.SCALAR_TARGET_NAMES_KEY]
    )
    num_vector_targets = len(
        example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )
    num_target_wavelengths = len(
        example_dict[example_utils.TARGET_WAVELENGTHS_KEY]
    )

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(HEIGHT_DIMENSION_KEY, num_heights)
    dataset_object.createDimension(
        SCALAR_PREDICTOR_DIM_KEY, num_scalar_predictors
    )
    dataset_object.createDimension(
        VECTOR_PREDICTOR_DIM_KEY, num_vector_predictors
    )
    dataset_object.createDimension(SCALAR_TARGET_DIM_KEY, num_scalar_targets)
    dataset_object.createDimension(VECTOR_TARGET_DIM_KEY, num_vector_targets)
    dataset_object.createDimension(
        TARGET_WAVELENGTH_DIM_KEY, num_target_wavelengths
    )

    example_id_strings = example_dict[example_utils.EXAMPLE_IDS_KEY]
    num_example_id_chars = numpy.max(numpy.array([
        len(id) for id in example_id_strings
    ]))
    dataset_object.createDimension(
        EXAMPLE_ID_CHAR_DIM_KEY, num_example_id_chars
    )

    field_names = (
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY] +
        example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY] +
        example_dict[example_utils.SCALAR_TARGET_NAMES_KEY] +
        example_dict[example_utils.VECTOR_TARGET_NAMES_KEY]
    )
    num_field_name_chars = numpy.max(numpy.array([
        len(n) for n in field_names
    ]))
    dataset_object.createDimension(
        FIELD_NAME_CHAR_DIM_KEY, num_field_name_chars
    )

    # Add heights.
    dataset_object.createVariable(
        example_utils.HEIGHTS_KEY, datatype=numpy.float32,
        dimensions=HEIGHT_DIMENSION_KEY
    )
    dataset_object.variables[example_utils.HEIGHTS_KEY][:] = (
        example_dict[example_utils.HEIGHTS_KEY]
    )

    # Add target wavelengths.
    dataset_object.createVariable(
        example_utils.TARGET_WAVELENGTHS_KEY, datatype=numpy.float32,
        dimensions=TARGET_WAVELENGTH_DIM_KEY
    )
    dataset_object.variables[example_utils.TARGET_WAVELENGTHS_KEY][:] = (
        example_dict[example_utils.TARGET_WAVELENGTHS_KEY]
    )

    # Add standard-atmosphere flags.
    dataset_object.createVariable(
        example_utils.STANDARD_ATMO_FLAGS_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[example_utils.STANDARD_ATMO_FLAGS_KEY][:] = (
        example_dict[example_utils.STANDARD_ATMO_FLAGS_KEY]
    )

    # Add valid times.
    dataset_object.createVariable(
        example_utils.VALID_TIMES_KEY, datatype=numpy.int32,
        dimensions=EXAMPLE_DIMENSION_KEY
    )
    dataset_object.variables[example_utils.VALID_TIMES_KEY][:] = (
        example_dict[example_utils.VALID_TIMES_KEY]
    )

    # Add example IDs.
    this_string_format = 'S{0:d}'.format(num_example_id_chars)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        example_utils.EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[example_utils.EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array
    )

    # Add field names.
    this_string_format = 'S{0:d}'.format(num_field_name_chars)

    scalar_predictor_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[example_utils.SCALAR_PREDICTOR_NAMES_KEY],
        dtype=this_string_format
    ))
    dataset_object.createVariable(
        example_utils.SCALAR_PREDICTOR_NAMES_KEY, datatype='S1',
        dimensions=(SCALAR_PREDICTOR_DIM_KEY, FIELD_NAME_CHAR_DIM_KEY)
    )
    dataset_object.variables[example_utils.SCALAR_PREDICTOR_NAMES_KEY][:] = (
        numpy.array(scalar_predictor_names_char_array)
    )

    vector_predictor_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[example_utils.VECTOR_PREDICTOR_NAMES_KEY],
        dtype=this_string_format
    ))
    dataset_object.createVariable(
        example_utils.VECTOR_PREDICTOR_NAMES_KEY, datatype='S1',
        dimensions=(VECTOR_PREDICTOR_DIM_KEY, FIELD_NAME_CHAR_DIM_KEY)
    )
    dataset_object.variables[example_utils.VECTOR_PREDICTOR_NAMES_KEY][:] = (
        numpy.array(vector_predictor_names_char_array)
    )

    scalar_target_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[example_utils.SCALAR_TARGET_NAMES_KEY],
        dtype=this_string_format
    ))
    dataset_object.createVariable(
        example_utils.SCALAR_TARGET_NAMES_KEY, datatype='S1',
        dimensions=(SCALAR_TARGET_DIM_KEY, FIELD_NAME_CHAR_DIM_KEY)
    )
    dataset_object.variables[example_utils.SCALAR_TARGET_NAMES_KEY][:] = (
        numpy.array(scalar_target_names_char_array)
    )

    vector_target_names_char_array = netCDF4.stringtochar(numpy.array(
        example_dict[example_utils.VECTOR_TARGET_NAMES_KEY],
        dtype=this_string_format
    ))
    dataset_object.createVariable(
        example_utils.VECTOR_TARGET_NAMES_KEY, datatype='S1',
        dimensions=(VECTOR_TARGET_DIM_KEY, FIELD_NAME_CHAR_DIM_KEY)
    )
    dataset_object.variables[example_utils.VECTOR_TARGET_NAMES_KEY][:] = (
        numpy.array(vector_target_names_char_array)
    )

    # Add main data.
    dataset_object.createVariable(
        example_utils.SCALAR_PREDICTOR_VALS_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, SCALAR_PREDICTOR_DIM_KEY)
    )
    dataset_object.variables[example_utils.SCALAR_PREDICTOR_VALS_KEY][:] = (
        example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY]
    )

    dataset_object.createVariable(
        example_utils.VECTOR_PREDICTOR_VALS_KEY, datatype=numpy.float32,
        dimensions=(
            EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
            VECTOR_PREDICTOR_DIM_KEY
        )
    )
    dataset_object.variables[example_utils.VECTOR_PREDICTOR_VALS_KEY][:] = (
        example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY]
    )

    dataset_object.createVariable(
        example_utils.SCALAR_TARGET_VALS_KEY, datatype=numpy.float32,
        dimensions=(
            EXAMPLE_DIMENSION_KEY, TARGET_WAVELENGTH_DIM_KEY,
            SCALAR_TARGET_DIM_KEY
        )
    )
    dataset_object.variables[example_utils.SCALAR_TARGET_VALS_KEY][:] = (
        example_dict[example_utils.SCALAR_TARGET_VALS_KEY]
    )

    dataset_object.createVariable(
        example_utils.VECTOR_TARGET_VALS_KEY, datatype=numpy.float32,
        dimensions=(
            EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
            TARGET_WAVELENGTH_DIM_KEY, VECTOR_TARGET_DIM_KEY
        )
    )
    dataset_object.variables[example_utils.VECTOR_TARGET_VALS_KEY][:] = (
        example_dict[example_utils.VECTOR_TARGET_VALS_KEY]
    )

    dataset_object.close()
