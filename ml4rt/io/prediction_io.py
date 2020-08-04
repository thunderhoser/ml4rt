"""Input/output methods for model predictions."""

import os.path
import numpy
import netCDF4
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import longitude_conversion as lng_conversion
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
VECTOR_TARGET_DIMENSION_KEY = 'vector_target'
SCALAR_TARGET_DIMENSION_KEY = 'scalar_target'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'

MODEL_FILE_KEY = 'model_file_name'
SCALAR_TARGETS_KEY = 'scalar_target_matrix'
SCALAR_PREDICTIONS_KEY = 'scalar_prediction_matrix'
VECTOR_TARGETS_KEY = 'vector_target_matrix'
VECTOR_PREDICTIONS_KEY = 'vector_prediction_matrix'
HEIGHTS_KEY = 'heights_m_agl'
EXAMPLE_IDS_KEY = 'example_id_strings'

ONE_PER_EXAMPLE_KEYS = [
    SCALAR_TARGETS_KEY, SCALAR_PREDICTIONS_KEY,
    VECTOR_TARGETS_KEY, VECTOR_PREDICTIONS_KEY, EXAMPLE_IDS_KEY
]

DEFAULT_MAX_PMM_PERCENTILE_LEVEL = 99.
MAX_ZENITH_ANGLE_RADIANS = numpy.pi / 2

ZENITH_ANGLE_BIN_KEY = 'zenith_angle_bin'
MONTH_KEY = 'month'
GRID_ROW_KEY = 'grid_row'
GRID_COLUMN_KEY = 'grid_column'
METADATA_KEYS = [ZENITH_ANGLE_BIN_KEY, MONTH_KEY, GRID_ROW_KEY, GRID_COLUMN_KEY]

GRID_ROW_DIMENSION_KEY = 'row'
GRID_COLUMN_DIMENSION_KEY = 'column'
LATITUDES_KEY = 'latitude_deg_n'
LONGITUDES_KEY = 'longitude_deg_e'


def find_file(directory_name, zenith_angle_bin=None, month=None, grid_row=None,
              grid_column=None, raise_error_if_missing=True):
    """Finds NetCDF file with predictions.

    :param directory_name: Name of directory where file is expected.
    :param zenith_angle_bin: Zenith-angle bin (non-negative integer).  If file
        does not contain predictions for a specific zenith-angle bin, leave this
        alone.
    :param month: Month (integer from 1...12).  If file does not contain
        predictions for a specific month, leave this alone.
    :param grid_row: Grid row (non-negative integer).  If file does not contain
        predictions for a specific spatial region, leave this alone.
    :param grid_column: Same but for grid column.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: prediction_file_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(directory_name)

    if zenith_angle_bin is not None:
        error_checking.assert_is_integer(zenith_angle_bin)
        error_checking.assert_is_geq(zenith_angle_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, ZENITH_ANGLE_BIN_KEY.replace('_', '-'),
            zenith_angle_bin
        )

    elif month is not None:
        error_checking.assert_is_integer(month)
        error_checking.assert_is_geq(month, 1)
        error_checking.assert_is_leq(month, 12)

        prediction_file_name = '{0:s}/predictions_{1:s}={2:02d}.nc'.format(
            directory_name, MONTH_KEY.replace('_', '-'), month
        )

    elif grid_row is not None or grid_column is not None:
        error_checking.assert_is_integer(grid_row)
        error_checking.assert_is_geq(grid_row, 0)
        error_checking.assert_is_integer(grid_column)
        error_checking.assert_is_geq(grid_column, 0)

        prediction_file_name = (
            '{0:s}/{1:s}={2:03d}/predictions_{1:s}={2:03d}_{3:s}={4:03d}.nc'
        ).format(
            directory_name, GRID_ROW_KEY.replace('_', '-'), grid_row,
            GRID_COLUMN_KEY.replace('_', '-'), grid_column
        )

    else:
        prediction_file_name = '{0:s}/predictions.nc'.format(
            directory_name, grid_row, grid_column
        )

    if raise_error_if_missing and not os.path.isfile(prediction_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            prediction_file_name
        )
        raise ValueError(error_string)

    return prediction_file_name


def file_name_to_metadata(prediction_file_name):
    """Parses metadata from file name.

    This method is the inverse of `find_file`.

    :param prediction_file_name: Path to NetCDF file with predictions.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict['zenith_angle_bin']: See input doc for `find_file`.
    metadata_dict['month']: Same.
    metadata_dict['grid_row']: Same.
    metadata_dict['grid_column']: Same.
    """

    error_checking.assert_is_string(prediction_file_name)

    metadata_dict = dict()
    for this_key in METADATA_KEYS:
        metadata_dict[this_key] = None

    pathless_file_name = os.path.split(prediction_file_name)[-1]
    extensionless_file_name = os.path.splitext(pathless_file_name)[0]
    words = extensionless_file_name.split('_')

    for this_key in METADATA_KEYS:
        this_key_with_dashes = this_key.replace('_', '-')
        if this_key_with_dashes not in words[-1]:
            continue

        metadata_dict[this_key] = int(
            words[-1].replace(this_key_with_dashes + '=', '')
        )
        break

    if metadata_dict[GRID_COLUMN_KEY] is not None:
        this_key_with_dashes = GRID_ROW_KEY.replace('_', '-')
        metadata_dict[GRID_ROW_KEY] = int(
            words[-2].replace(this_key_with_dashes + '=', '')
        )

    return metadata_dict


def write_file(
        netcdf_file_name, scalar_target_matrix, vector_target_matrix,
        scalar_prediction_matrix, vector_prediction_matrix, heights_m_agl,
        example_id_strings, model_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    H = number of heights
    T_s = number of scalar targets
    T_v = number of vector targets

    :param netcdf_file_name: Path to output file.
    :param scalar_target_matrix: numpy array (E x T_s) with actual values of
        scalar targets.
    :param vector_target_matrix: numpy array (E x H x T_v) with actual values of
        vector targets.
    :param scalar_prediction_matrix: Same as `scalar_target_matrix` but with
        predicted values.
    :param vector_prediction_matrix: Same as `vector_target_matrix` but with
        predicted values.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param example_id_strings: length-E list of IDs created by
        `example_utils.create_example_ids`.
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)
    error_checking.assert_is_numpy_array(scalar_target_matrix, num_dimensions=2)

    error_checking.assert_is_numpy_array_without_nan(scalar_prediction_matrix)
    error_checking.assert_is_numpy_array(
        scalar_prediction_matrix,
        exact_dimensions=numpy.array(scalar_target_matrix.shape, dtype=int)
    )

    error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
    error_checking.assert_is_numpy_array(vector_target_matrix, num_dimensions=3)

    num_examples = scalar_target_matrix.shape[0]
    expected_dim = numpy.array(
        (num_examples,) + vector_target_matrix.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        vector_target_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(vector_prediction_matrix)
    error_checking.assert_is_numpy_array(
        vector_prediction_matrix,
        exact_dimensions=numpy.array(vector_target_matrix.shape, dtype=int)
    )

    num_heights = vector_target_matrix.shape[1]
    error_checking.assert_is_greater_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(
        heights_m_agl,
        exact_dimensions=numpy.array([num_heights], dtype=int)
    )

    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings),
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    example_utils.parse_example_ids(example_id_strings)

    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)

    num_examples = vector_target_matrix.shape[0]
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        HEIGHT_DIMENSION_KEY, vector_target_matrix.shape[1]
    )
    dataset_object.createDimension(
        VECTOR_TARGET_DIMENSION_KEY, vector_target_matrix.shape[2]
    )

    num_scalar_targets = scalar_target_matrix.shape[1]
    if num_scalar_targets > 0:
        dataset_object.createDimension(
            SCALAR_TARGET_DIMENSION_KEY, scalar_target_matrix.shape[1]
        )

    if num_examples == 0:
        num_id_characters = 1
    else:
        num_id_characters = numpy.max(numpy.array([
            len(id) for id in example_id_strings
        ]))

    dataset_object.createDimension(EXAMPLE_ID_CHAR_DIM_KEY, num_id_characters)

    this_string_format = 'S{0:d}'.format(num_id_characters)
    example_ids_char_array = netCDF4.stringtochar(numpy.array(
        example_id_strings, dtype=this_string_format
    ))

    dataset_object.createVariable(
        EXAMPLE_IDS_KEY, datatype='S1',
        dimensions=(EXAMPLE_DIMENSION_KEY, EXAMPLE_ID_CHAR_DIM_KEY)
    )
    dataset_object.variables[EXAMPLE_IDS_KEY][:] = numpy.array(
        example_ids_char_array
    )

    dataset_object.createVariable(
        HEIGHTS_KEY, datatype=numpy.float32, dimensions=HEIGHT_DIMENSION_KEY
    )
    dataset_object.variables[HEIGHTS_KEY][:] = heights_m_agl

    if num_scalar_targets > 0:
        dataset_object.createVariable(
            SCALAR_TARGETS_KEY, datatype=numpy.float32,
            dimensions=(EXAMPLE_DIMENSION_KEY, SCALAR_TARGET_DIMENSION_KEY)
        )
        dataset_object.variables[SCALAR_TARGETS_KEY][:] = scalar_target_matrix

        dataset_object.createVariable(
            SCALAR_PREDICTIONS_KEY, datatype=numpy.float32,
            dimensions=(EXAMPLE_DIMENSION_KEY, SCALAR_TARGET_DIMENSION_KEY)
        )
        dataset_object.variables[SCALAR_PREDICTIONS_KEY][:] = (
            scalar_prediction_matrix
        )

    these_dimensions = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY, VECTOR_TARGET_DIMENSION_KEY
    )

    dataset_object.createVariable(
        VECTOR_TARGETS_KEY, datatype=numpy.float32, dimensions=these_dimensions
    )
    dataset_object.variables[VECTOR_TARGETS_KEY][:] = vector_target_matrix

    dataset_object.createVariable(
        VECTOR_PREDICTIONS_KEY, datatype=numpy.float32,
        dimensions=these_dimensions
    )
    dataset_object.variables[VECTOR_PREDICTIONS_KEY][:] = (
        vector_prediction_matrix
    )

    dataset_object.close()


def read_file(netcdf_file_name):
    """Reads predictions from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: prediction_dict: Dictionary with the following keys.
    prediction_dict['scalar_target_matrix']: See doc for `write_file`.
    prediction_dict['scalar_prediction_matrix']: Same.
    prediction_dict['vector_target_matrix']: Same.
    prediction_dict['vector_prediction_matrix']: Same.
    prediction_dict['example_id_strings']: Same.
    prediction_dict['model_file_name']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        VECTOR_TARGETS_KEY: dataset_object.variables[VECTOR_TARGETS_KEY][:],
        VECTOR_PREDICTIONS_KEY:
            dataset_object.variables[VECTOR_PREDICTIONS_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
    }

    if HEIGHTS_KEY in dataset_object.variables:
        prediction_dict[HEIGHTS_KEY] = dataset_object.variables[HEIGHTS_KEY][:]
    else:
        model_metafile_name = neural_net.find_metafile(
            model_dir_name=os.path.split(prediction_dict[MODEL_FILE_KEY])[0],
            raise_error_if_missing=True
        )

        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        generator_option_dict = (
            model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]
        )
        prediction_dict[HEIGHTS_KEY] = (
            generator_option_dict[neural_net.HEIGHTS_KEY]
        )

    if SCALAR_TARGETS_KEY in dataset_object.variables:
        prediction_dict[SCALAR_TARGETS_KEY] = (
            dataset_object.variables[SCALAR_TARGETS_KEY][:]
        )
        prediction_dict[SCALAR_PREDICTIONS_KEY] = (
            dataset_object.variables[SCALAR_PREDICTIONS_KEY][:]
        )
    else:
        num_examples = prediction_dict[VECTOR_TARGETS_KEY].shape[0]
        prediction_dict[SCALAR_TARGETS_KEY] = numpy.full((num_examples, 0), 0.)
        prediction_dict[SCALAR_PREDICTIONS_KEY] = numpy.full(
            (num_examples, 0), 0.
        )

    dataset_object.close()
    return prediction_dict


def find_grid_metafile(prediction_dir_name, raise_error_if_missing=True):
    """Finds file with metadata for grid.

    This file is needed only if prediction files are split by space (one per
    grid cell).

    :param prediction_dir_name: Name of directory with prediction files.  The
        metafile is expected here.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: grid_metafile_name: File path.
    :raises: ValueError: if file is missing
        and `raise_error_if_missing == True`.
    """

    error_checking.assert_is_string(prediction_dir_name)
    grid_metafile_name = '{0:s}/grid_metadata.nc'.format(prediction_dir_name)

    if raise_error_if_missing and not os.path.isfile(grid_metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            grid_metafile_name
        )
        raise ValueError(error_string)

    return grid_metafile_name


def write_grid_metafile(grid_point_latitudes_deg, grid_point_longitudes_deg,
                        netcdf_file_name):
    """Writes metadata for grid to NetCDF file.

    This file is needed only if prediction files are split by space (one per
    grid cell).

    M = number of rows in grid
    N = number of columns in grid

    :param grid_point_latitudes_deg: length-M numpy array of latitudes (deg N).
    :param grid_point_longitudes_deg: length-N numpy array of longitudes
        (deg E).
    :param netcdf_file_name: Path to output file.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        grid_point_latitudes_deg, num_dimensions=1
    )
    error_checking.assert_is_valid_lat_numpy_array(grid_point_latitudes_deg)

    error_checking.assert_is_numpy_array(
        grid_point_longitudes_deg, num_dimensions=1
    )
    grid_point_longitudes_deg = lng_conversion.convert_lng_positive_in_west(
        grid_point_longitudes_deg
    )

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.createDimension(
        GRID_ROW_DIMENSION_KEY, len(grid_point_latitudes_deg)
    )
    dataset_object.createDimension(
        GRID_COLUMN_DIMENSION_KEY, len(grid_point_longitudes_deg)
    )

    dataset_object.createVariable(
        LATITUDES_KEY, datatype=numpy.float32, dimensions=GRID_ROW_DIMENSION_KEY
    )
    dataset_object.variables[LATITUDES_KEY][:] = grid_point_latitudes_deg

    dataset_object.createVariable(
        LONGITUDES_KEY, datatype=numpy.float32,
        dimensions=GRID_COLUMN_DIMENSION_KEY
    )
    dataset_object.variables[LONGITUDES_KEY][:] = grid_point_longitudes_deg

    dataset_object.close()


def read_grid_metafile(netcdf_file_name):
    """Reads metadata for grid from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: grid_point_latitudes_deg: See doc for `write_grid_metafile`.
    :return: grid_point_longitudes_deg: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    grid_point_latitudes_deg = dataset_object.variables[LATITUDES_KEY][:]
    grid_point_longitudes_deg = dataset_object.variables[LONGITUDES_KEY][:]
    dataset_object.close()

    return grid_point_latitudes_deg, grid_point_longitudes_deg


def average_predictions(
        prediction_dict, use_pmm,
        max_pmm_percentile_level=DEFAULT_MAX_PMM_PERCENTILE_LEVEL):
    """Averages predicted and target values over many examples.

    H = number of heights
    T_s = number of scalar targets
    T_v = number of vector targets

    :param prediction_dict: See doc for `write_file`.
    :param use_pmm: Boolean flag.  If True, will use probability-matched means
        for vector fields (vertical profiles).  If False, will use arithmetic
        means for vector fields.
    :param max_pmm_percentile_level: [used only if `use_pmm == True`]
        Max percentile level for probability-matched means.
    :return: mean_prediction_dict: Dictionary with the following keys.
    mean_prediction_dict['scalar_target_matrix']: numpy array (1 x T_s) with
        mean target (actual) values for scalar variables.
    mean_prediction_dict['scalar_prediction_matrix']: Same but with predicted
        values.
    mean_prediction_dict['vector_target_matrix']: numpy array (1 x H x T_v) with
        mean target (actual) values for vector variables.
    mean_prediction_dict['vector_prediction_matrix']: Same but with predicted
        values.
    mean_prediction_dict['heights_m_agl']: length-H numpy array of heights
        (metres above ground level).
    mean_prediction_dict['model_file_name']: Path to file with trained model
        (readable by `neural_net.read_model`).
    """

    error_checking.assert_is_boolean(use_pmm)
    error_checking.assert_is_geq(max_pmm_percentile_level, 90.)
    error_checking.assert_is_leq(max_pmm_percentile_level, 100.)

    mean_scalar_target_matrix = numpy.mean(
        prediction_dict[SCALAR_TARGETS_KEY], axis=0
    )
    mean_scalar_target_matrix = numpy.expand_dims(
        mean_scalar_target_matrix, axis=0
    )

    mean_scalar_prediction_matrix = numpy.mean(
        prediction_dict[SCALAR_PREDICTIONS_KEY], axis=0
    )
    mean_scalar_prediction_matrix = numpy.expand_dims(
        mean_scalar_prediction_matrix, axis=0
    )

    if use_pmm:
        mean_vector_target_matrix = pmm.run_pmm_many_variables(
            input_matrix=prediction_dict[VECTOR_TARGETS_KEY],
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        mean_vector_target_matrix = numpy.mean(
            prediction_dict[VECTOR_TARGETS_KEY], axis=0
        )

    mean_vector_target_matrix = numpy.expand_dims(
        mean_vector_target_matrix, axis=0
    )

    if use_pmm:
        mean_vector_prediction_matrix = pmm.run_pmm_many_variables(
            input_matrix=prediction_dict[VECTOR_PREDICTIONS_KEY],
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        mean_vector_prediction_matrix = numpy.mean(
            prediction_dict[VECTOR_PREDICTIONS_KEY], axis=0
        )

    mean_vector_prediction_matrix = numpy.expand_dims(
        mean_vector_prediction_matrix, axis=0
    )

    return {
        SCALAR_TARGETS_KEY: mean_scalar_target_matrix,
        SCALAR_PREDICTIONS_KEY: mean_scalar_prediction_matrix,
        VECTOR_TARGETS_KEY: mean_vector_target_matrix,
        VECTOR_PREDICTIONS_KEY: mean_vector_prediction_matrix,
        HEIGHTS_KEY: prediction_dict[HEIGHTS_KEY],
        MODEL_FILE_KEY: prediction_dict[MODEL_FILE_KEY]
    }


def subset_by_standard_atmo(prediction_dict, standard_atmo_enum):
    """Subsets examples by standard-atmosphere type.

    :param prediction_dict: See doc for `write_file`.
    :param standard_atmo_enum: See doc for
        `example_utils.check_standard_atmo_type`.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    example_utils.check_standard_atmo_type(standard_atmo_enum)

    all_standard_atmo_enums = example_utils.parse_example_ids(
        prediction_dict[EXAMPLE_IDS_KEY]
    )[example_utils.STANDARD_ATMO_FLAGS_KEY]

    good_indices = numpy.where(all_standard_atmo_enums == standard_atmo_enum)[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if isinstance(prediction_dict[this_key], list):
            prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in good_indices
            ]
        else:
            prediction_dict[this_key] = (
                prediction_dict[this_key][good_indices, ...]
            )

    return prediction_dict


def subset_by_zenith_angle(
        prediction_dict, min_zenith_angle_rad, max_zenith_angle_rad,
        max_inclusive=None):
    """Subsets examples by solar zenith angle.

    :param prediction_dict: See doc for `write_file`.
    :param min_zenith_angle_rad: Minimum zenith angle (radians).
    :param max_zenith_angle_rad: Max zenith angle (radians).
    :param max_inclusive: Boolean flag.  If True (False), `max_zenith_angle_rad`
        will be included in subset.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_geq(min_zenith_angle_rad, 0.)
    error_checking.assert_is_leq(max_zenith_angle_rad, MAX_ZENITH_ANGLE_RADIANS)
    error_checking.assert_is_greater(max_zenith_angle_rad, min_zenith_angle_rad)

    if max_inclusive is None:
        max_inclusive = max_zenith_angle_rad == MAX_ZENITH_ANGLE_RADIANS

    error_checking.assert_is_boolean(max_inclusive)

    all_zenith_angles_rad = example_utils.parse_example_ids(
        prediction_dict[EXAMPLE_IDS_KEY]
    )[example_utils.ZENITH_ANGLES_KEY]

    min_flags = all_zenith_angles_rad >= min_zenith_angle_rad

    if max_inclusive:
        max_flags = all_zenith_angles_rad <= max_zenith_angle_rad
    else:
        max_flags = all_zenith_angles_rad < max_zenith_angle_rad

    good_indices = numpy.where(numpy.logical_and(min_flags, max_flags))[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if isinstance(prediction_dict[this_key], list):
            prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in good_indices
            ]
        else:
            prediction_dict[this_key] = (
                prediction_dict[this_key][good_indices, ...]
            )

    return prediction_dict


def subset_by_month(prediction_dict, desired_month):
    """Subsets examples by month.

    :param prediction_dict: See doc for `write_file`.
    :param desired_month: Desired month (integer from 1...12).
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_integer(desired_month)
    error_checking.assert_is_geq(desired_month, 1)
    error_checking.assert_is_leq(desired_month, 12)

    all_times_unix_sec = example_utils.parse_example_ids(
        prediction_dict[EXAMPLE_IDS_KEY]
    )[example_utils.VALID_TIMES_KEY]

    all_months = numpy.array([
        int(time_conversion.unix_sec_to_string(t, '%m'))
        for t in all_times_unix_sec
    ], dtype=int)

    good_indices = numpy.where(all_months == desired_month)[0]

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if isinstance(prediction_dict[this_key], list):
            prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in good_indices
            ]
        else:
            prediction_dict[this_key] = (
                prediction_dict[this_key][good_indices, ...]
            )

    return prediction_dict


def subset_by_index(prediction_dict, desired_indices):
    """Subsets examples by index.

    :param prediction_dict: See doc for `write_file`.
    :param desired_indices: 1-D numpy array of desired indices.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_numpy_array(desired_indices, num_dimensions=1)
    error_checking.assert_is_integer_numpy_array(desired_indices)
    error_checking.assert_is_geq_numpy_array(desired_indices, 0)
    error_checking.assert_is_less_than_numpy_array(
        desired_indices, len(prediction_dict[EXAMPLE_IDS_KEY])
    )

    for this_key in ONE_PER_EXAMPLE_KEYS:
        if isinstance(prediction_dict[this_key], list):
            prediction_dict[this_key] = [
                prediction_dict[this_key][k] for k in desired_indices
            ]
        else:
            prediction_dict[this_key] = (
                prediction_dict[this_key][desired_indices, ...]
            )

    return prediction_dict
