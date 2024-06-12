"""Input/output methods for model predictions."""

import copy
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

TOLERANCE = 1e-6
METRES_TO_MICRONS = 1e6

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
TARGET_WAVELENGTH_DIMENSION_KEY = 'target_wavelength'
VECTOR_TARGET_DIMENSION_KEY = 'vector_target'
SCALAR_TARGET_DIMENSION_KEY = 'scalar_target'
ENSEMBLE_MEMBER_DIM_KEY = 'ensemble_member'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'

MODEL_FILE_KEY = 'model_file_name'
ISOTONIC_MODEL_FILE_KEY = 'isotonic_model_file_name'
UNCERTAINTY_CALIB_MODEL_FILE_KEY = 'uncertainty_calib_model_file_name'
NORMALIZATION_FILE_KEY = 'normalization_file_name'
SCALAR_TARGETS_KEY = 'scalar_target_matrix'
SCALAR_PREDICTIONS_KEY = 'scalar_prediction_matrix'
VECTOR_TARGETS_KEY = 'vector_target_matrix'
VECTOR_PREDICTIONS_KEY = 'vector_prediction_matrix'
HEIGHTS_KEY = 'heights_m_agl'
TARGET_WAVELENGTHS_KEY = 'target_wavelengths_metres'
EXAMPLE_IDS_KEY = 'example_id_strings'

ONE_PER_EXAMPLE_KEYS = [
    SCALAR_TARGETS_KEY, SCALAR_PREDICTIONS_KEY,
    VECTOR_TARGETS_KEY, VECTOR_PREDICTIONS_KEY, EXAMPLE_IDS_KEY
]

DEFAULT_MAX_PMM_PERCENTILE_LEVEL = 99.
MAX_ZENITH_ANGLE_RADIANS = numpy.pi

ZENITH_ANGLE_BIN_KEY = 'zenith_angle_bin'
ALBEDO_BIN_KEY = 'albedo_bin'
SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY = 'shortwave_sfc_down_flux_bin'
AEROSOL_OPTICAL_DEPTH_BIN_KEY = 'aerosol_optical_depth_bin'
SURFACE_TEMP_BIN_KEY = 'surface_temp_bin'
LONGWAVE_SFC_DOWN_FLUX_BIN_KEY = 'longwave_sfc_down_flux_bin'
LONGWAVE_TOA_UP_FLUX_BIN_KEY = 'longwave_toa_up_flux_bin'
MONTH_KEY = 'month'
GRID_ROW_KEY = 'grid_row'
GRID_COLUMN_KEY = 'grid_column'

METADATA_KEYS = [
    ZENITH_ANGLE_BIN_KEY, ALBEDO_BIN_KEY, SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY,
    AEROSOL_OPTICAL_DEPTH_BIN_KEY, SURFACE_TEMP_BIN_KEY,
    LONGWAVE_SFC_DOWN_FLUX_BIN_KEY, LONGWAVE_TOA_UP_FLUX_BIN_KEY,
    MONTH_KEY, GRID_ROW_KEY, GRID_COLUMN_KEY
]

GRID_ROW_DIMENSION_KEY = 'row'
GRID_COLUMN_DIMENSION_KEY = 'column'
LATITUDES_KEY = 'latitude_deg_n'
LONGITUDES_KEY = 'longitude_deg_e'


def find_file(
        directory_name, zenith_angle_bin=None, albedo_bin=None,
        shortwave_sfc_down_flux_bin=None, aerosol_optical_depth_bin=None,
        surface_temp_bin=None, longwave_sfc_down_flux_bin=None,
        longwave_toa_up_flux_bin=None, month=None, grid_row=None,
        grid_column=None, raise_error_if_missing=True):
    """Finds NetCDF file with predictions.

    :param directory_name: Name of directory where file is expected.
    :param zenith_angle_bin: Zenith-angle bin (non-negative integer).  If file
        does not contain predictions for a specific zenith-angle bin, leave this
        alone.
    :param albedo_bin: Albedo bin (non-negative integer).  If file does not
        contain predictions for a specific albedo bin, leave this alone.
    :param shortwave_sfc_down_flux_bin: Bin for shortwave surface downwelling
        flux (non-negative integer).  If file does not contain predictions for a
        specific flux bin, leave this alone.
    :param aerosol_optical_depth_bin: Bin for aerosol optical depth
        (AOD; non-negative integer).  If file does not contain predictions for a
        specific AOD bin, leave this alone.
    :param surface_temp_bin: Bin for surface temperature (non-negative integer).
        If file does not contain predictions for a specific surface-temperature
        bin, leave this alone.
    :param longwave_sfc_down_flux_bin: Same as `shortwave_sfc_down_flux_bin` but
        for longwave.
    :param longwave_toa_up_flux_bin: Same as `longwave_sfc_down_flux_bin` but
        for top-of-atmosphere upwelling flux.
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

    elif albedo_bin is not None:
        error_checking.assert_is_integer(albedo_bin)
        error_checking.assert_is_geq(albedo_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, ALBEDO_BIN_KEY.replace('_', '-'), albedo_bin
        )

    elif shortwave_sfc_down_flux_bin is not None:
        error_checking.assert_is_integer(shortwave_sfc_down_flux_bin)
        error_checking.assert_is_geq(shortwave_sfc_down_flux_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, SHORTWAVE_SFC_DOWN_FLUX_BIN_KEY.replace('_', '-'),
            shortwave_sfc_down_flux_bin
        )

    elif aerosol_optical_depth_bin is not None:
        error_checking.assert_is_integer(aerosol_optical_depth_bin)
        error_checking.assert_is_geq(aerosol_optical_depth_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, AEROSOL_OPTICAL_DEPTH_BIN_KEY.replace('_', '-'),
            aerosol_optical_depth_bin
        )

    elif month is not None:
        error_checking.assert_is_integer(month)
        error_checking.assert_is_geq(month, 1)
        error_checking.assert_is_leq(month, 12)

        prediction_file_name = '{0:s}/predictions_{1:s}={2:02d}.nc'.format(
            directory_name, MONTH_KEY.replace('_', '-'), month
        )

    elif surface_temp_bin is not None:
        error_checking.assert_is_integer(surface_temp_bin)
        error_checking.assert_is_geq(surface_temp_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, SURFACE_TEMP_BIN_KEY.replace('_', '-'),
            surface_temp_bin
        )

    elif longwave_sfc_down_flux_bin is not None:
        error_checking.assert_is_integer(longwave_sfc_down_flux_bin)
        error_checking.assert_is_geq(longwave_sfc_down_flux_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, LONGWAVE_SFC_DOWN_FLUX_BIN_KEY.replace('_', '-'),
            longwave_sfc_down_flux_bin
        )

    elif longwave_toa_up_flux_bin is not None:
        error_checking.assert_is_integer(longwave_toa_up_flux_bin)
        error_checking.assert_is_geq(longwave_toa_up_flux_bin, 0)

        prediction_file_name = (
            '{0:s}/predictions_{1:s}={2:03d}.nc'
        ).format(
            directory_name, LONGWAVE_TOA_UP_FLUX_BIN_KEY.replace('_', '-'),
            longwave_toa_up_flux_bin
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
    metadata_dict['albedo_bin']: Same.
    metadata_dict['shortwave_sfc_down_flux_bin']: Same.
    metadata_dict['aerosol_optical_depth_bin']: Same.
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
        target_wavelengths_metres, example_id_strings,
        model_file_name, isotonic_model_file_name,
        uncertainty_calib_model_file_name, normalization_file_name):
    """Writes predictions to NetCDF file.

    E = number of examples
    H = number of heights
    T_s = number of scalar targets
    T_v = number of vector targets
    S = number of ensemble members
    W = number of target wavelengths

    :param netcdf_file_name: Path to output file.
    :param scalar_target_matrix: numpy array (E x W x T_s) with actual values of
        scalar targets.
    :param vector_target_matrix: numpy array (E x H x W x T_v) with actual
        values of vector targets.
    :param scalar_prediction_matrix: numpy array (E x W x T_s x S) with
        predicted values of scalar targets.
    :param vector_prediction_matrix: numpy array (E x H x W x T_v x S) with
        predicted values of vector targets.
    :param heights_m_agl: length-H numpy array of heights (metres above ground
        level).
    :param target_wavelengths_metres: length-W numpy array of wavelengths.
    :param example_id_strings: length-E list of IDs created by
        `example_utils.create_example_ids`.
    :param model_file_name: Path to file with trained model (readable by
        `neural_net.read_model`).
    :param isotonic_model_file_name: Path to file with trained isotonic-
        regression models (readable by `isotonic_regression.read_file`) used to
        make predictions.  If isotonic regression was not used, leave this as
        None.
    :param uncertainty_calib_model_file_name: Path to file with trained
        uncertainty-calibration model (readable by
        `uncertainty_calibration.read_file`).  If predictions do not have
        calibrated uncertainty, make this None.
    :param normalization_file_name: Path to file with normalization params
        (readable by `example_io.read_file`).  If predictions were created with
        the same normalization params as used for model-training, leave this as
        None.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)
    error_checking.assert_is_numpy_array(scalar_target_matrix, num_dimensions=3)
    error_checking.assert_is_numpy_array_without_nan(scalar_prediction_matrix)
    error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)
    error_checking.assert_is_numpy_array_without_nan(vector_prediction_matrix)

    num_ensemble_members = scalar_prediction_matrix.shape[-1]
    these_dim = numpy.array(
        scalar_target_matrix.shape + (num_ensemble_members,), dtype=int
    )
    error_checking.assert_is_numpy_array(
        scalar_prediction_matrix, exact_dimensions=these_dim
    )

    num_examples = scalar_target_matrix.shape[0]
    expected_dim = numpy.array(
        (num_examples,) + vector_target_matrix.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        vector_target_matrix, exact_dimensions=expected_dim
    )

    these_dim = numpy.array(
        vector_target_matrix.shape + (num_ensemble_members,), dtype=int
    )
    error_checking.assert_is_numpy_array(
        vector_prediction_matrix, exact_dimensions=these_dim
    )

    num_heights = vector_target_matrix.shape[1]
    error_checking.assert_is_greater_numpy_array(heights_m_agl, 0.)
    error_checking.assert_is_numpy_array(
        heights_m_agl,
        exact_dimensions=numpy.array([num_heights], dtype=int)
    )

    num_wavelengths = vector_target_matrix.shape[2]
    error_checking.assert_is_greater_numpy_array(target_wavelengths_metres, 0.)
    error_checking.assert_is_numpy_array(
        target_wavelengths_metres,
        exact_dimensions=numpy.array([num_wavelengths], dtype=int)
    )

    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings),
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    example_utils.parse_example_ids(example_id_strings)

    error_checking.assert_is_string(model_file_name)

    if isotonic_model_file_name is None:
        isotonic_model_file_name = ''
    if uncertainty_calib_model_file_name is None:
        uncertainty_calib_model_file_name = ''
    if normalization_file_name is None:
        normalization_file_name = ''

    error_checking.assert_is_string(isotonic_model_file_name)
    error_checking.assert_is_string(uncertainty_calib_model_file_name)
    error_checking.assert_is_string(normalization_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF4_CLASSIC'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(ISOTONIC_MODEL_FILE_KEY, isotonic_model_file_name)
    dataset_object.setncattr(
        UNCERTAINTY_CALIB_MODEL_FILE_KEY, uncertainty_calib_model_file_name
    )
    dataset_object.setncattr(NORMALIZATION_FILE_KEY, normalization_file_name)

    num_examples = vector_target_matrix.shape[0]
    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        HEIGHT_DIMENSION_KEY, vector_target_matrix.shape[1]
    )
    dataset_object.createDimension(
        TARGET_WAVELENGTH_DIMENSION_KEY, vector_target_matrix.shape[2]
    )
    dataset_object.createDimension(
        VECTOR_TARGET_DIMENSION_KEY, vector_target_matrix.shape[3]
    )
    dataset_object.createDimension(
        ENSEMBLE_MEMBER_DIM_KEY, num_ensemble_members
    )

    num_scalar_targets = scalar_target_matrix.shape[-1]
    if num_scalar_targets > 0:
        dataset_object.createDimension(
            SCALAR_TARGET_DIMENSION_KEY, num_scalar_targets
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

    dataset_object.createVariable(
        TARGET_WAVELENGTHS_KEY, datatype=numpy.float32,
        dimensions=TARGET_WAVELENGTH_DIMENSION_KEY
    )
    dataset_object.variables[TARGET_WAVELENGTHS_KEY][:] = (
        target_wavelengths_metres
    )

    if num_scalar_targets > 0:
        dataset_object.createVariable(
            SCALAR_TARGETS_KEY, datatype=numpy.float32,
            dimensions=(
                EXAMPLE_DIMENSION_KEY, TARGET_WAVELENGTH_DIMENSION_KEY,
                SCALAR_TARGET_DIMENSION_KEY
            )
        )
        dataset_object.variables[SCALAR_TARGETS_KEY][:] = scalar_target_matrix

        these_dim = (
            EXAMPLE_DIMENSION_KEY, TARGET_WAVELENGTH_DIMENSION_KEY,
            SCALAR_TARGET_DIMENSION_KEY, ENSEMBLE_MEMBER_DIM_KEY
        )
        dataset_object.createVariable(
            SCALAR_PREDICTIONS_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[SCALAR_PREDICTIONS_KEY][:] = (
            scalar_prediction_matrix
        )

    these_dim = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
        TARGET_WAVELENGTH_DIMENSION_KEY, VECTOR_TARGET_DIMENSION_KEY
    )
    dataset_object.createVariable(
        VECTOR_TARGETS_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[VECTOR_TARGETS_KEY][:] = vector_target_matrix

    these_dim = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
        TARGET_WAVELENGTH_DIMENSION_KEY, VECTOR_TARGET_DIMENSION_KEY,
        ENSEMBLE_MEMBER_DIM_KEY
    )
    dataset_object.createVariable(
        VECTOR_PREDICTIONS_KEY, datatype=numpy.float32,
        dimensions=these_dim
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
    prediction_dict['heights_m_agl']: Same.
    prediction_dict['target_wavelengths_metres']: Same.
    prediction_dict['model_file_name']: Same.
    prediction_dict['isotonic_model_file_name']: Same.
    prediction_dict['uncertainty_calib_model_file_name']: Same.
    prediction_dict['normalization_file_name']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    prediction_dict = {
        VECTOR_TARGETS_KEY:
            dataset_object.variables[VECTOR_TARGETS_KEY][:],
        VECTOR_PREDICTIONS_KEY:
            dataset_object.variables[VECTOR_PREDICTIONS_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
    }

    if TARGET_WAVELENGTHS_KEY in dataset_object.variables:
        prediction_dict[TARGET_WAVELENGTHS_KEY] = (
            dataset_object.variables[TARGET_WAVELENGTHS_KEY][:]
        )
    else:
        prediction_dict[TARGET_WAVELENGTHS_KEY] = numpy.array(
            [example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES]
        )

    # Add ensemble dimension if necessary.
    if len(prediction_dict[VECTOR_PREDICTIONS_KEY].shape) == 3:
        prediction_dict[VECTOR_PREDICTIONS_KEY] = numpy.expand_dims(
            prediction_dict[VECTOR_PREDICTIONS_KEY], axis=-1
        )

    # Add wavelength dimension if necessary.
    if len(prediction_dict[VECTOR_PREDICTIONS_KEY].shape) == 4:
        prediction_dict[VECTOR_PREDICTIONS_KEY] = numpy.expand_dims(
            prediction_dict[VECTOR_PREDICTIONS_KEY], axis=-2
        )
        prediction_dict[VECTOR_TARGETS_KEY] = numpy.expand_dims(
            prediction_dict[VECTOR_TARGETS_KEY], axis=-2
        )

    try:
        prediction_dict[ISOTONIC_MODEL_FILE_KEY] = str(
            getattr(dataset_object, ISOTONIC_MODEL_FILE_KEY)
        )
    except:
        prediction_dict[ISOTONIC_MODEL_FILE_KEY] = ''

    if prediction_dict[ISOTONIC_MODEL_FILE_KEY] == '':
        prediction_dict[ISOTONIC_MODEL_FILE_KEY] = None

    try:
        prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = str(getattr(
            dataset_object, UNCERTAINTY_CALIB_MODEL_FILE_KEY
        ))
    except:
        prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = ''

    if prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] == '':
        prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY] = None

    try:
        prediction_dict[NORMALIZATION_FILE_KEY] = str(
            getattr(dataset_object, NORMALIZATION_FILE_KEY)
        )
    except:
        prediction_dict[NORMALIZATION_FILE_KEY] = ''

    if prediction_dict[NORMALIZATION_FILE_KEY] == '':
        prediction_dict[NORMALIZATION_FILE_KEY] = None

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

    num_target_wavelengths = len(prediction_dict[TARGET_WAVELENGTHS_KEY])

    if SCALAR_TARGETS_KEY in dataset_object.variables:
        prediction_dict[SCALAR_TARGETS_KEY] = (
            dataset_object.variables[SCALAR_TARGETS_KEY][:]
        )
        prediction_dict[SCALAR_PREDICTIONS_KEY] = (
            dataset_object.variables[SCALAR_PREDICTIONS_KEY][:]
        )
    else:
        num_examples = prediction_dict[VECTOR_TARGETS_KEY].shape[0]
        prediction_dict[SCALAR_TARGETS_KEY] = numpy.full(
            (num_examples, num_target_wavelengths, 0), 0.
        )
        prediction_dict[SCALAR_PREDICTIONS_KEY] = numpy.full(
            (num_examples, num_target_wavelengths, 0, 1), 0.
        )

    # Add ensemble dimension if necessary.
    if len(prediction_dict[SCALAR_PREDICTIONS_KEY].shape) == 2:
        prediction_dict[SCALAR_PREDICTIONS_KEY] = numpy.expand_dims(
            prediction_dict[SCALAR_PREDICTIONS_KEY], axis=-1
        )

    # Add wavelength dimension if necessary.
    if len(prediction_dict[SCALAR_PREDICTIONS_KEY].shape) == 3:
        prediction_dict[SCALAR_PREDICTIONS_KEY] = numpy.expand_dims(
            prediction_dict[SCALAR_PREDICTIONS_KEY], axis=-2
        )
        prediction_dict[SCALAR_TARGETS_KEY] = numpy.expand_dims(
            prediction_dict[SCALAR_TARGETS_KEY], axis=-2
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
        netcdf_file_name, 'w', format='NETCDF4_CLASSIC'
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


def average_predictions_many_examples(
        prediction_dict, use_pmm,
        max_pmm_percentile_level=DEFAULT_MAX_PMM_PERCENTILE_LEVEL,
        test_mode=False):
    """Averages predicted and target values over many examples.

    H = number of heights
    T_s = number of scalar targets
    T_v = number of vector targets
    W = number of target wavelengths

    :param prediction_dict: See doc for `write_file`.
    :param use_pmm: Boolean flag.  If True, will use probability-matched means
        for vector fields (vertical profiles).  If False, will use arithmetic
        means for vector fields.
    :param max_pmm_percentile_level: [used only if `use_pmm == True`]
        Max percentile level for probability-matched means.
    :param test_mode: Leave this alone.
    :return: mean_prediction_dict: Dictionary with the following keys.
    mean_prediction_dict['scalar_target_matrix']: numpy array (1 x W x T_s) with
        mean target (actual) values for scalar variables.
    mean_prediction_dict['scalar_prediction_matrix']: Same but with predicted
        values.
    mean_prediction_dict['vector_target_matrix']: numpy array (1 x H x W x T_v)
        with mean target (actual) values for vector variables.
    mean_prediction_dict['vector_prediction_matrix']: Same but with predicted
        values.
    mean_prediction_dict['heights_m_agl']: length-H numpy array of heights
        (metres above ground level).
    mean_prediction_dict['target_wavelengths_metres']: length-W numpy array of
        target wavelengths.
    mean_prediction_dict['model_file_name']: Path to file with trained model
        (readable by `neural_net.read_model`).
    mean_prediction_dict['isotonic_model_file_name']: Path to file with trained
        isotonic-regression models (readable by `isotonic_regression.read_file`)
        used to make predictions.  If isotonic regression was not used, this is
        None.
    mean_prediction_dict['normalization_file_name']: Path to file with
        normalization params (readable by `example_io.read_file`).  If
        predictions were created with the same normalization params as used for
        model-training, leave this as None.
    """

    error_checking.assert_is_boolean(test_mode)
    if not test_mode:
        prediction_dict = get_ensemble_mean(prediction_dict)

    error_checking.assert_is_boolean(use_pmm)
    error_checking.assert_is_geq(max_pmm_percentile_level, 90.)
    error_checking.assert_is_leq(max_pmm_percentile_level, 100.)

    mean_scalar_target_matrix = numpy.mean(
        prediction_dict[SCALAR_TARGETS_KEY], axis=0, keepdims=True
    )
    mean_scalar_prediction_matrix = numpy.mean(
        prediction_dict[SCALAR_PREDICTIONS_KEY], axis=0, keepdims=True
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
        TARGET_WAVELENGTHS_KEY: prediction_dict[TARGET_WAVELENGTHS_KEY],
        MODEL_FILE_KEY: prediction_dict[MODEL_FILE_KEY],
        ISOTONIC_MODEL_FILE_KEY: prediction_dict[ISOTONIC_MODEL_FILE_KEY],
        UNCERTAINTY_CALIB_MODEL_FILE_KEY:
            prediction_dict[UNCERTAINTY_CALIB_MODEL_FILE_KEY],
        NORMALIZATION_FILE_KEY: prediction_dict[NORMALIZATION_FILE_KEY]
    }


def get_ensemble_mean(prediction_dict):
    """Computes ensemble-mean prediction for each example and each target var.

    :param prediction_dict: See doc for `write_file`.
    :return: prediction_dict: Same but without last axis for the keys
        "scalar_prediction_matrix" and "vector_prediction_matrix".
    """

    prediction_dict[SCALAR_PREDICTIONS_KEY] = numpy.mean(
        prediction_dict[SCALAR_PREDICTIONS_KEY], axis=-1
    )
    prediction_dict[VECTOR_PREDICTIONS_KEY] = numpy.mean(
        prediction_dict[VECTOR_PREDICTIONS_KEY], axis=-1
    )

    return prediction_dict


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

    desired_indices = numpy.where(
        all_standard_atmo_enums == standard_atmo_enum
    )[0]
    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


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

    desired_indices = numpy.where(numpy.logical_and(min_flags, max_flags))[0]
    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


def subset_by_shortwave_sfc_down_flux(
        prediction_dict, min_flux_w_m02, max_flux_w_m02,
        wavelengths_metres=None):
    """Subsets examples by shortwave surface downwelling flux.

    :param prediction_dict: See doc for `write_file`.
    :param min_flux_w_m02: Minimum flux.
    :param max_flux_w_m02: Max flux.
    :param wavelengths_metres: 1-D numpy array of wavelengths over which to sum
        SW sfc downwelling flux.  If None, will sum over all wavelengths (i.e.,
        will use broadband flux).
    :return: prediction_dict: Same as input but with fewer examples.
    """

    if wavelengths_metres is None:
        try:
            w = example_utils.match_wavelengths(
                wavelengths_metres=prediction_dict[TARGET_WAVELENGTHS_KEY],
                desired_wavelength_metres=
                numpy.array([example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES])
            )
            wavelengths_metres = numpy.array([
                prediction_dict[TARGET_WAVELENGTHS_KEY][w]
            ])
        except ValueError:
            wavelengths_metres = prediction_dict[TARGET_WAVELENGTHS_KEY] + 0.

    wave_inds = example_utils.match_wavelengths(
        wavelengths_metres=prediction_dict[TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelengths_metres
    )

    error_checking.assert_is_geq(min_flux_w_m02, 0.)
    error_checking.assert_is_greater(
        max_flux_w_m02, min_flux_w_m02
    )

    model_file_name = prediction_dict[MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    t = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
        example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
    )
    actual_fluxes_w_m02 = numpy.sum(
        prediction_dict[SCALAR_TARGETS_KEY][..., t][:, wave_inds],
        axis=1
    )

    desired_indices = numpy.where(numpy.logical_and(
        actual_fluxes_w_m02 >= min_flux_w_m02,
        actual_fluxes_w_m02 <= max_flux_w_m02
    ))[0]

    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


def subset_by_longwave_sfc_down_flux(
        prediction_dict, min_flux_w_m02, max_flux_w_m02,
        wavelengths_metres=None):
    """Subsets examples by longwave surface downwelling flux.

    :param prediction_dict: See doc for `write_file`.
    :param min_flux_w_m02: Minimum flux.
    :param max_flux_w_m02: Max flux.
    :param wavelengths_metres: 1-D numpy array of wavelengths over which to sum
        LW sfc downwelling flux.  If None, will sum over all wavelengths (i.e.,
        will use broadband flux).
    :return: prediction_dict: Same as input but with fewer examples.
    """

    if wavelengths_metres is None:
        try:
            w = example_utils.match_wavelengths(
                wavelengths_metres=prediction_dict[TARGET_WAVELENGTHS_KEY],
                desired_wavelength_metres=
                numpy.array([example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES])
            )
            wavelengths_metres = numpy.array([
                prediction_dict[TARGET_WAVELENGTHS_KEY][w]
            ])
        except ValueError:
            wavelengths_metres = prediction_dict[TARGET_WAVELENGTHS_KEY] + 0.

    wave_inds = example_utils.match_wavelengths(
        wavelengths_metres=prediction_dict[TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelengths_metres
    )

    error_checking.assert_is_geq(min_flux_w_m02, 0.)
    error_checking.assert_is_greater(
        max_flux_w_m02, min_flux_w_m02
    )

    model_file_name = prediction_dict[MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    t = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
        example_utils.LONGWAVE_SURFACE_DOWN_FLUX_NAME
    )
    actual_fluxes_w_m02 = numpy.sum(
        prediction_dict[SCALAR_TARGETS_KEY][..., t][:, wave_inds],
        axis=1
    )

    desired_indices = numpy.where(numpy.logical_and(
        actual_fluxes_w_m02 >= min_flux_w_m02,
        actual_fluxes_w_m02 <= max_flux_w_m02
    ))[0]

    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


def subset_by_longwave_toa_up_flux(
        prediction_dict, min_flux_w_m02, max_flux_w_m02,
        wavelengths_metres=None):
    """Subsets examples by longwave TOA upwelling flux.

    :param prediction_dict: See doc for `write_file`.
    :param min_flux_w_m02: Minimum flux.
    :param max_flux_w_m02: Max flux.
    :param wavelengths_metres: 1-D numpy array of wavelengths over which to sum
        LW sfc downwelling flux.  If None, will sum over all wavelengths (i.e.,
        will use broadband flux).
    :return: prediction_dict: Same as input but with fewer examples.
    """

    if wavelengths_metres is None:
        try:
            w = example_utils.match_wavelengths(
                wavelengths_metres=prediction_dict[TARGET_WAVELENGTHS_KEY],
                desired_wavelength_metres=
                numpy.array([example_utils.DUMMY_BROADBAND_WAVELENGTH_METRES])
            )
            wavelengths_metres = numpy.array([
                prediction_dict[TARGET_WAVELENGTHS_KEY][w]
            ])
        except ValueError:
            wavelengths_metres = prediction_dict[TARGET_WAVELENGTHS_KEY] + 0.

    wave_inds = example_utils.match_wavelengths(
        wavelengths_metres=prediction_dict[TARGET_WAVELENGTHS_KEY],
        desired_wavelength_metres=wavelengths_metres
    )

    error_checking.assert_is_geq(min_flux_w_m02, 0.)
    error_checking.assert_is_greater(
        max_flux_w_m02, min_flux_w_m02
    )

    model_file_name = prediction_dict[MODEL_FILE_KEY]
    model_metafile_name = neural_net.find_metafile(
        model_dir_name=os.path.split(model_file_name)[0],
        raise_error_if_missing=True
    )

    print('Reading metadata from: "{0:s}"...'.format(model_metafile_name))
    model_metadata_dict = neural_net.read_metafile(model_metafile_name)
    training_option_dict = model_metadata_dict[neural_net.TRAINING_OPTIONS_KEY]

    t = training_option_dict[neural_net.SCALAR_TARGET_NAMES_KEY].index(
        example_utils.LONGWAVE_TOA_UP_FLUX_NAME
    )
    actual_fluxes_w_m02 = numpy.sum(
        prediction_dict[SCALAR_TARGETS_KEY][..., t][:, wave_inds],
        axis=1
    )

    desired_indices = numpy.where(numpy.logical_and(
        actual_fluxes_w_m02 >= min_flux_w_m02,
        actual_fluxes_w_m02 <= max_flux_w_m02
    ))[0]

    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


def subset_by_albedo(
        prediction_dict, min_albedo, max_albedo, max_inclusive=None):
    """Subsets examples by albedo.

    :param prediction_dict: See doc for `write_file`.
    :param min_albedo: Minimum albedo (unitless).
    :param max_albedo: Max albedo (unitless).
    :param max_inclusive: Boolean flag.  If True (False), `max_albedo` will be
        included in subset.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    error_checking.assert_is_geq(min_albedo, 0.)
    error_checking.assert_is_leq(max_albedo, 1.)
    error_checking.assert_is_greater(max_albedo, min_albedo)

    if max_inclusive is None:
        max_inclusive = max_albedo == 1.

    error_checking.assert_is_boolean(max_inclusive)

    all_albedos = example_utils.parse_example_ids(
        prediction_dict[EXAMPLE_IDS_KEY]
    )[example_utils.ALBEDOS_KEY]

    min_flags = all_albedos >= min_albedo

    if max_inclusive:
        max_flags = all_albedos <= max_albedo
    else:
        max_flags = all_albedos < max_albedo

    desired_indices = numpy.where(numpy.logical_and(min_flags, max_flags))[0]
    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


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

    desired_indices = numpy.where(all_months == desired_month)[0]
    return subset_by_index(
        prediction_dict=prediction_dict, desired_indices=desired_indices
    )


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


def concat_predictions(prediction_dicts):
    """Concatenates many dictionaries with predictions into one.

    :param prediction_dicts: List of dictionaries, each in the format returned
        by `read_file`.
    :return: prediction_dict: Single dictionary, also in the format returned by
        `read_file`.
    :raises: ValueError: if any two dictionaries have predictions created with
        different models.
    """

    prediction_dict = copy.deepcopy(prediction_dicts[0])
    keys_to_match = [
        MODEL_FILE_KEY, ISOTONIC_MODEL_FILE_KEY,
        UNCERTAINTY_CALIB_MODEL_FILE_KEY, NORMALIZATION_FILE_KEY
    ]

    for i in range(1, len(prediction_dicts)):
        if not numpy.allclose(
                prediction_dict[HEIGHTS_KEY], prediction_dicts[i][HEIGHTS_KEY],
                atol=TOLERANCE
        ):
            error_string = (
                '1st and {0:d}th dictionaries have different height coords '
                '(units are m AGL).  1st dictionary:\n{1:s}\n\n'
                '{0:d}th dictionary:\n{2:s}'
            ).format(
                i + 1,
                str(prediction_dict[HEIGHTS_KEY]),
                str(prediction_dicts[i][HEIGHTS_KEY])
            )

            raise ValueError(error_string)

        if not numpy.allclose(
                prediction_dict[TARGET_WAVELENGTHS_KEY],
                prediction_dicts[i][TARGET_WAVELENGTHS_KEY],
                atol=TOLERANCE
        ):
            error_string = (
                '1st and {0:d}th dictionaries have different target '
                'wavelengths (units are microns).  1st dictionary:\n{1:s}\n\n'
                '{0:d}th dictionary:\n{2:s}'
            ).format(
                i + 1,
                str(METRES_TO_MICRONS * prediction_dict[HEIGHTS_KEY]),
                str(METRES_TO_MICRONS * prediction_dicts[i][HEIGHTS_KEY])
            )

            raise ValueError(error_string)

        for this_key in keys_to_match:
            if prediction_dict[this_key] == prediction_dicts[i][this_key]:
                continue

            error_string = (
                '1st and {0:d}th dictionaries have different values for '
                '"{1:s}".  1st dictionary:\n{2:s}\n\n'
                '{0:d}th dictionary:\n{3:s}'
            ).format(
                i + 1, this_key, str(prediction_dict[this_key]),
                str(prediction_dicts[i][this_key])
            )

            raise ValueError(error_string)

        for this_key in ONE_PER_EXAMPLE_KEYS:
            if isinstance(prediction_dict[this_key], list):
                prediction_dict[this_key] += prediction_dicts[i][this_key]
            else:
                prediction_dict[this_key] = numpy.concatenate((
                    prediction_dict[this_key], prediction_dicts[i][this_key]
                ), axis=0)

    num_examples = len(prediction_dict[EXAMPLE_IDS_KEY])
    num_unique_examples = len(set(prediction_dict[EXAMPLE_IDS_KEY]))

    if num_examples == num_unique_examples:
        return prediction_dict

    error_string = (
        'Number of unique examples ({0:d}) is less than number of total '
        'examples ({1:d}).'
    ).format(num_unique_examples, num_examples)

    raise ValueError(error_string)
