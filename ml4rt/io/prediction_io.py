"""Input/output methods for model predictions."""

import numpy
import netCDF4
from gewittergefahr.gg_utils import prob_matched_means as pmm
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml4rt.io import example_io

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
EXAMPLE_IDS_KEY = 'example_id_strings'

ONE_PER_EXAMPLE_KEYS = [
    SCALAR_TARGETS_KEY, SCALAR_PREDICTIONS_KEY,
    VECTOR_TARGETS_KEY, VECTOR_PREDICTIONS_KEY, EXAMPLE_IDS_KEY
]

DEFAULT_MAX_PMM_PERCENTILE_LEVEL = 99.


def write_file(
        netcdf_file_name, scalar_target_matrix, vector_target_matrix,
        scalar_prediction_matrix, vector_prediction_matrix, example_id_strings,
        model_file_name):
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
    :param example_id_strings: length-E list of IDs created by
        `example_io.create_example_ids`.
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

    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings),
        exact_dimensions=numpy.array([num_examples], dtype=int)
    )
    example_io.parse_example_ids(example_id_strings)

    error_checking.assert_is_string(model_file_name)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)

    dataset_object.createDimension(
        EXAMPLE_DIMENSION_KEY, vector_target_matrix.shape[0]
    )
    dataset_object.createDimension(
        HEIGHT_DIMENSION_KEY, vector_target_matrix.shape[1]
    )
    dataset_object.createDimension(
        VECTOR_TARGET_DIMENSION_KEY, vector_target_matrix.shape[2]
    )
    dataset_object.createDimension(
        SCALAR_TARGET_DIMENSION_KEY, scalar_target_matrix.shape[1]
    )

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
        SCALAR_TARGETS_KEY: dataset_object.variables[SCALAR_TARGETS_KEY][:],
        SCALAR_PREDICTIONS_KEY:
            dataset_object.variables[SCALAR_PREDICTIONS_KEY][:],
        VECTOR_TARGETS_KEY: dataset_object.variables[VECTOR_TARGETS_KEY][:],
        VECTOR_PREDICTIONS_KEY:
            dataset_object.variables[VECTOR_PREDICTIONS_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY))
    }

    dataset_object.close()
    return prediction_dict


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
        MODEL_FILE_KEY: prediction_dict[MODEL_FILE_KEY]
    }


def subset_by_standard_atmo(prediction_dict, standard_atmo_enum):
    """Subsets examples by standard-atmosphere type.

    :param prediction_dict: See doc for `write_file`.
    :param standard_atmo_enum: See doc for
        `example_io.check_standard_atmo_type`.
    :return: prediction_dict: Same as input but with fewer examples.
    """

    example_io.check_standard_atmo_type(standard_atmo_enum)

    all_standard_atmo_enums = example_io.parse_example_ids(
        prediction_dict[EXAMPLE_IDS_KEY]
    )[-1]

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
