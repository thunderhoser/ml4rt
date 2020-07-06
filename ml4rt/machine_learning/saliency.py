"""Methods for computing, reading, and writing saliency maps."""

import os.path
import numpy
import netCDF4
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as saliency_utils
from ml4rt.io import example_io
from ml4rt.machine_learning import neural_net

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
SCALAR_PREDICTOR_DIM_KEY = 'scalar_predictor'
VECTOR_PREDICTOR_DIM_KEY = 'vector_predictor'
SCALAR_TARGET_DIM_KEY = 'scalar_target'
VECTOR_TARGET_DIM_KEY = 'vector_target'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'

MODEL_FILE_KEY = 'model_file_name'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'
TARGET_FIELD_KEY = 'target_field_name'
TARGET_HEIGHT_KEY = 'target_height_m_agl'

EXAMPLE_IDS_KEY = 'example_id_strings'
SCALAR_SALIENCY_KEY = 'scalar_saliency_matrix'
VECTOR_SALIENCY_KEY = 'vector_saliency_matrix'
SALIENCY_SCALAR_P_SCALAR_T_KEY = 'saliency_matrix_scalar_p_scalar_t'
SALIENCY_VECTOR_P_SCALAR_T_KEY = 'saliency_matrix_vector_p_scalar_t'
SALIENCY_SCALAR_P_VECTOR_T_KEY = 'saliency_matrix_scalar_p_vector_t'
SALIENCY_VECTOR_P_VECTOR_T_KEY = 'saliency_matrix_vector_p_vector_t'


def check_metadata(layer_name, neuron_indices, ideal_activation=None):
    """Checks metadata for errors.

    The "relevant neuron" is that whose activation will be used in the numerator
    of the saliency equation.  In other words, if the relevant neuron is n,
    the saliency of each predictor x will be d(a_n) / dx, where a_n is the
    activation of n.

    :param layer_name: Name of layer with relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of relevant neuron.
        Must have length D - 1, where D = number of dimensions in layer output.
        The first dimension is the batch dimension, which always has length
        `None` in Keras.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  If you specify this, the loss function will be
        (neuron_activation - ideal_activation)**2.  If you leave this as None,
        the loss function will be
        -sign(neuron_activation) * neuron_activation**2.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)

    if ideal_activation is not None:
        error_checking.assert_is_not_nan(ideal_activation)


def get_saliency_one_neuron(
        model_object, predictor_matrix, layer_name, neuron_indices,
        ideal_activation=None):
    """Computes saliency maps with respect to activation of one neuron.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: numpy array of predictors.  Must be formatted in
        the same way as for training and inference.
    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :return: saliency_matrix: Matrix of saliency values, with same shape as
        `predictor_matrix`.
    """

    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )
    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)

    activation_tensor = None

    for k in neuron_indices[::-1]:
        if activation_tensor is None:
            activation_tensor = (
                model_object.get_layer(name=layer_name).output[..., k]
            )
        else:
            activation_tensor = activation_tensor[..., k]

    if ideal_activation is None:
        loss_tensor = -K.sign(activation_tensor) * activation_tensor ** 2
    else:
        loss_tensor = (activation_tensor - ideal_activation) ** 2

    return saliency_utils.do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=[predictor_matrix]
    )[0]


def write_standard_file(
        netcdf_file_name, scalar_saliency_matrix, vector_saliency_matrix,
        example_id_strings, model_file_name, layer_name, neuron_indices,
        ideal_activation=None, target_field_name=None,
        target_height_m_agl=None):
    """Writes standard (non-averaged) saliency maps to NetCDF file.

    E = number of examples
    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors

    :param netcdf_file_name: Path to output file.
    :param scalar_saliency_matrix: numpy array with saliency values for scalar
        predictors.  If neural-net type is dense, this array should be E x P_s.
        Otherwise, this array should be E x H x P_s.
    :param vector_saliency_matrix: numpy array (E x H x P_v) with saliency
        values for vector predictors.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
    :param target_field_name: Name of target variable predicted by relevant
        neuron.  If relevant neuron is *not* an output neuron, leave this alone.
    :param target_height_m_agl: Height of target variable predicted by relevant
        neuron.  If relevant neuron is *not* an output neuron or if the target
        variable is scalar, leave this alone.
    """

    # Check input args.
    check_metadata(
        layer_name=layer_name, neuron_indices=neuron_indices,
        ideal_activation=ideal_activation
    )

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )
    num_examples = len(example_id_strings)

    error_checking.assert_is_numpy_array_without_nan(vector_saliency_matrix)
    error_checking.assert_is_numpy_array(
        vector_saliency_matrix, num_dimensions=3
    )
    expected_dim = numpy.array(
        (num_examples,) + vector_saliency_matrix.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        vector_saliency_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(scalar_saliency_matrix)
    num_scalar_dim = len(scalar_saliency_matrix.shape)
    error_checking.assert_is_geq(num_scalar_dim, 2)
    error_checking.assert_is_leq(num_scalar_dim, 3)

    if num_scalar_dim == 2 or vector_saliency_matrix.size == 0:
        expected_dim = numpy.array(
            (num_examples,) + scalar_saliency_matrix.shape[1:], dtype=int
        )
    else:
        num_heights = vector_saliency_matrix.shape[1]
        num_scalar_predictors = scalar_saliency_matrix.shape[2]
        expected_dim = numpy.array(
            [num_examples, num_heights, num_scalar_predictors], dtype=int
        )

    error_checking.assert_is_numpy_array(
        scalar_saliency_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_string(model_file_name)

    if target_field_name is None:
        target_height_m_agl = None
    else:
        example_io.check_field_name(target_field_name)

    if target_height_m_agl is not None:
        target_height_m_agl = int(numpy.round(target_height_m_agl))
        error_checking.assert_is_geq(target_height_m_agl, 0)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(LAYER_NAME_KEY, layer_name)
    dataset_object.setncattr(NEURON_INDICES_KEY, neuron_indices)
    dataset_object.setncattr(
        IDEAL_ACTIVATION_KEY,
        numpy.nan if ideal_activation is None else ideal_activation
    )
    dataset_object.setncattr(
        TARGET_FIELD_KEY,
        '' if target_field_name is None else target_field_name
    )
    dataset_object.setncattr(
        TARGET_HEIGHT_KEY,
        -1 if target_height_m_agl is None else target_height_m_agl
    )

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        SCALAR_PREDICTOR_DIM_KEY, scalar_saliency_matrix.shape[-1]
    )
    dataset_object.createDimension(
        HEIGHT_DIMENSION_KEY, vector_saliency_matrix.shape[1]
    )
    dataset_object.createDimension(
        VECTOR_PREDICTOR_DIM_KEY, vector_saliency_matrix.shape[2]
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

    if scalar_saliency_matrix.size > 0:
        if num_scalar_dim == 2:
            dataset_object.createVariable(
                SCALAR_SALIENCY_KEY, datatype=numpy.float32,
                dimensions=(EXAMPLE_DIMENSION_KEY, SCALAR_PREDICTOR_DIM_KEY)
            )
        else:
            these_dim = (
                EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
                SCALAR_PREDICTOR_DIM_KEY
            )
            dataset_object.createVariable(
                SCALAR_SALIENCY_KEY, datatype=numpy.float32,
                dimensions=these_dim
            )

        dataset_object.variables[SCALAR_SALIENCY_KEY][:] = (
            scalar_saliency_matrix
        )

    if vector_saliency_matrix.size > 0:
        these_dim = (
            EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
            VECTOR_PREDICTOR_DIM_KEY
        )
        dataset_object.createVariable(
            VECTOR_SALIENCY_KEY, datatype=numpy.float32, dimensions=these_dim
        )
        dataset_object.variables[VECTOR_SALIENCY_KEY][:] = (
            vector_saliency_matrix
        )

    dataset_object.close()


def read_standard_file(netcdf_file_name):
    """Reads standard (non-averaged) saliency maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['scalar_saliency_matrix']: See doc for `write_standard_file`.
    saliency_dict['vector_saliency_matrix']: Same.
    saliency_dict['example_id_strings']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['ideal_activation']: Same.
    saliency_dict['target_field_name']: Same.
    saliency_dict['target_height_m_agl']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    saliency_dict = {
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY),
        TARGET_FIELD_KEY: getattr(dataset_object, TARGET_FIELD_KEY),
        TARGET_HEIGHT_KEY: getattr(dataset_object, TARGET_HEIGHT_KEY)
    }

    if numpy.isnan(saliency_dict[IDEAL_ACTIVATION_KEY]):
        saliency_dict[IDEAL_ACTIVATION_KEY] = None
    if saliency_dict[TARGET_FIELD_KEY] == '':
        saliency_dict[TARGET_FIELD_KEY] = None
    if saliency_dict[TARGET_HEIGHT_KEY] < 0:
        saliency_dict[TARGET_HEIGHT_KEY] = None

    num_examples = dataset_object.dimensions[EXAMPLE_DIMENSION_KEY].size
    num_scalar_predictors = (
        dataset_object.dimensions[SCALAR_PREDICTOR_DIM_KEY].size
    )
    num_vector_predictors = (
        dataset_object.dimensions[VECTOR_PREDICTOR_DIM_KEY].size
    )
    num_heights = dataset_object.dimensions[HEIGHT_DIMENSION_KEY].size

    if SCALAR_SALIENCY_KEY in dataset_object.variables:
        saliency_dict[SCALAR_SALIENCY_KEY] = (
            dataset_object.variables[SCALAR_SALIENCY_KEY][:]
        )
    else:
        model_metafile_name = neural_net.find_metafile(
            model_dir_name=os.path.split(saliency_dict[MODEL_FILE_KEY])[0]
        )
        model_metadata_dict = neural_net.read_metafile(model_metafile_name)
        net_type_string = model_metadata_dict[neural_net.NET_TYPE_KEY]

        if net_type_string == neural_net.DENSE_NET_TYPE_STRING:
            these_dim = (num_examples, num_scalar_predictors)
        else:
            these_dim = (num_examples, num_heights, num_scalar_predictors)

        saliency_dict[SCALAR_SALIENCY_KEY] = numpy.full(these_dim, 0.)

    if VECTOR_SALIENCY_KEY in dataset_object.variables:
        saliency_dict[VECTOR_SALIENCY_KEY] = (
            dataset_object.variables[VECTOR_SALIENCY_KEY][:]
        )
    else:
        these_dim = (num_examples, num_heights, num_vector_predictors)
        saliency_dict[VECTOR_SALIENCY_KEY] = numpy.full(these_dim, 0.)

    dataset_object.close()
    return saliency_dict


def write_all_targets_file(
        netcdf_file_name, saliency_matrix_scalar_p_scalar_t,
        saliency_matrix_vector_p_scalar_t, saliency_matrix_scalar_p_vector_t,
        saliency_matrix_vector_p_vector_t, example_id_strings, model_file_name,
        ideal_activation=None):
    """Writes saliency maps for all target variables to NetCDF file.

    E = number of examples
    H = number of heights
    P_s = number of scalar predictors
    P_v = number of vector predictors
    T_s = number of scalar targets
    T_v = number of vector targets

    :param netcdf_file_name: Path to output file.
    :param saliency_matrix_scalar_p_scalar_t: numpy array (E x P_s x T_s) with
        saliency maps for each scalar target with respect to each scalar
        predictor.
    :param saliency_matrix_vector_p_scalar_t: numpy array (E x H x P_v x T_s)
        with saliency maps for each scalar target with respect to each vector
        predictor.
    :param saliency_matrix_scalar_p_vector_t: numpy array (E x P_s x H x T_v)
        with saliency maps for each vector target with respect to each scalar
        predictor.
    :param saliency_matrix_vector_p_vector_t: numpy array
        (E x H x P_v x H x T_v) with saliency maps for each vector target with
        respect to each vector predictor.
    :param example_id_strings: See doc for `write_standard_file`.
    :param model_file_name: Same.
    :param ideal_activation: Same.
    """

    error_checking.assert_is_string_list(example_id_strings)
    error_checking.assert_is_numpy_array(
        numpy.array(example_id_strings), num_dimensions=1
    )

    error_checking.assert_is_numpy_array_without_nan(
        saliency_matrix_scalar_p_scalar_t
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_scalar_p_scalar_t, num_dimensions=3
    )

    num_examples = len(example_id_strings)
    num_scalar_predictors = saliency_matrix_scalar_p_scalar_t.shape[1]
    num_scalar_targets = saliency_matrix_scalar_p_scalar_t.shape[2]

    expected_dim = numpy.array(
        [num_examples, num_scalar_predictors, num_scalar_targets], dtype=int
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_scalar_p_scalar_t, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(
        saliency_matrix_vector_p_scalar_t
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_vector_p_scalar_t, num_dimensions=4
    )

    num_heights = saliency_matrix_vector_p_scalar_t.shape[1]
    num_vector_predictors = saliency_matrix_vector_p_scalar_t.shape[2]

    expected_dim = numpy.array(
        [num_examples, num_heights, num_vector_predictors, num_scalar_targets],
        dtype=int
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_vector_p_scalar_t, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(
        saliency_matrix_scalar_p_vector_t
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_scalar_p_vector_t, num_dimensions=4
    )

    num_vector_targets = saliency_matrix_scalar_p_vector_t.shape[3]

    expected_dim = numpy.array(
        [num_examples, num_scalar_predictors, num_heights, num_vector_targets],
        dtype=int
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_scalar_p_vector_t, exact_dimensions=expected_dim
    )

    error_checking.assert_is_numpy_array_without_nan(
        saliency_matrix_vector_p_vector_t
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix_vector_p_vector_t, num_dimensions=5
    )

    expected_dim = numpy.array([
        num_examples, num_heights, num_vector_predictors, num_heights,
        num_vector_targets
    ], dtype=int)
    error_checking.assert_is_numpy_array(
        saliency_matrix_vector_p_vector_t, exact_dimensions=expected_dim
    )

    error_checking.assert_is_string(model_file_name)
    if ideal_activation is not None:
        error_checking.assert_is_not_nan(ideal_activation)

    # Write to NetCDF file.
    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    dataset_object = netCDF4.Dataset(
        netcdf_file_name, 'w', format='NETCDF3_64BIT_OFFSET'
    )

    dataset_object.setncattr(MODEL_FILE_KEY, model_file_name)
    dataset_object.setncattr(
        IDEAL_ACTIVATION_KEY,
        numpy.nan if ideal_activation is None else ideal_activation
    )

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        SCALAR_PREDICTOR_DIM_KEY, num_scalar_predictors
    )
    dataset_object.createDimension(
        VECTOR_PREDICTOR_DIM_KEY, num_vector_predictors
    )
    dataset_object.createDimension(HEIGHT_DIMENSION_KEY, num_heights)
    dataset_object.createDimension(SCALAR_TARGET_DIM_KEY, num_scalar_targets)
    dataset_object.createDimension(VECTOR_TARGET_DIM_KEY, num_vector_targets)

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

    if saliency_matrix_scalar_p_scalar_t.size > 0:
        these_dim = (
            EXAMPLE_DIMENSION_KEY, SCALAR_PREDICTOR_DIM_KEY,
            SCALAR_TARGET_DIM_KEY
        )
        dataset_object.createVariable(
            SALIENCY_SCALAR_P_SCALAR_T_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[SALIENCY_SCALAR_P_SCALAR_T_KEY][:] = (
            saliency_matrix_scalar_p_scalar_t
        )

    if saliency_matrix_vector_p_scalar_t.size > 0:
        these_dim = (
            EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
            VECTOR_PREDICTOR_DIM_KEY, SCALAR_TARGET_DIM_KEY
        )
        dataset_object.createVariable(
            SALIENCY_VECTOR_P_SCALAR_T_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[SALIENCY_VECTOR_P_SCALAR_T_KEY][:] = (
            saliency_matrix_vector_p_scalar_t
        )

    if saliency_matrix_scalar_p_vector_t.size > 0:
        these_dim = (
            EXAMPLE_DIMENSION_KEY, SCALAR_PREDICTOR_DIM_KEY,
            HEIGHT_DIMENSION_KEY, VECTOR_TARGET_DIM_KEY
        )
        dataset_object.createVariable(
            SALIENCY_SCALAR_P_VECTOR_T_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[SALIENCY_SCALAR_P_VECTOR_T_KEY][:] = (
            saliency_matrix_scalar_p_vector_t
        )

    if saliency_matrix_vector_p_vector_t.size > 0:
        these_dim = (
            EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
            VECTOR_PREDICTOR_DIM_KEY, HEIGHT_DIMENSION_KEY,
            VECTOR_TARGET_DIM_KEY
        )
        dataset_object.createVariable(
            SALIENCY_VECTOR_P_VECTOR_T_KEY, datatype=numpy.float32,
            dimensions=these_dim
        )
        dataset_object.variables[SALIENCY_VECTOR_P_VECTOR_T_KEY][:] = (
            saliency_matrix_vector_p_vector_t
        )

    dataset_object.close()


def read_all_targets_file(netcdf_file_name):
    """Reads saliency maps for all target variables from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['saliency_matrix_scalar_p_scalar_t']: See doc for
        `write_all_targets_file`.
    saliency_dict['saliency_matrix_vector_p_scalar_t']: Same.
    saliency_dict['saliency_matrix_scalar_p_vector_t']: Same.
    saliency_dict['saliency_matrix_vector_p_vector_t']: Same.
    saliency_dict['example_id_strings']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['ideal_activation']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    saliency_dict = {
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY)
    }

    num_examples = dataset_object.dimensions[EXAMPLE_DIMENSION_KEY].size
    num_scalar_predictors = (
        dataset_object.dimensions[SCALAR_PREDICTOR_DIM_KEY].size
    )
    num_vector_predictors = (
        dataset_object.dimensions[VECTOR_PREDICTOR_DIM_KEY].size
    )
    num_heights = dataset_object.dimensions[HEIGHT_DIMENSION_KEY].size
    num_scalar_targets = dataset_object.dimensions[SCALAR_TARGET_DIM_KEY].size
    num_vector_targets = dataset_object.dimensions[VECTOR_TARGET_DIM_KEY].size

    if SALIENCY_SCALAR_P_SCALAR_T_KEY in dataset_object.variables:
        saliency_dict[SALIENCY_SCALAR_P_SCALAR_T_KEY] = (
            dataset_object.variables[SALIENCY_SCALAR_P_SCALAR_T_KEY][:]
        )
    else:
        these_dim = (num_examples, num_scalar_predictors, num_scalar_targets)
        saliency_dict[SALIENCY_SCALAR_P_SCALAR_T_KEY] = numpy.full(
            these_dim, 0.
        )

    if SALIENCY_VECTOR_P_SCALAR_T_KEY in dataset_object.variables:
        saliency_dict[SALIENCY_VECTOR_P_SCALAR_T_KEY] = (
            dataset_object.variables[SALIENCY_VECTOR_P_SCALAR_T_KEY][:]
        )
    else:
        these_dim = (
            num_examples, num_heights, num_vector_predictors, num_scalar_targets
        )
        saliency_dict[SALIENCY_VECTOR_P_SCALAR_T_KEY] = numpy.full(
            these_dim, 0.
        )

    if SALIENCY_SCALAR_P_VECTOR_T_KEY in dataset_object.variables:
        saliency_dict[SALIENCY_SCALAR_P_VECTOR_T_KEY] = (
            dataset_object.variables[SALIENCY_SCALAR_P_VECTOR_T_KEY][:]
        )
    else:
        these_dim = (
            num_examples, num_scalar_predictors, num_heights, num_vector_targets
        )
        saliency_dict[SALIENCY_SCALAR_P_VECTOR_T_KEY] = numpy.full(
            these_dim, 0.
        )

    if SALIENCY_VECTOR_P_VECTOR_T_KEY in dataset_object.variables:
        saliency_dict[SALIENCY_VECTOR_P_VECTOR_T_KEY] = (
            dataset_object.variables[SALIENCY_VECTOR_P_VECTOR_T_KEY][:]
        )
    else:
        these_dim = (
            num_examples, num_heights, num_vector_predictors,
            num_heights, num_vector_targets
        )
        saliency_dict[SALIENCY_VECTOR_P_VECTOR_T_KEY] = numpy.full(
            these_dim, 0.
        )

    if numpy.isnan(saliency_dict[IDEAL_ACTIVATION_KEY]):
        saliency_dict[IDEAL_ACTIVATION_KEY] = None

    dataset_object.close()
    return saliency_dict
