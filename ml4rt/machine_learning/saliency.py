"""Methods for computing, reading, and writing saliency maps."""

import numpy
import netCDF4
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as saliency_utils
from ml4rt.io import example_io

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
SCALAR_PREDICTOR_DIM_KEY = 'scalar_predictor'
VECTOR_PREDICTOR_DIM_KEY = 'vector_predictor'
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
    :param scalar_saliency_matrix: numpy array (E x P_s) with saliency values
        for scalar predictors.
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

    error_checking.assert_is_numpy_array_without_nan(scalar_saliency_matrix)
    error_checking.assert_is_numpy_array(
        scalar_saliency_matrix, num_dimensions=2
    )
    expected_dim = numpy.array(
        (num_examples,) + scalar_saliency_matrix.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        scalar_saliency_matrix, exact_dimensions=expected_dim
    )

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
        SCALAR_PREDICTOR_DIM_KEY, scalar_saliency_matrix.shape[1]
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

    dataset_object.createVariable(
        SCALAR_SALIENCY_KEY, datatype=numpy.float32,
        dimensions=(EXAMPLE_DIMENSION_KEY, SCALAR_PREDICTOR_DIM_KEY)
    )
    dataset_object.variables[SCALAR_SALIENCY_KEY][:] = scalar_saliency_matrix

    these_dim = (
        EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY, VECTOR_PREDICTOR_DIM_KEY
    )
    dataset_object.createVariable(
        VECTOR_SALIENCY_KEY, datatype=numpy.float32, dimensions=these_dim
    )
    dataset_object.variables[VECTOR_SALIENCY_KEY][:] = vector_saliency_matrix

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
        SCALAR_SALIENCY_KEY: dataset_object.variables[SCALAR_SALIENCY_KEY][:],
        VECTOR_SALIENCY_KEY: dataset_object.variables[VECTOR_SALIENCY_KEY][:],
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

    dataset_object.close()
    return saliency_dict
