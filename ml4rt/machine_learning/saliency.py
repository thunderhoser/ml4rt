"""Methods for computing, reading, and writing saliency maps."""

import numpy
import netCDF4
from keras import backend as K
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as saliency_utils

EXAMPLE_DIMENSION_KEY = 'example'
HEIGHT_DIMENSION_KEY = 'height'
CHANNEL_DIMENSION_KEY = 'channel'
EXAMPLE_ID_CHAR_DIM_KEY = 'example_id_char'

MODEL_FILE_KEY = 'model_file_name'
LAYER_NAME_KEY = 'layer_name'
NEURON_INDICES_KEY = 'neuron_indices'
IDEAL_ACTIVATION_KEY = 'ideal_activation'

EXAMPLE_IDS_KEY = 'example_id_strings'
SALIENCY_KEY = 'saliency_matrix'


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
        netcdf_file_name, saliency_matrix, example_id_strings, model_file_name,
        layer_name, neuron_indices, ideal_activation=None):
    """Writes standard (non-averaged) saliency maps to NetCDF file.

    E = number of examples

    :param netcdf_file_name: Path to output file.
    :param saliency_matrix: numpy array (either 2-D or 3-D) of saliency values,
        where the first axis has length E.
    :param example_id_strings: length-E list of example IDs.
    :param model_file_name: Path to file with neural net used to create saliency
        maps (readable by `neural_net.read_model`).
    :param layer_name: See doc for `check_metadata`.
    :param neuron_indices: Same.
    :param ideal_activation: Same.
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

    error_checking.assert_is_numpy_array_without_nan(saliency_matrix)
    num_saliency_dim = len(saliency_matrix.shape)
    error_checking.assert_is_geq(num_saliency_dim, 2)
    error_checking.assert_is_leq(num_saliency_dim, 3)

    num_examples = len(example_id_strings)
    expected_dim = numpy.array(
        (num_examples,) + saliency_matrix.shape[1:], dtype=int
    )
    error_checking.assert_is_numpy_array(
        saliency_matrix, exact_dimensions=expected_dim
    )

    error_checking.assert_is_string(model_file_name)

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

    dataset_object.createDimension(EXAMPLE_DIMENSION_KEY, num_examples)
    dataset_object.createDimension(
        CHANNEL_DIMENSION_KEY, saliency_matrix.shape[-1]
    )

    if num_saliency_dim == 3:
        dataset_object.createDimension(
            HEIGHT_DIMENSION_KEY, saliency_matrix.shape[1]
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

    if num_saliency_dim == 2:
        dataset_object.createVariable(
            SALIENCY_KEY, datatype=numpy.float32,
            dimensions=(EXAMPLE_DIMENSION_KEY, CHANNEL_DIMENSION_KEY)
        )
    else:
        dataset_object.createVariable(
            SALIENCY_KEY, datatype=numpy.float32,
            dimensions=(
                EXAMPLE_DIMENSION_KEY, HEIGHT_DIMENSION_KEY,
                CHANNEL_DIMENSION_KEY
            )
        )

    dataset_object.variables[SALIENCY_KEY][:] = saliency_matrix
    dataset_object.close()


def read_standard_file(netcdf_file_name):
    """Reads standard (non-averaged) saliency maps from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: saliency_dict: Dictionary with the following keys.
    saliency_dict['saliency_matrix']: See doc for `write_standard_file`.
    saliency_dict['example_id_strings']: Same.
    saliency_dict['model_file_name']: Same.
    saliency_dict['layer_name']: Same.
    saliency_dict['neuron_indices']: Same.
    saliency_dict['ideal_activation']: Same.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)

    saliency_dict = {
        SALIENCY_KEY: dataset_object.variables[SALIENCY_KEY][:],
        EXAMPLE_IDS_KEY: [
            str(id) for id in
            netCDF4.chartostring(dataset_object.variables[EXAMPLE_IDS_KEY][:])
        ],
        MODEL_FILE_KEY: str(getattr(dataset_object, MODEL_FILE_KEY)),
        LAYER_NAME_KEY: str(getattr(dataset_object, LAYER_NAME_KEY)),
        NEURON_INDICES_KEY: numpy.array(
            getattr(dataset_object, NEURON_INDICES_KEY), dtype=int
        ),
        IDEAL_ACTIVATION_KEY: getattr(dataset_object, IDEAL_ACTIVATION_KEY)
    }

    print(saliency_dict[IDEAL_ACTIVATION_KEY])

    if numpy.isnan(saliency_dict[IDEAL_ACTIVATION_KEY]):
        saliency_dict[IDEAL_ACTIVATION_KEY] = None

    print(saliency_dict[IDEAL_ACTIVATION_KEY])

    dataset_object.close()
    return saliency_dict
