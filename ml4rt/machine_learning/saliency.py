"""Methods for computing, reading, and writing saliency maps."""

from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import saliency_maps as saliency_utils


def get_saliency_one_neuron(
        model_object, layer_name, neuron_indices, predictor_matrix,
        ideal_activation=None):
    """Computes saliency maps with respect to activation of one neuron.

    :param model_object: Trained neural net (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param layer_name: Name of layer with relevant neuron.
    :param neuron_indices: 1-D numpy array with indices of relevant neuron.
        Must have length D - 1, where D = number of dimensions in layer output.
        The first dimension is the batch dimension, which always has length
        `None` in Keras.
    :param predictor_matrix: numpy array of predictors.  Must be formatted in
        the same way as for training and inference.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  If you specify this, the loss function will be
        (neuron_activation - ideal_activation)**2.  If you leave this as None,
        the loss function will be
        -sign(neuron_activation) * neuron_activation**2.
    :return: saliency_matrix: Matrix of saliency values, with same shape as
        `predictor_matrix`.
    """

    error_checking.assert_is_string(layer_name)
    error_checking.assert_is_integer_numpy_array(neuron_indices)
    error_checking.assert_is_geq_numpy_array(neuron_indices, 0)
    error_checking.assert_is_numpy_array(neuron_indices, num_dimensions=1)
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
        error_checking.assert_is_not_nan(ideal_activation)
        loss_tensor = (activation_tensor - ideal_activation) ** 2

    return saliency_utils.do_saliency_calculations(
        model_object=model_object, loss_tensor=loss_tensor,
        list_of_input_matrices=[predictor_matrix]
    )[0]
