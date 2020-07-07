"""Helper methods for Grad-CAM (gradient-weighted class-activation maps)."""

import numpy
from keras import backend as K
from gewittergefahr.gg_utils import error_checking
from gewittergefahr.deep_learning import gradcam as gradcam_utils

DEFAULT_IDEAL_ACTIVATION = 2.


def _normalize_tensor(input_tensor):
    """Normalizes tensor by its L2 norm.

    :param input_tensor: Unnormalized tensor.
    :return: output_tensor: Normalized tensor.
    """

    rms_tensor = K.sqrt(K.mean(K.square(input_tensor)))
    return input_tensor / (rms_tensor + K.epsilon())


def run_gradcam(
        model_object, predictor_matrix, activation_layer_name,
        vector_output_layer_name, output_neuron_indices,
        ideal_activation=DEFAULT_IDEAL_ACTIVATION):
    """Runs the Grad-CAM algorithm.

    H = number of heights
    P = number of predictor variables

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: H-by-P numpy array of predictor values.
    :param activation_layer_name: Name of activation layer.
    :param vector_output_layer_name: Name of layer that outputs predictions for
        vector target variables.
    :param output_neuron_indices: length-2 numpy array with indices of output
        neuron (height index, then channel index).  Class activation will be
        computed with respect to the output of this neuron.
    :param ideal_activation: Ideal neuron activation, used to define loss
        function.  The loss function will be
        (output_neuron_activation - ideal_activation)**2.
    :return: layer_activation_matrix: length-H numpy array of class activations.
    """

    # Check input args.
    error_checking.assert_is_numpy_array_without_nan(predictor_matrix)
    error_checking.assert_is_numpy_array(predictor_matrix, num_dimensions=2)
    predictor_matrix = numpy.expand_dims(predictor_matrix, axis=0)

    error_checking.assert_is_integer_numpy_array(output_neuron_indices)
    error_checking.assert_is_geq_numpy_array(output_neuron_indices, 0)
    error_checking.assert_is_numpy_array(
        output_neuron_indices, exact_dimensions=numpy.array([2], dtype=int)
    )

    error_checking.assert_is_not_nan(ideal_activation)

    # Set up loss function.
    output_tensor = model_object.get_layer(
        name=vector_output_layer_name
    ).output[:, output_neuron_indices[0], output_neuron_indices[1]]

    loss_tensor = (output_tensor - ideal_activation) ** 2

    # Set up gradient function.
    layer_activation_tensor = (
        model_object.get_layer(name=activation_layer_name).output
    )
    gradient_tensor = (
        K.gradients(loss_tensor, [layer_activation_tensor])[0]
    )
    gradient_tensor = _normalize_tensor(gradient_tensor)
    gradient_function = K.function(
        [model_object.input], [layer_activation_tensor, gradient_tensor]
    )

    # Evaluate gradient function.
    layer_activation_matrix, gradient_matrix = gradient_function(
        [predictor_matrix]
    )
    layer_activation_matrix = layer_activation_matrix[0, ...]
    gradient_matrix = gradient_matrix[0, ...]

    # Compute class-activation map in activation layer's space.
    mean_weight_by_filter = numpy.mean(gradient_matrix, axis=0)
    class_activation_matrix = numpy.ones(layer_activation_matrix.shape[:-1])
    num_filters = len(mean_weight_by_filter)

    for k in range(num_filters):
        class_activation_matrix += (
            mean_weight_by_filter[k] * layer_activation_matrix[..., k]
        )

    num_input_heights = predictor_matrix.shape[1]
    class_activation_matrix = gradcam_utils._upsample_cam(
        class_activation_matrix=class_activation_matrix,
        new_dimensions=numpy.array([num_input_heights], dtype=int)
    )

    return numpy.maximum(class_activation_matrix, 0.)
