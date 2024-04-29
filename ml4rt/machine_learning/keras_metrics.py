"""Evaluation scores used during training.

Definitions:

E = number of examples
H = number of heights
W = number of wavelengths
T = number of target variables
S = ensemble size
"""

from tensorflow.keras import backend as K


def mean_bias(target_tensor, prediction_tensor):
    """Computes mean bias (mean signed error).

    Assumes deterministic model -- i.e., prediction_tensor should not have an
    extra axis for ensemble member.

    :param target_tensor: Keras tensor with target values.
    :param prediction_tensor: Keras tensor with predicted values.
    :return: mean_bias: Mean bias.
    """

    # return prediction_tensor.get_shape()[1]
    return K.mean(prediction_tensor - target_tensor)


def mean_absolute_error(target_tensor, prediction_tensor):
    """Computes mean absolute error.

    Assumes deterministic model -- i.e., prediction_tensor should not have an
    extra axis for ensemble member.

    :param target_tensor: Keras tensor with target values.
    :param prediction_tensor: Keras tensor with predicted values.
    :return: mean_absolute_error: Mean absolute error.
    """

    return K.mean(K.abs(prediction_tensor - target_tensor))


def mae_skill_score(target_tensor, prediction_tensor):
    """Computes mean-absolute-error (MAE) skill score.

    Assumes deterministic model -- i.e., prediction_tensor should not have an
    extra axis for ensemble member.

    :param target_tensor: Keras tensor with target values.
    :param prediction_tensor: Keras tensor with predicted values.
    :return: mae_skill_score: MAE skill score.
    """

    mean_target_tensor = K.mean(target_tensor, axis=0, keepdims=True)

    mae_actual = mean_absolute_error(target_tensor, prediction_tensor)
    mae_climo = mean_absolute_error(target_tensor, mean_target_tensor)
    return (mae_climo - mae_actual) / mae_climo


def mean_squared_error(target_tensor, prediction_tensor):
    """Computes mean squared error.

    Assumes deterministic model -- i.e., prediction_tensor should not have an
    extra axis for ensemble member.

    :param target_tensor: Keras tensor with target values.
    :param prediction_tensor: Keras tensor with predicted values.
    :return: mean_squared_error: Mean squared error.
    """

    return K.mean((prediction_tensor - target_tensor) ** 2)


def mse_skill_score(target_tensor, prediction_tensor):
    """Computes mean-squared-error (MSE) skill score.

    Assumes deterministic model -- i.e., prediction_tensor should not have an
    extra axis for ensemble member.

    :param target_tensor: Keras tensor with target values.
    :param prediction_tensor: Keras tensor with predicted values.
    :return: mse_skill_score: MSE skill score.
    """

    mean_target_tensor = K.mean(target_tensor, axis=0, keepdims=True)

    mse_actual = mean_squared_error(target_tensor, prediction_tensor)
    mse_climo = mean_squared_error(target_tensor, mean_target_tensor)
    return (mse_climo - mse_actual) / mse_climo


def correlation(target_tensor, prediction_tensor):
    """Computes correlation.

    Assumes deterministic model -- i.e., prediction_tensor should not have an
    extra axis for ensemble member.

    :param target_tensor: Keras tensor with target values.
    :param prediction_tensor: Keras tensor with predicted values.
    :return: correlation: Correlation.
    """

    numerator = K.sum(
        (target_tensor - K.mean(target_tensor)) *
        (prediction_tensor - K.mean(prediction_tensor))
    )
    sum_squared_target_diffs = K.sum(
        (target_tensor - K.mean(target_tensor)) ** 2
    )
    sum_squared_prediction_diffs = K.sum(
        (prediction_tensor - K.mean(prediction_tensor)) ** 2
    )

    return (
        numerator /
        K.sqrt(sum_squared_target_diffs * sum_squared_prediction_diffs)
    )
