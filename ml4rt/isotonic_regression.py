"""Methods for building, training, and applying isotonic-regression models."""

import os
import sys
import dill
import numpy
from sklearn.isotonic import IsotonicRegression

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking


def train_models(
        orig_vector_prediction_matrix, orig_scalar_prediction_matrix,
        vector_target_matrix, scalar_target_matrix):
    """Trains isotonic-regression models.

    E = number of examples
    H = number of heights
    T_v = number of vector target variables
    T_s = number of scalar target variables
    W = number of wavelengths
    S = number of ensemble members

    :param orig_vector_prediction_matrix: numpy array (E x H x W x T_v x S) of
        predicted values for vector target variables.
    :param orig_scalar_prediction_matrix: numpy array (E x W x T_s x S) of
        predicted values for scalar target variables.
    :param vector_target_matrix: numpy array (E x H x W x T_v) of actual values
        for vector target variables.
    :param scalar_target_matrix: numpy array (E x W x T_s) of actual values for
        scalar target variables.
    :return: scalar_model_object_matrix: numpy array of models
        (instances of `sklearn.isotonic.IsotonicRegression`) for scalar target
        variables.  Dimensions are W x T_s.
    :return: vector_model_object_matrix: numpy array of models
        (instances of `sklearn.isotonic.IsotonicRegression`) for vector target
        variables.  Dimensions are H x W x T_v.
    """

    # Check input args.
    num_examples = None
    num_heights = 0
    num_wavelengths = None
    num_vector_targets = 0
    num_scalar_targets = 0
    ensemble_size = None

    have_vectors = (
        orig_vector_prediction_matrix is not None
        or vector_target_matrix is not None
    )

    if have_vectors:
        orig_vector_prediction_matrix = orig_vector_prediction_matrix.astype(
            numpy.float32
        )
        vector_target_matrix = vector_target_matrix.astype(numpy.float32)

        error_checking.assert_is_numpy_array(
            orig_vector_prediction_matrix, num_dimensions=5
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_vector_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            vector_target_matrix,
            exact_dimensions=numpy.array(
                orig_vector_prediction_matrix.shape[:-1], dtype=int
            )
        )
        error_checking.assert_is_numpy_array_without_nan(vector_target_matrix)

        num_examples = vector_target_matrix.shape[0]
        num_heights = vector_target_matrix.shape[1]
        num_wavelengths = vector_target_matrix.shape[2]
        num_vector_targets = vector_target_matrix.shape[3]
        ensemble_size = orig_vector_prediction_matrix.shape[4]

    have_scalars = (
        orig_scalar_prediction_matrix is not None
        or scalar_target_matrix is not None
    )

    if have_scalars:
        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, num_dimensions=4
        )

        if num_examples is None:
            num_examples = orig_scalar_prediction_matrix.shape[0]
        if num_wavelengths is None:
            num_wavelengths = orig_scalar_prediction_matrix.shape[1]
        if ensemble_size is None:
            ensemble_size = orig_scalar_prediction_matrix.shape[3]

        expected_dim = numpy.array([
            num_examples, num_wavelengths,
            orig_scalar_prediction_matrix.shape[2], ensemble_size
        ], dtype=int)

        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_scalar_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            scalar_target_matrix,
            exact_dimensions=numpy.array(
                orig_scalar_prediction_matrix.shape[:-1], dtype=int
            )
        )
        error_checking.assert_is_numpy_array_without_nan(scalar_target_matrix)

        num_scalar_targets = scalar_target_matrix.shape[2]

    # Do actual stuff.
    scalar_model_object_matrix = numpy.full(
        (num_wavelengths, num_scalar_targets), '', dtype=object
    )
    vector_model_object_matrix = numpy.full(
        (num_heights, num_wavelengths, num_vector_targets), '', dtype=object
    )

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            print((
                'Training isotonic-regression model for {0:d}th of {1:d} '
                'scalar target variables at {2:d}th of {3:d} wavelengths...'
            ).format(
                t + 1, num_scalar_targets,
                w + 1, num_wavelengths
            ))

            this_prediction_matrix = (
                orig_scalar_prediction_matrix[:, w, t, :] + 0.
            )
            this_target_matrix = numpy.expand_dims(
                scalar_target_matrix[:, w, t], axis=-1
            )
            this_target_matrix = numpy.repeat(
                this_target_matrix, axis=-1, repeats=ensemble_size
            )

            scalar_model_object_matrix[w, t] = IsotonicRegression(
                increasing=True, out_of_bounds='clip'
            )
            scalar_model_object_matrix[w, t].fit(
                X=numpy.ravel(this_prediction_matrix),
                y=numpy.ravel(this_target_matrix)
            )

        print('\n')

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            for h in range(num_heights):
                print((
                    'Training isotonic-regression model for {0:d}th of {1:d} '
                    'vector target variables at {2:d}th of {3:d} wavelengths '
                    'and {4:d}th of {5:d} heights...'
                ).format(
                    t + 1, num_vector_targets,
                    w + 1, num_wavelengths,
                    h + 1, num_heights
                ))

                vector_model_object_matrix[h, w, t] = IsotonicRegression(
                    increasing=True, out_of_bounds='clip'
                )

                this_prediction_matrix = (
                    orig_vector_prediction_matrix[:, h, w, t, :] + 0.
                )
                this_target_matrix = numpy.expand_dims(
                    vector_target_matrix[:, h, w, t], axis=-1
                )
                this_target_matrix = numpy.repeat(
                    this_target_matrix, axis=-1, repeats=ensemble_size
                )

                vector_model_object_matrix[h, w, t].fit(
                    X=numpy.ravel(this_prediction_matrix),
                    y=numpy.ravel(this_target_matrix)
                )

            print('\n')

    return scalar_model_object_matrix, vector_model_object_matrix


def apply_models(
        orig_vector_prediction_matrix, orig_scalar_prediction_matrix,
        scalar_model_object_matrix, vector_model_object_matrix):
    """Applies isotonic-regression models.

    :param orig_vector_prediction_matrix: See doc for `train_models`.
    :param orig_scalar_prediction_matrix: Same.
    :param scalar_model_object_matrix: Same.
    :param vector_model_object_matrix: Same.
    :return: new_vector_prediction_matrix: Same as
        `orig_vector_prediction_matrix` but with transformed values.
    :return: new_scalar_prediction_matrix: Same as
        `orig_scalar_prediction_matrix` but with transformed values.
    """

    # Check input args.
    num_examples = None
    num_heights = 0
    num_wavelengths = None
    num_vector_targets = 0
    num_scalar_targets = 0
    ensemble_size = None

    have_vectors = (
        orig_vector_prediction_matrix is not None
        or vector_model_object_matrix.size > 0
    )

    if have_vectors:
        orig_vector_prediction_matrix = orig_vector_prediction_matrix.astype(
            numpy.float16
        )

        error_checking.assert_is_numpy_array(
            orig_vector_prediction_matrix, num_dimensions=5
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_vector_prediction_matrix
        )

        error_checking.assert_is_numpy_array(
            vector_model_object_matrix, num_dimensions=3
        )

        num_heights = orig_vector_prediction_matrix.shape[1]
        num_wavelengths = orig_vector_prediction_matrix.shape[2]
        num_vector_targets = orig_vector_prediction_matrix.shape[3]
        expected_dim = numpy.array(
            [num_heights, num_wavelengths, num_vector_targets], dtype=int
        )
        error_checking.assert_is_numpy_array(
            vector_model_object_matrix, exact_dimensions=expected_dim
        )

        num_examples = orig_vector_prediction_matrix.shape[0]
        ensemble_size = orig_vector_prediction_matrix.shape[4]

    have_scalars = (
        orig_scalar_prediction_matrix is not None
        or scalar_model_object_matrix.size > 0
    )

    if have_scalars:
        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, num_dimensions=4
        )

        if num_examples is None:
            num_examples = orig_scalar_prediction_matrix.shape[0]
        if ensemble_size is None:
            ensemble_size = orig_scalar_prediction_matrix.shape[3]

        num_scalar_targets = orig_scalar_prediction_matrix.shape[2]
        expected_dim = numpy.array(
            [num_examples, num_wavelengths, num_scalar_targets, ensemble_size],
            dtype=int
        )

        error_checking.assert_is_numpy_array(
            orig_scalar_prediction_matrix, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array_without_nan(
            orig_scalar_prediction_matrix
        )

        expected_dim = numpy.array(
            [num_wavelengths, num_scalar_targets], dtype=int
        )
        error_checking.assert_is_numpy_array(
            numpy.array(scalar_model_object_matrix),
            exact_dimensions=expected_dim
        )

    if have_vectors:
        new_vector_prediction_matrix = numpy.full(
            orig_vector_prediction_matrix.shape, numpy.nan, dtype=numpy.float32
        )
    else:
        new_vector_prediction_matrix = numpy.full(
            (num_examples, 0, num_wavelengths, 0, ensemble_size),
            numpy.nan,
            dtype=numpy.float32
        )

    if have_scalars:
        new_scalar_prediction_matrix = numpy.full(
            orig_scalar_prediction_matrix.shape, numpy.nan
        )
    else:
        new_scalar_prediction_matrix = numpy.full(
            (num_examples, num_wavelengths, 0, ensemble_size), numpy.nan
        )

    for t in range(num_scalar_targets):
        for w in range(num_wavelengths):
            print((
                'Applying isotonic-regression model to {0:d}th of {1:d} scalar '
                'target variables at {2:d}th of {3:d} wavelengths...'
            ).format(
                t + 1, num_scalar_targets,
                w + 1, num_wavelengths
            ))

            for s in range(ensemble_size):
                new_scalar_prediction_matrix[:, w, t, s] = (
                    scalar_model_object_matrix[w, t].predict(
                        orig_scalar_prediction_matrix[:, w, t, s]
                    )
                )

        print('\n')

    for t in range(num_vector_targets):
        for w in range(num_wavelengths):
            for h in range(num_heights):
                print((
                    'Applying isotonic-regression model to {0:d}th of {1:d} '
                    'vector target variables at {2:d}th of {3:d} heights '
                    'and {4:d}th of {5:d} wavelengths...'
                ).format(
                    t + 1, num_vector_targets,
                    h + 1, num_heights,
                    w + 1, num_wavelengths
                ))

                for s in range(ensemble_size):
                    new_vector_prediction_matrix[:, h, w, t, s] = (
                        vector_model_object_matrix[h, w, t].predict(
                            orig_vector_prediction_matrix[:, h, w, t, s]
                        )
                    )

            print('\n')

    return new_vector_prediction_matrix, new_scalar_prediction_matrix


def find_file(model_dir_name, raise_error_if_missing=True):
    """Finds Dill file with set of isotonic-regression models.

    :param model_dir_name: Name of directory.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: dill_file_name: Path to Dill file with models.
    """

    error_checking.assert_is_string(model_dir_name)
    error_checking.assert_is_boolean(raise_error_if_missing)

    dill_file_name = '{0:s}/isotonic_regression.dill'.format(model_dir_name)

    if raise_error_if_missing and not os.path.isfile(dill_file_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            dill_file_name
        )
        raise ValueError(error_string)

    return dill_file_name


def write_file(dill_file_name, scalar_model_object_matrix,
               vector_model_object_matrix):
    """Writes set of isotonic-regression models to Dill file.

    :param dill_file_name: Path to output file.
    :param scalar_model_object_matrix: See doc for `train_models`.
    :param vector_model_object_matrix: Same.
    """

    error_checking.assert_is_numpy_array(
        numpy.array(scalar_model_object_matrix), num_dimensions=2
    )
    error_checking.assert_is_numpy_array(
        vector_model_object_matrix, num_dimensions=3
    )

    file_system_utils.mkdir_recursive_if_necessary(file_name=dill_file_name)

    dill_file_handle = open(dill_file_name, 'wb')
    dill.dump(scalar_model_object_matrix, dill_file_handle)
    dill.dump(vector_model_object_matrix, dill_file_handle)
    dill_file_handle.close()


def read_file(dill_file_name):
    """Reads set of isotonic-regression models from Dill file.

    :param dill_file_name: Path to input file.
    :return: scalar_model_object_matrix: See doc for `train_models`.
    :return: vector_model_object_matrix: Same.
    """

    error_checking.assert_file_exists(dill_file_name)

    dill_file_handle = open(dill_file_name, 'rb')
    scalar_model_object_matrix = dill.load(dill_file_handle)
    vector_model_object_matrix = dill.load(dill_file_handle)
    dill_file_handle.close()

    scalar_model_object_matrix = numpy.array(scalar_model_object_matrix)
    if len(scalar_model_object_matrix.shape) == 1:
        scalar_model_object_matrix = numpy.expand_dims(
            scalar_model_object_matrix, axis=0
        )

    if len(vector_model_object_matrix.shape) == 1:
        vector_model_object_matrix = numpy.expand_dims(
            vector_model_object_matrix, axis=-2
        )

    return scalar_model_object_matrix, vector_model_object_matrix
