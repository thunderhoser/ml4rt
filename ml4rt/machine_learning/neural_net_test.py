"""Unit tests for neural_net.py."""

import copy
import unittest
import numpy
from ml4rt.utils import example_utils
from ml4rt.machine_learning import neural_net

TOLERANCE = 1e-6

# The following constants are used to test predictors_dict_to_numpy,
# predictors_numpy_to_dict, targets_dict_to_numpy, and targets_numpy_to_dict.
HEIGHTS_M_AGL = numpy.array([100, 500], dtype=float)
TARGET_WAVELENGTHS_METRES = numpy.array([1e-6, 2e-6, 3e-6])
VALID_TIMES_UNIX_SEC = numpy.array([0, 300, 600, 1200], dtype=int)
STANDARD_ATMO_FLAGS = numpy.array([0, 1, 2, 3], dtype=int)

SCALAR_PREDICTOR_NAMES = [
    example_utils.ZENITH_ANGLE_NAME, example_utils.LATITUDE_NAME
]
ZENITH_ANGLES_RADIANS = numpy.array([0, 1, 2, 3], dtype=float)
LATITUDES_DEG_N = numpy.array([40.02, 40.02, 40.02, 40.02])
SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (ZENITH_ANGLES_RADIANS, LATITUDES_DEG_N)
))

VECTOR_PREDICTOR_NAMES = [
    example_utils.TEMPERATURE_NAME, example_utils.SPECIFIC_HUMIDITY_NAME
]
TEMPERATURE_MATRIX_KELVINS = numpy.array([
    [290, 295],
    [289, 294],
    [288, 293],
    [287, 292.5]
])
SPEC_HUMIDITY_MATRIX_KG_KG01 = numpy.array([
    [0.008, 0.009],
    [0.007, 0.008],
    [0.005, 0.006],
    [0.0075, 0.01]
])
VECTOR_PREDICTOR_MATRIX = numpy.stack(
    (TEMPERATURE_MATRIX_KELVINS, SPEC_HUMIDITY_MATRIX_KG_KG01), axis=-1
)

SCALAR_TARGET_NAMES = [example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME]
SCALAR_TARGET_MATRIX = numpy.array([
    [200, 150, 50],
    [400, 300, 60],
    [300, 70, 75],
    [100, 100, 100]
], dtype=float)
SCALAR_TARGET_MATRIX = numpy.expand_dims(SCALAR_TARGET_MATRIX, axis=-1)

VECTOR_TARGET_NAMES = [
    example_utils.SHORTWAVE_DOWN_FLUX_NAME, example_utils.SHORTWAVE_UP_FLUX_NAME
]
DOWNWELLING_FLUX_MATRIX_W_M02 = numpy.array([
    [300, 200],
    [500, 300],
    [450, 450],
    [200, 100]
], dtype=float)
UPWELLING_FLUX_MATRIX_W_M02 = numpy.array([
    [150, 150],
    [200, 150],
    [300, 350],
    [400, 100]
], dtype=float)
VECTOR_TARGET_MATRIX_1WAVELENGTH = numpy.stack(
    (DOWNWELLING_FLUX_MATRIX_W_M02, UPWELLING_FLUX_MATRIX_W_M02), axis=-1
)

VECTOR_TARGET_MATRIX = numpy.stack((
    VECTOR_TARGET_MATRIX_1WAVELENGTH - 100,
    VECTOR_TARGET_MATRIX_1WAVELENGTH,
    VECTOR_TARGET_MATRIX_1WAVELENGTH + 100
), axis=-2)

EXAMPLE_DICT = {
    example_utils.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_utils.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_utils.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_utils.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_utils.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_utils.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_utils.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_utils.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_utils.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_utils.TARGET_WAVELENGTHS_KEY: TARGET_WAVELENGTHS_METRES,
    example_utils.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    example_utils.STANDARD_ATMO_FLAGS_KEY: STANDARD_ATMO_FLAGS
}

THIS_ZENITH_ANGLE_MATRIX = numpy.array([
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3]
], dtype=float)

THIS_LATITUDE_MATRIX = numpy.full((4, 2), 40.02)
THIS_SCALAR_PREDICTOR_MATRIX = numpy.stack(
    (THIS_ZENITH_ANGLE_MATRIX, THIS_LATITUDE_MATRIX), axis=-1
)
NN_PREDICTOR_MATRIX = numpy.concatenate(
    (VECTOR_PREDICTOR_MATRIX, THIS_SCALAR_PREDICTOR_MATRIX), axis=-1
)
NN_PREDICTOR_NAME_MATRIX = numpy.array([
    [example_utils.TEMPERATURE_NAME, example_utils.SPECIFIC_HUMIDITY_NAME,
     example_utils.ZENITH_ANGLE_NAME, example_utils.LATITUDE_NAME],
    [example_utils.TEMPERATURE_NAME, example_utils.SPECIFIC_HUMIDITY_NAME,
     example_utils.ZENITH_ANGLE_NAME, example_utils.LATITUDE_NAME]
])
NN_HEIGHT_MATRIX_M_AGL = numpy.array([
    [100, 100, numpy.nan, numpy.nan],
    [500, 500, numpy.nan, numpy.nan]
])

NN_TARGET_MATRICES = [
    VECTOR_TARGET_MATRIX + 0., SCALAR_TARGET_MATRIX + 0.
]
NN_TARGET_MATRICES_NO_SCALARS = [VECTOR_TARGET_MATRIX + 0.]

# The following constants are used to test neuron_indices_to_target_var and
# target_var_to_neuron_indices.
NEURON_INDICES_LIST = [
    numpy.array([0, 2, 0], dtype=int),
    numpy.array([0, 1, 1], dtype=int),
    numpy.array([1, 0, 0], dtype=int),
    numpy.array([1, 2, 1], dtype=int),
    numpy.array([0, 0], dtype=int),
    numpy.array([1, 0], dtype=int)
]
NN_HEIGHTS_M_AGL = [100, 100, 500, 500, None, None]
NN_TARGET_NAMES = [
    example_utils.SHORTWAVE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_UP_FLUX_NAME,
    example_utils.SHORTWAVE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_UP_FLUX_NAME,
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME,
    example_utils.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
]
NN_TARGET_WAVELENGTHS_METRES = numpy.array([
    3e-6, 2e-6, 1e-6, 3e-6, 1e-6, 2e-6
])


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_predictors_dict_to_numpy(self):
        """Ensures correct output from predictors_dict_to_numpy."""

        (
            this_predictor_matrix,
            this_predictor_name_matrix,
            this_height_matrix_m_agl
        ) = neural_net.predictors_dict_to_numpy(EXAMPLE_DICT)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, NN_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_predictor_name_matrix, NN_PREDICTOR_NAME_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_height_matrix_m_agl, NN_HEIGHT_MATRIX_M_AGL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_predictors_numpy_to_dict(self):
        """Ensures correct output from predictors_numpy_to_dict."""

        this_example_dict = neural_net.predictors_numpy_to_dict(
            predictor_matrix=NN_PREDICTOR_MATRIX,
            example_dict=EXAMPLE_DICT
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_utils.VECTOR_PREDICTOR_VALS_KEY],
            VECTOR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_utils.SCALAR_PREDICTOR_VALS_KEY],
            SCALAR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))

    def test_targets_dict_to_numpy(self):
        """Ensures correct output from targets_dict_to_numpy."""

        these_matrices = neural_net.targets_dict_to_numpy(EXAMPLE_DICT)
        self.assertTrue(len(these_matrices) == len(NN_TARGET_MATRICES))

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], NN_TARGET_MATRICES[i], atol=TOLERANCE
            ))

    def test_targets_dict_to_numpy_no_scalars(self):
        """Ensures correct output from targets_dict_to_numpy.

        In this case, there are no scalar target variables.
        """

        this_scalar_target_matrix = (
            EXAMPLE_DICT[example_utils.SCALAR_TARGET_VALS_KEY] + 0.
        )
        this_scalar_target_matrix = numpy.full(
            (this_scalar_target_matrix.shape[0], 0), 0.
        )

        this_example_dict = copy.deepcopy(EXAMPLE_DICT)
        this_example_dict[example_utils.SCALAR_TARGET_VALS_KEY] = (
            this_scalar_target_matrix
        )

        these_matrices = neural_net.targets_dict_to_numpy(this_example_dict)
        self.assertTrue(
            len(these_matrices) == len(NN_TARGET_MATRICES_NO_SCALARS)
        )

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], NN_TARGET_MATRICES_NO_SCALARS[i],
                atol=TOLERANCE
            ))

    def test_targets_numpy_to_dict(self):
        """Ensures correct output from targets_numpy_to_dict."""

        this_example_dict = neural_net.targets_numpy_to_dict(
            NN_TARGET_MATRICES
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_utils.VECTOR_TARGET_VALS_KEY],
            VECTOR_TARGET_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_utils.SCALAR_TARGET_VALS_KEY],
            SCALAR_TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_targets_numpy_to_dict_no_scalars(self):
        """Ensures correct output from targets_numpy_to_dict.

        In this case, there are no scalar target variables.
        """

        this_example_dict = neural_net.targets_numpy_to_dict(
            NN_TARGET_MATRICES_NO_SCALARS
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_utils.VECTOR_TARGET_VALS_KEY],
            VECTOR_TARGET_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(
            this_example_dict[example_utils.SCALAR_TARGET_VALS_KEY].size == 0
        )

    def test_neuron_indices_to_target_var(self):
        """Ensures correct output from neuron_indices_to_target_var."""

        num_tests = len(NEURON_INDICES_LIST)

        for i in range(num_tests):
            this_target_name, this_height_m_agl, this_wavelength_metres = (
                neural_net.neuron_indices_to_target_var(
                    neuron_indices=NEURON_INDICES_LIST[i],
                    example_dict=copy.deepcopy(EXAMPLE_DICT)
                )
            )

            self.assertTrue(this_target_name == NN_TARGET_NAMES[i])
            self.assertTrue(this_height_m_agl == NN_HEIGHTS_M_AGL[i])
            self.assertTrue(numpy.isclose(
                this_wavelength_metres, NN_TARGET_WAVELENGTHS_METRES[i]
            ))

    def test_target_var_to_neuron_indices(self):
        """Ensures correct output from target_var_to_neuron_indices."""

        num_tests = len(NEURON_INDICES_LIST)

        for i in range(num_tests):
            these_neuron_indices = neural_net.target_var_to_neuron_indices(
                example_dict=copy.deepcopy(EXAMPLE_DICT),
                target_name=NN_TARGET_NAMES[i],
                wavelength_metres=NN_TARGET_WAVELENGTHS_METRES[i],
                height_m_agl=NN_HEIGHTS_M_AGL[i]
            )

            self.assertTrue(numpy.array_equal(
                these_neuron_indices, NEURON_INDICES_LIST[i]
            ))


if __name__ == '__main__':
    unittest.main()
