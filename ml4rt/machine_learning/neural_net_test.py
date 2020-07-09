"""Unit tests for neural_net.py."""

import copy
import unittest
import numpy
from ml4rt.io import example_io
from ml4rt.machine_learning import neural_net

TOLERANCE = 1e-6

# The following constants are used to test predictors_dict_to_numpy,
# predictors_numpy_to_dict, targets_dict_to_numpy, and targets_numpy_to_dict.
HEIGHTS_M_AGL = numpy.array([100, 500], dtype=float)
VALID_TIMES_UNIX_SEC = numpy.array([0, 300, 600, 1200], dtype=int)
STANDARD_ATMO_FLAGS = numpy.array([0, 1, 2, 3], dtype=int)

SCALAR_PREDICTOR_NAMES = [
    example_io.ZENITH_ANGLE_NAME, example_io.LATITUDE_NAME
]
ZENITH_ANGLES_RADIANS = numpy.array([0, 1, 2, 3], dtype=float)
LATITUDES_DEG_N = numpy.array([40.02, 40.02, 40.02, 40.02])
SCALAR_PREDICTOR_MATRIX = numpy.transpose(numpy.vstack(
    (ZENITH_ANGLES_RADIANS, LATITUDES_DEG_N)
))

VECTOR_PREDICTOR_NAMES = [
    example_io.TEMPERATURE_NAME, example_io.SPECIFIC_HUMIDITY_NAME
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

SCALAR_TARGET_NAMES = [example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME]
SURFACE_DOWN_FLUXES_W_M02 = numpy.array([200, 200, 200, 200], dtype=float)
SCALAR_TARGET_MATRIX = numpy.reshape(
    SURFACE_DOWN_FLUXES_W_M02, (len(SURFACE_DOWN_FLUXES_W_M02), 1)
)

VECTOR_TARGET_NAMES = [
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME
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

VECTOR_TARGET_MATRIX = numpy.stack(
    (DOWNWELLING_FLUX_MATRIX_W_M02, UPWELLING_FLUX_MATRIX_W_M02), axis=-1
)

EXAMPLE_DICT = {
    example_io.SCALAR_PREDICTOR_NAMES_KEY: SCALAR_PREDICTOR_NAMES,
    example_io.SCALAR_PREDICTOR_VALS_KEY: SCALAR_PREDICTOR_MATRIX,
    example_io.VECTOR_PREDICTOR_NAMES_KEY: VECTOR_PREDICTOR_NAMES,
    example_io.VECTOR_PREDICTOR_VALS_KEY: VECTOR_PREDICTOR_MATRIX,
    example_io.SCALAR_TARGET_NAMES_KEY: SCALAR_TARGET_NAMES,
    example_io.SCALAR_TARGET_VALS_KEY: SCALAR_TARGET_MATRIX,
    example_io.VECTOR_TARGET_NAMES_KEY: VECTOR_TARGET_NAMES,
    example_io.VECTOR_TARGET_VALS_KEY: VECTOR_TARGET_MATRIX,
    example_io.HEIGHTS_KEY: HEIGHTS_M_AGL,
    example_io.VALID_TIMES_KEY: VALID_TIMES_UNIX_SEC,
    example_io.STANDARD_ATMO_FLAGS_KEY: STANDARD_ATMO_FLAGS
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
CNN_PREDICTOR_MATRIX = numpy.concatenate(
    (VECTOR_PREDICTOR_MATRIX, THIS_SCALAR_PREDICTOR_MATRIX), axis=-1
)
CNN_PREDICTOR_NAME_MATRIX = numpy.array([
    [example_io.TEMPERATURE_NAME, example_io.SPECIFIC_HUMIDITY_NAME,
     example_io.ZENITH_ANGLE_NAME, example_io.LATITUDE_NAME],
    [example_io.TEMPERATURE_NAME, example_io.SPECIFIC_HUMIDITY_NAME,
     example_io.ZENITH_ANGLE_NAME, example_io.LATITUDE_NAME]
])
CNN_HEIGHT_MATRIX_M_AGL = numpy.array([
    [100, 100, numpy.nan, numpy.nan],
    [500, 500, numpy.nan, numpy.nan]
])

U_NET_PREDICTOR_MATRIX = CNN_PREDICTOR_MATRIX + 0.
U_NET_PREDICTOR_NAME_MATRIX = copy.deepcopy(CNN_PREDICTOR_NAME_MATRIX)
U_NET_HEIGHT_MATRIX_M_AGL = CNN_HEIGHT_MATRIX_M_AGL + 0

CNN_TARGET_MATRICES_DEFAULT_LOSS = [
    VECTOR_TARGET_MATRIX + 0., SCALAR_TARGET_MATRIX + 0.
]
U_NET_TARGET_MATRICES = [VECTOR_TARGET_MATRIX + 0.]

THIS_SCALAR_TARGET_MATRIX = numpy.array([
    [150, 300, 200],
    [150, 500, 200],
    [350, 450, 200],
    [100, 200, 200]
], dtype=float)

CNN_TARGET_MATRICES_CUSTOM_LOSS = [
    VECTOR_TARGET_MATRIX + 0., THIS_SCALAR_TARGET_MATRIX + 0.
]
CNN_TARGET_MATRICES_NO_SCALARS = [VECTOR_TARGET_MATRIX + 0.]

DENSE_NET_PREDICTOR_MATRIX = numpy.array([
    [290, 295, 0.008, 0.009, 0, 40.02],
    [289, 294, 0.007, 0.008, 1, 40.02],
    [288, 293, 0.005, 0.006, 2, 40.02],
    [287, 292.5, 0.0075, 0.01, 3, 40.02]
])
DENSE_NET_PREDICTOR_NAME_MATRIX = numpy.array([
    example_io.TEMPERATURE_NAME, example_io.TEMPERATURE_NAME,
    example_io.SPECIFIC_HUMIDITY_NAME, example_io.SPECIFIC_HUMIDITY_NAME,
    example_io.ZENITH_ANGLE_NAME, example_io.LATITUDE_NAME
])
DENSE_NET_HEIGHT_MATRIX_M_AGL = numpy.array(
    [100, 500, 100, 500, numpy.nan, numpy.nan]
)

DENSE_NET_TARGET_MATRIX = numpy.array([
    [300, 200, 150, 150, 200],
    [500, 300, 200, 150, 200],
    [450, 450, 300, 350, 200],
    [200, 100, 400, 100, 200]
], dtype=float)

DENSE_NET_TARGET_MATRICES = [DENSE_NET_TARGET_MATRIX]

# The following constants are used to test neuron_indices_to_target_var and
# target_var_to_neuron_indices.
CNN_NEURON_INDICES_LIST = [
    numpy.array([0, 0], dtype=int),
    numpy.array([0, 1], dtype=int),
    numpy.array([1, 0], dtype=int),
    numpy.array([1, 1], dtype=int),
    numpy.array([0], dtype=int)
]
CNN_HEIGHTS_M_AGL = [100, 100, 500, 500, None]
CNN_TARGET_NAMES = [
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME,
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME,
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
]

DENSE_NET_NEURON_INDICES_LIST = [
    numpy.array([0], dtype=int),
    numpy.array([1], dtype=int),
    numpy.array([2], dtype=int),
    numpy.array([3], dtype=int),
    numpy.array([4], dtype=int)
]
DENSE_NET_HEIGHTS_M_AGL = [100, 500, 100, 500, None]
DENSE_NET_TARGET_NAMES = [
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_DOWN_FLUX_NAME,
    example_io.SHORTWAVE_UP_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME,
    example_io.SHORTWAVE_SURFACE_DOWN_FLUX_NAME
]

U_NET_NEURON_INDICES_LIST = [
    numpy.array([0, 0], dtype=int),
    numpy.array([0, 1], dtype=int),
    numpy.array([1, 0], dtype=int),
    numpy.array([1, 1], dtype=int)
]
U_NET_HEIGHTS_M_AGL = [100, 100, 500, 500]
U_NET_TARGET_NAMES = [
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME,
    example_io.SHORTWAVE_DOWN_FLUX_NAME, example_io.SHORTWAVE_UP_FLUX_NAME
]


class NeuralNetTests(unittest.TestCase):
    """Each method is a unit test for neural_net.py."""

    def test_predictors_dict_to_numpy_cnn(self):
        """Ensures correct output from predictors_dict_to_numpy.

        In this case, neural-net type is CNN.
        """

        (
            this_predictor_matrix, this_predictor_name_matrix,
            this_height_matrix_m_agl
        ) = neural_net.predictors_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.CNN_TYPE_STRING
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, CNN_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_predictor_name_matrix, CNN_PREDICTOR_NAME_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_height_matrix_m_agl, CNN_HEIGHT_MATRIX_M_AGL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_predictors_dict_to_numpy_dense_net(self):
        """Ensures correct output from predictors_dict_to_numpy.

        In this case, neural-net type is dense.
        """

        (
            this_predictor_matrix, this_predictor_name_matrix,
            this_height_matrix_m_agl
        ) = neural_net.predictors_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.DENSE_NET_TYPE_STRING
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, DENSE_NET_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_predictor_name_matrix, DENSE_NET_PREDICTOR_NAME_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_height_matrix_m_agl, DENSE_NET_HEIGHT_MATRIX_M_AGL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_predictors_dict_to_numpy_u_net(self):
        """Ensures correct output from predictors_dict_to_numpy.

        In this case, neural-net type is U-net.
        """

        (
            this_predictor_matrix, this_predictor_name_matrix,
            this_height_matrix_m_agl
        ) = neural_net.predictors_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.U_NET_TYPE_STRING
        )

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, U_NET_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(
            this_predictor_name_matrix, U_NET_PREDICTOR_NAME_MATRIX
        ))
        self.assertTrue(numpy.allclose(
            this_height_matrix_m_agl, U_NET_HEIGHT_MATRIX_M_AGL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_predictors_numpy_to_dict_cnn(self):
        """Ensures correct output from predictors_numpy_to_dict.

        In this case, neural-net type is CNN.
        """

        this_example_dict = neural_net.predictors_numpy_to_dict(
            predictor_matrix=CNN_PREDICTOR_MATRIX, example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.CNN_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY],
            VECTOR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY],
            SCALAR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))

    def test_predictors_numpy_to_dict_dense_net(self):
        """Ensures correct output from predictors_numpy_to_dict.

        In this case, neural-net type is dense.
        """

        this_example_dict = neural_net.predictors_numpy_to_dict(
            predictor_matrix=DENSE_NET_PREDICTOR_MATRIX,
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.DENSE_NET_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY],
            VECTOR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY],
            SCALAR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))

    def test_predictors_numpy_to_dict_u_net(self):
        """Ensures correct output from predictors_numpy_to_dict.

        In this case, neural-net type is U-net.
        """

        this_example_dict = neural_net.predictors_numpy_to_dict(
            predictor_matrix=U_NET_PREDICTOR_MATRIX,
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.U_NET_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_PREDICTOR_VALS_KEY],
            VECTOR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.SCALAR_PREDICTOR_VALS_KEY],
            SCALAR_PREDICTOR_MATRIX, atol=TOLERANCE
        ))

    def test_targets_dict_to_numpy_cnn_default_loss(self):
        """Ensures correct output from targets_dict_to_numpy.

        In this case, neural-net type is CNN with default loss function.
        """

        these_matrices = neural_net.targets_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.CNN_TYPE_STRING,
            is_loss_constrained_mse=False
        )

        self.assertTrue(
            len(these_matrices) == len(CNN_TARGET_MATRICES_DEFAULT_LOSS)
        )

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], CNN_TARGET_MATRICES_DEFAULT_LOSS[i],
                atol=TOLERANCE
            ))

    def test_targets_dict_to_numpy_cnn_custom_loss(self):
        """Ensures correct output from targets_dict_to_numpy.

        In this case, neural-net type is CNN with custom loss function.
        """

        these_matrices = neural_net.targets_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.CNN_TYPE_STRING,
            is_loss_constrained_mse=True
        )

        self.assertTrue(
            len(these_matrices) == len(CNN_TARGET_MATRICES_CUSTOM_LOSS)
        )

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], CNN_TARGET_MATRICES_CUSTOM_LOSS[i],
                atol=TOLERANCE
            ))

    def test_targets_dict_to_numpy_cnn_no_scalars(self):
        """Ensures correct output from targets_dict_to_numpy.

        In this case, neural-net type is CNN with no scalar outputs.
        """

        this_scalar_target_matrix = (
            EXAMPLE_DICT[example_io.SCALAR_TARGET_VALS_KEY] + 0.
        )
        this_scalar_target_matrix = numpy.full(
            (this_scalar_target_matrix.shape[0], 0), 0.
        )

        this_example_dict = copy.deepcopy(EXAMPLE_DICT)
        this_example_dict[example_io.SCALAR_TARGET_VALS_KEY] = (
            this_scalar_target_matrix
        )

        these_matrices = neural_net.targets_dict_to_numpy(
            example_dict=this_example_dict,
            net_type_string=neural_net.CNN_TYPE_STRING,
            is_loss_constrained_mse=False
        )

        self.assertTrue(
            len(these_matrices) == len(CNN_TARGET_MATRICES_NO_SCALARS)
        )

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], CNN_TARGET_MATRICES_NO_SCALARS[i],
                atol=TOLERANCE
            ))

    def test_targets_dict_to_numpy_dense_net(self):
        """Ensures correct output from targets_dict_to_numpy.

        In this case, neural-net type is dense.
        """

        these_matrices = neural_net.targets_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.DENSE_NET_TYPE_STRING
        )

        self.assertTrue(len(these_matrices) == len(DENSE_NET_TARGET_MATRICES))

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], DENSE_NET_TARGET_MATRICES[i], atol=TOLERANCE
            ))

    def test_targets_dict_to_numpy_u_net(self):
        """Ensures correct output from targets_dict_to_numpy.

        In this case, neural-net type is U-net.
        """

        these_matrices = neural_net.targets_dict_to_numpy(
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.U_NET_TYPE_STRING
        )

        self.assertTrue(len(these_matrices) == len(U_NET_TARGET_MATRICES))

        for i in range(len(these_matrices)):
            self.assertTrue(numpy.allclose(
                these_matrices[i], U_NET_TARGET_MATRICES[i], atol=TOLERANCE
            ))

    def test_targets_numpy_to_dict_cnn_default(self):
        """Ensures correct output from targets_numpy_to_dict.

        In this case, neural-net type is default CNN.
        """

        this_example_dict = neural_net.targets_numpy_to_dict(
            target_matrices=CNN_TARGET_MATRICES_DEFAULT_LOSS,
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.CNN_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_TARGET_VALS_KEY],
            VECTOR_TARGET_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.SCALAR_TARGET_VALS_KEY],
            SCALAR_TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_targets_numpy_to_dict_cnn_no_scalars(self):
        """Ensures correct output from targets_numpy_to_dict.

        In this case, neural-net type is CNN with no scalar outputs.
        """

        this_example_dict = neural_net.targets_numpy_to_dict(
            target_matrices=CNN_TARGET_MATRICES_NO_SCALARS,
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.CNN_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_TARGET_VALS_KEY],
            VECTOR_TARGET_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(
            this_example_dict[example_io.SCALAR_TARGET_VALS_KEY].size == 0
        )

    def test_targets_numpy_to_dict_dense_net(self):
        """Ensures correct output from targets_numpy_to_dict.

        In this case, neural-net type is dense.
        """

        this_example_dict = neural_net.targets_numpy_to_dict(
            target_matrices=DENSE_NET_TARGET_MATRICES,
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.DENSE_NET_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_TARGET_VALS_KEY],
            VECTOR_TARGET_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.SCALAR_TARGET_VALS_KEY],
            SCALAR_TARGET_MATRIX, atol=TOLERANCE
        ))

    def test_targets_numpy_to_dict_u_net(self):
        """Ensures correct output from targets_numpy_to_dict.

        In this case, neural-net type is U-net.
        """

        this_example_dict = neural_net.targets_numpy_to_dict(
            target_matrices=U_NET_TARGET_MATRICES,
            example_dict=EXAMPLE_DICT,
            net_type_string=neural_net.U_NET_TYPE_STRING
        )
        self.assertTrue(numpy.allclose(
            this_example_dict[example_io.VECTOR_TARGET_VALS_KEY],
            VECTOR_TARGET_MATRIX, atol=TOLERANCE
        ))
        self.assertTrue(
            this_example_dict[example_io.SCALAR_TARGET_VALS_KEY].size == 0
        )

    def test_neuron_indices_to_target_var_cnn(self):
        """Ensures correct output from neuron_indices_to_target_var.

        In this case, neural-net type is CNN.
        """

        num_tests = len(CNN_NEURON_INDICES_LIST)

        for i in range(num_tests):
            this_target_name, this_height_m_agl = (
                neural_net.neuron_indices_to_target_var(
                    neuron_indices=CNN_NEURON_INDICES_LIST[i],
                    example_dict=copy.deepcopy(EXAMPLE_DICT),
                    net_type_string=neural_net.CNN_TYPE_STRING
                )
            )

            self.assertTrue(this_target_name == CNN_TARGET_NAMES[i])
            self.assertTrue(this_height_m_agl == CNN_HEIGHTS_M_AGL[i])

    def test_neuron_indices_to_target_var_dense_net(self):
        """Ensures correct output from neuron_indices_to_target_var.

        In this case, neural-net type is dense.
        """

        num_tests = len(DENSE_NET_NEURON_INDICES_LIST)

        for i in range(num_tests):
            this_target_name, this_height_m_agl = (
                neural_net.neuron_indices_to_target_var(
                    neuron_indices=DENSE_NET_NEURON_INDICES_LIST[i],
                    example_dict=copy.deepcopy(EXAMPLE_DICT),
                    net_type_string=neural_net.DENSE_NET_TYPE_STRING
                )
            )

            self.assertTrue(this_target_name == DENSE_NET_TARGET_NAMES[i])
            self.assertTrue(this_height_m_agl == DENSE_NET_HEIGHTS_M_AGL[i])

    def test_neuron_indices_to_target_var_u_net(self):
        """Ensures correct output from neuron_indices_to_target_var.

        In this case, neural-net type is U-net.
        """

        num_tests = len(U_NET_NEURON_INDICES_LIST)

        for i in range(num_tests):
            this_target_name, this_height_m_agl = (
                neural_net.neuron_indices_to_target_var(
                    neuron_indices=U_NET_NEURON_INDICES_LIST[i],
                    example_dict=copy.deepcopy(EXAMPLE_DICT),
                    net_type_string=neural_net.U_NET_TYPE_STRING
                )
            )

            self.assertTrue(this_target_name == U_NET_TARGET_NAMES[i])
            self.assertTrue(this_height_m_agl == U_NET_HEIGHTS_M_AGL[i])

    def test_target_var_to_neuron_indices_cnn(self):
        """Ensures correct output from target_var_to_neuron_indices.

        In this case, neural-net type is CNN.
        """

        num_tests = len(CNN_NEURON_INDICES_LIST)

        for i in range(num_tests):
            these_neuron_indices = neural_net.target_var_to_neuron_indices(
                example_dict=copy.deepcopy(EXAMPLE_DICT),
                net_type_string=neural_net.CNN_TYPE_STRING,
                target_name=CNN_TARGET_NAMES[i],
                height_m_agl=CNN_HEIGHTS_M_AGL[i]
            )

            self.assertTrue(numpy.array_equal(
                these_neuron_indices, CNN_NEURON_INDICES_LIST[i]
            ))

    def test_target_var_to_neuron_indices_dense_net(self):
        """Ensures correct output from target_var_to_neuron_indices.

        In this case, neural-net type is dense.
        """

        num_tests = len(DENSE_NET_NEURON_INDICES_LIST)

        for i in range(num_tests):
            these_neuron_indices = neural_net.target_var_to_neuron_indices(
                example_dict=copy.deepcopy(EXAMPLE_DICT),
                net_type_string=neural_net.DENSE_NET_TYPE_STRING,
                target_name=DENSE_NET_TARGET_NAMES[i],
                height_m_agl=DENSE_NET_HEIGHTS_M_AGL[i]
            )

            self.assertTrue(numpy.array_equal(
                these_neuron_indices, DENSE_NET_NEURON_INDICES_LIST[i]
            ))

    def test_target_var_to_neuron_indices_u_net(self):
        """Ensures correct output from target_var_to_neuron_indices.

        In this case, neural-net type is U-net.
        """

        num_tests = len(U_NET_NEURON_INDICES_LIST)

        for i in range(num_tests):
            these_neuron_indices = neural_net.target_var_to_neuron_indices(
                example_dict=copy.deepcopy(EXAMPLE_DICT),
                net_type_string=neural_net.U_NET_TYPE_STRING,
                target_name=U_NET_TARGET_NAMES[i],
                height_m_agl=U_NET_HEIGHTS_M_AGL[i]
            )

            self.assertTrue(numpy.array_equal(
                these_neuron_indices, U_NET_NEURON_INDICES_LIST[i]
            ))


if __name__ == '__main__':
    unittest.main()
