"""Unit tests for permutation.py."""

import unittest
import numpy
from ml4rt.machine_learning import permutation

TOLERANCE = 1e-6

# The following constants are used to _permute_values and _depermute_values.
PREDICTOR_MATRIX_2D = numpy.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24]
], dtype=float)

CHANNEL_INDEX_2D = 2

PREDICTOR_MATRIX_3D = numpy.stack((
    PREDICTOR_MATRIX_2D, PREDICTOR_MATRIX_2D * 2, PREDICTOR_MATRIX_2D * 3
), axis=-2)

CHANNEL_INDEX_3D = 1
HEIGHT_INDEX_3D = 0

# The following constants are used to test _predictor_indices_to_metadata.
PREDICTOR_NAME_MATRIX = numpy.array([
    ['foo', 'bar', 'moo', 'hal'],
    ['foo', 'bar', 'moo', 'hal'],
    ['foo', 'bar', 'moo', 'hal']
])

HEIGHT_MATRIX_M_AGL = numpy.array([
    [10, 10, 10, 10],
    [100, 100, 100, 100],
    [2000, 2000, 2000, 2000]
], dtype=int)

FIRST_RESULT_DICT = {
    permutation.PERMUTED_CHANNELS_KEY: numpy.array([0, 3, 2, 1], dtype=int),
    permutation.PERMUTED_HEIGHTS_KEY: None
}

FIRST_PREDICTOR_NAMES = ['foo', 'hal', 'moo', 'bar']
FIRST_HEIGHTS_M_AGL = numpy.array([numpy.nan, numpy.nan, numpy.nan, numpy.nan])

SECOND_RESULT_DICT = {
    permutation.PERMUTED_CHANNELS_KEY:
        numpy.array([0, 2, 2, 1, 0, 2, 3, 1, 1, 0, 3, 3], dtype=int),
    permutation.PERMUTED_HEIGHTS_KEY:
        numpy.array([2, 1, 2, 2, 0, 0, 2, 1, 0, 1, 0, 1], dtype=int)
}

SECOND_PREDICTOR_NAMES = [
    'foo', 'moo', 'moo', 'bar', 'foo', 'moo', 'hal', 'bar', 'bar', 'foo',
    'hal', 'hal'
]
SECOND_HEIGHTS_M_AGL = numpy.array(
    [2000, 100, 2000, 2000, 10, 10, 2000, 100, 10, 100, 10, 100], dtype=int
)

# The following constants are used to test make_cost_function.
FIRST_TARGET_MATRIX = numpy.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24]
], dtype=float)

FIRST_TARGET_MATRIX = numpy.stack((
    FIRST_TARGET_MATRIX, FIRST_TARGET_MATRIX + 24, FIRST_TARGET_MATRIX + 48
), axis=-2)

SECOND_TARGET_MATRIX = numpy.array([
    [10, 20],
    [30, 40],
    [50, 60],
    [60, 50],
    [40, 30],
    [20, 10]
], dtype=float)

# ACTUAL_NET_FLUXES_W_M02 = numpy.array([-10, -10, -10, 10, 10, 10], dtype=float)
# PREDICTED_NET_FLUXES_W_M02 = numpy.array([-9, -9, -9, 9, 9, 9], dtype=float)

FIRST_PREDICTION_MATRIX = 0.75 * FIRST_TARGET_MATRIX
SECOND_PREDICTION_MATRIX = 0.9 * SECOND_TARGET_MATRIX
TARGET_MATRICES = [FIRST_TARGET_MATRIX, SECOND_TARGET_MATRIX]
PREDICTION_MATRICES = [FIRST_PREDICTION_MATRIX, SECOND_PREDICTION_MATRIX]

ACTUAL_HEATING_RATES_K_DAY01 = numpy.linspace(1, 72, num=72, dtype=float)
DWMSE_FOR_HEATING_RATES = numpy.mean(
    ACTUAL_HEATING_RATES_K_DAY01 * (0.25 * ACTUAL_HEATING_RATES_K_DAY01) ** 2
)
MSE_FOR_FLUXES_SANS_NET = numpy.mean((0.1 * SECOND_TARGET_MATRIX) ** 2)
MSE_FOR_FLUXES_WITH_NET = numpy.average(
    a=numpy.array([MSE_FOR_FLUXES_SANS_NET, 1]),
    weights=numpy.array([2, 1])
)

DOWN_FLUX_INDICES = numpy.array([0], dtype=int)
UP_FLUX_INDICES = numpy.array([1], dtype=int)

FIRST_HEATING_RATE_WEIGHT = 1.
FIRST_FLUX_WEIGHT = 0.
FIRST_INCLUDE_NET_FLUX_FLAG = True
FIRST_COST = DWMSE_FOR_HEATING_RATES + 0.

SECOND_HEATING_RATE_WEIGHT = 0.
SECOND_FLUX_WEIGHT = 1.
SECOND_INCLUDE_NET_FLUX_FLAG = True
SECOND_COST = MSE_FOR_FLUXES_WITH_NET + 0.

THIRD_HEATING_RATE_WEIGHT = 0.
THIRD_FLUX_WEIGHT = 1.
THIRD_INCLUDE_NET_FLUX_FLAG = False
THIRD_COST = MSE_FOR_FLUXES_SANS_NET + 0.

FOURTH_HEATING_RATE_WEIGHT = 2.
FOURTH_FLUX_WEIGHT = 5.
FOURTH_INCLUDE_NET_FLUX_FLAG = True
FOURTH_COST = 2 * DWMSE_FOR_HEATING_RATES + 5 * MSE_FOR_FLUXES_WITH_NET


class PermutationTests(unittest.TestCase):
    """Each method is a unit test for permutation.py."""

    def test_permute_values_2d(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix is 2-D.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_2D + 0.,
                channel_index=CHANNEL_INDEX_2D, height_index=None,
                permuted_value_matrix=None
            )
        )

        num_channels = new_predictor_matrix.shape[-1]
        indices_to_compare = numpy.arange(num_channels) != CHANNEL_INDEX_2D

        self.assertTrue(numpy.allclose(
            new_predictor_matrix[..., indices_to_compare],
            PREDICTOR_MATRIX_2D[..., indices_to_compare], atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_2D + 0.,
            channel_index=CHANNEL_INDEX_2D, height_index=None,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_permute_values_3d(self):
        """Ensures correct output from _permute_values.

        In this case, the predictor matrix is 3-D.
        """

        new_predictor_matrix, permuted_value_matrix = (
            permutation._permute_values(
                predictor_matrix=PREDICTOR_MATRIX_3D + 0.,
                channel_index=CHANNEL_INDEX_3D, height_index=HEIGHT_INDEX_3D,
                permuted_value_matrix=None
            )
        )

        num_heights = new_predictor_matrix.shape[1]
        num_channels = new_predictor_matrix.shape[2]
        height_indices = numpy.arange(num_heights) != HEIGHT_INDEX_3D
        channel_indices = numpy.arange(num_channels) != CHANNEL_INDEX_3D

        self.assertTrue(numpy.allclose(
            new_predictor_matrix[..., channel_indices][..., height_indices, :],
            PREDICTOR_MATRIX_3D[..., channel_indices][..., height_indices, :],
            atol=TOLERANCE
        ))

        newnew_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_3D + 0.,
            channel_index=CHANNEL_INDEX_3D, height_index=HEIGHT_INDEX_3D,
            permuted_value_matrix=permuted_value_matrix
        )[0]

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, newnew_predictor_matrix, atol=TOLERANCE
        ))

    def test_depermute_values_2d(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix is 2-D.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_2D + 0.,
            channel_index=CHANNEL_INDEX_2D, height_index=None,
            permuted_value_matrix=None
        )[0]

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_2D, atol=TOLERANCE
        ))

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_2D,
            channel_index=CHANNEL_INDEX_2D, height_index=None
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_2D, atol=TOLERANCE
        ))

    def test_depermute_values_3d(self):
        """Ensures correct output from _depermute_values.

        In this case, the predictor matrix is 3-D.
        """

        new_predictor_matrix = permutation._permute_values(
            predictor_matrix=PREDICTOR_MATRIX_3D + 0.,
            channel_index=CHANNEL_INDEX_3D, height_index=HEIGHT_INDEX_3D,
            permuted_value_matrix=None
        )[0]

        self.assertFalse(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_3D, atol=TOLERANCE
        ))

        new_predictor_matrix = permutation._depermute_values(
            predictor_matrix=new_predictor_matrix,
            clean_predictor_matrix=PREDICTOR_MATRIX_3D,
            channel_index=CHANNEL_INDEX_3D, height_index=HEIGHT_INDEX_3D
        )

        self.assertTrue(numpy.allclose(
            new_predictor_matrix, PREDICTOR_MATRIX_3D, atol=TOLERANCE
        ))

    def test_predictor_indices_to_metadata_first(self):
        """Ensures correct output from _predictor_indices_to_metadata.

        In this case, using first set of indices.
        """

        these_predictor_names, these_heights_m_agl = (
            permutation._predictor_indices_to_metadata(
                all_predictor_name_matrix=PREDICTOR_NAME_MATRIX,
                all_height_matrix_m_agl=HEIGHT_MATRIX_M_AGL,
                one_step_result_dict=FIRST_RESULT_DICT
            )
        )

        self.assertTrue(these_predictor_names == FIRST_PREDICTOR_NAMES)
        self.assertTrue(numpy.allclose(
            these_heights_m_agl, FIRST_HEIGHTS_M_AGL,
            atol=TOLERANCE, equal_nan=True
        ))

    def test_predictor_indices_to_metadata_second(self):
        """Ensures correct output from _predictor_indices_to_metadata.

        In this case, using second set of indices.
        """

        these_predictor_names, these_heights_m_agl = (
            permutation._predictor_indices_to_metadata(
                all_predictor_name_matrix=PREDICTOR_NAME_MATRIX,
                all_height_matrix_m_agl=HEIGHT_MATRIX_M_AGL,
                one_step_result_dict=SECOND_RESULT_DICT
            )
        )

        self.assertTrue(these_predictor_names == SECOND_PREDICTOR_NAMES)
        self.assertTrue(numpy.allclose(
            these_heights_m_agl, SECOND_HEIGHTS_M_AGL, atol=TOLERANCE
        ))

    def test_make_cost_function_first(self):
        """Ensures correct output from make_cost_function.

        In this case, using first set of inputs.
        """

        this_cost_function = permutation.make_cost_function(
            heating_rate_weight=FIRST_HEATING_RATE_WEIGHT,
            flux_weight=FIRST_FLUX_WEIGHT,
            include_net_flux=FIRST_INCLUDE_NET_FLUX_FLAG,
            down_flux_indices=DOWN_FLUX_INDICES,
            up_flux_indices=UP_FLUX_INDICES
        )

        this_cost = this_cost_function(
            target_matrices=TARGET_MATRICES,
            prediction_matrices=PREDICTION_MATRICES
        )

        self.assertTrue(numpy.isclose(this_cost, FIRST_COST, atol=TOLERANCE))

    def test_make_cost_function_second(self):
        """Ensures correct output from make_cost_function.

        In this case, using second set of inputs.
        """

        this_cost_function = permutation.make_cost_function(
            heating_rate_weight=SECOND_HEATING_RATE_WEIGHT,
            flux_weight=SECOND_FLUX_WEIGHT,
            include_net_flux=SECOND_INCLUDE_NET_FLUX_FLAG,
            down_flux_indices=DOWN_FLUX_INDICES,
            up_flux_indices=UP_FLUX_INDICES
        )

        this_cost = this_cost_function(
            target_matrices=TARGET_MATRICES,
            prediction_matrices=PREDICTION_MATRICES
        )

        self.assertTrue(numpy.isclose(this_cost, SECOND_COST, atol=TOLERANCE))

    def test_make_cost_function_third(self):
        """Ensures correct output from make_cost_function.

        In this case, using third set of inputs.
        """

        this_cost_function = permutation.make_cost_function(
            heating_rate_weight=THIRD_HEATING_RATE_WEIGHT,
            flux_weight=THIRD_FLUX_WEIGHT,
            include_net_flux=THIRD_INCLUDE_NET_FLUX_FLAG,
            down_flux_indices=DOWN_FLUX_INDICES,
            up_flux_indices=UP_FLUX_INDICES
        )

        this_cost = this_cost_function(
            target_matrices=TARGET_MATRICES,
            prediction_matrices=PREDICTION_MATRICES
        )

        self.assertTrue(numpy.isclose(this_cost, THIRD_COST, atol=TOLERANCE))

    def test_make_cost_function_fourth(self):
        """Ensures correct output from make_cost_function.

        In this case, using fourth set of inputs.
        """

        this_cost_function = permutation.make_cost_function(
            heating_rate_weight=FOURTH_HEATING_RATE_WEIGHT,
            flux_weight=FOURTH_FLUX_WEIGHT,
            include_net_flux=FOURTH_INCLUDE_NET_FLUX_FLAG,
            down_flux_indices=DOWN_FLUX_INDICES,
            up_flux_indices=UP_FLUX_INDICES
        )

        this_cost = this_cost_function(
            target_matrices=TARGET_MATRICES,
            prediction_matrices=PREDICTION_MATRICES
        )

        self.assertTrue(numpy.isclose(this_cost, FOURTH_COST, atol=TOLERANCE))


if __name__ == '__main__':
    unittest.main()
