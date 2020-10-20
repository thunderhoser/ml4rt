"""Unit tests for evaluation.py."""

import unittest
import numpy
from ml4rt.utils import evaluation

TOLERANCE = 1e-6

# The following constants are used to test _get_mse_one_scalar,
# _get_mse_ss_one_scalar, _get_mae_one_scalar, _get_mae_ss_one_scalar,
# _get_bias_one_scalar, and _get_correlation_one_scalar.
SIMPLE_TARGET_VALUES = numpy.array([0, 10, 50, 100, 1000], dtype=float)
SIMPLE_PREDICTED_VALUES = numpy.array([10, 10, 25, 150, 700], dtype=float)
SIMPLE_MEAN_TRAINING_VALUE = 500.

SIMPLE_MSE = 93225. / 5
SIMPLE_MSE_CLIMO = 1102600. / 5
SIMPLE_MSE_SKILL_SCORE = (SIMPLE_MSE_CLIMO - SIMPLE_MSE) / SIMPLE_MSE_CLIMO

SIMPLE_MAE = 385. / 5
SIMPLE_MAE_CLIMO = 2340. / 5
SIMPLE_MAE_SKILL_SCORE = (SIMPLE_MAE_CLIMO - SIMPLE_MAE) / SIMPLE_MAE_CLIMO

SIMPLE_BIAS = -265. / 5
SIMPLE_CORRELATION = numpy.corrcoef(
    SIMPLE_TARGET_VALUES, SIMPLE_PREDICTED_VALUES
)[0, 1]

MEAN_TARGET_VALUE = 232.
MEAN_PREDICTED_VALUE = 179.
STDEV_TARGET_VALUE = numpy.std(SIMPLE_TARGET_VALUES, ddof=1)
STDEV_PREDICTED_VALUE = numpy.std(SIMPLE_PREDICTED_VALUES, ddof=1)

VARIANCE_BIAS = (
    (STDEV_PREDICTED_VALUE / MEAN_PREDICTED_VALUE) *
    (STDEV_TARGET_VALUE / MEAN_TARGET_VALUE) ** -1
)
MEAN_BIAS = MEAN_PREDICTED_VALUE / MEAN_TARGET_VALUE

SIMPLE_KGE = 1. - numpy.sqrt(
    (SIMPLE_CORRELATION - 1.) ** 2 +
    (VARIANCE_BIAS - 1.) ** 2 +
    (MEAN_BIAS - 1.) ** 2
)

# The following constants are used to test _get_prmse_one_variable.
TARGET_MATRIX = numpy.array([
    [200, 175, 150, 125, 100],
    [100, 200, 300, 200, 100],
    [40, 30, 20, 10, 0]
], dtype=float)

PREDICTION_MATRIX = numpy.array([
    [200, 200, 200, 100, 100],
    [100, 250, 350, 200, 110],
    [35, 30, 20, 15, 0]
], dtype=float)

PRMSE_FIRST_EXAMPLE = numpy.sqrt(3750. / 5)
PRMSE_SECOND_EXAMPLE = numpy.sqrt(5100. / 5)
PRMSE_THIRD_EXAMPLE = numpy.sqrt(50. / 5)
PRMSE = numpy.mean(numpy.array(
    [PRMSE_FIRST_EXAMPLE, PRMSE_SECOND_EXAMPLE, PRMSE_THIRD_EXAMPLE]
))

# The following constants are used to test _get_rel_curve_one_scalar.
NUM_BINS = 5
MIN_BIN_EDGE = 0.
MAX_BIN_EDGE = 10.

TARGET_VALUES_NO_EMPTY_BINS = numpy.array(
    [1, 2, 5, 1, 9, 1, 4, 4, 6, 1, 5, 6, 3, 4, 1, 7, 3, 8, 9, 8], dtype=float
)
PREDICTED_VALUES_NO_EMPTY_BINS = numpy.array(
    [0, 5, 5, 5, 12, 4, 4, 1, 6, 3, 4, 10, 3, 6, 5, 12, 6, 3, 9, 4], dtype=float
)
X_COORDS_NO_EMPTY_BINS = numpy.array([0.5, 3, 4.5, 6, 10.75])
Y_COORDS_NO_EMPTY_BINS = numpy.array([2.5, 4, 3.375, 4.333333333, 7.75])
COUNTS_NO_EMPTY_BINS = numpy.array([2, 3, 8, 3, 4], dtype=int)

TARGET_VALUES_ONE_EMPTY_BIN = numpy.array(
    [2, 5, 1, 9, 1, 4, 6, 1, 5, 6, 3, 4, 1, 7, 3, 8, 9, 8], dtype=float
)
PREDICTED_VALUES_ONE_EMPTY_BIN = numpy.array(
    [5, 5, 5, 12, 4, 4, 6, 3, 4, 10, 3, 6, 5, 12, 6, 3, 9, 4], dtype=float
)

X_COORDS_ONE_EMPTY_BIN = numpy.array([numpy.nan, 3, 4.5, 6, 10.75])
Y_COORDS_ONE_EMPTY_BIN = numpy.array([numpy.nan, 4, 3.375, 4.333333333, 7.75])
COUNTS_ONE_EMPTY_BIN = numpy.array([0, 3, 8, 3, 4], dtype=int)


class EvaluationTests(unittest.TestCase):
    """Each method is a unit test for evaluation.py."""

    def test_get_mse_one_scalar(self):
        """Ensures correct output from _get_mse_one_scalar."""

        this_mse = evaluation._get_mse_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES
        )

        self.assertTrue(numpy.isclose(this_mse, SIMPLE_MSE, atol=TOLERANCE))

    def test_get_mse_ss_one_scalar(self):
        """Ensures correct output from _get_mse_ss_one_scalar."""

        this_mse_skill_score = evaluation._get_mse_ss_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES,
            mean_training_target_value=SIMPLE_MEAN_TRAINING_VALUE
        )

        self.assertTrue(numpy.isclose(
            this_mse_skill_score, SIMPLE_MSE_SKILL_SCORE, atol=TOLERANCE
        ))

    def test_get_mae_one_scalar(self):
        """Ensures correct output from _get_mae_one_scalar."""

        this_mae = evaluation._get_mae_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES
        )

        self.assertTrue(numpy.isclose(this_mae, SIMPLE_MAE, atol=TOLERANCE))

    def test_get_mae_ss_one_scalar(self):
        """Ensures correct output from _get_mae_ss_one_scalar."""

        this_mae_skill_score = evaluation._get_mae_ss_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES,
            mean_training_target_value=SIMPLE_MEAN_TRAINING_VALUE
        )

        self.assertTrue(numpy.isclose(
            this_mae_skill_score, SIMPLE_MAE_SKILL_SCORE, atol=TOLERANCE
        ))

    def test_get_bias_one_scalar(self):
        """Ensures correct output from _get_bias_one_scalar."""

        this_bias = evaluation._get_bias_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES
        )

        self.assertTrue(numpy.isclose(this_bias, SIMPLE_BIAS, atol=TOLERANCE))

    def test_get_correlation_one_scalar(self):
        """Ensures correct output from _get_correlation_one_scalar."""

        this_correlation = evaluation._get_correlation_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES
        )

        self.assertTrue(numpy.isclose(
            this_correlation, SIMPLE_CORRELATION, atol=TOLERANCE
        ))

    def test_get_kge_one_scalar(self):
        """Ensures correct output from _get_kge_one_scalar."""

        this_kge = evaluation._get_kge_one_scalar(
            target_values=SIMPLE_TARGET_VALUES,
            predicted_values=SIMPLE_PREDICTED_VALUES
        )

        self.assertTrue(numpy.isclose(this_kge, SIMPLE_KGE, atol=TOLERANCE))

    def test_get_prmse_one_variable(self):
        """Ensures correct output from _get_prmse_one_variable."""

        this_prmse = evaluation._get_prmse_one_variable(
            target_matrix=TARGET_MATRIX, prediction_matrix=PREDICTION_MATRIX
        )
        self.assertTrue(numpy.isclose(this_prmse, PRMSE, atol=TOLERANCE))

    def test_get_rel_curve_no_empty_bins(self):
        """Ensures correct output from _get_rel_curve_one_scalar.

        In this case there should be no empty bins.
        """

        these_x_coords, these_y_coords, these_counts = (
            evaluation._get_rel_curve_one_scalar(
                target_values=TARGET_VALUES_NO_EMPTY_BINS,
                predicted_values=PREDICTED_VALUES_NO_EMPTY_BINS,
                num_bins=NUM_BINS, min_bin_edge=MIN_BIN_EDGE,
                max_bin_edge=MAX_BIN_EDGE
            )
        )

        self.assertTrue(numpy.allclose(
            these_x_coords, X_COORDS_NO_EMPTY_BINS, atol=TOLERANCE
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords, Y_COORDS_NO_EMPTY_BINS, atol=TOLERANCE
        ))
        self.assertTrue(numpy.array_equal(these_counts, COUNTS_NO_EMPTY_BINS))

    def test_get_rel_curve_one_empty_bin(self):
        """Ensures correct output from _get_rel_curve_one_scalar.

        In this case there should one empty bin.
        """

        these_x_coords, these_y_coords, these_counts = (
            evaluation._get_rel_curve_one_scalar(
                target_values=TARGET_VALUES_ONE_EMPTY_BIN,
                predicted_values=PREDICTED_VALUES_ONE_EMPTY_BIN,
                num_bins=NUM_BINS, min_bin_edge=MIN_BIN_EDGE,
                max_bin_edge=MAX_BIN_EDGE
            )
        )

        self.assertTrue(numpy.allclose(
            these_x_coords, X_COORDS_ONE_EMPTY_BIN, atol=TOLERANCE,
            equal_nan=True
        ))
        self.assertTrue(numpy.allclose(
            these_y_coords, Y_COORDS_ONE_EMPTY_BIN, atol=TOLERANCE,
            equal_nan=True
        ))
        self.assertTrue(numpy.array_equal(these_counts, COUNTS_ONE_EMPTY_BIN))


if __name__ == '__main__':
    unittest.main()
