"""Unit tests for backwards_optimization.py."""

import copy
import unittest
import numpy
from ml4rt.utils import normalization
from ml4rt.machine_learning import neural_net
from ml4rt.machine_learning import backwards_optimization as bwo

TOLERANCE = 1e-6

PREDICTOR_DIMENSIONS = numpy.array([1, 73, 13], dtype=int)

THIS_OPTION_DICT = {
    neural_net.PREDICTOR_NORM_TYPE_KEY: normalization.Z_SCORE_NORM_STRING
}
METADATA_DICT_Z_SCORE = {
    neural_net.TRAINING_OPTIONS_KEY: copy.deepcopy(THIS_OPTION_DICT)
}
PREDICTOR_MATRIX_Z_SCORE = numpy.full((1, 73, 13), 0.)

THIS_OPTION_DICT = {
    neural_net.PREDICTOR_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    neural_net.PREDICTOR_MIN_NORM_VALUE_KEY: 0.,
    neural_net.PREDICTOR_MAX_NORM_VALUE_KEY: 1.
}
METADATA_DICT_MINMAX_FIRST = {
    neural_net.TRAINING_OPTIONS_KEY: copy.deepcopy(THIS_OPTION_DICT)
}
PREDICTOR_MATRIX_MINMAX_FIRST = numpy.full((1, 73, 13), 0.5)

THIS_OPTION_DICT = {
    neural_net.PREDICTOR_NORM_TYPE_KEY: normalization.MINMAX_NORM_STRING,
    neural_net.PREDICTOR_MIN_NORM_VALUE_KEY: -5.5,
    neural_net.PREDICTOR_MAX_NORM_VALUE_KEY: 10.
}
METADATA_DICT_MINMAX_SECOND = {
    neural_net.TRAINING_OPTIONS_KEY: copy.deepcopy(THIS_OPTION_DICT)
}
PREDICTOR_MATRIX_MINMAX_SECOND = numpy.full((1, 73, 13), 2.25)


class BackwardsOptimizationTests(unittest.TestCase):
    """Each method is a unit test for backwards_optimization.py."""

    def test_create_climo_initializer_z_score(self):
        """Ensures correct output from create_climo_initializer.

        In this case, normalization type is z-score.
        """

        this_init_function = bwo.create_climo_initializer(METADATA_DICT_Z_SCORE)
        this_predictor_matrix = this_init_function(PREDICTOR_DIMENSIONS)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_Z_SCORE, atol=TOLERANCE
        ))

    def test_create_climo_initializer_minmax_first(self):
        """Ensures correct output from create_climo_initializer.

        In this case, normalization type is min-max with first set of params.
        """

        this_init_function = (
            bwo.create_climo_initializer(METADATA_DICT_MINMAX_FIRST)
        )
        this_predictor_matrix = this_init_function(PREDICTOR_DIMENSIONS)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_MINMAX_FIRST,
            atol=TOLERANCE
        ))

    def test_create_climo_initializer_minmax_second(self):
        """Ensures correct output from create_climo_initializer.

        In this case, normalization type is min-max with second set of params.
        """

        this_init_function = (
            bwo.create_climo_initializer(METADATA_DICT_MINMAX_SECOND)
        )
        this_predictor_matrix = this_init_function(PREDICTOR_DIMENSIONS)

        self.assertTrue(numpy.allclose(
            this_predictor_matrix, PREDICTOR_MATRIX_MINMAX_SECOND,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
