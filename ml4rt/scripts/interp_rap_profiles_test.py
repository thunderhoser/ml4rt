"""Unit tests for interp_rap_profiles.py."""

import unittest
import numpy
from ml4rt.scripts import interp_rap_profiles

TOLERANCE = 1e-6

# The following constants are used to test _find_jumps.
ORIG_DATA_MATRIX = numpy.array([
    [0, 1, 1],
    [1, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 2],
    [0, 0, 0],
    [0, 0, 2],
    [0, 1, 3],
    [0, 0, 0]
])
ORIG_DATA_MATRIX = numpy.transpose(ORIG_DATA_MATRIX)

ORIG_HEIGHTS_METRES = numpy.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float
)
FIRST_NEW_HEIGHTS_METRES = ORIG_HEIGHTS_METRES + 0.

FIRST_JUMP_FLAG_MATRIX = numpy.array([
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1]
], dtype=bool)

FIRST_JUMP_FLAG_MATRIX = numpy.transpose(FIRST_JUMP_FLAG_MATRIX)

SECOND_NEW_HEIGHTS_METRES = numpy.array([
    0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 9.5, 9.75, 10, 11
])

SECOND_JUMP_FLAG_MATRIX = numpy.array([
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1],
    [0, 1, 1]
], dtype=bool)

SECOND_JUMP_FLAG_MATRIX = numpy.transpose(SECOND_JUMP_FLAG_MATRIX)

# The following constants are used to test _conserve_mass_one_variable.
ORIG_CONC_MATRIX_KG_KG01 = numpy.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21],
    [22, 23, 24],
    [25, 26, 27],
    [28, 29, 30]
], dtype=float)

ORIG_CONC_MATRIX_KG_KG01 = numpy.transpose(ORIG_CONC_MATRIX_KG_KG01)

ORIG_AIR_DENS_MATRIX_KG_M03 = numpy.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0.5, 0.5, 0.5]
])

ORIG_AIR_DENS_MATRIX_KG_M03 = numpy.transpose(ORIG_AIR_DENS_MATRIX_KG_M03)

NEW_HEIGHTS_METRES = numpy.array([
    0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 9.5, 9.75, 10, 11
])

NEW_CONC_MATRIX_KG_KG01 = numpy.array([
    [0.3, 0.8, 1.5],
    [0.4, 1.0, 1.8],
    [0.5, 1.3, 2.1],
    [0.7, 1.6, 2.5],
    [1.0, 2.0, 3.0],
    [2.0, 3.2, 4.2],
    [4.0, 5.0, 6.0],
    [5.3, 6.3, 7.3],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0],
    [16.0, 17.0, 18.0],
    [19.0, 20.0, 21.0],
    [22.0, 23.0, 24.0],
    [25.0, 26.0, 27.0],
    [26.5, 27.5, 28.5],
    [27.2, 28.2, 29.2],
    [28.0, 29.0, 30.0],
    [31.4, 32.3, 33.3]
])

NEW_CONC_MATRIX_KG_KG01 = numpy.transpose(NEW_CONC_MATRIX_KG_KG01)

NEW_AIR_DENS_MATRIX_KG_M03 = numpy.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0.5, 0.5, 0.5]
])

NEW_AIR_DENS_MATRIX_KG_M03 = numpy.transpose(NEW_AIR_DENS_MATRIX_KG_M03)

ORIG_MASSES_KG_M02 = numpy.array([905, 955, 1005], dtype=float)
NEW_MASSES_KG_M02 = numpy.array([1751.775, 1836.5, 1921.525], dtype=float)

NEW_CONC_MATRIX_KG_M03 = (
    NEW_CONC_MATRIX_KG_KG01 * NEW_AIR_DENS_MATRIX_KG_M03
)

NEW_CONC_MATRIX_KG_M03 = NEW_CONC_MATRIX_KG_M03 + 0.
NEW_CONC_MATRIX_KG_M03[0, :] *= (
    ORIG_MASSES_KG_M02[0] / NEW_MASSES_KG_M02[0]
)
NEW_CONC_MATRIX_KG_M03[1, :] *= (
    ORIG_MASSES_KG_M02[1] / NEW_MASSES_KG_M02[1]
)
NEW_CONC_MATRIX_KG_M03[2, :] *= (
    ORIG_MASSES_KG_M02[2] / NEW_MASSES_KG_M02[2]
)

NEW_CONC_MATRIX_CONSERVED_KG_KG01 = (
    NEW_CONC_MATRIX_KG_M03 / NEW_AIR_DENS_MATRIX_KG_M03
)


class InterpRapProfilesTests(unittest.TestCase):
    """Each method is a unit test for interp_rap_profiles.py."""

    def test_find_jumps_first(self):
        """Ensures correct output from _find_jumps.

        In this case, using first set of heights.
        """

        this_flag_matrix = interp_rap_profiles._find_jumps(
            orig_data_matrix=ORIG_DATA_MATRIX,
            orig_heights_metres=ORIG_HEIGHTS_METRES,
            new_heights_metres=FIRST_NEW_HEIGHTS_METRES
        )

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, FIRST_JUMP_FLAG_MATRIX
        ))

    def test_find_jumps_second(self):
        """Ensures correct output from _find_jumps.

        In this case, using second set of heights.
        """

        this_flag_matrix = interp_rap_profiles._find_jumps(
            orig_data_matrix=ORIG_DATA_MATRIX,
            orig_heights_metres=ORIG_HEIGHTS_METRES,
            new_heights_metres=SECOND_NEW_HEIGHTS_METRES
        )

        self.assertTrue(numpy.array_equal(
            this_flag_matrix, SECOND_JUMP_FLAG_MATRIX
        ))

    def test_conserve_mass_one_variable(self):
        """Ensures correct output from _conserve_mass_one_variable."""

        this_conc_matrix_kg_kg01 = (
            interp_rap_profiles._conserve_mass_one_variable(
                orig_conc_matrix_kg_kg01=ORIG_CONC_MATRIX_KG_KG01,
                orig_air_dens_matrix_kg_m03=ORIG_AIR_DENS_MATRIX_KG_M03,
                orig_heights_metres=ORIG_HEIGHTS_METRES,
                new_conc_matrix_kg_kg01=NEW_CONC_MATRIX_KG_KG01,
                new_air_dens_matrix_kg_m03=NEW_AIR_DENS_MATRIX_KG_M03,
                new_heights_metres=NEW_HEIGHTS_METRES, test_mode=True
            )
        )

        self.assertTrue(numpy.allclose(
            this_conc_matrix_kg_kg01, NEW_CONC_MATRIX_CONSERVED_KG_KG01,
            atol=TOLERANCE
        ))


if __name__ == '__main__':
    unittest.main()
