"""Unit tests for aerosols.py"""

import unittest
import numpy
from ml4rt.utils import aerosols

EXAMPLE_LATITUDES_DEG_N = numpy.array([
    60, -60,
    40, 40, 40, 40, 40, 40, 40, 47,
    30, 30, 30, 30, 30,
    10, 10, 10, 10, 10,
    50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30,
    0, 0, 0, 0, 0, 0, 0, 0,
    40, 40, 40, 40
], dtype=float)

EXAMPLE_LONGITUDES_DEG_E = numpy.array([
    100, 100,
    80, 100, 120, 140, 160, 180, -160, -179,
    80, 100, 120, 140, 160,
    80, 100, 120, 140, 160,
    -20, 0, 20, 40, 60,
    -60, -40, -20, 0, 20, 40, 60,
    -80, -60, -40, -20, 0, 20, 40, 60,
    -120, -100, -80, -60
], dtype=float)

POLAR = aerosols.POLAR_REGION_NAME
LAND = aerosols.LAND_REGION_NAME
OCEAN = aerosols.OCEAN_REGION_NAME
DUST1 = aerosols.FIRST_DESERT_DUST_REGION_NAME
DUST2 = aerosols.SECOND_DESERT_DUST_REGION_NAME
URBAN1 = aerosols.FIRST_URBAN_REGION_NAME
URBAN2 = aerosols.SECOND_URBAN_REGION_NAME
BIOMASS = aerosols.BIOMASS_BURNING_REGION_NAME

REGION_NAME_BY_EXAMPLE = [
    POLAR, POLAR,
    LAND, LAND, DUST2, URBAN2, OCEAN, OCEAN, OCEAN, DUST2,
    URBAN2, URBAN2, URBAN2, URBAN2, OCEAN,
    URBAN2, BIOMASS, BIOMASS, BIOMASS, OCEAN,
    OCEAN, URBAN1, URBAN1, LAND, LAND,
    OCEAN, DUST1, DUST1, DUST1, DUST1, DUST1, LAND,
    LAND, BIOMASS, BIOMASS, BIOMASS, BIOMASS, BIOMASS, BIOMASS, OCEAN,
    LAND, LAND, URBAN1, OCEAN
]


class AerosolsTests(unittest.TestCase):
    """Each method is a unit test for aerosols.py."""

    def test_assign_examples_to_regions(self):
        """Ensures correct output from assign_examples_to_regions."""

        these_region_names = aerosols.assign_examples_to_regions(
            example_latitudes_deg_n=EXAMPLE_LATITUDES_DEG_N,
            example_longitudes_deg_e=EXAMPLE_LONGITUDES_DEG_E
        )
        self.assertTrue(these_region_names == REGION_NAME_BY_EXAMPLE)


if __name__ == '__main__':
    unittest.main()
