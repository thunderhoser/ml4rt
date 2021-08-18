"""Unit tests for prepare_gfs_for_rrtm.py."""

import copy
import unittest
import numpy
import xarray
from gewittergefahr.gg_utils import time_conversion
from ml4rt.scripts import prepare_gfs_for_rrtm as prepare_gfs

TOLERANCE = 1e-12

THESE_TIME_STRINGS = ['20210101', '20210101', '20210101']
THESE_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y%m%d') for t in THESE_TIME_STRINGS
], dtype=int)

THIS_METADATA_DICT = {
    prepare_gfs.TIME_DIMENSION_ORIG: THESE_TIMES_UNIX_SEC,
    prepare_gfs.SITE_DIMENSION_ORIG: numpy.array([0, 1, 2, 3], dtype=int)
}

THESE_LATITUDES_DEG_N = numpy.array([0, 10, 40, 50], dtype=float)
THESE_LONGITUDES_DEG_E = numpy.array([255, 255, 255, 255], dtype=float)
THESE_DIMENSIONS = (prepare_gfs.SITE_DIMENSION_ORIG,)
THIS_DATA_DICT = {
    prepare_gfs.LATITUDE_KEY_ORIG_DEG_N:
        (THESE_DIMENSIONS, THESE_LATITUDES_DEG_N),
    prepare_gfs.LONGITUDE_KEY_ORIG_DEG_E:
        (THESE_DIMENSIONS, THESE_LONGITUDES_DEG_E)
}

FIRST_GFS_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_DATA_DICT, coords=THIS_METADATA_DICT
)

NEW_HEIGHTS_M_AGL = numpy.linspace(1000, 100000, num=100, dtype=float)
NUM_NEW_HEIGHTS = len(NEW_HEIGHTS_M_AGL)

FIRST_NUM_TIMES = len(THIS_METADATA_DICT[prepare_gfs.TIME_DIMENSION_ORIG])
FIRST_NUM_SITES = len(THIS_METADATA_DICT[prepare_gfs.SITE_DIMENSION_ORIG])
FIRST_INTERP_DATA_DICT = {
    prepare_gfs.TEMPERATURE_KEY_ORIG_KELVINS:
        numpy.full((FIRST_NUM_TIMES, FIRST_NUM_SITES, NUM_NEW_HEIGHTS), 300.)
}

THESE_TIME_STRINGS = ['20210101', '20210404', '20210707', '20211010']
THESE_TIMES_UNIX_SEC = numpy.array([
    time_conversion.string_to_unix_sec(t, '%Y%m%d') for t in THESE_TIME_STRINGS
], dtype=int)

THIS_METADATA_DICT = {
    prepare_gfs.TIME_DIMENSION_ORIG: THESE_TIMES_UNIX_SEC,
    prepare_gfs.SITE_DIMENSION_ORIG: numpy.array([0, 1, 2], dtype=int)
}

THESE_LATITUDES_DEG_N = numpy.array([50, 50, 50], dtype=float)
THESE_LONGITUDES_DEG_E = numpy.array([255, 255, 255], dtype=float)
THESE_DIMENSIONS = (prepare_gfs.SITE_DIMENSION_ORIG,)
THIS_DATA_DICT = {
    prepare_gfs.LATITUDE_KEY_ORIG_DEG_N:
        (THESE_DIMENSIONS, THESE_LATITUDES_DEG_N),
    prepare_gfs.LONGITUDE_KEY_ORIG_DEG_E:
        (THESE_DIMENSIONS, THESE_LONGITUDES_DEG_E)
}

SECOND_GFS_TABLE_XARRAY = xarray.Dataset(
    data_vars=THIS_DATA_DICT, coords=THIS_METADATA_DICT
)

SECOND_NUM_TIMES = len(THIS_METADATA_DICT[prepare_gfs.TIME_DIMENSION_ORIG])
SECOND_NUM_SITES = len(THIS_METADATA_DICT[prepare_gfs.SITE_DIMENSION_ORIG])
SECOND_INTERP_DATA_DICT = {
    prepare_gfs.TEMPERATURE_KEY_ORIG_KELVINS:
        numpy.full((SECOND_NUM_TIMES, SECOND_NUM_SITES, NUM_NEW_HEIGHTS), 300.)
}


class PrepareGfsForRrtmTests(unittest.TestCase):
    """Each method is a unit test for prepare_gfs_for_rrtm.py."""

    def test_add_trace_gases_first(self):
        """Ensures correct output from _add_trace_gases.

        In this case, using first set of inputs.
        """

        this_interp_data_dict = prepare_gfs._add_trace_gases(
            orig_gfs_table_xarray=FIRST_GFS_TABLE_XARRAY,
            new_heights_m_agl=NEW_HEIGHTS_M_AGL,
            interp_data_dict=copy.deepcopy(FIRST_INTERP_DATA_DICT),
            test_mode=True
        )[0]

        # Differences across times should all be zero.
        for j in range(FIRST_NUM_SITES):
            this_diff_matrix = numpy.absolute(numpy.diff(
                this_interp_data_dict[
                    prepare_gfs.CH4_MIXR_KEY_ORIG_KG_KG01
                ][:, j, :],
                axis=0
            ))

            self.assertTrue(numpy.all(this_diff_matrix <= TOLERANCE))

        # Some differences across sites should be non-zero.
        for i in range(FIRST_NUM_TIMES):
            this_diff_matrix = numpy.absolute(numpy.diff(
                this_interp_data_dict[
                    prepare_gfs.CH4_MIXR_KEY_ORIG_KG_KG01
                ][i, ...],
                axis=0
            ))

            self.assertFalse(numpy.all(this_diff_matrix <= TOLERANCE))

    def test_add_trace_gases_second(self):
        """Ensures correct output from _add_trace_gases.

        In this case, using second set of inputs.
        """

        this_interp_data_dict = prepare_gfs._add_trace_gases(
            orig_gfs_table_xarray=SECOND_GFS_TABLE_XARRAY,
            new_heights_m_agl=NEW_HEIGHTS_M_AGL,
            interp_data_dict=copy.deepcopy(SECOND_INTERP_DATA_DICT),
            test_mode=True
        )[0]

        # Some differences across times should be non-zero.
        for j in range(SECOND_NUM_SITES):
            this_diff_matrix = numpy.absolute(numpy.diff(
                this_interp_data_dict[
                    prepare_gfs.CH4_MIXR_KEY_ORIG_KG_KG01
                ][:, j, :],
                axis=0
            ))

            self.assertFalse(numpy.all(this_diff_matrix <= TOLERANCE))

        # Differences across sites should all be zero.
        for i in range(SECOND_NUM_TIMES):
            this_diff_matrix = numpy.absolute(numpy.diff(
                this_interp_data_dict[
                    prepare_gfs.CH4_MIXR_KEY_ORIG_KG_KG01
                ][i, ...],
                axis=0
            ))

            self.assertTrue(numpy.all(this_diff_matrix <= TOLERANCE))


if __name__ == '__main__':
    unittest.main()
