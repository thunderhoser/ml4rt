"""Plots results on training, validation, and testing data for one model."""

import argparse
import numpy
from gewittergefahr.gg_utils import error_checking
from ml4rt.utils import evaluation
from ml4rt.utils import example_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TOLERANCE = 1e-6

HEIGHTS_ARG_NAME = 'heights_m_agl'
CONFIDENCE_LEVEL_ARG_NAME = 'confidence_level'
TRAINING_EVAL_DIR_ARG_NAME = 'input_training_eval_dir_name'
VALIDN_EVAL_DIR_ARG_NAME = 'input_validn_eval_dir_name'
TESTING_EVAL_DIR_ARG_NAME = 'input_testing_eval_dir_name'

HEIGHTS_HELP_STRING = (
    'Will compute scores for net flux and heating rate at these heights (metres'
    ' above ground level).'
)
CONFIDENCE_LEVEL_HELP_STRING = 'Confidence level for bootstrapping.'
TRAINING_EVAL_DIR_HELP_STRING = (
    'Name of top-level directory with evaluation results on training data.'
)
VALIDN_EVAL_DIR_HELP_STRING = (
    'Name of top-level directory with evaluation results on validation data.'
)
TESTING_EVAL_DIR_HELP_STRING = (
    'Name of top-level directory with evaluation results on testing data.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + HEIGHTS_ARG_NAME, type=int, nargs='+', required=True,
    help=HEIGHTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + CONFIDENCE_LEVEL_ARG_NAME, type=float, required=False, default=0.99,
    help=CONFIDENCE_LEVEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TRAINING_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=TRAINING_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + VALIDN_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=VALIDN_EVAL_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TESTING_EVAL_DIR_ARG_NAME, type=str, required=True,
    help=TESTING_EVAL_DIR_HELP_STRING
)


def _read_one_file(evaluation_file_name, heights_m_agl, confidence_level):
    """Reads results from one evaluation file.

    :param evaluation_file_name: Path to input file (will be read by
        `evaluation.read_file`).
    :param heights_m_agl: See documentation at top of file.
    :param confidence_level: Same.
    """

    min_percentile = 50. * (1. - confidence_level)
    max_percentile = 50. * (1. + confidence_level)

    print('Reading data from: "{0:s}"...'.format(evaluation_file_name))
    result_table_xarray = evaluation.read_file(evaluation_file_name)

    net_flux_index = numpy.where(
        result_table_xarray.coords[evaluation.AUX_TARGET_FIELD_DIM].values ==
        evaluation.SHORTWAVE_NET_FLUX_NAME
    )[0][0]

    net_flux_mse_skill_scores = (
        result_table_xarray[evaluation.AUX_MSE_SKILL_KEY].values[
            net_flux_index, :
        ]
    )
    print('MSE skill score for net flux = {0:.3g} [{1:.3g}, {2:.3g}]'.format(
        numpy.mean(net_flux_mse_skill_scores),
        numpy.percentile(net_flux_mse_skill_scores, min_percentile),
        numpy.percentile(net_flux_mse_skill_scores, max_percentile)
    ))

    net_flux_biases = (
        result_table_xarray[evaluation.AUX_BIAS_KEY].values[net_flux_index, :]
    )
    print('Bias for net flux = {0:.3g} [{1:.3g}, {2:.3g}]'.format(
        numpy.mean(net_flux_biases),
        numpy.percentile(net_flux_biases, min_percentile),
        numpy.percentile(net_flux_biases, max_percentile)
    ))

    heating_rate_index = numpy.where(
        result_table_xarray.coords[evaluation.VECTOR_FIELD_DIM].values ==
        example_utils.SHORTWAVE_HEATING_RATE_NAME
    )[0][0]

    num_heights = len(heights_m_agl)

    for k in range(num_heights):
        these_diffs = numpy.absolute(
            result_table_xarray.coords[evaluation.HEIGHT_DIM].values -
            heights_m_agl[k]
        )
        this_height_index = numpy.where(these_diffs <= TOLERANCE)[0][0]

        these_mse_skill_scores = (
            result_table_xarray[evaluation.VECTOR_MSE_SKILL_KEY].values[
                this_height_index, heating_rate_index, :
            ]
        )
        print((
            'MSE skill score for heating rate at {0:d} m AGL = {1:.3g} '
            '[{2:.3g}, {3:.3g}]'
        ).format(
            heights_m_agl[k],
            numpy.mean(these_mse_skill_scores),
            numpy.percentile(these_mse_skill_scores, min_percentile),
            numpy.percentile(these_mse_skill_scores, max_percentile)
        ))

        these_biases = (
            result_table_xarray[evaluation.VECTOR_BIAS_KEY].values[
                this_height_index, heating_rate_index, :
            ]
        )
        print((
            'Bias for heating rate at {0:d} m AGL = {1:.3g} [{2:.3g}, {3:.3g}]'
        ).format(
            heights_m_agl[k],
            numpy.mean(these_biases),
            numpy.percentile(these_biases, min_percentile),
            numpy.percentile(these_biases, max_percentile)
        ))


def _run(heights_m_agl, confidence_level, training_eval_dir_name,
         validn_eval_dir_name, testing_eval_dir_name):
    """Plots results on training, validation, and testing data for one model.

    This is effectively the main method.

    :param heights_m_agl: See documentation at top of file.
    :param confidence_level: Same.
    :param training_eval_dir_name: Same.
    :param validn_eval_dir_name: Same.
    :param testing_eval_dir_name: Same.
    """

    error_checking.assert_is_geq(confidence_level, 0.9)
    error_checking.assert_is_less_than(confidence_level, 1.)

    training_eval_overall_file_name = (
        '{0:s}/evaluation.nc'.format(training_eval_dir_name)
    )
    _read_one_file(
        evaluation_file_name=training_eval_overall_file_name,
        heights_m_agl=heights_m_agl, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    training_eval_by_cloud_file_name = (
        '{0:s}/by_cloud_regime/multi_layer_cloud/evaluation.nc'
    ).format(training_eval_dir_name)

    _read_one_file(
        evaluation_file_name=training_eval_by_cloud_file_name,
        heights_m_agl=heights_m_agl, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    validn_eval_overall_file_name = (
        '{0:s}/evaluation.nc'.format(validn_eval_dir_name)
    )
    _read_one_file(
        evaluation_file_name=validn_eval_overall_file_name,
        heights_m_agl=heights_m_agl, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    validn_eval_by_cloud_file_name = (
        '{0:s}/by_cloud_regime/multi_layer_cloud/evaluation.nc'
    ).format(validn_eval_dir_name)

    _read_one_file(
        evaluation_file_name=validn_eval_by_cloud_file_name,
        heights_m_agl=heights_m_agl, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    testing_eval_overall_file_name = (
        '{0:s}/evaluation.nc'.format(testing_eval_dir_name)
    )
    _read_one_file(
        evaluation_file_name=testing_eval_overall_file_name,
        heights_m_agl=heights_m_agl, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)

    testing_eval_by_cloud_file_name = (
        '{0:s}/by_cloud_regime/multi_layer_cloud/evaluation.nc'
    ).format(testing_eval_dir_name)

    _read_one_file(
        evaluation_file_name=testing_eval_by_cloud_file_name,
        heights_m_agl=heights_m_agl, confidence_level=confidence_level
    )
    print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        heights_m_agl=numpy.array(
            getattr(INPUT_ARG_OBJECT, HEIGHTS_ARG_NAME), dtype=int
        ),
        confidence_level=getattr(INPUT_ARG_OBJECT, CONFIDENCE_LEVEL_ARG_NAME),
        training_eval_dir_name=getattr(
            INPUT_ARG_OBJECT, TRAINING_EVAL_DIR_ARG_NAME
        ),
        validn_eval_dir_name=getattr(
            INPUT_ARG_OBJECT, VALIDN_EVAL_DIR_ARG_NAME
        ),
        testing_eval_dir_name=getattr(
            INPUT_ARG_OBJECT, TESTING_EVAL_DIR_ARG_NAME
        )
    )
