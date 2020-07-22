"""Averages many interpretation maps."""

import argparse
import numpy
from gewittergefahr.gg_utils import prob_matched_means as pmm
from ml4rt.machine_learning import saliency
from ml4rt.machine_learning import gradcam
from ml4rt.machine_learning import backwards_optimization as bwo

SALIENCY_TYPE_STRING = 'saliency'
SALIENCY_ALL_TARGETS_TYPE_STRING = 'saliency_all_targets'
GRADCAM_TYPE_STRING = 'gradcam'
GRADCAM_ALL_TARGETS_TYPE_STRING = 'gradcam_all_targets'
BWO_TYPE_STRING = 'bwo'

VALID_FILE_TYPE_STRINGS = [
    SALIENCY_TYPE_STRING, SALIENCY_ALL_TARGETS_TYPE_STRING,
    GRADCAM_TYPE_STRING, GRADCAM_ALL_TARGETS_TYPE_STRING,
    BWO_TYPE_STRING
]

INPUT_FILE_ARG_NAME = 'input_file_name'
FILE_TYPE_ARG_NAME = 'file_type_string'
USE_PMM_ARG_NAME = 'use_pmm'
MAX_PERCENTILE_ARG_NAME = 'max_pmm_percentile_level'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_FILE_HELP_STRING = (
    'Path to input file, containing interpretation maps to average.  Will be '
    'read by `saliency.read_file`, `saliency.read_all_targets_file`, '
    '`gradcam.read_file`, `gradcam.read_all_targets_file`, or '
    '`backwards_optimization.read_file`.'
)

FILE_TYPE_HELP_STRING = (
    'Type of interpretation maps in file.  Must be in the following list:'
    '\n{0:s}'
).format(str(VALID_FILE_TYPE_STRINGS))

USE_PMM_HELP_STRING = (
    'Boolean flag.  If 1 (0), will use probability-matched (arithmetic) means '
    'for vertical profiles.'
)
MAX_PERCENTILE_HELP_STRING = (
    '[used only if `{0:s}` = 1] Max percentile level for probability-matched '
    'means.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Average interpretation map will be written here by '
    '`saliency.write_file`, `saliency.write_all_targets_file`, '
    '`gradcam.write_file`, `gradcam.write_all_targets_file`, or '
    '`backwards_optimization.write_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_FILE_ARG_NAME, type=str, required=True,
    help=INPUT_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FILE_TYPE_ARG_NAME, type=str, required=True,
    help=FILE_TYPE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_PMM_ARG_NAME, type=int, required=True, help=USE_PMM_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MAX_PERCENTILE_ARG_NAME, type=float, required=False, default=99.,
    help=MAX_PERCENTILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _average_saliency_maps(
        input_file_name, use_pmm, max_pmm_percentile_level, output_file_name):
    """Averages many saliency maps.

    :param input_file_name: See documentation at top of file.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    print('Reading saliency maps from: "{0:s}"...'.format(input_file_name))
    saliency_dict = saliency.read_file(input_file_name)

    vector_saliency_matrix = saliency_dict[saliency.VECTOR_SALIENCY_KEY]
    scalar_saliency_matrix = saliency_dict[saliency.VECTOR_SALIENCY_KEY]

    if vector_saliency_matrix.size == 0:
        vector_saliency_matrix = vector_saliency_matrix[0, ...]
    elif use_pmm:
        vector_saliency_matrix = pmm.run_pmm_many_variables(
            input_matrix=vector_saliency_matrix,
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        vector_saliency_matrix = numpy.mean(vector_saliency_matrix, axis=0)

    if scalar_saliency_matrix.size == 0:
        scalar_saliency_matrix = scalar_saliency_matrix[0, ...]
    elif use_pmm and len(scalar_saliency_matrix.shape) == 3:
        scalar_saliency_matrix = pmm.run_pmm_many_variables(
            input_matrix=scalar_saliency_matrix,
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        scalar_saliency_matrix = numpy.mean(scalar_saliency_matrix, axis=0)

    vector_saliency_matrix = numpy.expand_dims(vector_saliency_matrix, axis=0)
    scalar_saliency_matrix = numpy.expand_dims(scalar_saliency_matrix, axis=0)

    if use_pmm:
        example_id_strings = [saliency.DUMMY_EXAMPLE_ID_PMM]
    else:
        example_id_strings = [saliency.DUMMY_EXAMPLE_ID_AVERAGE]

    print('Writing average saliency map to: "{0:s}"...'.format(
        output_file_name
    ))

    saliency.write_file(
        netcdf_file_name=output_file_name,
        scalar_saliency_matrix=scalar_saliency_matrix,
        vector_saliency_matrix=vector_saliency_matrix,
        example_id_strings=example_id_strings,
        model_file_name=saliency_dict[saliency.MODEL_FILE_KEY],
        layer_name=saliency_dict[saliency.LAYER_NAME_KEY],
        neuron_indices=saliency_dict[saliency.NEURON_INDICES_KEY],
        ideal_activation=saliency_dict[saliency.IDEAL_ACTIVATION_KEY],
        target_field_name=saliency_dict[saliency.TARGET_FIELD_KEY],
        target_height_m_agl=saliency_dict[saliency.TARGET_HEIGHT_KEY]
    )


def _average_saliency_maps_all_targets(
        input_file_name, use_pmm, max_pmm_percentile_level, output_file_name):
    """Averages saliency maps for each target variable.

    :param input_file_name: See documentation at top of file.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    print('Reading saliency maps from: "{0:s}"...'.format(input_file_name))
    saliency_dict = saliency.read_all_targets_file(input_file_name)

    saliency_matrix_vector_p_scalar_t = (
        saliency_dict[saliency.SALIENCY_VECTOR_P_SCALAR_T_KEY]
    )
    num_scalar_targets = saliency_matrix_vector_p_scalar_t.shape[-1]

    if saliency_matrix_vector_p_scalar_t.size != 0:
        for k in range(num_scalar_targets):
            if use_pmm:
                saliency_matrix_vector_p_scalar_t[0, ..., k] = (
                    pmm.run_pmm_many_variables(
                        input_matrix=saliency_matrix_vector_p_scalar_t[..., k],
                        max_percentile_level=max_pmm_percentile_level
                    )
                )
            else:
                saliency_matrix_vector_p_scalar_t[0, ..., k] = numpy.mean(
                    saliency_matrix_vector_p_scalar_t[..., k], axis=0
                )

    saliency_matrix_scalar_p_scalar_t = (
        saliency_dict[saliency.SALIENCY_SCALAR_P_SCALAR_T_KEY]
    )

    if saliency_matrix_scalar_p_scalar_t.size != 0:
        for k in range(num_scalar_targets):
            if use_pmm and len(saliency_matrix_scalar_p_scalar_t.shape) == 4:
                saliency_matrix_scalar_p_scalar_t[0, ..., k] = (
                    pmm.run_pmm_many_variables(
                        input_matrix=saliency_matrix_scalar_p_scalar_t[..., k],
                        max_percentile_level=max_pmm_percentile_level
                    )
                )
            else:
                saliency_matrix_scalar_p_scalar_t[0, ..., k] = numpy.mean(
                    saliency_matrix_scalar_p_scalar_t[..., k], axis=0
                )

    saliency_matrix_vector_p_vector_t = (
        saliency_dict[saliency.SALIENCY_VECTOR_P_VECTOR_T_KEY]
    )
    num_heights = saliency_matrix_vector_p_vector_t.shape[-2]
    num_vector_targets = saliency_matrix_vector_p_vector_t.shape[-1]

    if saliency_matrix_vector_p_vector_t.size != 0:
        for j in range(num_heights):
            for k in range(num_vector_targets):
                if use_pmm:
                    saliency_matrix_vector_p_vector_t[0, ..., j, k] = (
                        pmm.run_pmm_many_variables(
                            input_matrix=
                            saliency_matrix_vector_p_vector_t[..., j, k],
                            max_percentile_level=max_pmm_percentile_level
                        )
                    )
                else:
                    saliency_matrix_vector_p_vector_t[0, ..., j, k] = (
                        numpy.mean(
                            saliency_matrix_vector_p_vector_t[..., j, k], axis=0
                        )
                    )

    saliency_matrix_scalar_p_vector_t = (
        saliency_dict[saliency.SALIENCY_SCALAR_P_VECTOR_T_KEY]
    )

    if saliency_matrix_scalar_p_vector_t.size != 0:
        for j in range(num_heights):
            for k in range(num_vector_targets):
                if (
                        use_pmm and
                        len(saliency_matrix_scalar_p_vector_t.shape) == 5
                ):
                    saliency_matrix_scalar_p_vector_t[0, ..., j, k] = (
                        pmm.run_pmm_many_variables(
                            input_matrix=
                            saliency_matrix_scalar_p_vector_t[..., j, k],
                            max_percentile_level=max_pmm_percentile_level
                        )
                    )
                else:
                    saliency_matrix_scalar_p_vector_t[0, ..., j, k] = (
                        numpy.mean(
                            saliency_matrix_scalar_p_vector_t[..., j, k], axis=0
                        )
                    )

    saliency_matrix_vector_p_scalar_t = (
        saliency_matrix_vector_p_scalar_t[[0], ...]
    )
    saliency_matrix_scalar_p_scalar_t = (
        saliency_matrix_scalar_p_scalar_t[[0], ...]
    )
    saliency_matrix_vector_p_vector_t = (
        saliency_matrix_vector_p_vector_t[[0], ...]
    )
    saliency_matrix_scalar_p_vector_t = (
        saliency_matrix_scalar_p_vector_t[[0], ...]
    )

    if use_pmm:
        example_id_strings = [saliency.DUMMY_EXAMPLE_ID_PMM]
    else:
        example_id_strings = [saliency.DUMMY_EXAMPLE_ID_AVERAGE]

    print('Writing average saliency maps to: "{0:s}"...'.format(
        output_file_name
    ))

    saliency.write_all_targets_file(
        netcdf_file_name=output_file_name,
        saliency_matrix_scalar_p_scalar_t=saliency_matrix_scalar_p_scalar_t,
        saliency_matrix_vector_p_scalar_t=saliency_matrix_vector_p_scalar_t,
        saliency_matrix_scalar_p_vector_t=saliency_matrix_scalar_p_vector_t,
        saliency_matrix_vector_p_vector_t=saliency_matrix_vector_p_vector_t,
        example_id_strings=example_id_strings,
        model_file_name=saliency_dict[saliency.MODEL_FILE_KEY],
        ideal_activation=saliency_dict[saliency.IDEAL_ACTIVATION_KEY]
    )


def _average_cams(
        input_file_name, use_pmm, max_pmm_percentile_level, output_file_name):
    """Averages many class-activation maps.

    :param input_file_name: See documentation at top of file.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    print('Reading class-activation maps from: "{0:s}"...'.format(
        input_file_name
    ))
    gradcam_dict = gradcam.read_file(input_file_name)
    class_activation_matrix = gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY]

    if use_pmm:
        class_activation_matrix = numpy.expand_dims(
            class_activation_matrix, axis=-1
        )
        class_activation_matrix = pmm.run_pmm_many_variables(
            input_matrix=class_activation_matrix,
            max_percentile_level=max_pmm_percentile_level
        )[..., 0]
    else:
        class_activation_matrix = numpy.mean(class_activation_matrix, axis=0)

    class_activation_matrix = numpy.expand_dims(class_activation_matrix, axis=0)

    if use_pmm:
        example_id_strings = [gradcam.DUMMY_EXAMPLE_ID_PMM]
    else:
        example_id_strings = [gradcam.DUMMY_EXAMPLE_ID_AVERAGE]

    print('Writing average class-activation map to: "{0:s}"...'.format(
        output_file_name
    ))

    gradcam.write_file(
        netcdf_file_name=output_file_name,
        class_activation_matrix=class_activation_matrix,
        example_id_strings=example_id_strings,
        model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
        activation_layer_name=gradcam_dict[gradcam.ACTIVATION_LAYER_KEY],
        vector_output_layer_name=gradcam_dict[gradcam.VECTOR_OUT_LAYER_KEY],
        output_neuron_indices=gradcam_dict[gradcam.OUTPUT_NEURONS_KEY],
        ideal_activation=gradcam_dict[gradcam.IDEAL_ACTIVATION_KEY]
    )


def _average_cams_all_targets(
        input_file_name, use_pmm, max_pmm_percentile_level, output_file_name):
    """Averages class-activation maps for each target variable.

    :param input_file_name: See documentation at top of file.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    print('Reading class-activation maps from: "{0:s}"...'.format(
        input_file_name
    ))
    gradcam_dict = gradcam.read_all_targets_file(input_file_name)

    class_activation_matrix = gradcam_dict[gradcam.CLASS_ACTIVATIONS_KEY]
    num_heights = class_activation_matrix.shape[-2]
    num_vector_targets = class_activation_matrix.shape[-1]

    for j in range(num_heights):
        for k in range(num_vector_targets):
            if use_pmm:
                this_activation_matrix = numpy.expand_dims(
                    class_activation_matrix[..., j, k], axis=-1
                )
                this_activation_matrix = pmm.run_pmm_many_variables(
                    input_matrix=this_activation_matrix,
                    max_percentile_level=max_pmm_percentile_level
                )[..., 0]
                class_activation_matrix[0, ..., j, k] = this_activation_matrix
            else:
                class_activation_matrix[0, ..., j, k] = numpy.mean(
                    class_activation_matrix[..., j, k], axis=0
                )

    class_activation_matrix = class_activation_matrix[[0], ...]

    if use_pmm:
        example_id_strings = [gradcam.DUMMY_EXAMPLE_ID_PMM]
    else:
        example_id_strings = [gradcam.DUMMY_EXAMPLE_ID_AVERAGE]

    print('Writing average class-activation maps to: "{0:s}"...'.format(
        output_file_name
    ))

    gradcam.write_all_targets_file(
        netcdf_file_name=output_file_name,
        class_activation_matrix=class_activation_matrix,
        example_id_strings=example_id_strings,
        model_file_name=gradcam_dict[gradcam.MODEL_FILE_KEY],
        activation_layer_name=gradcam_dict[gradcam.ACTIVATION_LAYER_KEY],
        vector_output_layer_name=gradcam_dict[gradcam.VECTOR_OUT_LAYER_KEY],
        ideal_activation=gradcam_dict[gradcam.IDEAL_ACTIVATION_KEY]
    )


def _average_bwo_results(
        input_file_name, use_pmm, max_pmm_percentile_level, output_file_name):
    """Averages results of backwards optimization.

    :param input_file_name: See documentation at top of file.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    """

    print('Reading backwards-optimization results from: "{0:s}"...'.format(
        input_file_name
    ))
    bwo_dict = bwo.read_file(input_file_name)

    init_vector_predictor_matrix = bwo_dict[bwo.INIT_VECTOR_PREDICTORS_KEY]
    final_vector_predictor_matrix = bwo_dict[bwo.FINAL_VECTOR_PREDICTORS_KEY]
    init_scalar_predictor_matrix = bwo_dict[bwo.INIT_SCALAR_PREDICTORS_KEY]
    final_scalar_predictor_matrix = bwo_dict[bwo.FINAL_SCALAR_PREDICTORS_KEY]

    if init_vector_predictor_matrix.size == 0:
        init_vector_predictor_matrix = init_vector_predictor_matrix[0, ...]
        final_vector_predictor_matrix = final_vector_predictor_matrix[0, ...]
    elif use_pmm:
        init_vector_predictor_matrix = pmm.run_pmm_many_variables(
            input_matrix=init_vector_predictor_matrix,
            max_percentile_level=max_pmm_percentile_level
        )
        final_vector_predictor_matrix = pmm.run_pmm_many_variables(
            input_matrix=final_vector_predictor_matrix,
            max_percentile_level=max_pmm_percentile_level
        )
    else:
        init_vector_predictor_matrix = numpy.mean(
            init_vector_predictor_matrix, axis=0
        )
        final_vector_predictor_matrix = numpy.mean(
            final_vector_predictor_matrix, axis=0
        )

    if init_scalar_predictor_matrix.size == 0:
        init_scalar_predictor_matrix = init_scalar_predictor_matrix[0, ...]
        final_scalar_predictor_matrix = final_scalar_predictor_matrix[0, ...]
    else:
        init_scalar_predictor_matrix = numpy.mean(
            init_scalar_predictor_matrix, axis=0
        )
        final_scalar_predictor_matrix = numpy.mean(
            final_scalar_predictor_matrix, axis=0
        )

    init_vector_predictor_matrix = numpy.expand_dims(
        init_vector_predictor_matrix, axis=0
    )
    final_vector_predictor_matrix = numpy.expand_dims(
        final_vector_predictor_matrix, axis=0
    )
    init_scalar_predictor_matrix = numpy.expand_dims(
        init_scalar_predictor_matrix, axis=0
    )
    final_scalar_predictor_matrix = numpy.expand_dims(
        final_scalar_predictor_matrix, axis=0
    )

    if use_pmm:
        example_id_strings = [bwo.DUMMY_EXAMPLE_ID_PMM]
    else:
        example_id_strings = [bwo.DUMMY_EXAMPLE_ID_AVERAGE]

    initial_activations = numpy.array([
        numpy.mean(bwo_dict[bwo.INITIAL_ACTIVATIONS_KEY])
    ])
    final_activations = numpy.array([
        numpy.mean(bwo_dict[bwo.FINAL_ACTIVATIONS_KEY])
    ])

    print((
        'Writing average backwards-optimization results to: "{0:s}"...'
    ).format(
        output_file_name
    ))

    bwo.write_file(
        netcdf_file_name=output_file_name,
        init_scalar_predictor_matrix=init_scalar_predictor_matrix,
        final_scalar_predictor_matrix=final_scalar_predictor_matrix,
        init_vector_predictor_matrix=init_vector_predictor_matrix,
        final_vector_predictor_matrix=final_vector_predictor_matrix,
        initial_activations=initial_activations,
        final_activations=final_activations,
        example_id_strings=example_id_strings,
        model_file_name=bwo_dict[bwo.MODEL_FILE_KEY],
        layer_name=bwo_dict[bwo.LAYER_NAME_KEY],
        neuron_indices=bwo_dict[bwo.NEURON_INDICES_KEY],
        ideal_activation=bwo_dict[bwo.IDEAL_ACTIVATION_KEY],
        num_iterations=bwo_dict[bwo.NUM_ITERATIONS_KEY],
        learning_rate=bwo_dict[bwo.LEARNING_RATE_KEY],
        l2_weight=bwo_dict[bwo.L2_WEIGHT_KEY]
    )


def _run(input_file_name, file_type_string, use_pmm, max_pmm_percentile_level,
         output_file_name):
    """Averages many interpretation maps.

    This is effectively the main method.

    :param input_file_name: See documentation at top of file.
    :param file_type_string: Same.
    :param use_pmm: Same.
    :param max_pmm_percentile_level: Same.
    :param output_file_name: Same.
    :raises: ValueError: if `file_type_string not in VALID_FILE_TYPE_STRINGS`.
    """

    if file_type_string not in VALID_FILE_TYPE_STRINGS:
        error_string = (
            '\nFile type ("{0:s}") is invalid.  Must be in the following list:'
            '\n{1:s}'
        ).format(file_type_string, str(VALID_FILE_TYPE_STRINGS))

        raise ValueError(error_string)

    if file_type_string == SALIENCY_TYPE_STRING:
        _average_saliency_maps(
            input_file_name=input_file_name, use_pmm=use_pmm,
            max_pmm_percentile_level=max_pmm_percentile_level,
            output_file_name=output_file_name
        )
    elif file_type_string == SALIENCY_ALL_TARGETS_TYPE_STRING:
        _average_saliency_maps_all_targets(
            input_file_name=input_file_name, use_pmm=use_pmm,
            max_pmm_percentile_level=max_pmm_percentile_level,
            output_file_name=output_file_name
        )
    elif file_type_string == GRADCAM_TYPE_STRING:
        _average_cams(
            input_file_name=input_file_name, use_pmm=use_pmm,
            max_pmm_percentile_level=max_pmm_percentile_level,
            output_file_name=output_file_name
        )
    elif file_type_string == GRADCAM_ALL_TARGETS_TYPE_STRING:
        _average_cams_all_targets(
            input_file_name=input_file_name, use_pmm=use_pmm,
            max_pmm_percentile_level=max_pmm_percentile_level,
            output_file_name=output_file_name
        )
    else:
        _average_bwo_results(
            input_file_name=input_file_name, use_pmm=use_pmm,
            max_pmm_percentile_level=max_pmm_percentile_level,
            output_file_name=output_file_name
        )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_file_name=getattr(INPUT_ARG_OBJECT, INPUT_FILE_ARG_NAME),
        file_type_string=getattr(INPUT_ARG_OBJECT, FILE_TYPE_ARG_NAME),
        use_pmm=bool(getattr(INPUT_ARG_OBJECT, USE_PMM_ARG_NAME)),
        max_pmm_percentile_level=getattr(
            INPUT_ARG_OBJECT, MAX_PERCENTILE_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )
