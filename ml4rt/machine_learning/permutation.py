"""IO and helper methods for permutation-based importance test."""

import numpy


def run_forward_test(
        model_object, predictor_matrix, target_matrices, model_metadata_dict,
        cost_function, preserve_vectors=True,
        num_bootstrap_reps=DEFAULT_NUM_BOOTSTRAP_REPS):
    """Runs multi-pass forward version of permutation test.

    S = number of steps executed in test
    B = number of bootstrap replicates
    P = number of predictor

    :param model_object: Trained model (instance of `keras.models.Model` or
        `keras.models.Sequential`).
    :param predictor_matrix: Predictor matrix (numpy array) in format accepted
        by model.
    :param target_matrices: 1-D list of target matrices (numpy arrays) in format
        accepted by model.
    :param model_metadata_dict: Dictionary read by `neural_net.read_metafile`.
    :param cost_function: Cost function.  Must be negatively oriented (lower is
        better), with the following inputs and outputs.
    Input: target_matrices: Same as input to this method.
    Input: prediction_matrices: Same as `target_matrices` but with predicted,
        rather than actual, values.
    Output: cost: Scalar value.

    :param preserve_vectors: Boolean flag.  If True, vector predictors
        (profiles) will be shuffled together.  If False, each scalar predictor
        (i.e., each variable at each height) will be shuffled on its own,
        independently of all the others.
    :param num_bootstrap_reps: Number of bootstrap replicates used to estimate
        cost function.  If you do not want bootstrapping, make this <= 1.

    :return: result_dict: Dictionary with the following keys.
    result_dict['best_predictor_names']: length-S list with name of best
        predictor at each step.
    result_dict['best_heights_m_agl']: length-S numpy array of corresponding
        heights (metres above ground level).
    result_dict['best_cost_matrix']: S-by-B numpy array of post-permutation cost
        values.
    result_dict["original_cost_array"]: length-B numpy array of costs before
        permutation.
    result_dict['step1_predictor_names']: length-P list of predictor names, in
        the order that they were shuffled in step 1 (the single-pass test).
    result_dict['step1_heights_m_agl']: length-P numpy array of corresponding
        heights (metres above ground level).
    result_dict['step1_cost_matrix']: P-by-B numpy array of post-permutation
        cost values in step 1.
    result_dict['backwards_test']: Boolean flag (always False).
    """