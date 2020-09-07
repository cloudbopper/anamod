"""Computes p-values for paired statistical tests over input vectors"""

import numpy as np
from numpy import asarray, compress, sqrt
from scipy.stats import find_repeats, rankdata, norm, ttest_rel

from anamod import constants, utils


def compute_empirical_p_value(baseline_test_statistic, perturbed_test_statistics):
    """Compute Monte Carlo estimate of empirical permutation-based p-value"""
    sample_count = len(perturbed_test_statistics)
    return (1 + sum(perturbed_test_statistics <= baseline_test_statistic)) / (1 + sample_count)


def compute_p_value(baseline, perturbed, test=constants.PAIRED_TTEST, alternative=constants.TWOSIDED):
    """Compute p-value using paired difference test on input numpy arrays"""
    # TODO: Implement one-sided t-tests
    baseline = utils.round_value(baseline, decimals=15)
    perturbed = utils.round_value(perturbed, decimals=15)
    # Perform statistical test
    valid_tests = [constants.PAIRED_TTEST, constants.WILCOXON_TEST]
    assert test in valid_tests, "Invalid test name %s" % test
    if test == constants.PAIRED_TTEST:
        # Two-tailed paired t-test
        pvalue = ttest_rel(baseline, perturbed).pvalue
        if np.isnan(pvalue):
            # Identical vectors
            pvalue = 1.0
        return pvalue
    # One-tailed Wilcoxon signed-rank test
    return wilcoxon_test(baseline, perturbed, alternative=alternative)


def wilcoxon_test(x, y, alternative):
    """
    One-sided Wilcoxon signed-rank test derived from Scipy's two-sided test
    e.g. for alternative == constants.LESS, rejecting the null means that median difference x - y < 0
    Returns p-value
    """
    # TODO: add unit tests to verify results identical to R's Wilcoxon test for a host of input values
    # pylint: disable = invalid-name, too-many-locals
    x, y = map(asarray, (x, y))
    d = x - y

    d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)

    r = rankdata(abs(d))
    T = np.sum((d > 0) * r, axis=0)

    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if se < 1e-20:
        return 1.  # Degenerate case

    _, repnum = find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    if alternative == constants.LESS:
        correction = -0.5
    elif alternative == constants.GREATER:
        correction = 0.5
    else:
        correction = 0.5 * np.sign(T - mn)  # two-sided

    z = (T - mn - correction) / se

    if alternative == constants.LESS:
        return norm.cdf(z)
    if alternative == constants.GREATER:
        return norm.sf(z)
    return 2 * min(norm.cdf(z), norm.sf(z))  # two-sided
