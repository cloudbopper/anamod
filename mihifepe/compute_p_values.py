"""Computes p-values for paired statistical tests over input vectors"""

import numpy as np
from numpy import asarray, compress, sqrt
from scipy.stats import find_repeats, rankdata, norm, ttest_rel

from mihifepe import constants


def compute_p_value(baseline, perturbed, test=constants.WILCOXON_TEST):
    """Compute p-value using paired difference test on input numpy arrays"""
    valid_tests = [constants.PAIRED_TTEST, constants.WILCOXON_TEST]
    assert test in valid_tests, "Invalid test name %s" % test
    if test == constants.PAIRED_TTEST:
        # Two-tailed paired t-test
        return ttest_rel(baseline, perturbed)[1]
    # One-tailed Wilcoxon signed-rank test
    return wilcoxon_test(baseline, perturbed, alternative="less")


def wilcoxon_test(x, y, alternative=constants.LESS):
    """
    One-sided Wilcoxon signed-rank test derived from Scipy's two-sided test
    Returns p-value
    """
    # pylint: disable = invalid-name, too-many-locals
    x, y = map(asarray, (x, y))
    d = x - y

    d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)

    r = rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    T = min(r_plus, r_minus)
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if se < 1e-20:
        return 1.  # Degenerate case

    _, repnum = find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    correction = 0.5 * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = norm.sf(abs(z))
    if alternative == constants.LESS:
        return prob if z < 0 else 1 - prob
    if alternative == constants.GREATER:
        return prob if z > 0 else 1 - prob
    return 2 * prob  # Two-sided
