"""Computes p-values for paired statistical tests over input vectors"""

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from scipy import stats

from . import constants

# pylint: disable = no-member, invalid-name

rstats = importr('stats', robject_translations={'format.perc': 'format_dot_perc', 'format_perc': 'format_dash_perc'})
numpy2ri.activate()

def compute_p_value(baseline, perturbed, test=constants.WILCOXON_TEST):
    """Compute p-value using paired difference test on input numpy arrays"""
    valid_tests = [constants.PAIRED_TTEST, constants.WILCOXON_TEST]
    assert test in valid_tests, "Invalid test name %s" % test
    if test == constants.PAIRED_TTEST:
        # Two-tailed paired t-test
        return stats.ttest_rel(baseline, perturbed)[1]

    # One-tailed Wilcoxon signed-rank test
    return rstats.wilcox_test(baseline, perturbed, paired=True, exact=False, alternative="less")[2][0]
