"""Implementation of algorithms used in hierarchical_fdr_control.py"""
# pylint: disable = invalid-name

import sys

import numpy as np

from anamod.core.constants import POSITIVE


def num_rejections(args, F, d, total_rejected):
    """Sub-procedure that returns number of rejections at given level"""
    # Choose initial value k - TODO check if value of k changes result?
    m = len(F[d - 1])  # number of hypotheses at current level
    r_t = m  # Hardcode step-up procedure since more powerful
    psi_t = psi(args, r_t, F, d, total_rejected)
    while r_t > psi_t:
        r_t = psi_t
        psi_t = psi(args, r_t, F, d, total_rejected)
        if r_t <= psi_t:
            return r_t
    while r_t <= psi_t:
        r_t = psi_t + 1
        psi_t = psi(args, r_t, F, d, total_rejected)
        if r_t > psi_t:
            return r_t - 1


def hierarchical_fdr_control(args, F, M):
    """
    General procedure that tests hierarchically ordered hypotheses
    Args:
        F: hypotheses organized by level/depth (0-indexed, unlike 1-indexed as used in paper)
        M: list of all hypotheses
    """
    procedure = getattr(sys.modules[__name__], args.procedure)
    return procedure(args, F, M)


def lynch_guo(args, F, M):
    """Lynch and Guo (2016) hierarchical FDR control procedure"""
    args.logger.info("Begin hypotheses testing procedure")
    # Process root node
    Rs = [0] * len(F)  # list of rejections at each level
    root = M[0]
    R = num_rejections(args, F, 1, 0)
    assert R in (0, 1)
    root.critical_constant = alpha(args, root, R)
    root.adjusted_pvalue = root.pvalue / root.critical_constant * args.alpha if root.critical_constant > 0 else 1
    if R == 0 or not root.pvalue <= root.critical_constant:
        return Rs
    root.rejected = True
    total_rejected = 1  # total number of hypotheses rejected so far
    Rs = [1]  # number of hypotheses rejected at each level
    # Process remaining nodes
    for depth, level in enumerate(F):
        if depth == 0:
            continue  # root already seen
        d = depth + 1  # depth is 0-indexed whereas d is assumed to be 1-indexed
        args.logger.info("Now processing depth %d" % d)
        Rs.append(num_rejections(args, F, d, total_rejected))
        rhs = 0
        for node in level:
            node.critical_constant = alpha_star(args, node, Rs[-1], total_rejected)
            node.adjusted_pvalue = node.pvalue / node.critical_constant * args.alpha if node.critical_constant > 0 else 1
            if node.pvalue <= node.critical_constant:
                node.rejected = True
                rhs += 1
        total_rejected += Rs[-1]
        assert Rs[-1] == rhs  # Self consistency verification
    args.logger.info("End hypotheses testing procedure")
    return Rs


def yekutieli(args, F, M):
    """Yekutieli (2008) hierarchical FDR control procedure"""
    args.logger.info("Begin hierarchical FDR controlled hypothesis testing using Yekutieli (2008)")
    Rs = [0] * len(F)  # list of rejections at each level
    # Handle root
    root = M[0]
    root.critical_constant = args.alpha
    root.adjusted_pvalue = root.pvalue
    if root.pvalue <= root.critical_constant:
        root.rejected = True
        Rs[0] = 1
    # Handle other (than root) nodes
    for level in F:
        for node in level:
            # For each parent, test its child nodes as a family
            if node.is_leaf or not node.rejected:
                continue
            adjusted_pvalues, rejected_hypotheses = bh_procedure([child.pvalue for child in node.children], args.alpha)
            for idx, child in enumerate(node.children):
                child.adjusted_pvalue = adjusted_pvalues[idx]
                child.rejected = rejected_hypotheses[idx]
            Rs[node.depth + 1] += sum(rejected_hypotheses)
    # Sanity check
    for node in M:
        if node.parent and node.rejected:
            assert node.parent.rejected
    args.logger.info("End hierarchical FDR controlled hypothesis testing using Yekutieli (2008)")
    return Rs


def alpha_star(args, node, r, total_rejected):
    """Alpha star"""
    if node.parent and not node.parent.rejected:
        return 0.
    return alpha(args, node, r + total_rejected)


def alpha(args, node, r):
    """Return critical constant for given r, i and dependence assumption"""
    value = (node.l * args.alpha * (node.m + r - 1)) / (node.root.l * node.m)
    if args.dependence_assumption == POSITIVE:
        return value
    # Arbitrary dependence otherwise:
    c = 1 + sum([1. / (node.m + j) for j in range(node.depth + 1, node.G_j_cardinality)])  # node.depth is 0-indexed but we need 1-indexed
    return value / c


def psi(args, r, F, d, total_rejected):
    """Implementation of function psi"""
    return sum([node.pvalue <= alpha_star(args, node, r, total_rejected) for node in F[d - 1]])  # sum counts True as 1


def bh_procedure(pvalues, significance_level):
    """Return adjusted p-values and rejected hypotheses computed according to Benjamini Hochberg procedure"""
    m = len(pvalues)
    hypotheses = list(zip(range(m), pvalues))
    hypotheses.sort(key=lambda x: x[1])
    max_idx = 0
    adjusted_pvalues = np.ones(m)
    rejected_hypotheses = [False] * m
    for idx, hypothesis in enumerate(hypotheses):
        _, pvalue = hypothesis
        i = idx + 1
        adjusted_pvalues[idx] = m / i * pvalue
        critical_constant = i * significance_level / m
        if pvalue < critical_constant:
            max_idx = i
    for idx in range(max_idx):
        rejected_hypotheses[idx] = True
    for idx in reversed(range(m - 1)):
        # Adjusted pvalues - see http://www.biostathandbook.com/multiplecomparisons.html
        adjusted_pvalues[idx] = min(adjusted_pvalues[idx], adjusted_pvalues[idx + 1])
    data = sorted(zip(hypotheses, adjusted_pvalues, rejected_hypotheses), key=lambda elem: elem[0][0])
    _, adjusted_pvalues, rejected_hypotheses = zip(*data)
    return adjusted_pvalues, rejected_hypotheses
