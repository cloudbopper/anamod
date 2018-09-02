"""Implementation of 'On Procedures Controlling the FDR for Testing Hierarchically Ordered Hypotheses'"""

import argparse
import codecs
import csv
import logging
import math
import os

import anytree
from anytree.exporter import DotExporter
from anytree.exporter import JsonExporter

from .. import constants
from .fdr_algorithms import hierarchical_fdr_control

# pylint: disable = invalid-name

def main():
    """Main"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir", help="name of output directory")
    parser.add_argument("-dependence_assumption", help="choice of dependence assumption used by Lynch and Guo (2016) procedure", choices=[constants.POSITIVE, constants.ARBITRARY], default=constants.POSITIVE)
    parser.add_argument("-alpha", type=float, default=0.05)
    parser.add_argument("-procedure", default=constants.YEKUTIELI, choices=[constants.YEKUTIELI, constants.LYNCH_GUO])
    parser.add_argument("csv_filename", help="CSV (with header) representing hierarchy, each row corresponding to one node:"
                        " the name of the node, the name of its parent node, the node's p-value and optionally a description of the node")
    parser.add_argument("-effect_name", default="AUROC")
    parser.add_argument("-tree_effect_size_threshold", help="while generating output tree of rejected hypotheses,"
                        " only show nodes within given threshold of root (i.e. all nodes erased) effect size", type=float, default=1)
    parser.add_argument("-color_scheme", default="ylorrd9", help="color scheme to use for shading nodes")
    parser.add_argument("-color_range", help="range for chosen color scheme", nargs=2, type=int, default=[1, 9])
    parser.add_argument("-sorting_param", help="parameter to sort on for color grading", default=constants.ADJUSTED_PVALUE, choices=[constants.ADJUSTED_PVALUE, constants.EFFECT_SIZE])
    parser.add_argument("-minimal_labels", help="do not write descriptions/effect sizes on node labels", action="store_true")
    parser.add_argument("-rectangle_leaves", help="enable to generate rectangular nodes for leaves of original hierarchy", action="store_true")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = "%s_alpha_%f_effect_threshold_%f" % (os.path.splitext(os.path.basename(args.csv_filename))[0],
                                                               args.alpha, args.tree_effect_size_threshold)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/hierarchical_fdr_control.log" % args.output_dir, format="%(asctime)s: %(message)s")
    logger = logging.getLogger()
    logger.info("Begin hierarchical_fdr_control")

    tree = build_tree(args, logger)
    F, M = process_tree(logger, tree)
    hierarchical_fdr_control(args, logger, F, M)
    write_outputs(args, logger, tree)
    logger.info("End hierarchical_fdr_control")


def write_outputs(args, logger, tree):
    """Write outputs"""
    logger.info("Begin writing outputs")
    # Export JSON using anytree
    with open("%s/%s.json" % (args.output_dir, constants.HIERARCHICAL_FDR_OUTPUTS), "w") as output_file:
        JsonExporter(indent=2).write(tree, output_file)
    # Write CSV with additional column for rejected or not
    with open("%s/%s.csv" % (args.output_dir, constants.HIERARCHICAL_FDR_OUTPUTS), "w") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([constants.NODE_NAME, constants.PARENT_NAME, constants.PVALUE_LOSSES, constants.REJECTED_STATUS, constants.ADJUSTED_PVALUE])
        for node in anytree.LevelOrderIter(tree):
            parent_name = ""
            if node.parent:
                parent_name = node.parent.name
            writer.writerow([node.name, parent_name, node.pvalue, int(node.rejected), node.adjusted_pvalue])
    # Generate tree of rejected hypotheses with colour grading based on adjusted p-value
    generate_tree_of_rejected_hypotheses(args, tree)
    logger.info("End writing outputs")


def generate_tree_of_rejected_hypotheses(args, tree):
    """Generate tree of rejected hypotheses with colour grading based on adjusted p-value"""
    # Generate tree of rejected hypotheses
    assert tree.rejected, "No hypothesis rejected - check your input p-values" # at least root must be rejected
    nodes = {}
    for node in anytree.LevelOrderIter(tree):
        if node.rejected:
            parent = nodes[node.parent.name] if node.parent else None
            newnode = anytree.Node(node.name, parent=parent, adjusted_pvalue=node.adjusted_pvalue, description=node.description, effect_size=node.effect_size, was_leaf=node.is_leaf)
            nodes[newnode.name] = newnode
    newtree = next(iter(nodes.values())).root # identify root
    prune_tree_on_effect_size(args, newtree)
    color_nodes(args, newtree)
    render_tree(args, newtree)


def render_tree(args, tree):
    """Render tree in ASCII and graphviz"""
    with codecs.open("{0}/{1}.txt".format(args.output_dir, constants.TREE), "w", encoding="utf8") as txt_file:
        for pre, _, node in anytree.RenderTree(tree):
            txt_file.write("%s%s: %s (%s: %s)\n" % (pre, node.name, node.description.title(), args.effect_name, str(node.effect_size)))
    graph_options = [] # Example: graph_options = ["dpi=300.0;", "style=filled;", "bgcolor=yellow;"]
    DotExporter(tree, options=graph_options, nodeattrfunc=lambda node: nodeattrfunc(args, node)).to_dotfile("{0}/{1}.dot".format(args.output_dir, constants.TREE))
    DotExporter(tree, options=graph_options, nodeattrfunc=lambda node: nodeattrfunc(args, node)).to_picture("{0}/{1}.png".format(args.output_dir, constants.TREE))


def prune_tree_on_effect_size(args, tree):
    """Prune tree by thresholding on effect size"""
    if not tree.effect_size:
        return # No effect_size column in input file
    effect_size_threshold = tree.effect_size * (1 + args.tree_effect_size_threshold)
    for node in anytree.LevelOrderIter(tree):
        if node.effect_size > effect_size_threshold:
            node.parent = None


def color_nodes(args, tree):
    """Add fill and font color to nodes based on partition in sorted list"""
    differentiator = lambda node: node.adjusted_pvalue if args.sorting_param == constants.ADJUSTED_PVALUE else node.effect_size
    nodes_sorted = sorted(anytree.LevelOrderIter(tree), key=differentiator, reverse=True) # sort nodes for color grading
    num_nodes = len(nodes_sorted)
    lower, upper = args.color_range
    num_colors = upper - lower + 1
    assert 1 <= lower <= upper <= 9
    for idx, node in enumerate(nodes_sorted):
        node = nodes_sorted[idx]
        node.color = idx + lower
        if num_nodes > num_colors:
            node.color = lower + (idx * num_colors) // num_nodes
        assert node.color in range(lower, upper + 1, 1)
    # Non-differentiated nodes should have the same color
    prev_node = None
    for node in nodes_sorted:
        if prev_node and differentiator(node) == differentiator(prev_node):
            node.color = prev_node.color
        prev_node = node
        node.fontcolor = "black" if node.color <= 5 else "white"


def nodeattrfunc(args, node):
    """Node attributes function"""
    label = node.name.upper()
    if not args.minimal_labels and node.description:
        label = "%s:\n%s" % (label, node.description)
    if not args.minimal_labels and node.effect_size:
        label = "%s\n%s: %0.3f" % (label, args.effect_name, node.effect_size)
    words = label.split(" ")
    words_per_line = 3
    lines = []
    for idx in range(0, len(words), words_per_line):
        line = " ".join(words[idx: min(len(words), idx + words_per_line)])
        lines.append(line)
    newlabel = "\n".join(lines)
    shape = "rectangle" if args.rectangle_leaves and node.was_leaf else "ellipse"
    return "fillcolor=\"/%s/%d\" label=\"%s\" style=filled fontname=\"helvetica bold\" fontsize=15.0 fontcolor=%s shape = %s" % (args.color_scheme, node.color, newlabel, node.fontcolor, shape)


def build_tree(args, logger):
    """Build tree from CSV file"""
    logger.info("Begin building tree")
    nodes = {} # map of all nodes in tree
    root = None
    with open(args.csv_filename) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            node_name = row[constants.NODE_NAME]
            parent_name = row[constants.PARENT_NAME]
            pvalue = float(row[constants.PVALUE_LOSSES])
            description = row[constants.DESCRIPTION] if constants.DESCRIPTION in row else ""
            effect_size = ""
            if constants.EFFECT_SIZE in row and row[constants.EFFECT_SIZE]:
                effect_size = float(row[constants.EFFECT_SIZE])
            # Create or retrieve current node
            if node_name not in nodes:
                node = anytree.Node(node_name)
                nodes[node_name] = node
            else:
                node = nodes[node_name]
                assert not node.parent # to ensure no rows have the same node name
            # Create or retrieve parent node
            if not parent_name:
                # Root node
                assert root is None # to ensure root has not already been assigned
                root = node
                parent = None
            elif parent_name not in nodes:
                parent = anytree.Node(parent_name)
                nodes[parent_name] = parent
            else:
                parent = nodes[parent_name]
            node.parent = parent
            node.pvalue = pvalue if not math.isnan(pvalue) else 1.
            node.description = description
            node.effect_size = effect_size
    assert root # to ensure root exists and has an assigned p-value
    logger.info("End building tree")
    return root


def process_tree(logger, tree):
    """Processes tree and builds intermediate data structures"""
    logger.info("Begin processing tree")
    F = [] # hypotheses organized by level/depth (0-indexed, unlike 1-indexed as used in paper)
    M = [] # list of all hypotheses
    for level in anytree.LevelOrderGroupIter(tree):
        level = sorted(level, key=lambda node: node.pvalue) # sorting by p-value at each level; TODO: decide if needed
        F.append(level)
        M.extend(level)
        for node in level:
            node.G_j_cardinality = len(level) # number of nodes in tree upto and including this level
            if node.parent:
                node.G_j_cardinality += node.parent.G_j_cardinality
            node.rejected = False # no hypothesis rejected to start with
            node.critical_constant = -1. # populated later
            node.adjusted_pvalue = -1. # populated later
    for level in reversed(F):
        # Iterate over tree bottom-up to identify l (number of leaves) and m (number of hypotheses) for each node
        for node in level:
            node.m = 1
            node.l = 1
            if not node.is_leaf:
                node.l = sum([child.l for child in node.children])
                node.m = 1 + sum([child.m for child in node.children])
    logger.info("End processing tree")
    return F, M


if __name__ == "__main__":
    main()
