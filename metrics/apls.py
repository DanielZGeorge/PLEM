"""
Raster-skeleton APLS (Average Path Length Similarity) approximation.

No vector road graph exists anywhere in this pipeline (datasets/common.py's
rasterize_lines collapses roads straight to a flat pixel mask, discarding
segment/graph structure), so this approximates the SpaceNet-style APLS
metric entirely from raster skeletons: skeletonize pred/GT masks
(skimage.morphology.skeletonize, as in cldice.py), build an 8-connected
pixel-adjacency graph via networkx, sample control-point pairs within the
same GT connected component, and compare shortest-path lengths between
matched GT/pred pairs.

This is deliberately sensitive to a failure mode DTAF1's per-pixel tolerance
matching misses: scattered surviving pixels after random dropout still fall
within DTAF1's tolerance radius of some GT pixel (so DTAF1 stays high), but
once the skeleton fragments, shortest paths between control points either
lengthen sharply or disappear entirely, so APLS collapses. See
metrics/dtaf1_topo.py for the composite that blends this signal into DTAF1.
"""

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree


DEFAULT_N_PAIRS = 100   # control-point pair budget per class
DEFAULT_SEED = 0        # RNG seed for deterministic control-point sampling

# The 4 "forward" 8-connectivity offsets: touches every undirected edge
# between adjacent skeleton pixels exactly once, no duplicate edge-adds.
_FORWARD_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1))


def _skeleton_graph(mask: np.ndarray) -> nx.Graph:
    """
    Build an 8-connected pixel-adjacency graph from a binary mask's skeleton.

    Nodes are (row, col) int tuples. Edge weight is Euclidean pixel distance
    (1.0 orthogonal, sqrt(2) diagonal).
    """
    skel = skeletonize(mask > 0)
    coords = [(int(r), int(c)) for r, c in zip(*np.nonzero(skel))]
    coord_set = set(coords)
    g = nx.Graph()
    g.add_nodes_from(coords)
    for (r, c) in coords:
        for dr, dc in _FORWARD_OFFSETS:
            nbr = (r + dr, c + dc)
            if nbr in coord_set:
                g.add_edge((r, c), nbr, weight=float(np.hypot(dr, dc)))
    return g


def _sample_control_pairs(gt_graph: nx.Graph, n_pairs: int, seed: int) -> list:
    """
    Deterministically sample up to `n_pairs` (source, [targets...]) groups,
    each restricted to a single connected component of `gt_graph` (a GT path
    only exists within a component, so cross-component pairs are excluded by
    construction rather than scored as failures).

    Grouped by source so the caller can run one
    nx.single_source_dijkstra_path_length per unique source instead of one
    independent Dijkstra call per pair.
    """
    components = [sorted(c) for c in nx.connected_components(gt_graph) if len(c) >= 2]
    if not components:
        return []

    rng = np.random.default_rng(seed)
    sizes = np.array([len(c) for c in components], dtype=float)
    weights = sizes / sizes.sum()
    n_sources = max(1, int(np.ceil(np.sqrt(n_pairs))))
    targets_per_source = max(1, int(np.ceil(n_pairs / n_sources)))
    total_nodes = sum(len(c) for c in components)

    grouped, used_sources, total, attempts = [], set(), 0, 0
    max_attempts = n_sources * 20 + 200
    while total < n_pairs and attempts < max_attempts and len(used_sources) < total_nodes:
        attempts += 1
        comp = components[rng.choice(len(components), p=weights)]
        source = comp[rng.integers(len(comp))]
        if source in used_sources:
            continue
        used_sources.add(source)
        remaining = [n for n in comp if n != source]
        k = min(targets_per_source, len(remaining), n_pairs - total)
        if k <= 0:
            continue
        idx = rng.choice(len(remaining), size=k, replace=False)
        targets = [remaining[i] for i in np.atleast_1d(idx)]
        grouped.append((source, targets))
        total += len(targets)
    return grouped


def _nearest_within_tolerance(tree, nodes, point, tolerance):
    """Nearest node to `point` if within `tolerance` px, else None."""
    if tree is None:
        return None
    dist, idx = tree.query(point)
    return nodes[idx] if dist <= tolerance else None


def apls(
    pred: np.ndarray,
    gt: np.ndarray,
    tolerance: float,
    n_pairs: int = DEFAULT_N_PAIRS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Raster-skeleton APLS approximation between a predicted and GT binary mask.

    Parameters
    ----------
    pred, gt   : H×W binary or integer array (>0 = foreground)
    tolerance  : snap-matching radius (px) for matching a GT skeleton node to
                 its nearest pred skeleton node -- reuse that class's DTAF1
                 tolerance.
    n_pairs    : control-point pair budget
    seed       : RNG seed for deterministic control-point sampling

    Returns
    -------
    dict with keys:
        "apls"             : scalar score in [0, 1]
        "n_pairs_sampled"  : number of GT control-point pairs evaluated
        "n_pairs_matched"  : of those, how many had both endpoints matched
                              to a connected pred path
        "n_gt_components"  : number of connected components in the GT skeleton

    Edge cases (mirrors dtaf1._class_dtf1 / cldice.cldice):
        both empty              -> 1.0
        gt empty, pred nonempty -> 0.0
        gt nonempty, pred empty -> 0.0
    """
    p, g = (pred > 0), (gt > 0)
    n_pred, n_gt = int(p.sum()), int(g.sum())

    if n_pred == 0 and n_gt == 0:
        return {"apls": 1.0, "n_pairs_sampled": 0, "n_pairs_matched": 0, "n_gt_components": 0}
    if n_pred == 0 or n_gt == 0:
        return {"apls": 0.0, "n_pairs_sampled": 0, "n_pairs_matched": 0, "n_gt_components": 0}

    gt_graph = _skeleton_graph(g)
    pred_graph = _skeleton_graph(p)
    n_gt_components = nx.number_connected_components(gt_graph)

    pred_nodes = list(pred_graph.nodes)
    pred_tree = cKDTree(np.asarray(pred_nodes)) if pred_nodes else None

    grouped_pairs = _sample_control_pairs(gt_graph, n_pairs, seed)

    if not grouped_pairs:
        # GT skeleton has no connected component with >=2 nodes (e.g. it
        # reduces to isolated single-pixel specks) -- there's no shortest-path
        # topology to compare. Fall back to a per-node tolerance-radius
        # presence fraction (the same tolerance-radius idea DTAF1 itself uses
        # at the pixel level) rather than an arbitrary constant.
        gt_nodes = list(gt_graph.nodes)
        matched = sum(
            1 for node in gt_nodes
            if _nearest_within_tolerance(pred_tree, pred_nodes, node, tolerance) is not None
        )
        score = matched / len(gt_nodes) if gt_nodes else 0.0
        return {"apls": float(score), "n_pairs_sampled": 0, "n_pairs_matched": 0,
                "n_gt_components": n_gt_components}

    scores, n_matched, n_sampled = [], 0, 0
    for source, targets in grouped_pairs:
        gt_dist = nx.single_source_dijkstra_path_length(gt_graph, source, weight="weight")
        pu = _nearest_within_tolerance(pred_tree, pred_nodes, source, tolerance)
        pred_dist = (nx.single_source_dijkstra_path_length(pred_graph, pu, weight="weight")
                     if pu is not None else None)
        for target in targets:
            n_sampled += 1
            gt_len = gt_dist[target]
            pv = _nearest_within_tolerance(pred_tree, pred_nodes, target, tolerance)
            if pred_dist is None or pv is None or pv not in pred_dist:
                scores.append(0.0)
                continue
            pred_len = pred_dist[pv]
            n_matched += 1
            scores.append(max(0.0, 1.0 - abs(pred_len - gt_len) / gt_len) if gt_len > 0 else 0.0)

    return {
        "apls": float(np.mean(scores)) if scores else 0.0,
        "n_pairs_sampled": n_sampled,
        "n_pairs_matched": n_matched,
        "n_gt_components": n_gt_components,
    }


def apls_multiclass(
    pred: np.ndarray,
    gt: np.ndarray,
    linear_classes: list,
    tolerances: dict,
    n_pairs: int = DEFAULT_N_PAIRS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """
    Apply apls() to selected classes in a multiclass label map.

    Parameters
    ----------
    pred, gt       : H×W integer label maps
    linear_classes : list of integer class IDs to evaluate with APLS
    tolerances     : {class_id: tolerance}

    Returns
    -------
    dict {class_id: apls_result_dict}, each result dict additionally carries
    "n_gt" (GT pixel count for that class) for pixel-area-weighted aggregation.
    """
    results = {}
    for cls in linear_classes:
        gt_mask = (gt == cls)
        result = apls(pred == cls, gt_mask, tolerances[cls], n_pairs=n_pairs, seed=seed)
        result["n_gt"] = int(gt_mask.sum())
        results[cls] = result
    return results


def mean_apls(
    pred: np.ndarray,
    gt: np.ndarray,
    linear_classes: list,
    tolerances: dict,
    reduction: str = "macro",
    n_pairs: int = DEFAULT_N_PAIRS,
    seed: int = DEFAULT_SEED,
) -> float:
    """
    Average APLS over the specified linear classes.

    reduction : "macro"    — unweighted mean of per-class APLS scores
                "weighted" — weight by number of GT pixels per class
    """
    per_class = apls_multiclass(pred, gt, linear_classes, tolerances, n_pairs, seed)
    scores = [v["apls"] for v in per_class.values()]
    if not scores:
        return 0.0
    if reduction == "macro":
        return float(np.mean(scores))
    elif reduction == "weighted":
        weights = np.array([v["n_gt"] for v in per_class.values()], dtype=float)
        total = weights.sum()
        return float(np.dot(weights, scores) / total) if total > 0 else 0.0
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}. Use 'macro' or 'weighted'.")
