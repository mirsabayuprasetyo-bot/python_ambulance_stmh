from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import hypot
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union
import networkx as nx

# astar_routing.py
# AStarRouter: mimic osmnx.shortest_path behavior but force A* search.
# - Supports single or paired lists of origins/destinations
# - Respects an edge weight attribute (default: "length")
# - Uses Euclidean heuristic based on node 'x'/'y' (if available), else falls back to Dijkstra (zero heuristic)
# - Optional lightweight parallelism with threads via cpus>1





Node = Hashable
Path = Optional[List[Node]]
WeightFunc = Callable[[Node, Node, Dict[str, Any]], float]
HeuristicFunc = Callable[[Node, Node], float]


class AStarRouter:
    def __init__(self, default_weight: Union[str, WeightFunc] = "length", default_heuristic: Union[str, HeuristicFunc, None] = "euclidean"):
        self.default_weight = default_weight
        self.default_heuristic = default_heuristic

    def shortest_path(
        self,
        G: nx.Graph,
        orig: Union[Node, Sequence[Node]],
        dest: Union[Node, Sequence[Node]],
        weight: Optional[Union[str, WeightFunc]] = None,
        heuristic: Optional[Union[str, HeuristicFunc]] = None,
        cpus: Optional[int] = 1,
    ) -> Union[Path, List[Path]]:
        """
        Compute shortest path(s) using A* between origin(s) and destination(s).

        Parameters:
        - G: NetworkX graph (DiGraph/MultiDiGraph) with edge weight attributes
             and, optionally, node attributes 'x' and 'y' for heuristic.
        - orig, dest: node id or sequence of node ids. If both are sequences:
            * lengths equal -> pairwise routes
            * one length is 1 -> broadcast to the other
        - weight: edge weight attribute name or callable (u, v, edge_data) -> cost. Defaults to "length".
        - heuristic: "euclidean", None/"none" (zero heuristic), or callable (u, v) -> estimate.
        - cpus: if >1, routes are computed in parallel using threads.

        Returns:
        - Single path (list of nodes) if inputs are singletons
        - Otherwise, a list of paths aligned to origin-destination pairs
        - Returns None for a pair if no path exists
        """
        weight = weight if weight is not None else self.default_weight
        heuristic = heuristic if heuristic is not None else self.default_heuristic

        od_pairs, single = self._normalize_pairs(orig, dest)

        # Precompute node coordinates once for heuristic
        node_xy = self._extract_node_xy(G)

        # Resolve functions
        weight_fn = self._resolve_weight_function(weight)
        heuristic_fn = self._resolve_heuristic_function(heuristic, node_xy)

        # Route computation
        if len(od_pairs) == 0:
            result: List[Path] = []
        elif cpus is None or cpus <= 1 or len(od_pairs) == 1:
            result = [self._astar_route(G, o, d, weight_fn, heuristic_fn) for o, d in od_pairs]
        else:
            result = [None] * len(od_pairs)
            with ThreadPoolExecutor(max_workers=cpus) as ex:
                futures = {ex.submit(self._astar_route, G, o, d, weight_fn, heuristic_fn): idx for idx, (o, d) in enumerate(od_pairs)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        result[idx] = fut.result()
                    except Exception:
                        result[idx] = None

        return result[0] if single else result

    # ----------------------
    # Internals
    # ----------------------

    def _is_sequence_of_nodes(self, x: Any) -> bool:
        if isinstance(x, (str, bytes)):
            return False
        return isinstance(x, Sequence)

    def _normalize_pairs(self, orig: Union[Node, Sequence[Node]], dest: Union[Node, Sequence[Node]]) -> Tuple[List[Tuple[Node, Node]], bool]:
        """
        Normalize origin/destination inputs to a list of (o, d) pairs
        and indicate whether the original inputs were both singletons.
        """
        orig_is_seq = self._is_sequence_of_nodes(orig)
        dest_is_seq = self._is_sequence_of_nodes(dest)

        if not orig_is_seq and not dest_is_seq:
            return [(orig, dest)], True

        if orig_is_seq and dest_is_seq:
            if len(orig) == len(dest):
                return list(zip(orig, dest)), False
            if len(orig) == 1 and len(dest) >= 1:
                return [(orig[0], d) for d in dest], False
            if len(dest) == 1 and len(orig) >= 1:
                return [(o, dest[0]) for o in orig], False
            raise ValueError("orig and dest sequences must have the same length or one of them must be length 1")

        if orig_is_seq and not dest_is_seq:
            return [(o, dest) for o in orig], False

        if not orig_is_seq and dest_is_seq:
            return [(orig, d) for d in dest], False

        # Should not reach here
        raise ValueError("Invalid orig/dest types")

    def _extract_node_xy(self, G: nx.Graph) -> Dict[Node, Optional[Tuple[float, float]]]:
        """
        Return mapping node -> (x, y) if available, else None.
        """
        node_xy: Dict[Node, Optional[Tuple[float, float]]] = {}
        for n, data in G.nodes(data=True):
            x = data.get("x", None)
            y = data.get("y", None)
            if x is None or y is None:
                node_xy[n] = None
            else:
                # Ensure floats
                try:
                    node_xy[n] = (float(x), float(y))
                except Exception:
                    node_xy[n] = None
        return node_xy

    def _resolve_weight_function(self, weight: Union[str, WeightFunc]) -> WeightFunc:
        """
        Convert weight parameter into a weight function weight(u, v, edge_data) -> float.

        For Multi(Di)Graph, edge_data provided by networkx.astar_path for each neighbor
        is typically the attribute dict of a single edge (not the key->attr mapping).
        We still defensively handle dict-of-dicts by taking the minimum available weight.
        """
        if callable(weight):
            return weight

        attr = str(weight)

        def weight_fn(u: Node, v: Node, data: Dict[str, Any]) -> float:
            # Fast path: standard edge attribute dict
            if attr in data:
                try:
                    return float(data.get(attr, 1.0))
                except Exception:
                    return 1.0

            # Defensive path: MultiEdge dict-of-dicts
            if isinstance(data, dict) and any(isinstance(val, dict) for val in data.values()):
                try:
                    vals = [float(d.get(attr, 1.0)) for d in data.values() if isinstance(d, dict)]
                    return min(vals) if vals else 1.0
                except Exception:
                    return 1.0

            return 1.0

        return weight_fn

    def _resolve_heuristic_function(
        self,
        heuristic: Union[str, HeuristicFunc, None],
        node_xy: Dict[Node, Optional[Tuple[float, float]]],
    ) -> HeuristicFunc:
        if heuristic is None:
            return lambda u, v: 0.0

        if isinstance(heuristic, str):
            name = heuristic.lower().strip()
            if name in ("none", "zero", "dijkstra"):
                return lambda u, v: 0.0
            if name in ("euclid", "euclidean", "manhattan"):  # support minor aliases; we still use Euclidean
                def h(u: Node, v: Node) -> float:
                    xu_yu = node_xy.get(u)
                    xv_yv = node_xy.get(v)
                    if not xu_yu or not xv_yv:
                        return 0.0
                    # Planar euclidean in same units as node x/y (usually degrees).
                    # Works as an admissible heuristic if edge weights are proportional to geometric distance.
                    return hypot(xu_yu[0] - xv_yv[0], xu_yu[1] - xv_yv[1])
                return h
            # Unknown string -> default to zero heuristic
            return lambda u, v: 0.0

        if callable(heuristic):
            return heuristic

        # Fallback
        return lambda u, v: 0.0

    def _astar_route(
        self,
        G: nx.Graph,
        o: Node,
        d: Node,
        weight_fn: WeightFunc,
        heuristic_fn: HeuristicFunc,
    ) -> Path:
        try:
            return nx.astar_path(G, o, d, heuristic=heuristic_fn, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None


# Example usage:
# router = AStarRouter(default_weight="length", default_heuristic="euclidean")
# path = router.shortest_path(G, orig_node, dest_node)
# paths = router.shortest_path(G, [o1, o2], [d1, d2], cpus=4)