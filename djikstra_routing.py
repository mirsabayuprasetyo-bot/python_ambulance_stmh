from __future__ import annotations
import heapq
from typing import Any, Dict, Iterable, List, Optional, Tuple
import osmnx as ox
import argparse

# djikstra_routing.py

class DijkstraRouter:
    """
    Compute shortest path on an OSMnx graph loaded from a GraphML file using Dijkstra's algorithm.

    Inputs mirror the common usage of osmnx.shortest_path(G, orig, dest, weight='length'),
    but this class loads the graph from a GraphML file path and runs its own Dijkstra.

    Parameters
    - graphml: Path to the .graphml file (as produced by osmnx.save_graphml)
    - start_node: Origin node id present in the graph
    - end_node: Destination node id present in the graph
    - weight: Edge attribute to minimize (default: 'length')

    Usage:
        router = DijkstraRouter("graph.graphml", start_node, end_node, weight="length")
        path = router.shortest_path()
    """

    def __init__(self, graphml: str, start_node: Any, end_node: Any, weight: str = "length", default_weight: float = 1.0):
        self.graphml = graphml
        self.start_node = start_node
        self.end_node = end_node
        self.weight = weight
        self.default_weight = float(default_weight)
        self.G = self._load_graph()

        if self.start_node not in self.G:
            raise KeyError(f"start_node {self.start_node!r} not found in graph")
        if self.end_node not in self.G:
            raise KeyError(f"end_node {self.end_node!r} not found in graph")

    def _load_graph(self):
        # ox.load_graphml preserves node ids/types and MultiDiGraph structure
        G = ox.load_graphml(self.graphml)
        # Ensure directed MultiDiGraph for correct routing semantics
        if not G.is_directed():
            G = G.to_directed()
        return G

    def _min_edge_weight(self, u: Any, v: Any) -> float:
        """
        For MultiDiGraph edges between u->v, return the minimal weight among parallel edges.
        """
        # self.G[u][v] is a dict: key -> edge_attrs
        edges_dict: Dict[Any, Dict[str, Any]] = self.G[u][v]  # type: ignore[index]
        best = None
        for _, data in edges_dict.items():
            w = data.get(self.weight, self.default_weight)
            try:
                w = float(w)
            except Exception:
                w = self.default_weight
            best = w if best is None else min(best, w)
        return best if best is not None else self.default_weight

    def _neighbors_with_weights(self, u: Any) -> Iterable[Tuple[Any, float]]:
        """
        Yield (v, w) for all successors v of u with the minimal parallel-edge weight.
        """
        for v in self.G.successors(u):
            yield v, self._min_edge_weight(u, v)

    def shortest_path(self) -> Optional[List[Any]]:
        """
        Run Dijkstra's algorithm and return the list of node ids from start_node to end_node.
        Returns None if no path exists. Matches osmnx.shortest_path() return shape.
        """
        if self.start_node == self.end_node:
            return [self.start_node]

        dist: Dict[Any, float] = {self.start_node: 0.0}
        prev: Dict[Any, Any] = {}
        pq: List[Tuple[float, Any]] = [(0.0, self.start_node)]
        visited: set = set()

        while pq:
            d_u, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == self.end_node:
                break

            for v, w_uv in self._neighbors_with_weights(u):
                if v in visited:
                    continue
                alt = d_u + w_uv
                if alt < dist.get(v, float("inf")):
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))

        if self.end_node not in prev and self.start_node != self.end_node:
            # Could still be reachable if end_node was directly discovered
            if self.end_node not in dist:
                return None

        # Reconstruct path
        path: List[Any] = []
        cur = self.end_node
        path.append(cur)
        while cur != self.start_node:
            if cur not in prev:
                # unreachable
                return None
            cur = prev[cur]
            path.append(cur)
        path.reverse()
        return path

    def path_length(self, path: List[Any]) -> float:
        """
        Compute the total weight along the given path using the same edge weight rule.
        """
        if not path or len(path) == 1:
            return 0.0
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += self._min_edge_weight(u, v)
        return total
