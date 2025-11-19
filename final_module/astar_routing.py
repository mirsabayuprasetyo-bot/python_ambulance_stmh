# astar_routing.py
import heapq
import math
from functools import lru_cache
from typing import Tuple, List, Optional, Any
import networkx as nx

class AStarRouter:
    """
    A* router for NetworkX / OSMnx graphs.
    - Instantiate with a graph: router = AStarRouter(G, weight='time_per_edge')
    - Use router.shortest_path(u, v) -> returns list of node ids (path)
    - Uses heuristic = euclidean (great-circle would be fine too), uses node attrs 'y' and 'x' if present.
    """

    def __init__(self, G: nx.Graph, weight: str = "time_per_edge"):
        self.G = G
        self.weight = weight
        # Use an undirected view for connectivity tests on directed graphs
        self._G_undirected = G.to_undirected(as_view=True)

    def _heuristic(self, u: Any, v: Any) -> float:
        """Euclidean distance in lat/lon degrees converted to meters approximated by haversine-ish small-dist."""
        # if coordinates are missing, fallback to 0
        lat1 = self.G.nodes[u].get("y", None)
        lon1 = self.G.nodes[u].get("x", None)
        lat2 = self.G.nodes[v].get("y", None)
        lon2 = self.G.nodes[v].get("x", None)
        if None in (lat1, lon1, lat2, lon2):
            return 0.0
        # approximate meters via simple haversine
        R = 6371000.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2.0)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    def _edge_cost(self, u: Any, v: Any) -> float:
        """
        Return minimal edge weight between u and v (works with MultiDiGraph).
        Falls back to 'length' if desired weight is missing.
        """
        try:
            data = self.G.get_edge_data(u, v)
            if data is None:
                return float("inf")
            # Multi edge case: data is dict keyed by keys; pick smallest weight
            if isinstance(data, dict):
                best = float("inf")
                # if graph is MultiGraph, items are keyed by key->attr dict
                for key, attr in data.items():
                    if isinstance(attr, dict):
                        val = attr.get(self.weight, attr.get("length", None))
                    else:
                        val = None
                    if val is None:
                        continue
                    try:
                        fv = float(val)
                    except Exception:
                        fv = float("inf")
                    if fv < best:
                        best = fv
                return best if best != float("inf") else (list(data.values())[0].get("length", float("inf")) if len(data)>0 else float("inf"))
            else:
                # single edge dict
                return float(data.get(self.weight, data.get("length", float("inf"))))
        except Exception:
            return float("inf")

    def shortest_path(self, source: Any, target: Any, return_cost: bool=False) -> Optional[List[Any]]:
        """
        A* search on the provided graph.
        Returns list of nodes OR (path, cost) if return_cost=True.
        If no path, returns None (or (None, inf) if return_cost=True).
        """
        if source == target:
            if return_cost:
                return [source], 0.0
            return [source]

        if not (source in self.G and target in self.G):
            if return_cost:
                return None, float("inf")
            return None

        # quick connectivity check (undirected)
        if not nx.has_path(self._G_undirected, source, target):
            if return_cost:
                return None, float("inf")
            return None

        open_heap = []
        gscore = {source: 0.0}
        fscore = {source: self._heuristic(source, target)}
        heapq.heappush(open_heap, (fscore[source], source))
        came_from = {}
        closed = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == target:
                # rebuild
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                total_cost = gscore[target]
                if return_cost:
                    return path, total_cost
                return path

            if current in closed:
                continue
            closed.add(current)

            for nbr in self.G.neighbors(current):
                # compute best cost for edge current->nbr
                step_cost = self._edge_cost(current, nbr)
                if step_cost == float("inf"):
                    continue
                tentative = gscore[current] + step_cost
                if tentative < gscore.get(nbr, float("inf")):
                    came_from[nbr] = current
                    gscore[nbr] = tentative
                    fscore[nbr] = tentative + self._heuristic(nbr, target)
                    heapq.heappush(open_heap, (fscore[nbr], nbr))

        # no path found
        if return_cost:
            return None, float("inf")
        return None


# Simple cached convenience wrapper (caches by graph id + node pair)
# Warning: caching assumes graph topology & weights are stable between calls.
@lru_cache(maxsize=20000)
def astar_cached_tuple(graph_id: int, source: Any, target: Any, weight: str = "time_per_edge") -> Tuple[Optional[Tuple[Any,...]], float]:
    """
    Low-level cached call. graph_id is an integer id representing the graph instance (use id(G)).
    Returns (tuple(path_nodes), cost) or (None, inf)
    """
    # This function is designed to be called by caller wrapper below which passes an actual graph and calls router.
    return None, float("inf")


def astar_shortest_path(G: nx.Graph, source: Any, target: Any, weight: str = "time_per_edge") -> Tuple[Optional[List[Any]], float]:
    """
    Convenience function: create router and run shortest path, returning (path_list, cost).
    """
    router = AStarRouter(G, weight=weight)
    path_cost = router.shortest_path(source, target, return_cost=True)
    if path_cost is None:
        return None, float("inf")
    return path_cost  # (path, cost)
