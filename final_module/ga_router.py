"""
ga_router.py

Pure-Python Genetic Algorithm router (no pygad dependency).

- Waypoint-based GA: chromosome = list of waypoint indices (integers) where -1 means "no waypoint".
- Segments between chosen waypoints (and start/end) are connected using NetworkX shortest_path with the provided weight.
- Designed as a drop-in replacement for previous pygad router.
- Public API:
    ga_shortest_path(G, orig, dest, weight='length', population_size=40, num_generations=80,
                     num_parents_mating=8, mutation_prob=0.15, num_waypoints=6,
                     candidate_pool_size=1200, include_start_end_neighbors=True,
                     allow_revisit=False, time_limit=None, random_seed=None, seed_with_nx=True)
    shortest_path(...)  # alias to ga_shortest_path

Notes:
- This implementation is conservative: it falls back to nx.shortest_path when GA fails.
- It uses lru_cache to cache segment shortest paths (speed).
- It expects a NetworkX graph (OSMnx MultiDiGraph) with edge weights accessible by `weight`.
"""

from __future__ import annotations
import random
import time
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple, Union
import networkx as nx

WeightType = Union[str, None]


def _edge_weight(G: nx.Graph, u: Any, v: Any, weight: WeightType, default: float = 1.0) -> float:
    """Return weight of the lightest edge between u and v (or default)."""
    # For MultiGraph/DiGraph the returned dict maps keys -> attr dicts
    try:
        data = G.get_edge_data(u, v, default={})
    except Exception:
        return float("inf")
    if not data:
        return float("inf")
    # if multigraph, data is a dict of keys -> attr dict
    if isinstance(data, dict):
        # choose minimum weight among parallel edges
        best = float("inf")
        for k, attr in data.items():
            if isinstance(attr, dict):
                w = attr.get(weight, attr.get("length", default))
            else:
                # fallback
                w = default
            try:
                wv = float(w)
            except Exception:
                wv = default
            if wv < best:
                best = wv
        return best
    else:
        # single edge data dict
        attr = data
        w = attr.get(weight, attr.get("length", default))
        try:
            return float(w)
        except Exception:
            return default


def _path_length(G: nx.Graph, path: Sequence[Any], weight: WeightType) -> float:
    """Sum of edge weights along a path. If any segment disconnected, returns inf."""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        w = _edge_weight(G, u, v, weight)
        if w == float("inf"):
            return float("inf")
        total += w
    return total


class GeneticRouter:
    """
    Waypoint-based Genetic Router (pure Python implementation).

    It selects a small set of candidate waypoints (from connected component),
    encodes a chromosome as `num_waypoints` integers in [-1 .. len(candidates)-1].
      - -1 means "no waypoint here"
      - otherwise integer is index into candidate list

    During fitness evaluation the chromosome is decoded into:
      origin -> waypoint1 -> ... -> destination
    with each segment computed by nx.shortest_path(weight=weight)
    """

    def __init__(
        self,
        G: nx.Graph,
        start: Any,
        end: Any,
        weight: WeightType = "length",
        population_size: int = 40,
        num_generations: int = 80,
        num_parents_mating: int = 8,
        mutation_prob: float = 0.15,
        crossover_rate: float = 0.9,
        num_waypoints: int = 6,
        candidate_pool_size: int = 1000,
        include_start_end_neighbors: bool = True,
        allow_revisit: bool = False,
        time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
        seed_with_nx: bool = True,
    ):
        if start not in G or end not in G:
            raise nx.NetworkXError("Start or end node not in graph.")
        self.G = G
        self.start = start
        self.end = end
        self.weight = weight
        self.population_size = max(4, int(population_size))
        self.num_generations = max(1, int(num_generations))
        self.num_parents_mating = max(2, int(num_parents_mating))
        self.mutation_prob = float(mutation_prob)
        self.crossover_rate = float(crossover_rate)
        self.num_waypoints = max(0, int(num_waypoints))
        self.candidate_pool_size = int(candidate_pool_size) if candidate_pool_size else 0
        self.include_start_end_neighbors = bool(include_start_end_neighbors)
        self.allow_revisit = bool(allow_revisit)
        self.time_limit = time_limit
        self.random_seed = random_seed
        self.seed_with_nx = bool(seed_with_nx)

        if self.random_seed is not None:
            random.seed(self.random_seed)

        # For directed graphs, check weak connectivity
        self._G_undirected = G.to_undirected(as_view=True)
        if not nx.has_path(self._G_undirected, start, end):
            raise nx.NetworkXNoPath("Start and end are disconnected in the weak sense.")

        # Build candidate pool
        self._candidates = self._build_candidate_pool()

        # Setup segment cache
        self._setup_segment_cache()

        # Precompute NX baseline path (for seeding and early stopping)
        try:
            self._nx_baseline_path = nx.shortest_path(self.G, self.start, self.end, weight=self.weight)
            self._nx_baseline_cost = _path_length(self.G, self._nx_baseline_path, self.weight)
        except Exception:
            self._nx_baseline_path = None
            self._nx_baseline_cost = None

        # best-so-far
        self._best_solution: Optional[List[Any]] = None
        self._best_cost = float("inf")

    def _setup_segment_cache(self):
        @lru_cache(maxsize=200_000)
        def _seg(u: Any, v: Any) -> Tuple[Tuple[Any, ...], float]:
            # compute weighted shortest path between u and v
            p = nx.shortest_path(self.G, u, v, weight=self.weight)
            cost = _path_length(self.G, p, self.weight)
            return tuple(p), float(cost)

        self._segment_path_cached = _seg

    def _build_candidate_pool(self) -> List[Any]:
        # nodes in connected component
        comp_nodes = list(nx.node_connected_component(self._G_undirected, self.start))
        # remove start and end
        pool = [n for n in comp_nodes if n != self.start and n != self.end]
        # optional bias: include neighbors of start/end first
        biased = []
        if self.include_start_end_neighbors:
            for node in (self.start, self.end):
                try:
                    neighbors = list(self._G_undirected.neighbors(node))
                except Exception:
                    neighbors = []
                for nb in neighbors:
                    if nb != self.start and nb != self.end and nb not in biased:
                        biased.append(nb)
        ordered = []
        seen = set()
        for n in biased + pool:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        # trim to candidate_pool_size
        if self.candidate_pool_size and len(ordered) > self.candidate_pool_size:
            head = ordered[: min(200, len(ordered))]
            tail = ordered[len(head):]
            k = max(0, self.candidate_pool_size - len(head))
            sampled = random.sample(tail, k) if k > 0 and len(tail) > 0 else []
            ordered = head + sampled
        if not ordered:
            # fallback: use neighbors of start
            try:
                ordered = list(self._G_undirected.neighbors(self.start))
            except Exception:
                ordered = []
        return ordered

    def _make_initial_population(self) -> List[List[int]]:
        # Represent genes as integers in [-1 .. len(candidates)-1]
        if self.num_waypoints <= 0:
            return []
        population = []
        # seed chromosome from nx baseline if available
        if self.seed_with_nx and self._nx_baseline_path and len(self._nx_baseline_path) > 2:
            internal = [n for n in self._nx_baseline_path[1:-1] if n in self._candidates]
            if internal:
                # convert to indices in candidates; pad with -1
                indices = []
                for wp in internal[: self.num_waypoints]:
                    try:
                        indices.append(self._candidates.index(wp))
                    except ValueError:
                        indices.append(-1)
                if len(indices) < self.num_waypoints:
                    indices += [-1] * (self.num_waypoints - len(indices))
                population.append(indices)
        # fill rest with random chromosomes
        for _ in range(max(0, self.population_size - len(population))):
            chrom = []
            for _ in range(self.num_waypoints):
                if random.random() < 0.5 or not self._candidates:
                    chrom.append(-1)
                else:
                    chrom.append(random.randrange(0, len(self._candidates)))
            population.append(chrom)
        return population

    def _decode_solution(self, genes: Sequence[int]) -> Tuple[List[Any], float, bool]:
        """Decode integer genes into full path (node list), compute cost; return (path, cost, feasible)."""
        current = self.start
        full_path = [self.start]
        feasible = True
        total_cost = 0.0
        # iterate genes
        for g in genes:
            if isinstance(g, (int,)) and g >= 0:
                if g >= len(self._candidates):
                    continue
                waypoint = self._candidates[g]
                if waypoint == current:
                    continue
                try:
                    seg_nodes, seg_cost = self._segment_path_cached(current, waypoint)
                except nx.NetworkXNoPath:
                    total_cost += 1e9
                    feasible = False
                    return full_path, total_cost, feasible
                # append segment except first node
                full_path.extend(list(seg_nodes)[1:])
                total_cost += seg_cost
                current = waypoint
            else:
                # -1 means skip
                continue
        # connect to destination
        try:
            seg_nodes, seg_cost = self._segment_path_cached(current, self.end)
        except nx.NetworkXNoPath:
            total_cost += 1e9
            feasible = False
            return full_path, total_cost, feasible
        full_path.extend(list(seg_nodes)[1:])
        total_cost += seg_cost

        if not self.allow_revisit:
            dup_count = len(full_path) - len(set(full_path))
            if dup_count > 0:
                total_cost *= (1.0 + 0.05 * dup_count)

        return full_path, total_cost, feasible

    def _fitness(self, genes: Sequence[int]) -> float:
        path, cost, feasible = self._decode_solution(genes)
        if cost < self._best_cost:
            self._best_cost = cost
            self._best_solution = path
        if not feasible:
            return 1.0 / (1.0 + cost)
        return 1.0 / (1.0 + max(cost, 0.0))

    # Simple GA operators (tournament selection, ordered crossover on indices, random mutation)
    def _tournament_select(self, population: List[List[int]], fitnesses: List[float], k: int = 3):
        best_idx = random.randrange(len(population))
        for _ in range(k - 1):
            cand = random.randrange(len(population))
            if fitnesses[cand] > fitnesses[best_idx]:
                best_idx = cand
        return list(population[best_idx])

    def _crossover(self, a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
        # uniform crossover
        n = len(a)
        if n <= 1:
            return list(a), list(b)
        if random.random() > self.crossover_rate:
            return list(a), list(b)
        # do two-point crossover
        i, j = sorted(random.sample(range(n), 2))
        ca = a[:i] + b[i:j] + a[j:]
        cb = b[:i] + a[i:j] + b[j:]
        return ca, cb

    def _mutate(self, chrom: List[int]) -> List[int]:
        n = len(chrom)
        if n == 0:
            return chrom
        # mutation: with mutation_prob, change gene to -1 or random candidate index
        for i in range(n):
            if random.random() < self.mutation_prob:
                if not self._candidates:
                    chrom[i] = -1
                else:
                    if random.random() < 0.5:
                        chrom[i] = -1
                    else:
                        chrom[i] = random.randrange(0, len(self._candidates))
        return chrom

    def solve(self) -> List[Any]:
        """Run GA and return best path (list of nodes)."""
        # if no waypoints desired, just return nx.shortest_path
        if self.num_waypoints <= 0:
            try:
                return list(nx.shortest_path(self.G, self.start, self.end, weight=self.weight))
            except Exception:
                return [self.start, self.end]

        start_time = time.time()
        population = self._make_initial_population()
        # ensure population_size
        while len(population) < self.population_size:
            chrom = []
            for _ in range(self.num_waypoints):
                if not self._candidates or random.random() < 0.5:
                    chrom.append(-1)
                else:
                    chrom.append(random.randrange(0, len(self._candidates)))
            population.append(chrom)

        # compute fitnesses
        fitnesses = [self._fitness(ch) for ch in population]

        for gen in range(self.num_generations):
            # optional time limit
            if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                break
            # elitism: keep top 1 or 2
            sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
            new_population = [population[sorted_idx[0]]]
            # produce children
            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitnesses, k=3)
                p2 = self._tournament_select(population, fitnesses, k=3)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.population_size:
                    new_population.append(c2)
            population = new_population
            fitnesses = [self._fitness(ch) for ch in population]
            # early stop if we beat nx baseline
            if self._nx_baseline_cost is not None and self._best_cost <= self._nx_baseline_cost:
                break

        # If best solution found and is valid (>=2 nodes), return it
        if self._best_solution is not None and len(self._best_solution) >= 2:
            return list(self._best_solution)

        # fallback: use NetworkX shortest path
        try:
            return list(nx.shortest_path(self.G, self.start, self.end, weight=self.weight))
        except Exception:
            # final fallback: direct nodes
            return [self.start, self.end]


def ga_shortest_path(
    G: nx.Graph,
    orig: Any,
    dest: Any,
    weight: WeightType = "length",
    population_size: int = 40,
    num_generations: int = 80,
    num_parents_mating: int = 8,
    mutation_prob: float = 0.15,
    num_waypoints: int = 6,
    candidate_pool_size: int = 1000,
    include_start_end_neighbors: bool = True,
    allow_revisit: bool = False,
    time_limit: Optional[float] = None,
    random_seed: Optional[int] = None,
    seed_with_nx: bool = True,
) -> List[Any]:
    """
    Convenience wrapper mirroring osmnx/previous prototype.

    Returns path as list of nodes from orig to dest.
    """
    router = GeneticRouter(
        G=G,
        start=orig,
        end=dest,
        weight=weight,
        population_size=population_size,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        mutation_prob=mutation_prob,
        num_waypoints=num_waypoints,
        candidate_pool_size=candidate_pool_size,
        include_start_end_neighbors=include_start_end_neighbors,
        allow_revisit=allow_revisit,
        time_limit=time_limit,
        random_seed=random_seed,
        seed_with_nx=seed_with_nx,
    )
    return router.solve()


def shortest_path(G: nx.Graph, orig: Any, dest: Any, weight: WeightType = "length", **kwargs) -> List[Any]:
    """Alias similar to osmnx.shortest_path for drop-in familiarity."""
    return ga_shortest_path(G, orig, dest, weight=weight, **kwargs)