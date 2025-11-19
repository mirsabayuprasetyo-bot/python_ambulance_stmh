from __future__ import annotations
import random
import time
from functools import lru_cache
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union
import networkx as nx
import pygad
from datetime import datetime

"""
Genetic algorithm based routing using pygad with a NetworkX/OSMnx-like interface.

- Accepts a NetworkX graph (e.g., OSMnx MultiDiGraph), start node, end node.
- Returns a list of node IDs forming the path, similar to osmnx.shortest_path.
- Uses a GA to pick intermediate waypoints; segments between waypoints are connected
    via NetworkX shortest paths to ensure feasibility.
"""


WeightType = Union[str, Callable[[Any, Any, dict], float], None]


def _edge_weight(
        G: nx.Graph, u: Any, v: Any, weight: WeightType, default: float = 1.0
) -> float:
        """Return the weight of the lightest edge between u and v."""
        if isinstance(G, (nx.MultiDiGraph, nx.MultiGraph)):
                data_dict = G.get_edge_data(u, v, default={})
                if not data_dict:
                        # No edge between u and v.
                        return float("inf")
                return data_dict.get(weight,default)


def _path_length(G: nx.Graph, path: Sequence[Any], weight: WeightType) -> float:
        """Sum of edge weights along a path (choose min parallel edge weight if multigraph)."""
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
        GA-based router that mimics osmnx.shortest_path behavior.

        Basic usage:
                router = GeneticRouter(G, start, end, weight="length")
                path = router.solve()  # returns list of nodes
        """

        def __init__(
                self,
                G: nx.Graph,
                start: Any,
                end: Any,
                weight: WeightType = "length",
                # GA topology and budget
                population_size: int = 40,
                num_generations: int = 80,
                num_parents_mating: int = 12,
                mutation_probability: float = 0.15,
                crossover_type: str = "single_point",
                selection_type: str = "sss",
                # Waypoint model
                num_waypoints: int = 6,
                candidate_pool_size: int = 1200,
                include_start_end_neighbors: bool = True,
                allow_revisit: bool = False,
                # Termination/penalties
                time_limit: Optional[float] = None,  # seconds
                penalty_no_path: float = 1e9,
                random_seed: Optional[int] = None,
                # Seeding with NX path
                seed_with_nx: bool = True,
        ):
                if start not in G or end not in G:
                        raise nx.NetworkXError("Start or end node not in graph.")

                self.G = G
                self.start = start
                self.end = end
                self.weight = weight
                self.population_size = int(population_size)
                self.num_generations = int(num_generations)
                self.num_parents_mating = int(num_parents_mating)
                self.mutation_probability = float(mutation_probability)
                self.crossover_type = crossover_type
                self.selection_type = selection_type
                self.num_waypoints = int(num_waypoints)
                self.candidate_pool_size = int(candidate_pool_size)
                self.include_start_end_neighbors = include_start_end_neighbors
                self.allow_revisit = bool(allow_revisit)
                self.time_limit = time_limit
                self.penalty_no_path = float(penalty_no_path)
                self.random_seed = random_seed
                self.seed_with_nx = bool(seed_with_nx)
                self.path_routing = []

                if self.random_seed is not None:
                        random.seed(self.random_seed)

                # Undirected view ensures checking weak connectivity on directed graphs.
                self._G_undirected = G.to_undirected(as_view=True)

                if not nx.has_path(self._G_undirected, start, end):
                        raise nx.NetworkXNoPath("Start and end are disconnected.")

                # Build candidate waypoint pool from the connected component of start/end.
                self._candidates = self._build_candidate_pool()

                # Gene space includes None to allow variable-length waypoint lists.
                self._gene_space = list(self._candidates) + [None]

                # Cache for segment shortest paths to speed up fitness evaluation.
                self._segment_cache_size = 200_000
                self._setup_segment_cache()

                # Precompute a baseline NX shortest path to optionally seed and to gauge progress.
                self._nx_baseline_path: Optional[List[Any]] = None
                self._nx_baseline_cost: Optional[float] = None
                try:
                        self._nx_baseline_path = nx.shortest_path(
                                self.G, self.start, self.end, weight=self.weight
                        )
                        self._nx_baseline_cost = _path_length(self.G, self._nx_baseline_path, self.weight)
                except Exception:
                        # Graph may be large or weight weird; it's fine to proceed without baseline.
                        self._nx_baseline_path = None
                        self._nx_baseline_cost = None

                self._best_solution: Optional[List[Any]] = None
                self._best_cost: float = float("inf")

        def _setup_segment_cache(self) -> None:
                @lru_cache(maxsize=self._segment_cache_size)
                def _seg(u: Any, v: Any) -> Tuple[Tuple[Any, ...], float]:
                        # Always use weighted shortest path between two nodes, return tuple for cache.
                        p = nx.shortest_path(self.G, u, v, weight=self.weight)
                        cost = _path_length(self.G, p, self.weight)
                        return tuple(p), float(cost)

                self._segment_path_cached = _seg

        def _build_candidate_pool(self) -> List[Any]:
                # Restrict to nodes in the weakly-connected component containing start/end.
                comp_nodes = nx.node_connected_component(self._G_undirected, self.start)
                pool: List[Any] = list(comp_nodes)

                # Avoid start and end in the candidate pool (we add via genes as None or existing step).
                pool = [n for n in pool if n != self.start and n != self.end]

                # Optionally bias by including neighbors of start/end first.
                biased: List[Any] = []
                if self.include_start_end_neighbors:
                        for node in (self.start, self.end):
                                try:
                                        neighbors = list(self._G_undirected.neighbors(node))
                                except Exception:
                                        neighbors = []
                                for nb in neighbors:
                                        if nb != self.start and nb != self.end:
                                                biased.append(nb)

                # Deduplicate while preserving bias order.
                seen = set()
                ordered: List[Any] = []
                for n in biased + pool:
                        if n not in seen:
                                seen.add(n)
                                ordered.append(n)

                if self.candidate_pool_size and len(ordered) > self.candidate_pool_size:
                        # Keep a biased head + random sample of the tail.
                        head = ordered[: min(200, len(ordered))]
                        tail = ordered[len(head) :]
                        k = max(0, self.candidate_pool_size - len(head))
                        sampled = random.sample(tail, k) if k > 0 and len(tail) > 0 else []
                        ordered = head + sampled

                if not ordered:
                        # As a fallback, include immediate neighbors or leave empty and rely on None genes.
                        ordered = list(self._G_undirected.neighbors(self.start))

                self.path_routing.append(
                       {
                              "time" : datetime.now(),
                              "node" : ordered,
                       }
                )

                return ordered

        def _make_initial_population(self) -> Optional[List[List[Any]]]:
                if not self.seed_with_nx or self._nx_baseline_path is None:
                        return None

                # Split baseline path into waypoints to seed a strong chromosome.
                path = self._nx_baseline_path
                if len(path) <= 2 or self.num_waypoints <= 0:
                        # Only start/end; no waypoints to seed.
                        # Use -1 instead of None as placeholder
                        seed = [-1] * self.num_waypoints
                        # Create multiple random variations to fill population
                        initial_pop = [seed]
                        
                        # Add random variations if we have candidates
                        if self._candidates and len(self._candidates) > 0:
                            for _ in range(min(self.population_size - 1, 9)):  # Add up to 9 more variations
                                random_chromosome = []
                                for _ in range(self.num_waypoints):
                                    # 50% chance of no waypoint (-1), 50% chance of random candidate
                                    if random.random() < 0.5:
                                        random_chromosome.append(-1)
                                    else:
                                        random_chromosome.append(random.randint(0, len(self._candidates) - 1))
                                initial_pop.append(random_chromosome)
                        
                        return initial_pop

                # Evenly spaced intermediate nodes as waypoints (avoid start/end).
                internal = path[1:-1]
                if len(internal) <= self.num_waypoints:
                        # Convert waypoints to indices in candidate list
                        waypoint_indices = []
                        for wp in internal:
                                try:
                                        idx = self._candidates.index(wp)
                                        waypoint_indices.append(idx)
                                except ValueError:
                                        # If waypoint not in candidates, use -1 as placeholder
                                        waypoint_indices.append(-1)
                        
                        # Pad with -1 if needed
                        waypoint_indices += [-1] * (self.num_waypoints - len(waypoint_indices))
                else:
                        # Sample indices evenly
                        idxs = [int(round(i * (len(internal) - 1) / (self.num_waypoints - 1))) for i in range(self.num_waypoints)]
                        waypoint_indices = []
                        for i in idxs:
                                wp = internal[i]
                                try:
                                        idx = self._candidates.index(wp)
                                        waypoint_indices.append(idx)
                                except ValueError:
                                        waypoint_indices.append(-1)

                # Create initial population with the seed plus random variations
                initial_pop = [waypoint_indices]
                
                # Add random variations
                if self._candidates and len(self._candidates) > 0:
                    for _ in range(min(self.population_size - 1, 9)):
                        random_chromosome = []
                        for _ in range(self.num_waypoints):
                            if random.random() < 0.5:  # 50% chance of no waypoint
                                random_chromosome.append(-1)
                            else:
                                random_chromosome.append(random.randint(0, len(self._candidates) - 1))
                        initial_pop.append(random_chromosome)
                
                return initial_pop

        def _decode_solution(self, genes: Sequence[Any]) -> Tuple[List[Any], float, bool]:
                """Return (full_path_nodes, total_cost, feasible) for a chromosome."""
                current = self.start
                full_path: List[Any] = [self.start]
                feasible = True
                total_cost = 0.0

                for g in genes:
                        # Convert gene to actual waypoint
                        if isinstance(g, (int, float)):
                                g_int = int(g)
                        else:
                                g_int = g
                        
                        if g_int == -1 or g_int < 0:  # No waypoint
                                continue
                        
                        if g_int >= len(self._candidates):  # Invalid index
                                continue
                        
                        waypoint = self._candidates[g_int]
                        
                        if waypoint == current:
                                continue
                        
                        try:
                                seg_nodes_tuple, seg_cost = self._segment_path_cached(current, waypoint)
                        except nx.NetworkXNoPath:
                                total_cost += self.penalty_no_path
                                feasible = False
                                return full_path, total_cost, feasible
                        # Append segment except the first node to avoid duplicates.
                        full_path.extend(list(seg_nodes_tuple)[1:])
                        total_cost += seg_cost
                        current = waypoint

                # Connect the last point to the end
                try:
                        seg_nodes_tuple, seg_cost = self._segment_path_cached(current, self.end)
                except nx.NetworkXNoPath:
                        total_cost += self.penalty_no_path
                        feasible = False
                        return full_path, total_cost, feasible

                full_path.extend(list(seg_nodes_tuple)[1:])
                total_cost += seg_cost

                if not self.allow_revisit:
                        dup_count = len(full_path) - len(set(full_path))
                        if dup_count > 0:
                                # Mild penalty for loops/revisits
                                total_cost *= (1.0 + 0.05 * dup_count)

                return full_path, total_cost, feasible

        
        def _fitness_func(self, ga_inst: "pygad.GA", solution: Sequence[Any], solution_idx: int) -> float:
                # solution is a numpy array possibly with dtype=object.
                genes = [g.item() if hasattr(g, "item") else g for g in solution]
                path, cost, feasible = self._decode_solution(genes)
                # Track the best encountered.
                if cost < self._best_cost:
                        self._best_cost = cost
                        self._best_solution = path
                # Convert cost to fitness (maximize).
                if not feasible:
                        return 1.0 / (1.0 + cost)
                return 1.0 / (1.0 + max(cost, 0.0))

        def solve(self) -> List[Any]:
                """Run the GA and return the best path as a list of node IDs."""
                start_time = time.time()
                initial_population = self._make_initial_population()

                # Build per-gene gene_space to allow -1 (no waypoint) at any position.
                if self.num_waypoints <= 0:
                        # No intermediate waypoints: just return NX shortest path for consistency.
                        path = nx.shortest_path(self.G, self.start, self.end, weight=self.weight)
                        return list(path)

                # Use indices instead of None values
                # -1 represents "no waypoint", 0 to len(candidates)-1 are valid waypoints
                gene_space = list(range(-1, len(self._candidates))) if self._candidates else [-1]
                gene_space_per_gene = [gene_space for _ in range(self.num_waypoints)]

                # Ensure num_parents_mating is valid
                effective_population_size = self.population_size
                if initial_population:
                    effective_population_size = max(self.population_size, len(initial_population))
                
                # Adjust num_parents_mating to be at most population_size
                effective_parents_mating = min(self.num_parents_mating, effective_population_size)
                
                # Ensure minimum viable GA parameters
                if effective_parents_mating < 2:
                    effective_parents_mating = min(2, effective_population_size)
                
                ga = pygad.GA(
                        num_generations=self.num_generations,
                        num_parents_mating=effective_parents_mating,
                        fitness_func=self._fitness_func,
                        sol_per_pop=effective_population_size,
                        num_genes=self.num_waypoints,
                        gene_space=gene_space_per_gene,
                        gene_type=int,  # Ensure integer genes
                        mutation_type="random",
                        mutation_probability=self.mutation_probability,
                        crossover_type=self.crossover_type,
                        parent_selection_type=self.selection_type,
                        initial_population=initial_population,
                        random_seed=self.random_seed,
                        keep_parents=2,
                )

                def _on_generation(ga_inst: "pygad.GA"):
                        # Early termination by time
                        if self.time_limit is not None and (time.time() - start_time) >= self.time_limit:
                                ga_inst.stop_generation = True
                                return
                        # If we have a known NX baseline and we've matched (or beaten) it, we can stop.
                        if self._nx_baseline_cost is not None and self._best_cost <= self._nx_baseline_cost:
                                ga_inst.stop_generation = True

                ga.on_generation = _on_generation

                ga.run()
                
                # If GA found something, return it; else fall back to NX shortest path.
                if self._best_solution is not None and len(self._best_solution) >= 2:
                        return list(self._best_solution)

                # Fallback
                print("GA did not find a valid path; falling back to NetworkX shortest path.")
                path = nx.shortest_path(self.G, self.start, self.end, weight=self.weight)
                return list(path)
        
        def get_simulation_records(self) -> List[dict]:
                """Return simulation records collected during the GA run."""
                # Placeholder: In a real implementation, this would return actual records.
                return self.path_routing


def ga_shortest_path(
        G: nx.Graph,
        orig: Any,
        dest: Any,
        weight: WeightType = "length",
        population_size: int = 40,
        num_generations: int = 80,
        num_parents_mating: int = 4,
        mutation_probability: float = 0.15,
        num_waypoints: int = 6,
        candidate_pool_size: int = 1200,
        include_start_end_neighbors: bool = True,
        allow_revisit: bool = False,
        time_limit: Optional[float] = None,
        random_seed: Optional[int] = None,
        seed_with_nx: bool = True,
) -> List[Any]:
        """
        Convenience function mirroring osmnx.shortest_path(G, orig, dest, weight=...).

        Returns a list of node IDs representing the path from orig to dest.
        """
        router = GeneticRouter(
                G=G,
                start=orig,
                end=dest,
                weight=weight,
                population_size=population_size,
                num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                mutation_probability=mutation_probability,
                num_waypoints=num_waypoints,
                candidate_pool_size=candidate_pool_size,
                include_start_end_neighbors=include_start_end_neighbors,
                allow_revisit=allow_revisit,
                time_limit=time_limit,
                random_seed=random_seed,
                seed_with_nx=seed_with_nx,
        )
        return router.solve()


# Alias similar to osmnx.shortest_path for drop-in familiarity.
def shortest_path(G: nx.Graph, orig: Any, dest: Any, weight: WeightType = "length", **kwargs) -> List[Any]:
        return ga_shortest_path(G, orig, dest, weight=weight, **kwargs)