from typing import List, Optional, Dict, Any, Tuple
import random
import math

# ga_routing.py
# Genetic algorithm based path finder with a similar interface to osmnx.shortest_path
# Requires: networkx graph (e.g., an OSMnx MultiDiGraph)


try:
    import networkx as nx  # type: ignore
except Exception:  # keep import optional in signature
    nx = None  # type: ignore


class GAShortestPath:
    """
    Genetic algorithm path finder.
    Usage:
        finder = GAShortestPath(population_size=80, generations=200, seed=42)
        path = finder.shortest_path(G, orig, dest, weight='length')
    Returns a list of node ids (like osmnx.shortest_path).
    """

    def __init__(
        self,
        population_size: int = 80,
        generations: int = 200,
        mutation_rate: float = 0.2,
        tournament_size: int = 4,
        elitism: int = 4,
        max_steps_factor: float = 4.0,
        heuristic_bias: float = 0.3,
        no_improve_limit: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = max(10, population_size)
        self.generations = max(1, generations)
        self.mutation_rate = min(1.0, max(0.0, mutation_rate))
        self.tournament_size = max(2, tournament_size)
        self.elitism = max(0, elitism)
        self.max_steps_factor = max(1.0, float(max_steps_factor))
        self.heuristic_bias = max(0.0, float(heuristic_bias))
        self.no_improve_limit = max(5, int(no_improve_limit))
        self.rng = random.Random(seed)

    def shortest_path(
        self,
        G: "nx.Graph",
        orig: Any,
        dest: Any,
        weight: str = "length",
    ) -> Optional[List[Any]]:
        if orig == dest:
            return [orig]
        if not (G.has_node(orig) and G.has_node(dest)):
            return None

        directed = G.is_directed() if hasattr(G, "is_directed") else True
        successors = (lambda n: G.successors(n)) if directed and hasattr(G, "successors") else (lambda n: G.neighbors(n))

        # Precompute node coords if available for heuristic
        coords = self._collect_coords(G)
        def heuristic(n: Any) -> float:
            if coords is None:
                return 0.0
            x1, y1 = coords.get(n, (None, None))
            x2, y2 = coords.get(dest, (None, None))
            if x1 is None or y1 is None or x2 is None or y2 is None:
                return 0.0
            dx, dy = x1 - x2, y1 - y2
            return math.hypot(dx, dy)

        # Max steps: proportional to graph size but bounded
        max_steps = int(self.max_steps_factor * max(10, int(G.number_of_nodes() ** 0.5)))
        max_steps = max(max_steps, 32)

        # Initialize population
        population: List[List[Any]] = []
        tries = 0
        target_init = self.population_size
        while len(population) < target_init and tries < target_init * 50:
            tries += 1
            path = self._random_path(G, successors, orig, dest, weight, heuristic, max_steps)
            if path is not None:
                population.append(path)

        if not population:
            return None

        best = min(population, key=lambda p: self._path_cost(G, p, weight))
        best_cost = self._path_cost(G, best, weight)
        stagnation = 0

        for gen in range(self.generations):
            # Evaluate fitness (lower is better)
            population.sort(key=lambda p: self._path_cost(G, p, weight))
            if self._path_cost(G, population[0], weight) + 1e-9 < best_cost:
                best = population[0]
                best_cost = self._path_cost(G, best, weight)
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= self.no_improve_limit:
                    break

            # Elitism
            new_pop: List[List[Any]] = population[: self.elitism]

            # Fill rest via selection, crossover, mutation
            while len(new_pop) < self.population_size:
                p1 = self._tournament_select(G, population, weight)
                p2 = self._tournament_select(G, population, weight)
                child = self._crossover(G, successors, p1, p2, orig, dest, weight, heuristic, max_steps)
                if self.rng.random() < self.mutation_rate:
                    child = self._mutate(G, successors, child, dest, weight, heuristic, max_steps)
                if child is not None:
                    new_pop.append(child)
                else:
                    # fallback random if crossover/mutation failed
                    rnd = self._random_path(G, successors, orig, dest, weight, heuristic, max_steps)
                    if rnd is not None:
                        new_pop.append(rnd)

            population = new_pop

        # Final best
        population.sort(key=lambda p: self._path_cost(G, p, weight))
        candidate = population[0]
        cand_cost = self._path_cost(G, candidate, weight)
        return candidate if math.isfinite(cand_cost) else None

    # ---------- GA components ----------

    def _path_cost(self, G: "nx.Graph", path: List[Any], weight: str) -> float:
        total = 0.0
        for u, v in zip(path, path[1:]):
            w = self._edge_weight(G, u, v, weight)
            if w is None or w == float("inf"):
                return float("inf")
            total += w
        # mild preference for shorter node count as tie-breaker
        return total + 1e-6 * len(path)

    def _tournament_select(self, G: "nx.Graph", population: List[List[Any]], weight: str) -> List[Any]:
        k = min(self.tournament_size, len(population))
        contenders = self.rng.sample(population, k)
        contenders.sort(key=lambda p: self._path_cost(G, p, weight))
        return contenders[0]

    def _crossover(
        self,
        G: "nx.Graph",
        successors,
        p1: List[Any],
        p2: List[Any],
        orig: Any,
        dest: Any,
        weight: str,
        heuristic,
        max_steps: int,
    ) -> Optional[List[Any]]:
        # Edge-recombination guided by parents; defaults to random path repair if stuck
        next1 = {u: v for u, v in zip(p1, p1[1:])}
        next2 = {u: v for u, v in zip(p2, p2[1:])}
        child = [orig]
        visited = {orig}
        current = orig
        steps = 0

        while current != dest and steps < max_steps:
            steps += 1
            candidates = []
            if current in next1:
                candidates.append(next1[current])
            if current in next2:
                candidates.append(next2[current])
            # Keep unique and not yet visited
            candidates = [n for i, n in enumerate(candidates) if n not in candidates[:i] and n not in visited]

            chosen = None
            if candidates:
                # Prefer candidate with lower local cost + heuristic
                def cand_score(n):
                    ew = self._edge_weight(G, current, n, weight) or float("inf")
                    return ew + self.heuristic_bias * heuristic(n)
                candidates.sort(key=cand_score)
                chosen = candidates[0]

            if chosen is None:
                # fallback to heuristic-biased random neighbor
                neigh = list(successors(current))
                neigh = [n for n in neigh if n not in visited] or list(successors(current))
                if not neigh:
                    break
                scores = []
                for n in neigh:
                    ew = self._edge_weight(G, current, n, weight)
                    ew = ew if ew is not None and ew > 0 else 1e6
                    h = heuristic(n)
                    score = 1.0 / (1e-9 + ew) * math.exp(-self.heuristic_bias * h)
                    scores.append(max(1e-12, score))
                chosen = self._weighted_choice(neigh, scores)

            if chosen is None:
                break

            child.append(chosen)
            visited.add(chosen)
            current = chosen

        if current != dest:
            # Try to repair the tail
            tail = self._random_path(G, successors, current, dest, weight, heuristic, max_steps)
            if tail is None or len(tail) < 2:
                return None
            child.extend(tail[1:])

        return child

    def _mutate(
        self,
        G: "nx.Graph",
        successors,
        path: List[Any],
        dest: Any,
        weight: str,
        heuristic,
        max_steps: int,
    ) -> Optional[List[Any]]:
        if path is None:
            return None
        if len(path) <= 2:
            return path
        # Pick a splice point and reroute from there
        i = self.rng.randrange(0, len(path) - 1)
        prefix = path[: i + 1]
        start = prefix[-1]
        tail = self._random_path(G, successors, start, dest, weight, heuristic, max_steps)
        if tail is None:
            return path
        return prefix + tail[1:]

    # ---------- Path generation ----------

    def _random_path(
        self,
        G: "nx.Graph",
        successors,
        orig: Any,
        dest: Any,
        weight: str,
        heuristic,
        max_steps: int,
    ) -> Optional[List[Any]]:
        for attempt in range(6):
            path = [orig]
            visited = {orig}
            current = orig
            steps = 0
            while current != dest and steps < max_steps:
                steps += 1
                neigh = list(successors(current))
                if not neigh:
                    break
                # Prefer unvisited, but allow visited if stuck
                unvisited = [n for n in neigh if n not in visited]
                cand = unvisited or neigh
                # Weighted choice inversely proportional to edge weight and biased by heuristic
                weights = []
                for n in cand:
                    ew = self._edge_weight(G, current, n, weight)
                    if ew is None or ew <= 0:
                        ew = 1e6
                    h = heuristic(n)
                    score = 1.0 / (1e-9 + ew) * math.exp(-self.heuristic_bias * h)
                    weights.append(max(1e-12, score))
                nxt = self._weighted_choice(cand, weights)
                if nxt is None:
                    break
                path.append(nxt)
                visited.add(nxt)
                current = nxt
            if current == dest:
                return path
            # else try again with same parameters
        return None

    # ---------- Utilities ----------

    def _edge_weight(self, G: "nx.Graph", u: Any, v: Any, weight: str) -> Optional[float]:
        # Handles Graph/Multi(Di)Graph: pick min weight across parallel edges
        try:
            data = G.get_edge_data(u, v, default=None)
        except Exception:
            data = None
        if data is None:
            return None
        if isinstance(data, dict) and any(isinstance(k, (int, str)) for k in data.keys()) and any(
            isinstance(vv, dict) for vv in data.values()
        ):
            # Multi-edge dict
            best = float("inf")
            found = False
            for _, attrs in data.items():
                w = attrs.get(weight, None)
                if w is None:
                    continue
                found = True
                if w < best:
                    best = w
            return best if found else 1.0  # default if missing
        elif isinstance(data, dict):
            w = data.get(weight, None)
            return float(w) if w is not None else 1.0
        return None

    def _weighted_choice(self, items: List[Any], weights: List[float]) -> Optional[Any]:
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(items) if items else None
        r = self.rng.random() * total
        upto = 0.0
        for item, w in zip(items, weights):
            upto += w
            if upto >= r:
                return item
        return items[-1] if items else None

    def _collect_coords(self, G: "nx.Graph") -> Optional[Dict[Any, Tuple[float, float]]]:
        # OSMnx usually has node attributes 'x' (lon) and 'y' (lat)
        sample = next(iter(G.nodes), None)
        if sample is None:
            return None
        node_attrs = G.nodes[sample]
        if not isinstance(node_attrs, dict):
            return None
        has_xy = ("x" in node_attrs) and ("y" in node_attrs)
        if not has_xy:
            return None
        coords: Dict[Any, Tuple[float, float]] = {}
        for n, d in G.nodes(data=True):
            try:
                coords[n] = (float(d.get("x")), float(d.get("y")))
            except Exception:
                coords[n] = (None, None)  # type: ignore
        return coords


# Convenience function mirroring osmnx.shortest_path signature
def ga_shortest_path(
    G: "nx.Graph",
    orig: Any,
    dest: Any,
    weight: str = "length",
    population_size: int = 80,
    generations: int = 200,
    mutation_rate: float = 0.2,
    tournament_size: int = 4,
    elitism: int = 4,
    max_steps_factor: float = 4.0,
    heuristic_bias: float = 0.3,
    no_improve_limit: int = 50,
    seed: Optional[int] = None,
) -> Optional[List[Any]]:
    finder = GAShortestPath(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elitism=elitism,
        max_steps_factor=max_steps_factor,
        heuristic_bias=heuristic_bias,
        no_improve_limit=no_improve_limit,
        seed=seed,
    )
    return finder.shortest_path(G, orig, dest, weight=weight)