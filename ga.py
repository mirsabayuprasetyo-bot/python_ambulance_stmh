# Full integrated script: GA + A* routing + sparse circular traffic zones + traffic-aware animation
# Supports callers requesting multiple ambulances
import numpy as np
import random
import math
import heapq
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import display, FileLink
from matplotlib.animation import FFMpegWriter

# -----------------------
# Parameters
# -----------------------
RND_SEED = 1
random.seed(RND_SEED)
np.random.seed(RND_SEED)

GRID_SIZE = 20
NUM_CALLS = 20
NUM_AMB = 11

POP_SIZE = 80
GENERATIONS = 960
TOURNAMENT_K = 3
MUT_RATE_SWAP = 0.15
MUT_RATE_CUT = 0.2
ELITE = 2

# Sparse circular traffic zones
NUM_ZONES = 5
ZONE_MIN_RADIUS = 1
ZONE_MAX_RADIUS = 3
ZONE_MIN_PEAK = 1.8   # multiplier at center
ZONE_MAX_PEAK = 3.0

# Animation / simulation
BASE_SPEED_UNIT = 1.0         # baseline unit for progress semantics
MIN_SPEED = 0.08              # minimum progress per frame (prevents freezing)
MAX_FRAMES = 5000


# -----------------------
# Problem setup: calls & homes
# -----------------------
call_coords = np.random.RandomState(RND_SEED).randint(0, GRID_SIZE, size=(NUM_CALLS, 2))

amb_home_coords = np.array([
    [0, 0],              # Hospital 1 → Ambulance 1

    [GRID_SIZE-1, 0],    # Hospital 2 → Ambulance 2
    [GRID_SIZE-1, 0],    # Hospital 2 → Ambulance 3

    [GRID_SIZE//2, GRID_SIZE-1],  # Hospital 3 → Ambulance 4
    [GRID_SIZE//2, GRID_SIZE-1],  # Hospital 3 → Ambulance 5
    [GRID_SIZE//2, GRID_SIZE-1],  # Hospital 3 → Ambulance 6

    [GRID_SIZE-8, GRID_SIZE//2],      # Hospital 4 → Ambulance 7
    [GRID_SIZE-8, GRID_SIZE//2],      # Hospital 4 → Ambulance 8
    [GRID_SIZE-8, GRID_SIZE//2],      # Hospital 4 → Ambulance 9
    [GRID_SIZE-8, GRID_SIZE//2],      # Hospital 4 → Ambulance 10
    [GRID_SIZE-8, GRID_SIZE//2]       # Hospital 4 → Ambulance 11 
], dtype=int)

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

# -----------------------
# Priority setup (3 levels)
# 'H' = High, 'M' = Medium, 'L' = Low
# reproducible assignment using RND_SEED
# -----------------------
_prng = np.random.RandomState(RND_SEED + 999)
# p(H)=0.2, p(M)=0.3, p(L)=0.5 (adjust if you want)
priority_choices = _prng.choice(['H', 'M', 'L'], size=NUM_CALLS, p=[0.2, 0.3, 0.5])
priority_levels = {i: priority_choices[i] for i in range(NUM_CALLS)}
priority_weight = {'H': 3.0, 'M': 2.0, 'L': 1.0}

# -----------------------
# Allow calls to request more than 1 ambulance
# -----------------------
# Example: random demand 1–2 ambulances per call (you can change upper bound)
# For deterministic reproducibility, use the seeded RNG above or np.random.RandomState
demand_rng = np.random.RandomState(RND_SEED + 2025)
# change high value to allow up to 3 requests, etc.
call_demand = demand_rng.randint(1, 3, size=NUM_CALLS)  # 1..2 ambulances per call by default

# Build expanded calls list (list of real call IDs, duplicated by demand)
EXPANDED_CALLS = []
for cid in range(NUM_CALLS):
    for _ in range(int(call_demand[cid])):
        EXPANDED_CALLS.append(cid)

NUM_EXPANDED = len(EXPANDED_CALLS)
print(f"NUM_CALLS = {NUM_CALLS}, NUM_EXPANDED (with demand) = {NUM_EXPANDED}")
print("call_demand (cid: demand):", {i: int(call_demand[i]) for i in range(NUM_CALLS)})

# -----------------------
# Build sparse circular traffic zones (per-cell multipliers)
# baseline = 1.0 (no slowdown); inside zones multiplier > 1
# -----------------------
traffic_grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=float)
rng = np.random.RandomState(RND_SEED + 12345)
zone_centers = []
for _ in range(NUM_ZONES):
    cx = int(rng.randint(0, GRID_SIZE))
    cy = int(rng.randint(0, GRID_SIZE))
    radius = int(rng.randint(ZONE_MIN_RADIUS, ZONE_MAX_RADIUS + 1))
    peak = float(rng.uniform(ZONE_MIN_PEAK, ZONE_MAX_PEAK))
    zone_centers.append((cx, cy, radius, peak))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            d = math.hypot(x - cx, y - cy)
            if d <= radius:
                # linear decay from peak at center to 1.0 at radius
                mul = 1.0 + (peak - 1.0) * max(0.0, (radius - d) / (radius + 1e-9))
                traffic_grid[y, x] = max(traffic_grid[y, x], mul)

# helper to read multiplier (note: traffic_grid indexed [row=y, col=x])
def traffic_multiplier_at_cell(x, y):
    ix = int(min(max(math.floor(x), 0), GRID_SIZE-1))
    iy = int(min(max(math.floor(y), 0), GRID_SIZE-1))
    return float(traffic_grid[iy, ix])

# -----------------------
# A* implementation on grid
# Cost to move from node u->v = traffic_multiplier_at_cell(v)
# 4-neighborhood (up/down/left/right)
# -----------------------
def neighbors(node):
    x, y = node
    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            yield (nx, ny)

def astar(start, goal):
    """Return (cost, path_nodes) from start to goal using A* with heuristic = Manhattan"""
    open_heap = []
    heapq.heappush(open_heap, (manhattan(start, goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    closed = set()
    while open_heap:
        f, g, node, parent = heapq.heappop(open_heap)
        if node in closed:
            continue
        came_from[node] = parent
        if node == goal:
            # rebuild path
            path = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return g, path
        closed.add(node)
        for nb in neighbors(node):
            tentative_g = g + traffic_multiplier_at_cell(nb[0], nb[1])
            if nb in gscore and tentative_g >= gscore[nb]:
                continue
            gscore[nb] = tentative_g
            fscore = tentative_g + manhattan(nb, goal)
            heapq.heappush(open_heap, (fscore, tentative_g, nb, node))
    return float('inf'), []

# caching for repeated A* queries (traffic static)
astar_cache = {}
def astar_cached(a, b):
    key = (a[0], a[1], b[0], b[1])
    if key in astar_cache:
        return astar_cache[key]
    cost, path = astar(a, b)
    astar_cache[key] = (cost, path)
    return cost, path

# -----------------------
# GA operators (use A* cost in fitness)
# Now operate on NUM_EXPANDED items (indices into EXPANDED_CALLS)
# -----------------------
def random_individual():
    perm = list(np.random.permutation(NUM_EXPANDED))
    # cuts partition NUM_EXPANDED into NUM_AMB segments
    # choose NUM_AMB-1 cut positions between 1..NUM_EXPANDED-1
    if NUM_AMB <= 1:
        cut_positions = []
    else:
        cut_positions = sorted(random.sample(range(1, NUM_EXPANDED), NUM_AMB-1))
    return perm, cut_positions

def decode_routes_safe(perm, cuts):
    routes = []
    start = 0
    for c in sorted(cuts):
        if c <= start:
            c = start + 1
        routes.append(perm[start:c])
        start = c
    routes.append(perm[start:NUM_EXPANDED])
    # ensure length == NUM_AMB (some ambulances might get empty route)
    while len(routes) < NUM_AMB:
        routes.append([])
    # if more routes than ambulances (shouldn't happen) cut extras
    if len(routes) > NUM_AMB:
        routes = routes[:NUM_AMB]
    return routes

def evaluate_individual(ind):
    """
    Fitness = total_travel_time + sum(priority_weight * waiting_time)
    waiting_time = effective arrival time per real caller (we use max of arrivals for that call)
    """
    perm, cuts = ind
    routes = decode_routes_safe(perm, cuts)

    # arrival lists per real caller (collect all arrivals for duplicates)
    arrival_times_multi = {cid: [] for cid in range(NUM_CALLS)}
    total_travel_time = 0.0

    for amb_idx, route in enumerate(routes):
        t = 0.0
        home = tuple(amb_home_coords[amb_idx])
        prev = home
        for expanded_idx in route:
            call_id = EXPANDED_CALLS[expanded_idx]
            call_pos = tuple(call_coords[call_id])
            c1, p1 = astar_cached(prev, call_pos)
            t += c1
            # record arrival for that copy
            arrival_times_multi[call_id].append(t)
            prev = call_pos
            # return to hospital after each call (rule A)
            c2, p2 = astar_cached(prev, home)
            t += c2
            prev = home
        total_travel_time += t

    # priority-weighted waiting penalty: use effective arrival per real caller
    penalty = 0.0
    for cid in range(NUM_CALLS):
        arrivals = arrival_times_multi[cid]
        if len(arrivals) == 0:
            # if no ambulance assigned to a call (should be rare unless demand > total amb copies),
            # assign a large penalty (here, we assume worst-case: GRID_SIZE*GRID_SIZE)
            effective_arrival = GRID_SIZE * GRID_SIZE * 2.0
        else:
            # effective arrival when the last requested ambulance arrives
            # (choose max; change to min/sum if another policy desired)
            effective_arrival = max(arrivals)
        pr = priority_levels[cid]
        w = priority_weight[pr]
        penalty += w * effective_arrival

    fitness = total_travel_time + penalty
    return fitness

def tournament_selection(pop, fitnesses, k=TOURNAMENT_K):
    best = random.randrange(len(pop))
    for _ in range(k-1):
        cand = random.randrange(len(pop))
        if fitnesses[cand] < fitnesses[best]:
            best = cand
    return deepcopy(pop[best])

def ordered_crossover(p1, p2):
    n = NUM_EXPANDED
    a, b = sorted(random.sample(range(n), 2))
    child = [-1]*n
    child[a:b+1] = p1[a:b+1]
    p2_idx = 0
    for i in range(n):
        if child[i] == -1:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
    return child

def crossover_cuts(c1, c2):
    # c1, c2 are lists of length NUM_AMB-1 (unless NUM_AMB<=1)
    if NUM_AMB <= 1:
        return []
    child = []
    for i in range(len(c1)):
        child.append(c1[i] if random.random() < 0.5 else c2[i])
    child = sorted(child)
    # enforce bounds relative to NUM_EXPANDED
    for idx in range(len(child)):
        min_val = 1 + idx
        max_val = NUM_EXPANDED - (len(child) - idx)
        child[idx] = max(min_val, min(max_val, child[idx]))
    return child

def mutate_individual_safe(ind):
    perm, cuts = ind
    if random.random() < MUT_RATE_SWAP and NUM_EXPANDED >= 2:
        i, j = random.sample(range(NUM_EXPANDED), 2)
        perm[i], perm[j] = perm[j], perm[i]
    if random.random() < MUT_RATE_CUT and NUM_AMB > 1 and len(cuts) > 0:
        k = random.randrange(len(cuts))
        shift = random.choice([-2, -1, 1, 2])
        new_val = cuts[k] + shift
        min_val = 1 + k
        max_val = NUM_EXPANDED - (len(cuts) - k)
        cuts[k] = max(min_val, min(max_val, new_val))
        cuts.sort()
    return perm, cuts

# -----------------------
# Run GA
# -----------------------
population = [random_individual() for _ in range(POP_SIZE)]
fitnesses = [evaluate_individual(ind) for ind in population]
best_history = []
best_fit = float('inf')
best_ind = None

for gen in range(GENERATIONS):
    order = np.argsort(fitnesses)
    population = [population[i] for i in order]
    fitnesses = [fitnesses[i] for i in order]
    if fitnesses[0] < best_fit:
        best_fit = fitnesses[0]
        best_ind = deepcopy(population[0])
    best_history.append(fitnesses[0])
    if gen % 10 == 0 or gen == GENERATIONS-1:
        print(f"Gen {gen:3d} best fitness = {fitnesses[0]:.2f}")
    new_pop = population[:ELITE]
    while len(new_pop) < POP_SIZE:
        p1 = tournament_selection(population, fitnesses)
        p2 = tournament_selection(population, fitnesses)
        child_perm = ordered_crossover(p1[0], p2[0])
        child_cuts = crossover_cuts(p1[1], p2[1])
        child = mutate_individual_safe((child_perm, child_cuts))
        new_pop.append(child)
    population = new_pop
    fitnesses = [evaluate_individual(ind) for ind in population]

print("\nGA finished. Best fitness:", best_fit)

# -----------------------
# Helper: compute arrival times and ambulances' simple route lists
# -----------------------
def compute_arrival_times_and_routes(ind):
    perm, cuts = ind
    routes = decode_routes_safe(perm, cuts)
    arrival_times_multi = {cid: [] for cid in range(NUM_CALLS)}
    amb_routes = []
    for amb_idx, route in enumerate(routes):
        t = 0.0
        home = tuple(amb_home_coords[amb_idx])
        prev = home
        for expanded_idx in route:
            call_id = EXPANDED_CALLS[expanded_idx]
            call_pos = tuple(call_coords[call_id])
            c1, p1 = astar_cached(prev, call_pos)
            t += c1
            arrival_times_multi[call_id].append(t)
            prev = call_pos
            c2, p2 = astar_cached(prev, home)
            t += c2
            prev = home
        amb_routes.append(route)
    # compute effective arrival per real caller
    effective_arrivals = np.zeros(NUM_CALLS, dtype=float)
    for cid in range(NUM_CALLS):
        arrs = arrival_times_multi[cid]
        if len(arrs) == 0:
            effective_arrivals[cid] = GRID_SIZE * GRID_SIZE * 2.0
        else:
            effective_arrivals[cid] = max(arrs)
    return effective_arrivals, amb_routes, arrival_times_multi

# -----------------------
# Decode best routes and build A* node paths for visualization (fixed concatenation)
# Each ambulance path: home -> call1 -> home -> call2 -> home -> ...
# -----------------------
best_perm, best_cuts = best_ind
best_routes = decode_routes_safe(best_perm, best_cuts)

amb_node_paths = []
for amb_idx, route in enumerate(best_routes):
    nodes = []
    home = tuple(amb_home_coords[amb_idx])
    nodes.append(home)
    for expanded_idx in route:
        call_id = EXPANDED_CALLS[expanded_idx]
        call_pos = tuple(call_coords[call_id])
        # path from current last node to call
        _, p_to_call = astar_cached(nodes[-1], call_pos)
        if len(p_to_call) > 1:
            nodes.extend(p_to_call[1:])
        elif len(p_to_call) == 1 and nodes[-1] != p_to_call[0]:
            nodes.append(p_to_call[0])
        # path from call back to home
        _, p_back = astar_cached(call_pos, home)
        if len(p_back) > 1:
            nodes.extend(p_back[1:])
        elif len(p_back) == 1 and nodes[-1] != p_back[0]:
            nodes.append(p_back[0])
    if len(nodes) == 0:
        nodes = [home]
    amb_node_paths.append(nodes)

# -----------------------
# Simulation: smooth traffic-aware movement along A* node paths
# -----------------------
n_amb = len(amb_node_paths)
progress = [0.0] * n_amb  # fractional index into node list (0..len-1)
positions_per_frame = []
frame = 0

while frame < MAX_FRAMES:
    all_done = True
    frame_positions = []
    for i in range(n_amb):
        nodes = amb_node_paths[i]
        if len(nodes) == 0:
            frame_positions.append(tuple(amb_home_coords[i]))
            continue
        # If at end, hold position
        if progress[i] >= len(nodes) - 1 - 1e-9:
            frame_positions.append(nodes[-1])
            continue
        all_done = False
        idx = int(math.floor(progress[i]))
        frac = progress[i] - idx
        # current continuous pos
        a = np.array(nodes[idx], dtype=float)
        b = np.array(nodes[idx+1], dtype=float)
        pos = a + (b - a) * frac
        # determine multiplier of destination cell (we use target node to decide speed)
        target_node = nodes[idx+1]
        mult = traffic_multiplier_at_cell(target_node[0], target_node[1])
        # effective progress increment per frame (node-to-node units)
        eff = (BASE_SPEED_UNIT / mult) * 0.12  # scale so animation speed looks good
        eff = max(MIN_SPEED, eff)
        progress[i] += eff
        # compute new interpolated pos
        idx_new = int(math.floor(progress[i]))
        frac_new = progress[i] - idx_new
        if idx_new >= len(nodes) - 1:
            new_pos = tuple(nodes[-1])
        else:
            a2 = np.array(nodes[idx_new], dtype=float)
            b2 = np.array(nodes[idx_new+1], dtype=float)
            new_pos = tuple((a2 + (b2 - a2) * frac_new).tolist())
        frame_positions.append(new_pos)
    positions_per_frame.append(frame_positions)
    frame += 1
    if all_done:
        break

# ensure minimal frames for animation
if len(positions_per_frame) < 10:
    last = positions_per_frame[-1] if positions_per_frame else [tuple(h) for h in amb_home_coords]
    for _ in range(10 - len(positions_per_frame)):
        positions_per_frame.append(last)

print("Simulated frames:", len(positions_per_frame))

# -----------------------
# Visualization: heatmap + halo routes + animation
# -----------------------
# convergence plot
plt.figure(figsize=(6,3))
plt.plot(best_history, marker='o')
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.title("GA Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

colors = plt.colormaps['tab20'].colors[:NUM_AMB]
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(-0.5, GRID_SIZE - 0.5)
ax.set_ylim(-0.5, GRID_SIZE - 0.5)
ax.set_xticks(range(0, GRID_SIZE))
ax.set_yticks(range(0, GRID_SIZE))
ax.grid(True, linewidth=0.35)
fig.suptitle(
    "Ambulance Deployment & Routing (A* + GA) with Traffic Zones - Multi-ambulance Requests",
    fontsize=14,
    y=0.98
)
ax.set_title("")  # clear subplot title

# draw traffic zones as semi-transparent circles
for (cx, cy, radius, peak) in zone_centers:
    intensity = (peak - 1.0) / (ZONE_MAX_PEAK - 1.0)
    alpha = 0.45 * (0.6 + 0.4 * intensity)
    circle = plt.Circle((cx, cy), radius + 0.35, color='red', alpha=alpha, zorder=1)
    ax.add_patch(circle)

# traffic heat overlay
ax.imshow(traffic_grid, origin='lower', cmap='Reds', alpha=0.28,
          extent=[-0.5, GRID_SIZE - 0.5, -0.5, GRID_SIZE - 0.5], zorder=1)

# faint grid lines
for x in range(GRID_SIZE-1):
    for y in range(GRID_SIZE):
        ax.plot([x, x+1], [y, y], color='lightgray', linewidth=0.6, zorder=1)
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE-1):
        ax.plot([x, x], [y, y+1], color='lightgray', linewidth=0.6, zorder=1)

# ---- calls: plot with shapes & colors by priority ----
priority_markers = {'H': 'h', 'M': '^', 'L': 'o'}  # hexagon, triangle, circle
priority_colors = {'H': 'red', 'M': 'gold', 'L': 'green'}

for cid in range(NUM_CALLS):
    x, y = call_coords[cid]
    pr = priority_levels[cid]
    demand = int(call_demand[cid])
    # size scaled by demand for quick visual cue
    s = 120 + 40 * (demand - 1)
    ax.scatter(
        x, y,
        marker=priority_markers[pr],
        s=s,
        color=priority_colors[pr],
        edgecolors='black',
        linewidths=0.8,
        zorder=6
    )
    ax.text(x, y + 0.22, f"C{cid} (d={demand})", ha='center', fontsize=8, zorder=7)

# ---- unique hospital labels ----
unique_hospitals = {}
hospital_counter = 1

for amb_idx, home in enumerate(amb_home_coords):
    home_tuple = tuple(home)
    # draw hospital square for each ambulance (same coord repeats visually)
    ax.scatter(
        home[0], home[1],
        c=[colors[amb_idx]], 
        s=110, marker='s', edgecolor='k', linewidth=0.6, zorder=7
    )
    # label only once
    if home_tuple not in unique_hospitals:
        label = f"H{hospital_counter}"
        unique_hospitals[home_tuple] = label
        ax.text(home[0] + 0.2, home[1] + 0.2, label, fontsize=9, zorder=8)
        hospital_counter += 1

# halo route lines (A* node sequences)
for i, nodes in enumerate(amb_node_paths):
    if len(nodes) < 2:
        continue
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    ax.plot(xs, ys, color='white', linewidth=4.2, solid_capstyle='round', zorder=4)
    ax.plot(xs, ys, color=colors[i], linewidth=2.2, solid_capstyle='round', zorder=5)

# ambulance markers (no labels)
amb_markers = []
for i in range(len(amb_node_paths)):
    m = ax.plot([], [], marker='o', markersize=9, color=colors[i], zorder=10)[0]
    amb_markers.append(m)

# place time text just under main title
time_text = fig.text(0.05, 0.94, '', fontsize=11)

def update_anim(f):
    poslist = positions_per_frame[f]
    for i, m in enumerate(amb_markers):
        x, y = poslist[i]
        m.set_data([x], [y])
    time_text.set_text(f"t = {f}")
    return amb_markers + [time_text]

# legend: priorities + hospitals
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker=priority_markers['H'], color='w', markerfacecolor=priority_colors['H'], markersize=10, label='High priority'),
    Line2D([0], [0], marker=priority_markers['M'], color='w', markerfacecolor=priority_colors['M'], markersize=10, label='Medium priority'),
    Line2D([0], [0], marker=priority_markers['L'], color='w', markerfacecolor=priority_colors['L'], markersize=10, label='Low priority'),
]
# hospital legend
for idx, (home_coord, label) in enumerate(unique_hospitals.items()):
    amb_idx_for_color = next(i for i, h in enumerate(amb_home_coords) if tuple(h) == home_coord)
    legend_handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=colors[amb_idx_for_color], markersize=10, label=label))

ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.85)

frames = len(positions_per_frame)
ani = FuncAnimation(fig, update_anim, frames=frames, interval=80, blit=False, repeat=False)

# save vid
writer = PillowWriter(fps=12)
MP4_PATH = "ambulance_animation_2.mp4"
writer = FFMpegWriter(fps=20, bitrate=3000)
ani.save(MP4_PATH, writer=writer)

# finalize display
for i in range(len(amb_markers)):
    amb_markers[i].set_data(
        [positions_per_frame[-1][i][0]],
        [positions_per_frame[-1][i][1]]
    )

display(fig)
display(FileLink(MP4_PATH))   # Show MP4 file link

# Summary prints
print("\nBest cuts:", best_cuts)
best_arrivals, best_amb_routes, arrivals_multi = compute_arrival_times_and_routes(best_ind)
print(f"Mean effective arrival time (best solution, per real call): {best_arrivals.mean():.2f}\n")
for i, route in enumerate(best_routes):
    # route is list of expanded indices
    real_call_counts = {}
    for expanded_idx in route:
        cid = EXPANDED_CALLS[expanded_idx]
        real_call_counts[cid] = real_call_counts.get(cid, 0) + 1
    print(f"Amb {i}: {len(route)} assigned expanded-calls, unique real-call targets = {len(real_call_counts)}, node steps = {len(amb_node_paths[i])}")

print("\nPer-call demand, assigned arrivals (example):")
for cid in range(NUM_CALLS):
    print(f"Call {cid}: demand={int(call_demand[cid])}, arrivals={sorted([round(x,2) for x in arrivals_multi[cid]])}, effective_arrival={round(best_arrivals[cid],2)}, priority={priority_levels[cid]}")

print("\nTraffic zones (cx,cy,radius,peak):")
for z in zone_centers:
    print(z)

print("\nDone.")
