import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from copy import deepcopy
from IPython.display import HTML, display, FileLink

# ============================================================
# Parameters
# ============================================================
RND_SEED = 42
np.random.seed(RND_SEED)
random.seed(RND_SEED)

GRID_SIZE = 20
NUM_CALLS = 15
NUM_AMB = 3
POP_SIZE = 40
GENERATIONS = 60
TOURNAMENT_K = 3
MUT_RATE_SWAP = 0.15
MUT_RATE_CUT = 0.2
ELITE = 2

# ============================================================
# Problem setup
# ============================================================
call_coords = np.random.randint(0, GRID_SIZE, size=(NUM_CALLS, 2))

amb_home_coords = np.array([
    [0, 0],
    [GRID_SIZE-1, 0],
    [GRID_SIZE//2, GRID_SIZE-1]
], dtype=int)

def manhattan(a, b):
    return int(abs(a[0]-b[0]) + abs(a[1]-b[1]))

# ============================================================
# GA Functions
# ============================================================
def random_individual():
    perm = list(np.random.permutation(NUM_CALLS))
    cut_positions = sorted(random.sample(range(1, NUM_CALLS), NUM_AMB-1))
    return perm, cut_positions

def decode_routes_safe(perm, cuts):
    routes = []
    start = 0
    cuts_sorted = sorted(cuts)
    for c in cuts_sorted:
        if c <= start:
            c = start + 1
        routes.append(perm[start:c])
        start = c
    routes.append(perm[start:NUM_CALLS])
    return routes

# ---------- Modified evaluate_individual ----------
# Enforce: home -> call -> home for every call
def evaluate_individual(ind):
    perm, cuts = ind
    routes = decode_routes_safe(perm, cuts)
    arrival_times = np.zeros(NUM_CALLS, dtype=float)
    for amb_idx, route in enumerate(routes):
        t = 0
        for call_idx in route:
            home = amb_home_coords[amb_idx]
            call_pos = call_coords[call_idx]
            # go to call
            d1 = manhattan(home, call_pos)
            t += d1
            arrival_times[call_idx] = t
            # return to home
            d2 = manhattan(call_pos, home)
            t += d2
        # always ends at home
    return arrival_times.mean()

def tournament_selection(pop, fitnesses, k=TOURNAMENT_K):
    best = random.randrange(len(pop))
    for _ in range(k-1):
        cand = random.randrange(len(pop))
        if fitnesses[cand] < fitnesses[best]:
            best = cand
    return deepcopy(pop[best])

def ordered_crossover(p1, p2):
    n = NUM_CALLS
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
    child = []
    for i in range(len(c1)):
        child.append(c1[i] if random.random()<0.5 else c2[i])
    child = sorted(child)
    for idx in range(len(child)):
        min_val = 1 + idx
        max_val = NUM_CALLS - (len(child)-idx)
        child[idx] = max(min_val, min(max_val, child[idx]))
    return sorted(child)

def mutate_individual_safe(ind):
    perm, cuts = ind
    if random.random() < MUT_RATE_SWAP:
        i, j = random.sample(range(NUM_CALLS), 2)
        perm[i], perm[j] = perm[j], perm[i]
    if random.random() < MUT_RATE_CUT:
        k = random.randrange(len(cuts))
        shift = random.choice([-2, -1, 1, 2])
        new_val = cuts[k] + shift
        min_val = 1 + k
        max_val = NUM_CALLS - (len(cuts) - k)
        cuts[k] = max(min_val, min(max_val, new_val))
        cuts.sort()
    return perm, cuts

# ============================================================
# Run GA
# ============================================================
population = [random_individual() for _ in range(POP_SIZE)]
fitnesses = [evaluate_individual(ind) for ind in population]

best_history = []
best_fit = float('inf')
best_ind = None

for gen in range(GENERATIONS):
    idx_sorted = np.argsort(fitnesses)
    population = [population[i] for i in idx_sorted]
    fitnesses = [fitnesses[i] for i in idx_sorted]
    if fitnesses[0] < best_fit:
        best_fit = fitnesses[0]
        best_ind = deepcopy(population[0])
    best_history.append(fitnesses[0])

    if gen % 10 == 0 or gen == GENERATIONS-1:
        print(f"Gen {gen:3d} best mean arrival = {fitnesses[0]:.2f}")

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

print("\nGA finished. Best mean arrival time:", best_fit)

# ============================================================
# Decode best individual
# ============================================================
best_perm, best_cuts = best_ind
best_routes = decode_routes_safe(best_perm, best_cuts)

# ---------- Modified build_paths ----------
# Build path: home->call->home for every call
def build_paths(routes, home_coords, call_coords):
    amb_paths_full = []
    for amb_idx, route in enumerate(routes):
        path = []
        for call_idx in route:
            home = tuple(home_coords[amb_idx])
            call_pos = tuple(call_coords[call_idx])
            # go to call
            x0, y0 = home
            x1, y1 = call_pos
            step = 1 if x1 >= x0 else -1
            for x in range(x0, x1, step):
                path.append((x+step, y0))
            step = 1 if y1 >= y0 else -1
            for y in range(y0, y1, step):
                path.append((x1, y+step))
            # return to home
            x0, y0 = x1, y1
            x1, y1 = home
            step = 1 if x1 >= x0 else -1
            for x in range(x0, x1, step):
                path.append((x+step, y0))
            step = 1 if y1 >= y0 else -1
            for y in range(y0, y1, step):
                path.append((x1, y+step))
        amb_paths_full.append(path)
    return amb_paths_full

amb_paths_full = build_paths(best_routes, amb_home_coords, call_coords)
max_len = max(len(p) for p in amb_paths_full)
frames = max_len + 5

# ============================================================
# Visualization
# ============================================================
plt.figure(figsize=(6,3))
plt.plot(best_history, marker='o')
plt.xlabel("Generation")
plt.ylabel("Best mean arrival time")
plt.title("GA Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

colors = plt.colormaps['tab10'].colors[:NUM_AMB]
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1, GRID_SIZE)
ax.set_ylim(-1, GRID_SIZE)
ax.set_xticks(range(0, GRID_SIZE))
ax.set_yticks(range(0, GRID_SIZE))
ax.grid(True, linewidth=0.4)
ax.set_title("Ambulance Routes with Return-to-Hospital Constraint")

ax.scatter(call_coords[:,0], call_coords[:,1], c='gray', alpha=0.7, label='Calls')
for i, home in enumerate(amb_home_coords):
    ax.scatter(home[0], home[1], c=[colors[i]], s=90, marker='s', edgecolor='k', linewidth=0.6)
    ax.text(home[0]+0.2, home[1]+0.2, f"H{i}", fontsize=8)

amb_markers = []
for i in range(NUM_AMB):
    m, = ax.plot([], [], marker='o', markersize=8, color=colors[i])
    amb_markers.append(m)

for i, path in enumerate(amb_paths_full):
    if path:
        xs = [amb_home_coords[i,0]] + [p[0] for p in path]
        ys = [amb_home_coords[i,1]] + [p[1] for p in path]
        ax.plot(xs, ys, linestyle='--', linewidth=0.8, color=colors[i], alpha=0.5)

def update(frame):
    for i in range(NUM_AMB):
        path = amb_paths_full[i]
        if not path:
            x, y = amb_home_coords[i]
        else:
            idx = min(frame, len(path)-1)
            x, y = path[idx]
        amb_markers[i].set_data([x], [y])
    return amb_markers

ani = FuncAnimation(fig, update, frames=frames, interval=200, blit=False, repeat=False)

gif_path = "ambulance_return_to_hospital.gif"
writer = PillowWriter(fps=5)
ani.save(gif_path, writer=writer)
print(f"Saved animation to {gif_path}")

for i in range(NUM_AMB):
    path = amb_paths_full[i]
    if path:
        x, y = path[-1]
    else:
        x, y = amb_home_coords[i]
    amb_markers[i].set_data([x], [y])
display(fig)

display(FileLink(gif_path))

print("\nBest cuts:", best_cuts)
for i, route in enumerate(best_routes):
    print(f"Amb {i}: {len(route)} calls, total trip count {len(route)} (each returns home)")