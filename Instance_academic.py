from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx


@dataclass(frozen=True)
class GridSpec:
    """Configuration for building a 2‑D grid graph."""
    rows: int
    cols: int
    edge_length: float = 1.0
    directed: bool = True


def create_graph(spec: GridSpec) -> Tuple[Dict[Tuple[int, int], str], nx.DiGraph]:
    """
    Build a directed graph from a 2‑D lattice and return a node id mapping,
    the directed graph, and its (edge‑)dual graph as a line graph.
    
    Node labels are strings of consecutive integers to match the user's path lists.
    
    Parameters
    ----------
    spec : GridSpec
        Grid and edge settings.
    
    Returns
    -------
    tuple
        (coord_to_id, G, dual_G)
        - coord_to_id: maps (row, col) -> node_id as string
        - G: directed graph with 'x','y' node attributes and 'length' edge attribute
    """
    # Base undirected lattice (4‑neighborhood)
    lattice = nx.grid_2d_graph(spec.rows, spec.cols)
    G = nx.DiGraph() if spec.directed else nx.Graph()
    G.graph["geo"] = False

    # Assign integer string IDs in a stable order and store coordinates
    coord_to_id: Dict[Tuple[int, int], str] = {}
    for i, (r, c) in enumerate(lattice.nodes):
        nid = str(i)
        coord_to_id[(r, c)] = nid
        G.add_node(nid, x=r, y=c)

    # Add symmetric arcs with a 'length' attribute
    for u_coord, v_coord in lattice.edges:
        u, v = coord_to_id[u_coord], coord_to_id[v_coord]
        G.add_edge(u, v, length=spec.edge_length)
        G.add_edge(v, u, length=spec.edge_length)

    return coord_to_id, G


# Predefined nominal flight templates (start offset, list of node IDs as strings)
BASE_FLIGHTS: Dict[str, Tuple[int, List[str]]] = {
    "D1": (0,  ["6","14","22","21","20","28","36","35","34","42","50","49","48","56","64"]),                # zigzag 1
    "D2": (7,  ["15","14","13","12","11","10","9","8"]),                                                   # vertical 1
    "D3": (13, ["5","13","21","29","37","45","53","61","69"]),                                             # horizontal 1
    "D4": (25, ["23","22","30","38","37","36","44","52","51","50","58","66"]),                             # zigzag 2
    "D5": (32, ["4","12","20","19","18","26","34","33","32","40","48"]),                                   # zigzag 3
}


@dataclass(frozen=True)
class TimingParamsV2:
    edge_length: float                 # length per edge (consistent units)
    v_min: float                       # min cruise speed
    v_max: float                       # max cruise speed
    ground_delay_max: float            # max ground delay (seconds)
    n_flight_levels: int               # total flight levels available (e.g., 2)
    climb_time_per_level: float        # seconds to climb one level
    earliest_climb_levels: int = 1     # levels climbed for earliest timestamps
    latest_climb_levels: int = 2       # levels climbed for latest timestamps


def _piecewise_offset(rep_index: int) -> int:
    if rep_index < 3:
        return rep_index * 115
    elif rep_index < 6:
        return rep_index * 100
    elif rep_index < 9:
        return rep_index * 85
    else:
        return rep_index * 70


def _cumulative_edge_times(num_edges: int, edge_length: float, v: float) -> List[float]:
    per_edge = edge_length / v
    return [k * per_edge for k in range(num_edges + 1)]  # include k=0


def generate_flight_intentions(
    repetitions: int,
    timing: TimingParamsV2,
    base_flights: Dict[str, Tuple[int, List[str]]] | None = None,
) -> Dict[str, Dict[str, object]]:
    """
    Enhanced flight intentions:
      - Earliest timestamps: ground delay = 0, climb = earliest_climb_levels.
      - Latest timestamps: ground delay = ground_delay_max, climb = latest_climb_levels.
      - Cruise speeds: use v_max (earliest) and v_min (latest).
    """
    base = BASE_FLIGHTS if base_flights is None else base_flights
    F: Dict[str, Dict[str, object]] = {}
    idx = 0

    for r in range(repetitions):
        add = _piecewise_offset(r)

        for _, (t0, path) in base.items():
            idx += 1
            dep_node, arr_node = path[0], path[-1]
            num_edges = max(0, len(path) - 1)

            cum_fast = _cumulative_edge_times(num_edges, timing.edge_length, timing.v_max)
            cum_slow = _cumulative_edge_times(num_edges, timing.edge_length, timing.v_min)

            t_sched = float(t0 + add)
            t0_min = t_sched + timing.earliest_climb_levels * timing.climb_time_per_level
            t0_max = t_sched + timing.ground_delay_max + timing.latest_climb_levels * timing.climb_time_per_level

            node_times = []
            for k, node in enumerate(path):
                t_min = t0_min + cum_fast[k]
                t_max = t0_max + cum_slow[k]
                node_times.append({"node": node, "t_min": t_min, "t_max": t_max})

            F[f"D{idx}"] = {
                "dep": dep_node,
                "arr": arr_node,
                "dep_time": int(t0 + add),
                "path": list(path),
                "params": {
                    **asdict(timing)
                },
                "node_times": node_times,
            }

    return F


def generate_separation_nodes(F_templates: Dict[str, Tuple[int, List[str]]],
                              rng: random.Random | None = None,
                              min_sep: int = 16,
                              max_sep: int = 28,
                              first_node_value: int = 60) -> Dict[str, int]:
    """
    Create a separation‑time requirement per node encountered in the *base* nominal paths.

    Rule used (explicit and reproducible):
      • For the first node of each nominal path: set separation = `first_node_value`.
      • For subsequent nodes in that path: draw an integer U[min_sep, max_sep].
      • If a node appears in multiple paths, keep the first assigned value (no overwrite).
      • Duplicate nodes within a single path are removed before processing.

    Parameters
    ----------
    F_templates : dict
        Base flight templates (not the replicated F). Keys ignored; values are (time, path).
    rng : random.Random or None
        Optional RNG to make results reproducible. If None, a new Random() is used.
    min_sep, max_sep : int
        Inclusive bounds for random separation values along paths.
    first_node_value : int
        Separation value assigned to the first node of each path.

    Returns
    -------
    dict
        Sep_Nodes mapping node_id (str) -> separation_time (int).
    """
    if rng is None:
        rng = random.Random()

    Sep_Nodes: Dict[str, int] = {}

    for _, (t0, path) in F_templates.items():
        # Remove duplicates while preserving order
        seen = set()
        dedup_path: List[str] = []
        for n in path:
            if n not in seen:
                seen.add(n)
                dedup_path.append(n)

        # Draw values for the remaining nodes of this path
        if dedup_path:
            first = dedup_path[0]
            if first not in Sep_Nodes:
                Sep_Nodes[first] = int(first_node_value)

            # For the rest of nodes, assign random values within bounds
            for node in dedup_path[1:]:
                if node not in Sep_Nodes:
                    Sep_Nodes[node] = int(rng.randint(min_sep, max_sep))

    return Sep_Nodes


def save_results_as_json(filepath: Path,
                         F: Dict[str, Dict[str, object]],
                         Sep_Nodes: Dict[str, int]) -> None:
    """
    Persist outputs to a single JSON file with two top‑level keys: 'F' and 'Sep_Nodes'.
    """
    payload = {"F": F, "Sep_Nodes": Sep_Nodes}
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


# -----------------------------
# Example run producing a file
# -----------------------------
if __name__ == "__main__":

    timing = TimingParamsV2(
        edge_length=60.0,
        v_min=2,
        v_max=10,
        ground_delay_max=120.0,
        n_flight_levels=2,
        climb_time_per_level=30.0,
        earliest_climb_levels=1,  # as requested
        latest_climb_levels=2     # as requested
    )
    
    # 1) Build a 9x8 directed grid with unit lengths and its line‑graph dual
    spec = GridSpec(rows=9, cols=8, edge_length=60.0, directed=True, make_dual=True)
    coord_to_id, G = create_graph(spec)

    # 2) Generate one replication of base flight intentions
    F = generate_flight_intentions(repetitions=1, 
                                   timing=timing,
                                   base_flights=BASE_FLIGHTS)

    # 3) Generate per‑node separation requirements from base templates
    Sep_Nodes = generate_separation_nodes(BASE_FLIGHTS, rng=None, min_sep=16, max_sep=28, first_node_value=60)

    # 4) Save to JSON
    out_path = Path("flight_data.json")
    save_results_as_json(out_path, F=F, Sep_Nodes=Sep_Nodes)

    print(f"JSON written to: {out_path.resolve()}")
    
