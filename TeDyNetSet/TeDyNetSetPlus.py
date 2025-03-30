from collections import defaultdict, deque
import bisect
from heapq import heappush, heappop
import json

class TemporalTransactionGraph:
    def __init__(self):
        self.edges = defaultdict(list)
        self.reverse_edges = defaultdict(list)
        self.last_cycles = []
        self.all_cycles = []
        self.cycle_stats = []

    def add_edge(self, u, v, timestamp, weight, max_depth=6, min_weight_threshold=0, USE_TEMPORAL_CHECK=True):
        print(f"\n[ADD] Edge ({u} -> {v}, t={timestamp}, w={weight})")
        self.last_cycles = []

        bisect.insort(self.edges[u], (timestamp, v, weight))
        bisect.insort(self.reverse_edges[v], (timestamp, u, weight))

        descendants = self._get_descendants(v)
        ancestors = self._get_ancestors(u)
        affected_nodes = descendants.intersection(ancestors).union({u, v})

        print(f"[INFO] Descendants of {v}: {descendants}")
        print(f"[INFO] Ancestors of {u}: {ancestors}")
        print(f"[INFO] Affected nodes: {affected_nodes}")

        if not affected_nodes:
            print("[SKIP] No structural cycle possible.")
            return

        if USE_TEMPORAL_CHECK:
            has_path = self._has_temporal_path(v, u, timestamp, affected_nodes, max_depth=max_depth)
            if not has_path:
                print(f"[SKIP] No temporal path from {v} to {u} before t={timestamp}")
                return

        while True:
            cycle = self._find_best_temporal_cycle_from_node(
                u, v, timestamp, affected_nodes,
                max_depth=max_depth,
                min_weight_threshold=min_weight_threshold
            )
            if not cycle:
                break
            print(f"[CYCLE FOUND] {cycle}")
            self.last_cycles.append(cycle)
            self.all_cycles.append(cycle)

            filtered = [(u1, v1, t1, w) for u1, v1, t1, w in cycle if u1 != "START"]
            min_w = min(w for _, _, _, w in filtered)
            total_w = sum(w for _, _, _, w in filtered)
            avg_w = total_w / len(filtered)
            min_t = min(t for _, _, t, _ in filtered)
            max_t = max(t for _, _, t, _ in filtered)

            self.cycle_stats.append({
                'min_timestamp': min_t,
                'max_timestamp': max_t,
                'length': len(filtered),
                'min_weight': min_w,
                'total_weight': total_w,
                'avg_weight': avg_w
            })

            self._resolve_cycle(cycle)

    def _has_temporal_path(self, start, target, t_max, allowed_nodes, max_depth=6):
        stack = [(start, 0, 0)]
        visited = set()
        while stack:
            node, curr_time, depth = stack.pop()
            if depth > max_depth:
                continue
            for t, neighbor, _ in self.edges[node]:
                if neighbor not in allowed_nodes or t < curr_time or t > t_max:
                    continue
                if neighbor == target:
                    return True
                state = (neighbor, t)
                if state not in visited:
                    visited.add(state)
                    stack.append((neighbor, t, depth + 1))
        return False

    def _find_best_temporal_cycle_from_node(self, u, v, t_max, allowed_nodes, max_depth=6, min_weight_threshold=0):
        heap = [(-0, v, [("START", v, 0, 0)], 0)]
        best_cycle = None
        best_weight = float('-inf')
        visited = set()

        while heap:
            neg_weight, node, path, curr_time = heappop(heap)
            total_weight = -neg_weight

            if len(path) > max_depth:
                continue

            for t, neighbor, weight in self.edges[node]:
                if neighbor not in allowed_nodes:
                    continue
                if t < curr_time or t > t_max:
                    continue

                new_weight = total_weight + weight
                new_path = path + [(node, neighbor, t, weight)]

                if neighbor == u:
                    cycle = new_path + [(u, v, t_max, self._get_edge_weight(u, v, t_max))]
                    cycle_weight = new_weight + self._get_edge_weight(u, v, t_max)
                    if cycle_weight > best_weight:
                        best_cycle = cycle
                        best_weight = cycle_weight
                    continue

                state = (neighbor, t)
                if state not in visited:
                    visited.add(state)
                    if new_weight >= min_weight_threshold:
                        heappush(heap, (-new_weight, neighbor, new_path, t))

        return best_cycle

    def _get_edge_weight(self, u, v, t):
        for ts, dest, w in self.edges[u]:
            if dest == v and ts == t:
                return w
        return 0

    def _get_ancestors(self, node):
        visited = set()
        stack = [node]
        while stack:
            u = stack.pop()
            for _, v, _ in self.reverse_edges[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return visited

    def _get_descendants(self, node):
        visited = set()
        stack = [node]
        while stack:
            u = stack.pop()
            for _, v, _ in self.edges[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        return visited

    def _resolve_cycle(self, cycle):
        min_weight = min(weight for u, v, t, weight in cycle if u != "START")
        print(f"[RESOLVE] Resolving cycle with min weight {min_weight}")
        for u, v, t, _ in cycle:
            edge_list = self.edges[u]
            for i, (ts, dest, w) in enumerate(edge_list):
                if dest == v and ts == t:
                    new_w = w - min_weight
                    if new_w <= 0:
                        edge_list.pop(i)
                    else:
                        edge_list[i] = (ts, dest, new_w)
                    break

            reverse_list = self.reverse_edges[v]
            for i, (ts, src, w) in enumerate(reverse_list):
                if src == u and ts == t:
                    if new_w <= 0:
                        reverse_list.pop(i)
                    else:
                        reverse_list[i] = (ts, src, new_w)
                    break

    def get_cycles_during_last_edge(self):
        return self.last_cycles

    def get_cycle_statistics(self):
        return self.cycle_stats

    def export_cycles_to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.all_cycles, f, indent=2)

    def print_graph(self):
        print("\n[GRAPH] Current Edgelist:")
        for u in sorted([key for key in self.edges if isinstance(key, int)]):
            for t, v, w in sorted(self.edges[u]):
                print(f"({u} -> {v}, t={t}, w={w})")

    def get_edgelist(self):
        edgelist = []
        for u in self.edges:
            if isinstance(u, str):
                continue  # skip 'START'
            for t, v, w in self.edges[u]:
                edgelist.append((u, v, {'t': t, 'w': w}))
        return sorted(edgelist, key=lambda x: x[2]['t'])