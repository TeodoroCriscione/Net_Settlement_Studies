import networkx as nx
# Shortest Path
# Dijkstra’s Algorithm O((V+E)logV)
def shortest_path_with_edges(G, source, target, weight="weight"):
    # Get shortest path nodes
    path_nodes = nx.shortest_path(G, source=source, target=target, weight=weight)
    
    # Extract edges and their attributes
    path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
    path_edges_with_data = [(u, v, G[u][v]) for u, v in path_edges]

    return path_edges_with_data

# Depth-First Search (DFS)
# O(V+E)
def is_predecessor(G, target_node, starting_node):
    visited = set()
    stack = [starting_node]
    while stack:
        node = stack.pop()
        if node == target_node:
            return True
        if node not in visited:
            visited.add(node)
            # Traverse backwards (predecessors are other nodes at distance 1)
            stack.extend(G.predecessors(node))  
    return False

# Update Graph
def update_graph(G, bundle):
    for candidate in bundle:
        if is_predecessor(G, candidate[1], candidate[0]):
            # path_edges_with_data
            path_edges_with_data = shortest_path_with_edges(G, candidate[1], candidate[0], weight="weight")
            # delete from graph
            G.remove_edges_from(path_edges_with_data)
            # find minimum weight
            path_edges_with_data.append(candidate)
            path_edges_with_data = sorted(path_edges_with_data, key= lambda x: x[2]['weight'])
            min_weight = path_edges_with_data[0][2]['weight']
            path_edges_to_add = [(s,t,{'weight':w['weight']-min_weight}) for s,t,w in path_edges_with_data[1:]]
            # update graph 
            G.add_edges_from(bundle)
    return G