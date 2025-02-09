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
            path_edges_with_data = sorted(path_edges_with_data, key= lambda x: x[2]['weight'],reverse=True)
            min_weight = path_edges_with_data[0][2]['weight']
            path_edges_to_add = [(s,t,{'weight':w['weight']-min_weight}) for s,t,w in path_edges_with_data[1:]]
            path_edges_to_add = [e for e in path_edges_to_add if e[2]['weight'] != 0]
            # update graph 
            G.add_edges_from(path_edges_to_add)
    return G

# to maximize the flow cycle cancelling using the Dijkstra algorithm
# the weights need to be negative
def negGraph (G, bundle):
    # Create a copy of the graph with negative edge weights
    G_neg = G.copy()
    for u, v, data in G_neg.edges(data=True):
        data['weight'] = -data['weight']  # Negate the weight
    # Create a copy of the bundle with negative weights
    bundle = [(s,t,{'weight':(-1)*w['weight']}) for s,t,w in bundle]
    # return
    return G_neg, bundle

# in case of concurrent cycles, choose the one with largest weights
def cycle_cancelling(G, bundle):
    G_neg, bundle = negGraph (G, bundle)
    G_neg = update_graph(G_neg, bundle)
    G = G_neg.copy()
    for u, v, data in G.edges(data=True):
        data['weight'] = (-1)*data['weight']  # positive weight
    return G 
