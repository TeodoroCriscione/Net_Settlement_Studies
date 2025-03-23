import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------- #
# ------------------- # DYNAMIC NET SETTLEMENT # --------------------------- #
# -------------------------------------------------------------------------- #

def sample_dag_incremental(G, edge_fraction=0.1, max_stagnant_attempts=1000):
    """
    Incrementally samples a DAG subgraph from a directed graph G.

    Parameters:
    - G: NetworkX DiGraph (original directed graph)
    - edge_fraction: Fraction of edges to sample (default 10%)
    - max_stagnant_attempts: Max iterations without adding a new edge

    Returns:
    - dag_subgraph: A NetworkX DiGraph that remains a DAG
    - sampled_edges: List of sampled edges (u, v, data)
    """
    if not nx.is_directed(G):
        raise ValueError("Graph must be directed (DiGraph)")

    num_edges = round(edge_fraction * G.number_of_edges())
    if num_edges == 0:
        return nx.DiGraph(), []

    # Initialize
    dag_subgraph = nx.DiGraph()
    all_edges = list(G.edges(data=True))
    random.shuffle(all_edges)

    seen_edges = set()
    sampled_edges = []

    # Start with a valid edge
    while all_edges:
        start_edge = all_edges.pop()
        u, v, data = start_edge
        dag_subgraph.add_edge(u, v, **data)
        if nx.is_directed_acyclic_graph(dag_subgraph):
            sampled_edges.append(start_edge)
            break
        else:
            dag_subgraph.remove_edge(u, v)
    else:
        return nx.DiGraph(), []  # No valid starting edge

    # Candidate edge pool
    candidate_edges = set(G.out_edges(v)).union(G.in_edges(u))
    stagnant_attempts = 0

    while len(sampled_edges) < num_edges and candidate_edges and stagnant_attempts < max_stagnant_attempts:
        edge = candidate_edges.pop()
        if edge in dag_subgraph.edges or edge in seen_edges:
            continue

        seen_edges.add(edge)
        u, v = edge
        data = G.get_edge_data(u, v)

        dag_subgraph.add_edge(u, v, **data)
        if nx.is_directed_acyclic_graph(dag_subgraph):
            sampled_edges.append((u, v, data))
            # Add new candidates
            candidate_edges.update(G.out_edges(v))
            candidate_edges.update(G.in_edges(u))
            stagnant_attempts = 0  # reset on success
        else:
            dag_subgraph.remove_edge(u, v)
            stagnant_attempts += 1


    # Extract largest weakly connected component (optional cleanup)
    if dag_subgraph.number_of_nodes() > 0:
        largest_wcc = max(nx.weakly_connected_components(dag_subgraph), key=len)
        dag_subgraph = dag_subgraph.subgraph(largest_wcc).copy()

    return dag_subgraph, sampled_edges

def dag_sampling(network):
    # DAG sampling
    dag_sg, sampled_edges = sample_dag_incremental(network, edge_fraction=0.1, max_stagnant_attempts=1000)
    # round for floating errors
    sg = nx.DiGraph()
    for e in dag_sg.edges(data=True):
        source, target, data = e
        sg.add_edge(source, target, weight = int(round(data['weight'])))
    not_sampled_edges = [(e[0], e[1],{'weight': int(round(e[2]['weight']))}) for e in network.edges(data=True) if e not in sampled_edges]
    #
    return sg, not_sampled_edges

def find_affected_nodes(network, source, target):
    """
    Finds the subgraph affected by the addition of an edge (source -> target).
    The affected nodes are those reachable from target and those that can reach source.
    
    Parameters:
    - network: The DAG where the cycle is checked.
    - source: The start node of the new edge.
    - target: The end node of the new edge.
    
    Returns:
    - affected_nodes: A set of nodes that belong to the cycles.
    """
    # Find nodes reachable from target (forward search)
    reachable_from_target = set(nx.descendants(network, target))
    reachable_from_target.add(target)

    # Find nodes that can reach source (reverse search)
    can_reach_source = set(nx.ancestors(network, source))
    can_reach_source.add(source)

    # The affected nodes are those in the intersection of both sets
    affected_nodes = reachable_from_target & can_reach_source
    return affected_nodes

def find_cycles_tarjan(network, affected_nodes, source, target, edge_weight):
    """
    Identifies all cycles using Tarjan's SCC algorithm and the minimum weight 
    among all edges in those cycles.
    
    Parameters:
    - network: The current DAG.
    - affected_nodes: Nodes involved in cycles.
    - source: Start node of the new edge.
    - target: End node of the new edge.
    - edge_weight: The weight of the edge being added.

    Returns:
    - cycles: List of cycles (each cycle is a list of edges).
    - min_weight: Minimum weight among all cycle edges.
    """
    # Extract the affected subgraph
    subgraph = network.subgraph(affected_nodes).copy()

    # Add the new edge *safely*
    subgraph.add_edge(source, target, weight=edge_weight)

    # Find Strongly Connected Components (SCCs)
    sccs = list(nx.strongly_connected_components(subgraph))

    # Identify cycles (SCCs with more than one node)
    cycles = []
    min_weight = float('inf')

    for scc in sccs:
        if len(scc) > 1:  # A cycle exists in this SCC
            cycle_nodes = list(scc)
            cycle_edges = [
                (u, v) for u in cycle_nodes for v in cycle_nodes if subgraph.has_edge(u, v)
            ]
            cycles.append(cycle_edges)

            # Find the minimum edge weight in this cycle
            for u, v in cycle_edges:
                weight = subgraph[u][v]['weight']
                min_weight = min(min_weight, weight)

    return cycles, min_weight if cycles else None

def remove_min_weight_edges_and_adjust(network, cycles, min_weight):
    """
    Removes all edges with the minimum weight from the cycles and subtracts
    the minimum weight from remaining edges in those cycles.

    Parameters:
    - network: The DAG where cycles are checked.
    - cycles: List of cycles detected.
    - min_weight: Minimum weight found in the cycles.
    """
    min_weight = min_weight
    if not cycles or min_weight is None:
        return  # Nothing to remove

    edges_to_remove = set()

    for cycle_edges in cycles:
        # Remove edges with min_weight and adjust the rest
        for u, v in cycle_edges:
            if network.has_edge(u, v):  # Check if edge still exists
                edge_weight = network[u][v]['weight']

                 # Remove the edge with exactly min_weight
                if abs(edge_weight - min_weight) < 1e-6:
                    edges_to_remove.add((u, v))  # Mark for removal
                else:
                    # Correct way to update weight in the graph
                    network[u][v]['weight'] -= min_weight
                    # Now check the updated weight in the graph
                    if abs(network[u][v]['weight']) < 1e-6:
                        edges_to_remove.add((u, v))

    # Remove all edges with the minimum weight
    network.remove_edges_from(edges_to_remove)

    # Recursively check if cycles remain after removal using affected nodes
    affected_nodes = {node for cycle in cycles for edge in cycle for node in edge}  # Collect nodes involved in cycles

    while True:
        remaining_cycles, new_min_weight = find_cycles_tarjan(network, affected_nodes, "", "", 0)  # Fix: Empty strings and 0 weight
        if not remaining_cycles:
            break  # Stop when no more cycles are detected
        remove_min_weight_edges_and_adjust(network, remaining_cycles, new_min_weight)

def cycle_cancelling(edge, network):
    """
    Adds an edge to the network and removes all cycles introduced by it using Tarjan's SCC.

    Parameters:
    - edge: The edge (u, v, {'weight': w}) to be added.
    - network: The DAG where the cycle is checked.

    Returns:
    - network: The updated DAG after cycle cancellation.
    """
    source, target, data = edge
    edge_weight = data['weight'] # Extract weight safely

    # Add the new edge to the network
    network.add_edge(source, target, **data)

    # Identify affected nodes
    affected_nodes = find_affected_nodes(network, source, target)
    
    if not affected_nodes:
        return network  # No cycles can exist, return early

    while True:
        # Find cycles and minimum weight in those cycles using Tarjan's SCC
        cycles, min_weight = find_cycles_tarjan(network, affected_nodes, source, target, edge_weight)
        
        if not cycles:
            break  # No more cycles

        # Remove min-weight edges and adjust remaining cycle edges
        remove_min_weight_edges_and_adjust(network, cycles, min_weight)

    return network

# -------------------------------------------------------------------------- #
# ------------------- # STATIC NET SETTLEMENT # --------------------------- #
# -------------------------------------------------------------------------- #


def min_cost_graph_transformation(network):

    if not nx.is_weakly_connected(network):
        print('Error: network not connected')
        return None

    originalGraph = nx.DiGraph()

    # define 'capacity'
    capacity = {(e[0], e[1]):int(round(e[2]['weight'])) for e in network.edges(data=True)}
    capacity =  {k: (1 if v == 0 else v) for k, v in capacity.items()}
    for e in capacity.keys():
        # capacity is rounded to avoid floating errors
        originalGraph.add_edge(e[0],e[1],capacity=int(round(capacity[(e[0],e[1])])))
    nx.set_edge_attributes(originalGraph, capacity, 'capacity')

    # define 'demand' based on the (rounded) capacity of edges
    demand = {n:0 for n in originalGraph.nodes()}
    for e in originalGraph.edges(data=True):
        demand[e[0]] -= e[2]['capacity']
        demand[e[1]] += e[2]['capacity']
    nx.set_node_attributes(originalGraph, demand, 'demand')

    # calculate flow
    flowDict = nx.min_cost_flow(originalGraph, demand='demand', capacity='capacity', weight=None)
    simplified_graph = {}
    for k in flowDict.keys():
        for j in flowDict[k].keys():
            if flowDict[k][j] > 0:
                simplified_graph[(k,j)] = int(flowDict[k][j])

    # create graph
    transformedGraph = nx.DiGraph()
    for e in simplified_graph.keys():
        transformedGraph.add_edge(e[0], e[1], weight=int(simplified_graph[(e[0],e[1])]))

    return transformedGraph

# -------------------------------------------------------------------------- #
# ------------------- # MEASURE NET SETTLEMENT # --------------------------- #
# -------------------------------------------------------------------------- #

# ecological network analysis (ENA): systemic sustainability #


def no_float_tranform(network):
    edgelist = {(e[0], e[1]):int(round(e[2]['weight'])) for e in network.edges(data=True)}
    edgelist =  {k: (1 if v == 0 else v) for k, v in edgelist.items()}
    n_network = nx.DiGraph()
    for e in edgelist.keys():
        n_network.add_edge(e[0], e[1], weight= int(round(edgelist[(e[0], e[1])])))
    return n_network

def ena_node_flow_calculator(network):
    node_inflow = {n:0 for n in network.nodes()}
    node_outflow = {n:0 for n in network.nodes()}
    for e in network.edges(data=True):
        node_inflow[e[1]] += e[2]['weight']
        node_outflow[e[0]] += e[2]['weight']
    return node_inflow, node_outflow

def ena_systemic_metrics_log2(network):
    network = no_float_tranform(network)
    node_inflow, node_outflow = ena_node_flow_calculator(network)
    volume = int(round(network.size('weight')))
    ascendency_results = []
    reserve_results = []
    capacity_results = []
    phi_results = []

    effective_flows_results = []
    effective_nodes_results = []
    effective_roles_results = []

    for e in network.edges(data=True):
        w = int(round(e[2]['weight']))

        edge_ascendency = w * np.log2((w*volume)/(node_outflow[e[0]]*node_inflow[e[1]]))
        edge_reserve = w * np.log2((w**2)/(node_outflow[e[0]]*node_inflow[e[1]]))
        edge_capacity = w * np.log2(w/volume)
        edge_phi = ((w**2)/(node_outflow[e[0]]*node_inflow[e[1]]))**((-1)*(1/2)*(w/volume))

        flow_eff = (w/volume)**((-1)*(w/volume))
        node_eff = ((volume**2/(node_inflow[e[1]]*node_outflow[e[0]])))**((1/2)*(w/volume))
        role_eff = ((w*volume)/(node_outflow[e[0]]*node_inflow[e[1]]))**(w/volume)

        ascendency_results.append(edge_ascendency)
        reserve_results.append(edge_reserve)
        capacity_results.append(edge_capacity)
        phi_results.append(edge_phi)

        effective_flows_results.append(flow_eff)
        effective_nodes_results.append(node_eff)
        effective_roles_results.append(role_eff)
    #
    ascendency = sum(ascendency_results)
    reserve = (-1)*sum(reserve_results)
    capacity = (-1)*sum(capacity_results)
    #phi = np.log2(np.prod(phi_results))
    #phi_connectivity = 2**(phi/2)
    #effective_flow = round(np.prod(effective_flows_results), 2)
    #effective_nodes = round(np.prod(effective_nodes_results), 2)
    #effective_roles = round(np.prod(effective_roles_results), 2)
    #average_mutual_information = np.log2(np.prod(effective_roles_results))

    return ascendency, reserve, capacity

# GINI index #

def gini(values):
    """Compute Gini index for a list or array of non-negative values."""
    values = np.array(values)
    if np.amin(values) < 0:
        raise ValueError("Gini index is not defined for negative values.")
    if np.all(values == 0):
        return 0.0

    sorted_vals = np.sort(values)
    n = len(values)
    cumulative_sum = np.cumsum(sorted_vals)
    gini_index = (2 * np.sum((np.arange(1, n+1) * sorted_vals))) / (n * np.sum(sorted_vals)) - (n + 1) / n
    return gini_index

def plot_lorenz(values, title="Lorenz Curve"):
    values = np.array(values)
    if np.any(values < 0):
        raise ValueError("Lorenz curve not defined for negative values.")
    if np.all(values == 0):
        values = np.ones_like(values)

    sorted_vals = np.sort(values)
    cumulative_vals = np.cumsum(sorted_vals)
    cumulative_vals = cumulative_vals / cumulative_vals[-1]
    cumulative_population = np.linspace(0, 1, len(values), endpoint=False)
    cumulative_population = np.append(cumulative_population, 1)
    cumulative_vals = np.append([0], cumulative_vals)

    plt.figure(figsize=(6, 6))
    plt.plot(cumulative_population, cumulative_vals, label='Lorenz Curve', color='blue')
    plt.plot([0, 1], [0, 1], label='Perfect Equality', linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel("Cumulative Share of Nodes")
    plt.ylabel("Cumulative Share of Weighted Degree")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
