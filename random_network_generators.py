import networkx as nx
import numpy as np

def ER_directed_weighted_graph(n, p, x_min, x_max):
    #x_min Minimum weight
    #x_max Maximum weight
    alpha = 1.85 - 1  # Pareto shape parameter for weight distribution (power-law exponent - 1)
    # Create directed ER graph
    G = nx.erdos_renyi_graph(n, p, directed=True)
    # Assign weights from Pareto distribution
    for u, v in G.edges():
        weight = x_min * (1 + np.random.pareto(alpha))  # Sample from Pareto
        weight = min(weight, x_max)  # Cap max weight at 5000
        G[u][v]['weight'] = int(round(round(weight,1)*10))
    return G


# Generate power-law degrees safely
def generate_power_law_degrees(n, gamma, min_degree, max_degree=1000):
    """ Generate degrees following a power-law distribution with limits. """
    degrees = (np.random.pareto(gamma - 1, n) + 1) * min_degree
    degrees = np.clip(degrees, min_degree, max_degree)  # Avoid extreme values
    return degrees.astype(int)  # Ensure integer degrees

# Configuration model generator
def CM_directed_weighted_graph(n, x_min, x_max):
    gamma = 1.5  # Power-law exponent
    min_degree = 1  # Minimum degree
    max_degree = 500  # Max degree to avoid extreme values
    alpha = 1.85 - 1  # Pareto exponent for weights
    frac_source = 0.2  # Fraction of source nodes (only out-degree)
    frac_sink = 0.2  # Fraction of sink nodes (only in-degree)

    num_source = int(n * frac_source)
    num_sink = int(n * frac_sink)
    num_both = n - num_source - num_sink

    while True:
        # Use np.int64 to prevent overflow
        out_degrees = np.zeros(n, dtype=np.int64)
        in_degrees = np.zeros(n, dtype=np.int64)

        # Assign degrees with limits
        out_degrees[:num_source] = generate_power_law_degrees(num_source, gamma, min_degree, max_degree)
        in_degrees[num_source:num_source + num_sink] = generate_power_law_degrees(num_sink, gamma, min_degree, max_degree)
        in_degrees[num_source + num_sink:] = generate_power_law_degrees(num_both, gamma, min_degree, max_degree)
        out_degrees[num_source + num_sink:] = generate_power_law_degrees(num_both, gamma, min_degree, max_degree)

        # Ensure degree sum balance
        diff = int(in_degrees.sum() - out_degrees.sum())

        if diff == 0:
            break  # Balanced, exit loop
        elif diff > 0:
            # Reduce in-degree from nodes with > min_degree
            reduce_indices = np.where(in_degrees > min_degree)[0]
            for i in reduce_indices:
                if diff == 0: break
                in_degrees[i] -= 1
                diff -= 1
        else:
            # Reduce out-degree from nodes with > min_degree
            reduce_indices = np.where(out_degrees > min_degree)[0]
            for i in reduce_indices:
                if diff == 0: break
                out_degrees[i] -= 1
                diff += 1

    # Create directed configuration model
    G = nx.directed_configuration_model(out_degrees, in_degrees)
    G = nx.DiGraph(G)  # Convert to simple directed graph
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops

    # Assign weights from Pareto distribution
    for u, v in G.edges():
        weight = x_min * (1 + np.random.pareto(alpha))
        weight = min(weight, x_max)  # Cap weight at x_max
        G[u][v]['weight'] = int(round(round(weight,1)*10))

    return G
