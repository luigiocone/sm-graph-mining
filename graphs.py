import os
import json
import pytz
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from preprocessing import DS_TRANSFORMED
from external_script.encoder import CompactJSONEncoder

# matplotlib colors
mpl_colors = ["mediumpurple", "pink", "salmon", "khaki", "lightgreen", "lightskyblue", "azure", "silver", "black"]
# mpl_colors = ["mediumpurple", "crimson", "salmon", "khaki", "lightgreen", "seagreen", "steelblue", "silver", "black"]
# mpl_colors = ["mediumpurple", "violet", "salmon", "khaki", "lightgreen", "seagreen", "steelblue", "silver", "black"]


def plot_graph(g: ig.Graph):
    g["title"] = "Test network"
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Node color based on entry time-slot
    colors = []
    min_hour = min(g.vs['entry']).hour
    max_hour = max(g.vs['entry']).hour
    for entry in g.vs['entry']:
        color_id = entry.hour - min_hour # This ID is in range(0, 8)
        colors.append(mpl_colors[color_id])
    
    # Edge thickness based on contact duration
    edge_width = []
    for w in g.es['intensity']:
        if w < 10: value = 0.2
        else: value = 2
        edge_width.append(value)
    
    """# Edge thickness=3 if that edge is a bridge, else 0.1
    edge_width = [0.1] * len(g.es)
    for bridge in g.bridges():
        edge_width[bridge] = 3"""
    
    # Building legend labels and handles 
    labels = []
    handles = []
    for i in range(max_hour-min_hour+1):
        labels.append(f'{min_hour+i}:00 - {min_hour+i}:59')
        #handle = plt.Line2D([], [], linewidth=1.5, marker='o', color = "black", markeredgewidth=1.5, markersize=8, markerfacecolor=mpl_colors[i])
        handle = plt.Line2D([], [], linestyle='', marker='o', color=mpl_colors[i], markersize=10)
        handles.append(handle)

    labels.append(r'$< 10$ min')
    handles.append(plt.Line2D([], [], linestyle='-', linewidth=0.2, marker='', color='black'))
    labels.append(r'$ \geq 10$ min')
    handles.append(plt.Line2D([], [], linestyle='-', linewidth=2, marker='', color='black'))

    # Graph and legend plots
    ig.plot(
        g,
        target=ax,
        vertex_size=0.3,
        vertex_color=colors,
        edge_width=edge_width,
        #vertex_label=g.vs['name'],
        #layout="circle", # print nodes in a circular layout
        #vertex_frame_width=4.0,
        #vertex_frame_color="white",
        #vertex_label_size=7.0,
        #edge_color=["#7142cf" if married else "#AAA" for married in g.es["married"]]
    )
    
    plt.legend(handles=handles, labels=labels, fontsize=11)
    plt.tight_layout()
    plt.show()


def build_graph(df: pd.DataFrame) -> ig.Graph:
    # Build graph from a list of edges
    edges = list(zip(df['node1'], df['node2']))
    g = ig.Graph(edges=edges, directed=False)
    
    # Assign weights to edges and IDs to vertices
    g.es['intensity'] = df['intensity'].to_list()
    g.vs['name'] = [v.index for v in g.vs]
    
    # Convert Dublin timestamp to datetime 
    tz = pytz.timezone('Europe/Dublin')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['timestamp'] = df['timestamp'].dt.tz_localize(tz)
    # Melt is equal to the SQL UNPIVOT operation
    df = pd.melt(df, id_vars=['timestamp', 'intensity'], value_vars=['node1', 'node2'], value_name='node')
    df = df[['timestamp', 'node', 'intensity']]
    # Group by node (hence drop duplicates) and keep the row with the minumum timestamp
    df_entry = df.loc[df.groupby('node', sort=False)['timestamp'].idxmin()]
    df_exit = df.loc[df.groupby('node', sort=False)['timestamp'].idxmax()]
    # Finally, assign entry and exit timestamps to vertices
    df_entry = df_entry.set_index('node')
    df_exit = df_exit.set_index('node')
    g.vs['entry'] = [df_entry.loc[v.index, 'timestamp'] for v in g.vs]
    g.vs['exit'] = [df_exit.loc[v.index, 'timestamp'] for v in g.vs]
    # Add the last contact duration to the exit timestamp
    g.vs['exit'] = [v['exit'] + timedelta(seconds=int(df_exit.loc[v.index, 'intensity']) *60) for v in g.vs]
    return g


def get_brokers(g, max_degree=3, btw_quantile=0.85):
    # To modify to get a list of nodes
    btw = g.betweenness()
    degrees = g.degree()
    max_btw = np.quantile(btw, btw_quantile)
    
    brokers = []
    for triple in zip(btw, degrees, g.vs['name']):
        if triple[0] >= max_btw and triple[1] <= max_degree:
            brokers.append(triple[2])
    return brokers


def katz_centrality(largest_cc: ig.Graph, alpha):
    """ Implementation of: c_K = (I - alpha * A')^(-1)*ones - ones """
    # Get the boolean adjacency matrix
    A = largest_cc.get_adjacency()
    A = np.array(A.data)
    
    # Check if "alpha < 1/max(eigenvalues)". Should be used a common alpha for all networks
    sr = get_spectral_radius(largest_cc)
    while alpha >= 1/sr:
        alpha -= 0.001
    print("alpha used during katz centrality computation: " + str(alpha))
    
    # Equation elements
    A = np.array(A.data)
    n = A.shape[0]
    I = np.identity(n)
    ones = np.ones(n)
    
    res = I - alpha * np.transpose(A)      # (I - alpha * A')
    res = np.linalg.inv(res)               # (I - alpha * A')^(-1)
    res = np.dot(res, ones) - ones         # (I - alpha * A')^(-1)*ones - ones
    # res = res / np.linalg.norm(res)      # Normalization used by networkx
    return list(res)


def get_largest_component(g: ig.Graph) -> ig.Graph:
    components = g.connected_components()
    largest = max(components, key=len)
    return g.induced_subgraph(largest)


def get_components_size(g: ig.Graph) -> ig.Graph:
    cc_sizes = [len(c) for c in g.connected_components()]
    cc_sizes.sort()
    return cc_sizes


def get_spectral_radius(largest_cc: ig.Graph) -> ig.Graph:
    A = largest_cc.get_adjacency()
    A = np.array(A.data)
    eig, _ = np.linalg.eig(A)
    modulus = np.abs(max(eig)) # Numpy sometimes return a '+0j' complex part
    return modulus


if __name__ == '__main__':
    # Enable LaTeX text rendering (requires latex installed and accessible from python environment)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    # Or else a different font
    # plt.rcParams.update({"font.family": "monospace"})
    
    rows = {}
    for file_name in os.listdir(DS_TRANSFORMED):
        if not file_name.startswith("listcontacts"): continue
        df = pd.read_csv(os.path.join(DS_TRANSFORMED, file_name), delimiter=',')
        g = build_graph(df)
        # if file_name.endswith('04_28.csv'): plot_graph(g)
        
        weights = [1/w for w in g.es['intensity']]  # igraph consider weights as distances instead of connection strengths
        largest_cc = get_largest_component(g)
        
        # Build a metrics row for current network
        rows[file_name] = {
            "num_nodes": len(g.vs),
            "num_edges": len(g.es),
            "diameter": g.diameter(),
            "density": g.density(loops=False),
            "bridges": len(g.bridges()),
            "brokers": get_brokers(g),
            "brokers_entry_time": [g.vs['entry'][b].strftime('%H:%M') for b in get_brokers(g)],
            "connected_components_size": get_components_size(g),
            "degree": g.degree(),
            "edge_intensities": [i for i in g.es['intensity']] ,
            "closeness": g.closeness(),
            "betweenness": g.betweenness(),                               # [b/len(g.vs)**2 for b in g.betweenness()],
            #"weighted_betweenness": g.betweenness(weights=weights),      # Weights considered as distances
            "katz": katz_centrality(largest_cc, alpha=0.04),              # 0.04 is "1/max(spectral_radius)" 
            "nodes_entry_time": [entry.strftime('%H:%M') for entry in g.vs['entry']],
            "nodes_visit_duration": [(exit - entry).total_seconds()/60 for entry, exit in zip(g.vs['entry'], g.vs['exit'])],
            "largest_connected_component": largest_cc.vs['name'],
            "largest_cc_spectral_radius": get_spectral_radius(largest_cc),
            "largest_cc_density": largest_cc.density(loops=False),
            "largest_cc_num_nodes": len(largest_cc.vs),
            "largest_cc_num_edges": len(largest_cc.es),
            "largest_cc_mean_closeness": np.mean(largest_cc.closeness()),
            "local_clustering_coeff": g.transitivity_local_undirected(mode=ig.TRANSITIVITY_ZERO),
            "global_clustering_coeff": g.transitivity_undirected()
        }
    
    text = json.dumps(rows, cls=CompactJSONEncoder)
    # text = text.replace('nan', 'null')
    with open('metrics.json', 'w') as output:
        output.write(text)


"""
# Katz centrality with networkx. Returns the same result
import networkx as nx
def katz_centrality(g : ig.Graph, alpha = 0.1):
    # Get the adjacency matrix of the connected component with the highest number of nodes
    components = g.connected_components()
    longest = max(components, key=len)
    g = g.induced_subgraph(longest)
    A = g.get_adjacency(attribute='intensity')
    A = np.array(A.data)

    # Graph creation from adjacency matrix
    G = nx.from_numpy_array(A)
    weights = {(u, v): A[u, v] for u, v in G.edges()}
    nx.set_edge_attributes(G, values=weights, name='intensity')
    
    # Build alpha parameter
    eig, _ = np.linalg.eig(A)
    max_eig = max(eig)
    while alpha >= 1/max_eig: 
        alpha -= 0.05
    print("alpha used during katz centrality computation: " + alpha)
    
    # Build beta parameter
    beta_vector = alpha * np.dot(np.transpose(A), np.ones(A.shape[0]))
    beta = {}
    i = 0
    for node in G.nodes():
        beta[node] = beta_vector[i]
        i+=1
    
    res = nx.katz_centrality(G, alpha=alpha, beta=beta, normalized=True, weight='intensity')
    return list(res.values())
"""