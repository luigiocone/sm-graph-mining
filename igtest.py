import os
import json
import pytz
import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import DS_TRANSFORMED
from external_script.encoder import CompactJSONEncoder

# matplotlib colors
mpl_colors = ["mediumpurple", "pink", "salmon", "khaki", "lightgreen", "lightskyblue", "azure", "silver", "black"]
# mpl_colors = ["mediumpurple", "crimson", "salmon", "khaki", "lightgreen", "seagreen", "steelblue", "silver", "black"]
# mpl_colors = ["mediumpurple", "violet", "salmon", "khaki", "lightgreen", "seagreen", "steelblue", "silver", "black"]

def plot_graph(g : ig.Graph):
    g["title"] = "Test network"
    fig, ax = plt.subplots(figsize=(5,5))
    
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
        if (w < 10): value = 0.2
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
    # Enable LaTeX text rendering (requires latex istalled and accessible from python environment)
    #plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    #plt.rcParams.update({"font.family": "monospace"})
    #ax.legend(handles=handles, labels=labels, fontsize=15)
    plt.show()

def build_graph(df : pd.DataFrame) -> ig.Graph:
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
    df = pd.melt(df, id_vars='timestamp', value_vars=['node1', 'node2'], value_name='node')
    df = df[['timestamp', 'node']]
    # Group by node (hence drop duplicates) and keep the row with the minumum timestamp
    df = df.loc[df.groupby('node', sort=False)['timestamp'].idxmin()]
    # Finally, assign entry datetime to vertices
    df = df.set_index('node')
    g.vs['entry'] = [df.loc[v.index, 'timestamp'] for v in g.vs]
    return g

def get_brokers(g, max_degree=3, btw_quantile=0.85):
    # To modify to get a list of nodes
    btw = g.betweenness()
    degrees = g.degree()
    max_b = np.quantile(btw, btw_quantile); 

    brokers = []
    for pair in zip(btw, degrees):
        if pair[0] >= max_b and pair[1] <= max_degree:
            brokers.append(pair)
    return brokers


if __name__ == '__main__':
    rows = {}
    for file_name in os.listdir(DS_TRANSFORMED):
        if not file_name.startswith("listcontacts"): continue
        df = pd.read_csv(os.path.join(DS_TRANSFORMED, file_name), delimiter=',')
        g = build_graph(df)
        # plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
        # plot_graph(g)
        
        # Build a metrics row for current network
        rows[file_name] = {
            "num_nodes" : len(g.vs),
            "num_edges" : len(g.es),
            "diameter": g.diameter(),
            "density" : g.density(loops=False),
            "bridges" : len(g.bridges()),
            "brokers" : len(get_brokers(g)),
            "connected_components" : len(g.connected_components()),
            "degree" : g.degree(),
            "betweenness" : g.betweenness(),
            "closeness" : g.closeness(),
            "eigenvector" : g.eigenvector_centrality(),   # To modify to take into account weights
            "local_clustering_coeff" : g.transitivity_local_undirected(mode=ig.TRANSITIVITY_ZERO),
            "global_clustering_coeff" : g.transitivity_undirected()
        }
    
    text = json.dumps(rows, cls=CompactJSONEncoder)
    # text = text.replace('nan', 'null')
    with open('metrics.json', 'w') as output:
        output.write(text)
    