import os
import pytz
import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import DS_REDUCED

# matplotlib colors
mpl_colors = ["mediumpurple", "crimson", "salmon", "khaki", "lightgreen", "seagreen", "steelblue", "silver", "black"]

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
    
    # Building legend labels and handles 
    labels = []
    handles = []
    for i in range(max_hour-min_hour+1):
        labels.append(f'{min_hour+i}:00 - {min_hour+i}:59')
        #handle = plt.Line2D([], [], linewidth=1.5, marker='o', color = "black", markeredgewidth=1.5, markersize=8, markerfacecolor=mpl_colors[i])
        handle = plt.Line2D([], [], linestyle='', marker='o', color=mpl_colors[i], markersize=8)
        handles.append(handle)

    labels.append(r'$< 10$ min')
    handles.append(plt.Line2D([], [], linestyle='-', linewidth=0.5, marker='', color='black'))
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
    ax.legend(handles=handles, labels=labels, fontsize=8)
    plt.show()

def build_graph(df : pd.DataFrame):
    # Add a column of ones representing the intensity of a contact. One is the lowest intensity, will be aggregated later 
    df['intensity'] = pd.Series(1, df.index)
    
    # Drop duplicated edges and aggregate weights and timestamp for each edge removal
    df = df.groupby(['node1', 'node2'], as_index=False, sort=False).agg({'intensity':'sum', 'timestamp':'first'})
    
    # Build graph from a list of edges
    edges = list(zip(df['node1'], df['node2']))
    g = ig.Graph(edges=edges, directed=False)
    
    # Assign weights to edges (contacts between humans) and name to vertices (visitor)
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


if __name__ == '__main__':
    metrics = []
    for file_name in os.listdir(DS_REDUCED):
        if not file_name.startswith("listcontacts"): continue
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(DS_REDUCED, file_name), 
            delimiter=',',
            header=None,
            names=['timestamp', 'node1', 'node2']
        )
        g = build_graph(df)
        # plot_graph(g)
        
        # Build a metrics row for current network
        row = {
            "file": file_name,
            "diameter": g.diameter(),
            "avg_closeness": sum(g.closeness()) / len(g.closeness()),
            "avg_degree": sum(g.degree()) / len(g.degree()),
            "avg_betweenness": sum(g.betweenness()) / len(g.betweenness()),
            "avg_local_clustering_coeff": g.transitivity_avglocal_undirected(),
            "avg_global_clustering_coeff": g.transitivity_undirected()
        }
        metrics.append(row)
    
    df = pd.DataFrame(metrics)
    df.to_csv(path_or_buf='metrics.csv', index=False)
