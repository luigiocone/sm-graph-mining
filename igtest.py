import os
import igraph as ig
import pandas as pd

DS_REDUCED = os.path.join("dataset", "reduced")

metrics = []
for file_name in os.listdir(DS_REDUCED):
    if not file_name.startswith("listcontacts"): continue
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(DS_REDUCED, file_name), 
        delimiter=',',
        header=None,
        names=['timestamp', 'node1', 'node2']
    )
    
    # ASK TO PROF
    df = df.drop_duplicates(subset=['node1', 'node2'], keep='first')
    
    # Incidence matrix
    edges = list(zip(df['node1'], df['node2']))
    g = ig.Graph(edges)
    
    # Some metrics
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


#n_vertex = max(df['node1'].max(), df['node2'].max()) + 1
