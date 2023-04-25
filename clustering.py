import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

MAX_DISTANCE = 0.08
CLUSTERING_FOLDER = "clustering_data"
CLUSTERING_FILE = os.path.join(CLUSTERING_FOLDER, "clusters.json")

def min_max_normalization(df):
    min = df.min()
    max = df.max()
    return ((df - min) / (max - min))

def get_clusters_dict(linkage_matrix, max_distance=MAX_DISTANCE):
    # Set the threshold for the maximum distance between points (used to cut the tree)
    clusters = sch.fcluster(linkage_matrix, max_distance, criterion='distance')
    
    # Create a dictionary of clusters and the elements belonging to each cluster
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[int(cluster_id)] = []
        cluster_dict[int(cluster_id)].append(i)
    return cluster_dict

if __name__ == '__main__':
    if not os.path.exists(CLUSTERING_FOLDER):
        os.mkdir(CLUSTERING_FOLDER)
    df = pd.read_csv(filepath_or_buffer="0.netsummary.csv")
    
    # Normalize dataframe and save it to csv
    df['num_nodes_norm'] = min_max_normalization(df['num_nodes'])
    df['num_edges_norm'] = min_max_normalization(df['num_edges'])
    df.to_csv(path_or_buf=os.path.join(CLUSTERING_FOLDER, 'netsummary_normalized.csv'))
    
    # Compute the distance matrix using the Euclidean distance metric
    data = df[['num_nodes_norm', 'num_edges_norm']]
    dist_matrix = sch.distance.pdist(data, metric='euclidean')   
    
    # Perform hierarchical clustering using the computed distance matrix and centroid linkage method
    linkage_matrix = sch.linkage(dist_matrix, method='centroid')
    
    # Plot the dendrogram and the distance threshold to visualize the results
    dendrogram = sch.dendrogram(linkage_matrix)
    plt.axhline(y=MAX_DISTANCE, color='r', linestyle='-')
    plt.show()
    
    # Get clusters at MAX_DISTANCE
    cluster_dict = get_clusters_dict(linkage_matrix)
    cluster_dict = dict(sorted(cluster_dict.items()))
    string = json.dumps(cluster_dict)
    # A pretty way to dump this json file
    string = string.replace('], ', '],\n\t')
    string = string.replace('{', '{\n\t')
    string = string.replace('}', '\n}')
    with open(os.path.join(CLUSTERING_FOLDER, "clusters.json"), 'w') as output:
        output.write(string)

