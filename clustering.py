import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from external_script.encoder import CompactJSONEncoder

MAX_HEIGHT = 0.25
CLUSTERING_FOLDER = "clustering_data"
CLUSTERING_FILE = os.path.join(CLUSTERING_FOLDER, "clusters.json")
full_df = None


def min_max_normalization(df):
    min = df.min()
    max = df.max()
    return ((df - min) / (max - min))


def max_tot_norm(cluster):
    return df.filter(items=cluster, axis=0)['tot_norm'].max()    


def map_group(date, clusters):
    for key, cluster in clusters.items():
        if date in cluster: return key


def get_clusters_dict(linkage_matrix, threshold=MAX_HEIGHT):
    # Set the threshold for the maximum distance between points (used to cut the tree)
    clusters = sch.fcluster(Z=linkage_matrix, t=threshold, criterion='distance')
    
    # Create a dictionary of clusters and the elements belonging to each cluster
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[int(cluster_id)] = []
        cluster_dict[int(cluster_id)].append(i)
    return cluster_dict


def get_centroids(data : pd.DataFrame, linkage_matrix, threshold):
    cluster_assignments = sch.fcluster(linkage_matrix, t=threshold, criterion='distance')
    unique_clusters = np.unique(cluster_assignments)
    
    centroid_indices = []
    for cluster in unique_clusters:
        cluster_points = data[cluster_assignments == cluster]
        centroid = np.mean(cluster_points, axis=0)
        # Calculate the Euclidean distance between each cluster point and the centroid vector
        dist_from_centroid = np.linalg.norm(cluster_points.sub(centroid, axis='columns'), axis=1)
        # Add the point having the least distance
        closest_point = np.argmin(dist_from_centroid)
        centroid_indices.append(cluster_points.index[closest_point])
    
    return df.iloc[centroid_indices]


if __name__ == '__main__':
    with open('metrics.json', 'r') as metrics:
        json_data = json.load(metrics)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    
    full_df = pd.DataFrame.from_dict(json_data, orient='index')
    full_df.reset_index(level=0, inplace=True, names=['file'])
    df = full_df
    df['mean_degree'] = df['degree'].apply(lambda x: np.mean(x))
    
    # Normalize dataframe
    df['num_nodes_norm'] = min_max_normalization(df['num_nodes'])
    df['mean_degree_norm'] = min_max_normalization(df['mean_degree'])
    df['tot_norm'] = df['num_nodes_norm'] + df['mean_degree_norm']
    
    # Compute the distance matrix using the Euclidean distance metric
    data = df[['num_nodes_norm', 'mean_degree_norm']]
    dist_matrix = sch.distance.pdist(data, metric='euclidean')   
    
    # Perform hierarchical clustering using the computed distance matrix and ward linkage method
    linkage_matrix = sch.linkage(dist_matrix, method='centroid')
    
    # Print the centroid elements
    centroid_values = get_centroids(data, linkage_matrix, threshold=MAX_HEIGHT)
    print('\nCentroids: \n')
    print(centroid_values)
    
    # Plot the dendrogram
    labels = df['file'].str[18:-4].to_list()
    dendrogram = sch.dendrogram(linkage_matrix, color_threshold=MAX_HEIGHT, labels=labels)
    plt.tight_layout()
    plt.show()
    
    # 1) Get clusters under MAX_HEIGHT; 2) Sort them by the 'tot_norm' column; 3) Assign them a date label 
    cluster_dict = get_clusters_dict(linkage_matrix)
    cluster_dict = dict(enumerate(sorted(cluster_dict.values(), key=max_tot_norm), start=1))
    for key, value in cluster_dict.items():
        value = [labels[v] for v in value]
        cluster_dict[key] = value
    
    # Assign cluster id to metrics days
    json_data = dict(json_data)
    for key, value in json_data.items():
        value['cluster_id'] = map_group(key[18:-4], cluster_dict)
    with open('metrics.json', 'w') as a:
        json.dump(json_data, a, cls=CompactJSONEncoder)
    
    # A pretty way to dump this kind of json file
    string = json.dumps(cluster_dict)
    string = string.replace('], ', '],\n\t')
    string = string.replace('{', '{\n\t')
    string = string.replace('}', '\n}')
    
    # Save clusters in a file
    if not os.path.exists(CLUSTERING_FOLDER):
        os.mkdir(CLUSTERING_FOLDER)
    with open(os.path.join(CLUSTERING_FOLDER, "clusters.json"), 'w') as output:
        output.write(string)

