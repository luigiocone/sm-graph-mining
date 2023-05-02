import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

MAX_HEIGHT = 0.25
CLUSTERING_FOLDER = "clustering_data"
CLUSTERING_FILE = os.path.join(CLUSTERING_FOLDER, "clusters.json")
full_df = pd.read_csv(filepath_or_buffer="metrics.csv")

def min_max_normalization(df):
    min = df.min()
    max = df.max()
    return ((df - min) / (max - min))

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

def max_tot_norm(cluster):
    return df.filter(items=cluster, axis=0)['tot_norm'].max()    

def map_group(date, clusters):
    for key, cluster in clusters.items():
        if date in cluster: return key

if __name__ == '__main__':
    if not os.path.exists(CLUSTERING_FOLDER):
        os.mkdir(CLUSTERING_FOLDER)
    df = full_df
    
    # Normalize dataframe and save it to csv
    df['num_nodes_norm'] = min_max_normalization(df['num_nodes'])
    df['num_edges_norm'] = min_max_normalization(df['num_edges'])
    df['tot_norm'] = df['num_nodes_norm'] + df['num_edges_norm']
    
    # Compute the distance matrix using the Euclidean distance metric
    data = df[['num_nodes_norm', 'num_edges_norm']]
    dist_matrix = sch.distance.pdist(data, metric='euclidean')   
    
    # Perform hierarchical clustering using the computed distance matrix and ward linkage method
    linkage_matrix = sch.linkage(dist_matrix, method='ward')
    
    # Plot the dendrogram
    labels = df['file'].str[18:-4].to_list()
    dendrogram = sch.dendrogram(linkage_matrix, color_threshold=MAX_HEIGHT, labels=labels)
    plt.axhline(y=MAX_HEIGHT, color='r', linestyle='-')
    plt.show()
    
    # 1) Get clusters under MAX_HEIGHT; 2) Sort them by the 'tot_norm' column; 3) Assign them a date label 
    cluster_dict = get_clusters_dict(linkage_matrix)
    cluster_dict = dict(enumerate(sorted(cluster_dict.values(), key=max_tot_norm), start=1))
    for key, value in cluster_dict.items():
        value = [labels[v] for v in value]
        cluster_dict[key] = value

    # Assign cluster id to files
    df['cluster_id'] = df['file'].apply(lambda str : map_group(str[18:-4], cluster_dict))
    df.to_csv(path_or_buf='metrics.csv')

    # A pretty way to dump this kind of json file
    string = json.dumps(cluster_dict)
    string = string.replace('], ', '],\n\t')
    string = string.replace('{', '{\n\t')
    string = string.replace('}', '\n}')
    with open(os.path.join(CLUSTERING_FOLDER, "clusters.json"), 'w') as output:
        output.write(string)

