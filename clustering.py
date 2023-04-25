import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import pandas as pd
import json

MAX_DISTANCE = 0.08

def min_max_normalization(df):
    min = df.min()
    max = df.max()
    return ((df - min) / (max - min))

df = pd.read_csv(filepath_or_buffer="0.netsummary.csv")
df['num_nodes'] = min_max_normalization(df['num_nodes'])
df['num_edges'] = min_max_normalization(df['num_edges'])

# Compute the distance matrix using the Euclidean distance metric
data = df[['num_nodes', 'num_edges']]
dist_matrix = sch.distance.pdist(data, metric='euclidean')

# Perform hierarchical clustering using the computed distance matrix and chosen linkage method
linkage_matrix = sch.linkage(dist_matrix, method='centroid')

# Plot the dendrogram and the cut-line to visualize the results
dendrogram = sch.dendrogram(linkage_matrix)
plt.axhline(y=MAX_DISTANCE, color='r', linestyle='-')
plt.show()

# get the cluster assignments
# set the threshold for the maximum distance between points
clusters = sch.fcluster(linkage_matrix, MAX_DISTANCE, criterion='distance')

# create a dictionary of clusters and the elements belonging to each cluster
cluster_dict = {}
for i, cluster in enumerate(clusters):
    if cluster not in cluster_dict:
        cluster_dict[int(cluster)] = []
    cluster_dict[int(cluster)].append(i)

# Serialize data into file:
cluster_dict = dict(sorted(cluster_dict.items()))
with open("clusters.json", 'w') as output:
    string = json.dumps(cluster_dict)
    # A prittier way to dump the json file
    string = string.replace('], ', '],\n\t')
    string = string.replace('{', '{\n\t')
    string = string.replace('}', '\n}')
    output.write(string)

