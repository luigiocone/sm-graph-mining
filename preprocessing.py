import os
import datetime
import numpy as np
import pandas as pd

DS_ORIGINAL = os.path.join("dataset", "original")
DS_REDUCED = os.path.join("dataset", "reduced")

def net_summary(dir, delimiter) -> pd.DataFrame:
    """ Return a dataframe reporting the number of nodes and edges for each network file """
    rows = []
    for file_name in os.listdir(dir):
        # Read current file in dataset directory
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(dir, file_name), 
            delimiter=delimiter,
            header=None,
            names=['timestamp', 'node1', 'node2']
        )
        
        # Get num of distinct IDs in current file
        n1_ids = df['node1'].unique()
        n2_ids = df['node2'].unique()
        tot_ids = np.union1d(n1_ids, n2_ids)
        
        # Save this data as a row in the final dataframe
        row = {
            'file': file_name,
            'num_nodes': tot_ids.shape[0],
            'num_edges': df.shape[0]
        }
        rows.append(row)
    return pd.DataFrame(rows)

def to_weekday(file_name):
    file_name = file_name[13:-4]
    weekday = datetime.datetime.strptime(file_name, '%Y_%m_%d').strftime('%a')
    return weekday

def reduce_edges(df : pd.DataFrame, duration : int):
    """ Remove connections between the nodes pairs that are less than 'duration' seconds apart """    
    i = 0
    while i < df.shape[0]:
        row = df.iloc[i]
        rm_mask = (
            (df.index > i) & 
            (df['timestamp'] < (row['timestamp'] + duration)) & 
            (df['node1'] == row['node1']) & 
            (df['node2'] == row['node2'])
        )
        df = df[~rm_mask]
        df = df.reset_index(drop=True)
        i += 1
    return df

if __name__ == '__main__':
    # Get a net summary of tsv files (tab-separated-value) and save it in a csv file
    df = net_summary(DS_ORIGINAL, delimiter='\t')
    print('Total nodes: ' + str(df['num_nodes'].sum()))
    print('Total edges: ' + str(df['num_edges'].sum()))
    df['weekday'] = df['file'].apply(lambda x: to_weekday(x))
    df.to_csv(path_or_buf='0.netsummary.csv', index=False)
    
    # Stop execution if output dir already exists
    if os.path.exists(DS_REDUCED):
        print(f'Directory "{DS_REDUCED}" already exists')
        exit(0)
    os.mkdir(DS_REDUCED)
    
    # Apply reductions and transformations to all dataset files
    for file_name in os.listdir(DS_ORIGINAL):
        df = pd.read_csv(
            filepath_or_buffer=os.path.join(DS_ORIGINAL, file_name), 
            delimiter='\t', # tsv (tab-separated-value) file
            header=None,    # files don't have an header row
            names=['timestamp', 'node1', 'node2']
        )
        
        # Replace complex IDs with incremental IDs
        curr = 0
        tot_ids = np.union1d(df['node1'].unique(), df['node2'].unique())
        for id in tot_ids:
            df[['node1','node2']] = df[['node1','node2']].replace(id, curr)
            curr += 1
        
        # Since the network is undirected, use a standard connection order of the nodes for each row
        df[['node1', 'node2']] = np.sort(df[['node1', 'node2']], axis=1)
        
        # Reduce number of edges of network files
        df = reduce_edges(df, duration=60)
        
        # Save reduced file
        df.to_csv(path_or_buf=os.path.join(DS_REDUCED, file_name), index=False, header=False)
        print("File " + file_name + " reduced")
    df = net_summary(DS_REDUCED, delimiter=',')
    df.to_csv(path_or_buf=os.path.join(DS_REDUCED, '1.netsummary_(reduced).csv'), index=False)
