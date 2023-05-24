import os
import networkx as nx
import matplotlib.pyplot as plt
import csv

folder_path = "dataset/reduced"
file_names = os.listdir(folder_path)
global_clustering_coeffs = []
average_clustering_coeffs = []

for file_name in file_names:
    # Percorso completo del file di testo
    file_path = os.path.join(folder_path, file_name)
    if not file_name.startswith('listcontacts'): continue

    # Apertura del file di testo
    with open(file_path, 'r') as file:
        # Lettura dei dati dal file di testo
        data = [line.split("\t") for line in file.readlines() if line.strip()]

    # Controllo se il file è vuoto
    if not data:
        continue

    # Estrazione delle colonne dai dati letti
    source_nodes = [int(row[1]) for row in data]
    destination_nodes = [int(row[2]) for row in data]

    # Creazione di un insieme unico di coppie di nodi
    node_pairs = list(set(zip(source_nodes, destination_nodes)))

    # Creazione del grafo non diretto
    G = nx.Graph()
    G.add_edges_from(node_pairs)

    # Calcolo del coefficiente di clustering globale
    global_clustering_coeff = nx.transitivity(G)
    global_clustering_coeffs.append(global_clustering_coeff)

    # Calcolo del coefficiente di clustering medio
    average_clustering_coeff = nx.average_clustering(G)
    average_clustering_coeffs.append(average_clustering_coeff)


    # Controllo se il grafo è connesso
    if not nx.is_connected(G):
        # Prendi solo la componente più grande del grafo
        largest_component = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_component)


# Rimozione del prefisso e del suffisso dai nomi dei file
file_names = [file_name.replace("listcontacts_2009_", "").replace(".txt", "") for file_name in file_names if file_name.startswith('listcontacts')]

# Salvataggio dei risultati in un file CSV
output_file = "graph_metrics.csv"
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['File', 'Global Clustering Coefficient', 'Average Clustering Coefficient'])
    for i in range(len(file_names)):
        writer.writerow([file_names[i], global_clustering_coeffs[i], average_clustering_coeffs[i]])

# Plotting dei grafici
plt.figure(figsize=(12, 4))

# Grafico per il coefficiente di clustering globale
plt.subplot(2, 1, 1)
plt.plot(file_names, global_clustering_coeffs, marker='o')
plt.xlabel('Reti (Giorni)')
plt.ylabel('Global Clustering Coefficient')
'''plt.title('Andamento del coefficiente di clustering globale')'''
plt.ylim([0, 1])
plt.xticks(rotation=90)  # Ruota i nomi dei file in verticale

# Grafico per il coefficiente di clustering medio
plt.subplot(2, 1, 2)
plt.plot(file_names, average_clustering_coeffs, marker='o')
plt.xlabel('Reti (Giorni)')
plt.ylabel('Average Clustering Coefficient')
'''plt.title('Andamento del coefficiente di clustering medio')'''
plt.ylim([0, 1]) 
plt.xticks(rotation=90)  # Ruota i nomi dei file in verticale

# Mostrare il grafico del coefficiente di clustering medio
'''plt.tight_layout()'''
plt.show()
