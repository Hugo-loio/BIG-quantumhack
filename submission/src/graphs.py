import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def draw_random_graph(n, p):
    # Generate a random graph
    G = nx.erdos_renyi_graph(n, p)    
    return(G)

def square_graph(size):
    # Define the size of the square lattice 
    G = nx.grid_2d_graph(size,size)
    return (G)

def rectangle_graph(rows,cols):
    # Define the size of the square lattice 
    G = nx.grid_2d_graph(rows,cols)
    return (G)

def plotting_square(G):
    import src.graphs as graph

    size=int(np.sqrt(G.number_of_nodes()))

    mapping = {}
    counter = 1
    for i in range(size):
        for j in range(size):
            mapping[(i, j)] = counter
            counter += 1

    #Create labels using the mapping
    labels = {node: str(mapping[node]) for node in G.nodes()}

    #Draw the 2D grid graph
    pos = dict((n, n) for n in G.nodes())  # Use node coordinates as positions

    nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color="lightblue", font_size=10, font_weight="bold", edge_color="gray")


def plotting_random(G):
    import src.graphs as graph
    
    n=G.number_of_nodes()
    pos = nx.spring_layout(G)

    #Draw the 2D grid graph
    pos = nx.spring_layout(G)  # Position nodes using Fruchterman-Reingold force-directed algorithm

    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold", edge_color="gray")
