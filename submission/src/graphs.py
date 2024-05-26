import networkx as nx
import matplotlib.pyplot as plt

def draw_random_graph(n, p):
    # Generate a random graph
    G = nx.erdos_renyi_graph(n, p)    
    return(G)

def square_graph(size):
    # Define the size of the square lattice 
    rows, cols = size, size
    G = nx.grid_2d_graph(rows, cols)
    return (G)

def plotting_square(G,size):
    import src.graphs as graph
    # G=graph.square_graph(size)

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


def plotting_random():
    import src.graphs as graph
    G=graph.square_graph(size)

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
