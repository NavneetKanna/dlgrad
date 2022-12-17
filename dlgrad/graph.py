import networkx as nx
import os
nodes = {}


# A better way to write this ??

# One of the input can be none such as in activation functions
def add_nodes(op, output, input1, input2=None):
    # Check if the output and input1/input2 object is the same 
    if nodes:
        if nodes[list(nodes.keys())[-1]][-1] in [input1, input2]:
            if input2:
                nodes[op] = [input1, input2, output]
            else:
                nodes[op] = [input1, output]
    else:
        if input2:
            nodes[op] = [input1, input2, output]
        else:
            nodes[op] = [input1, output]
    

def draw_cg():
    G = nx.DiGraph() 
    for i in nodes:
        G.add_node(i)
        G.add_edges_from([nodes[i]])
    nx.drawing.nx_pydot.write_dot(G, 'net.dot')
    os.system('dot -Tsvg net.dot -o net.svg')

