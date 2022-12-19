import networkx as nx
import pydot
import os
nodes = {}
edges = []
nodess = []
labels = {}
idx = 0
# A better way to write this ??
# Should we check object or shape ??

# One of the input can be none such as in activation functions
def add_nodes(op, output, input1, input2=None):
    # Check if the output of the previous and input1/input2 shape is the same 
    if nodes:
        last_op = list(nodes.keys())[-1]
        if op == 'Relu':
            nodes[f'{last_op} + Relu'] = nodes.pop(last_op)
            last_op = list(nodes.keys())[-1]
            return
        if nodes[last_op][-1] in [input1, input2]:
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
        for j in nodes[i][:-1]:
            edges.append((j, i))
       
        edges.append((i, nodes[i][-1]))

    G.add_edges_from(edges) 
    nx.drawing.nx_pydot.write_dot(G, r'/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad/net.dot')
    (graph,) = pydot.graph_from_dot_file(r'/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad/net.dot')
    graph.write_png(r'/mnt/c/Users/navne/Documents/vs_code/dlgrad_main/dlgrad/net.png')

